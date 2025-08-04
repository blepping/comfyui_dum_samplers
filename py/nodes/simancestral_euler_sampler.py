# By https://github.com/blepping
# LICENSE: Apache2
# Usage: Place this file in the custom_nodes directory and restart ComfyUI+refresh browser.
#        It will add a SimilarityAncestralEulerSampler node that can be used with SamplerCustom, etc.
from __future__ import annotations

from typing import Callable, NamedTuple

import torch
from comfy import model_patcher, model_sampling
from comfy.samplers import KSAMPLER
from tqdm import tqdm
from tqdm.auto import trange

from .. import utils
from .base import DumInputTypes, DumLazyInputTypes


class Config(NamedTuple):
    first_ancestral_step: int = 0
    last_ancestral_step: int = -1
    blend_mode: str = "lerp"
    dim: int = 1
    flatten: bool = False
    absolute: bool = False
    flipped: bool = False
    similarity_offset: float = 0.0
    similarity_multiplier: float = 1.0
    pingpong_threshold_high: float = -1.0
    pingpong_threshold_low: float = -1.0
    similarity_mode: str = "scaled"
    step_scale: float = 1.0
    target_a: str = "uncond"
    target_b: str = "cond"
    operation_a: Callable | None = None
    operation_b: Callable | None = None
    operation_sim: Callable | None = None


# Modifiedfrom ComfyUI.
def _get_ancestral_step_simple(
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_up = sigma_next.minimum(
        eta * (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2) ** 0.5,
    )
    sigma_down = (sigma_next**2 - sigma_up**2) ** 0.5
    eta_mask = eta > 0.0
    sigma_up = torch.where(eta_mask, sigma_up, 0.0)
    sigma_down = torch.where(eta_mask, sigma_down, sigma_next)
    return sigma_down, sigma_up


def get_ancestral_step_ext(
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float | torch.Tensor = 1.0,
    is_rf=False,
):
    orig_dtype = sigma.dtype
    sigma_next_orig = sigma_next
    sigma = sigma.to(dtype=torch.float64)
    sigma_next = sigma_next.to(dtype=torch.float64)
    if not isinstance(eta, torch.Tensor):
        if eta == 0:
            return sigma_next_orig, sigma_next_orig * 0, 1.0
        eta = sigma.new_full(sigma.shape, 1.0)
    if eta.allclose(eta.new_full((), 0.0)) or sigma_next.allclose(
        sigma_next.new_full((), 0.0),
    ):
        return sigma_next_orig, sigma_next_orig * 0, 1.0
    if eta.dtype != torch.float64:
        eta = eta.to(dtype=torch.float64)
    if not is_rf:
        sigma_down, sigma_up = _get_ancestral_step_simple(sigma, sigma_next, eta=eta)
        return sigma_down.to(dtype=orig_dtype), sigma_up.to(dtype=orig_dtype), 1.0
    # Referenced from ComfyUI.
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_ip1, alpha_down = 1.0 - sigma_next, 1.0 - sigma_down
    sigma_up = (sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2) ** 0.5
    x_coeff = alpha_ip1 / alpha_down
    eta_mask = eta > 0.0
    sigma_up = torch.where(eta_mask, sigma_up, 0.0).to(dtype=orig_dtype)
    sigma_down = torch.where(eta_mask, sigma_down, sigma_next).to(dtype=orig_dtype)
    x_coeff = torch.where(eta_mask, x_coeff, 1.0).to(dtype=orig_dtype)
    return sigma_down, sigma_up, x_coeff


def internal_step(
    x: torch.Tensor,
    denoised: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    noise_sampler=None,
    sigma_down: float | torch.Tensor | None = None,
    sigma_up: float | torch.Tensor = 0.0,
    x_coeff: float | torch.Tensor = 1.0,
    blend_function=torch.lerp,
) -> torch.Tensor:
    x = blend_function(
        denoised,
        x,
        (sigma_down if sigma_down is not None else sigma_next) / sigma,
    )
    if (
        noise_sampler is None  # noqa: PLR0916
        or sigma_down is None
        or (isinstance(sigma_up, float) and sigma_up == 0.0)
        or (
            isinstance(sigma_up, torch.Tensor)
            and sigma_up.allclose(sigma_up.new_full((), 0.0))
        )
    ):
        return x
    noise = noise_sampler(sigma, sigma_next).mul_(sigma_up)
    if isinstance(x_coeff, torch.Tensor) or x_coeff != 1:
        # x gets scaled for flow models.
        x *= x_coeff
    return x.add_(noise)


def fix_step_range(steps, start, end):
    if start < 0:
        start = steps + start
    if end < 0:
        end = steps + end
    start = max(0, min(steps - 1, start))
    end = max(0, min(steps - 1, end))
    return (end, start) if start > end else (start, end)


class SimAncestralEulerSampler:
    def __init__(
        self,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        noise_sampler=None,
        eta=1.0,
        s_noise=1.0,
        blep_sa_config=None,
        verbose=False,
        **_kwargs: dict,
    ):
        self.model_ = model
        self.sigmas = sigmas
        self.x = x
        self.s_in = x.new_ones((x.shape[0],))
        self.extra_args = extra_args if extra_args is not None else {}
        self.disable = disable
        self.callback_ = callback
        self.config = blep_sa_config if blep_sa_config is not None else Config()
        self.verbose = verbose
        self.eta = max(0.0, eta)
        self.s_noise = s_noise
        self.is_rf = isinstance(
            model.inner_model.inner_model.model_sampling,
            model_sampling.CONST,
        )
        if (
            self.eta == 0
            and self.config.pingpong_threshold_high < 0
            and self.config.pingpong_threshold_low < 0
        ):
            self.noise_sampler = None
        else:
            if noise_sampler is None:

                def noise_sampler(*_unused: list) -> torch.Tensor:
                    return torch.randn_like(x)

            self.noise_sampler = noise_sampler

    @classmethod
    def go(
        cls,
        model,
        x,
        sigmas: torch.Tensor,
        extra_args=None,
        callback=None,
        disable=None,
        noise_sampler=None,
        eta=1.0,
        s_noise=1.0,
        verbose=False,
        **kwargs: dict,
    ):
        return cls(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            noise_sampler=noise_sampler,
            eta=eta,
            s_noise=s_noise,
            verbose=verbose,
            **kwargs,
        )()

    def model(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        *,
        simple_mode: bool = False,
        **kwargs: dict,
    ) -> tuple:
        if simple_mode:
            return (
                self.model_(x, sigma * self.s_in, **self.extra_args, **kwargs),
                None,
                None,
            )
        cond = uncond = None

        def post_cfg_function(args):
            nonlocal cond, uncond
            cond, uncond = args["cond_denoised"], args["uncond_denoised"]
            return args["denoised"]

        extra_args = self.extra_args.copy()
        orig_model_options = extra_args.get("model_options", {})
        model_options = orig_model_options.copy()
        model_options["disable_cfg1_optimization"] = True
        extra_args["model_options"] = model_patcher.set_model_options_post_cfg_function(
            model_options,
            post_cfg_function,
        )
        denoised = self.model_(x, sigma * self.s_in, **extra_args, **kwargs)
        return denoised, cond, uncond

    def callback(
        self,
        idx: int,
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoised: torch.Tensor,
    ) -> None:
        if self.callback_ is None:
            return
        self.callback_({
            "i": idx,
            "x": x,
            "sigma": sigma,
            "sigma_hat": sigma,
            "denoised": denoised,
        })

    def __call__(self) -> torch.Tensor:
        x = self.x
        noise_sampler = self.noise_sampler
        config = self.config
        smode = config.similarity_mode
        abs_mode = config.absolute
        flipped = config.flipped
        if smode in {"simple", "simple_zeromean"}:
            flipped = not flipped
        blend_function = utils.BLENDING_MODES[config.blend_mode]
        steps = len(self.sigmas) - 1
        first_ancestral_step, last_ancestral_step = fix_step_range(
            steps,
            config.first_ancestral_step,
            config.last_ancestral_step,
        )
        denoised_prev = uncond_prev = cond_prev = x_prev = None
        for idx in trange(steps, disable=self.disable):
            target_a_s = target_b_s = sim = x_next = denoised = uncond = cond = eta = (
                None
            )
            sigma, sigma_next = self.sigmas[idx : idx + 2]
            sigma_next = (
                sigma + (sigma_next - sigma) * config.step_scale
                if sigma_next != 0
                else sigma_next
            )
            use_eta = noise_sampler is not None and (
                first_ancestral_step <= idx <= last_ancestral_step
            )
            denoised, cond, uncond = self.model(x, sigma, simple_mode=not use_eta)
            if sigma_next <= 1e-06:
                return denoised
            targets = {
                "denoised": denoised,
                "cond": cond,
                "uncond": uncond,
                "x": x / sigma,
                "denoised_prev": denoised_prev,
                "cond_prev": cond_prev,
                "uncond_prev": uncond_prev,
                "x_prev": x_prev,
            }
            denoised_prev = denoised.clone()
            cond_prev = cond.clone() if cond is not None else None
            uncond_prev = uncond.clone() if uncond is not None else None
            x_prev = targets["x"].clone()
            self.callback(idx, x, sigma, denoised)
            if not use_eta or (
                self.eta == 0
                and config.pingpong_threshold_high < 0
                and config.pingpong_threshold_low < 0
            ):
                x = internal_step(
                    x,
                    denoised,
                    sigma,
                    sigma_next,
                    blend_function=blend_function,
                )
                continue
            target_a = targets.get(config.target_a)
            target_b = targets.get(config.target_b)
            if target_a is not None and config.operation_a is not None:
                target_a = config.operation_a(latent=target_a)
            if target_b is not None and config.operation_b is not None:
                target_b = config.operation_b(latent=target_b)
            del targets
            if (
                target_a is not None
                and target_b is not None
                and (target_a.shape, target_a.dtype) == (denoised.shape, denoised.dtype)
                and (target_b.shape, target_b.dtype) == (denoised.shape, denoised.dtype)
            ):
                target_a_s = (
                    target_a.flatten(start_dim=config.dim)
                    if config.flatten
                    else target_a
                )
                target_b_s = (
                    target_b.flatten(start_dim=config.dim)
                    if config.flatten
                    else target_b
                )
                del target_a, target_b
                if smode in {"pearson_correlation", "simple_zeromean"}:
                    target_a_s -= target_a_s.mean(
                        dim=-1 if config.flatten else config.dim,
                        keepdim=True,
                    )
                    target_b_s -= target_b_s.mean(
                        dim=-1 if config.flatten else config.dim,
                        keepdim=True,
                    )
                if smode in {"simple", "simple_zeromean"}:
                    sim = target_a_s - target_b_s
                    sim -= sim.amin(dim=tuple(range(1, sim.ndim)), keepdim=True)
                    sim *= 1.0 / (
                        sim.amax(dim=tuple(range(1, sim.ndim)), keepdim=True) + 1e-07
                    )
                    sim = sim.sub_(0.5).mul_(2.0)
                else:
                    sim = torch.cosine_similarity(
                        target_a_s,
                        target_b_s,
                        dim=-1 if config.flatten else config.dim,
                    )
                sim = sim.clamp_(-1.0, 1.0)
                sim = sim.abs_() if abs_mode else sim.mul_(0.5).add_(0.5)
                if flipped:
                    sim = 1.0 - sim
                need_clamp = False
                if config.similarity_offset != 0:
                    sim = sim.add_(config.similarity_offset)
                    need_clamp = True
                if config.similarity_multiplier != 1:
                    sim *= config.similarity_multiplier
                    need_clamp = True
                if smode in {"simple", "simple_zeromean"}:
                    sim = sim.reshape(denoised.shape)
                elif config.flatten:
                    sim = sim.reshape(
                        *denoised.shape[: config.dim],
                        *((1,) * (denoised.ndim - config.dim)),
                    )
                else:
                    sim = sim.unsqueeze(config.dim)
                if need_clamp:
                    sim = sim.clamp_(0.0, 1.0)
                    need_clamp = False
                if config.operation_sim is not None:
                    sim = config.operation_sim(latent=sim)
                    need_clamp = True
                if need_clamp:
                    sim = sim.clamp_(0.0, 1.0)
                if self.verbose:
                    tqdm.write(
                        f"SimAncestralSampler: sim mean={sim.mean().item():.4f}, min={sim.min().item():.4f}, max={sim.max().item():.4f}, shape={sim.shape}",
                    )
                eta = self.eta * sim
            else:
                sim = None
                eta = 0.0
            if isinstance(eta, torch.Tensor):
                sigma_down, sigma_up, x_coeff = get_ancestral_step_ext(
                    sigma,
                    sigma_next,
                    eta=eta,
                    is_rf=self.is_rf,
                )
                sigma_up *= self.s_noise
                x_next = internal_step(
                    x,
                    denoised,
                    sigma,
                    sigma_next,
                    sigma_down=sigma_down,
                    sigma_up=sigma_up,
                    x_coeff=x_coeff,
                    noise_sampler=noise_sampler,
                    blend_function=blend_function,
                )
            else:
                x_next = internal_step(
                    x,
                    denoised,
                    sigma,
                    sigma_next,
                    blend_function=blend_function,
                )
            if sim is None or (
                config.pingpong_threshold_high < 0 and config.pingpong_threshold_low < 0
            ):
                x = x_next
                continue
            ppmask = torch.full(eta.shape, False, dtype=torch.bool, device=eta.device)  # noqa: FBT003
            if config.pingpong_threshold_low >= 0:
                ppmask |= sim <= config.pingpong_threshold_low
            if config.pingpong_threshold_high >= 0:
                ppmask |= sim >= config.pingpong_threshold_high
            if not torch.any(ppmask):
                del ppmask
                x = x_next
                continue
            noise = noise_sampler(sigma, sigma_next).mul_(self.s_noise)
            x_pp = (
                blend_function(denoised, noise, sigma_next)
                if self.is_rf
                else denoised.add_(noise.mul_(sigma_next))
            )
            x = torch.where(ppmask, x_pp, x_next)
            del x_pp
        return x


class SimilarityAncestralEulerSamplerNode:
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    _valid_targets = (
        "cond",
        "uncond",
        "denoised",
        "x",
        "cond_prev",
        "uncond_prev",
        "denoised_prev",
        "x_prev",
    )
    INPUT_TYPES = DumLazyInputTypes(
        lambda valid_targets=_valid_targets: DumInputTypes()
        .req_float_eta(
            default=1.0,
            min=0.0,
            tooltip="This controls ancestralness. ETA will get multiplied with the similarity for non-pingpong steps so this is going to be the upper bound on ancestralness. You can set this to 0 to only use ancestralness through the pingpong thresholds.",
        )
        .req_float_s_noise(
            default=1.0,
            tooltip="Scale for added noise.",
        )
        .req_selectblend_blend_mode(
            tooltip="Blending function used when calculating steps. LERP is basically the only thing that works well, but experimenting is possible.",
        )
        .req_float_step_scale(
            default=1.0,
            tooltip="Generally should be left at 1.0. Setting it to a value over 1.0 will remove more noise than expected, setting it to a value under 1.0 will remove less. A little goes a long way, if you want to change it try an increment of something like 0.01 or less.",
        )
        .req_float_pingpong_threshold_low(
            default=1.0,
            min=-1.0,
            tooltip="Threshold (inclusive) for doing a pingpong step. Pingpong steps completely replace the noise. Uses whatever granularity you have set for similarity. Any negative value disables using the threshold, otherwise it's considered active where similarity is less or equal to this value.",
        )
        .req_float_pingpong_threshold_high(
            default=1.0,
            min=-1.0,
            tooltip="Threshold (inclusive) for doing a pingpong step. Pingpong steps completely replace the noise. Uses whatever granularity you have set for similarity. Any negative value disables using the threshold, otherwise it's considered active where similarity is greater or equal to this value.",
        )
        .req_int_dim(
            default=1,
            tooltip="Controls the dimension where flattening starts and for pearson_correlation and cosine_similarity modes what dimension the calculation uses. For simple modes, this only affects flattening.",
        )
        .req_bool_flatten(
            default=True,
            tooltip="Determines whether the tensor is flattened starting from the specified dimension.",
        )
        .req_bool_flipped(
            default=True,
            tooltip="When enabled, the similarity is flipped. I.E. exactly the same (1.0) becomes 0.0.",
        )
        .req_bool_absolute(tooltip="Use absolute values when calculating similarity.")
        .req_int_first_ancestral_step(
            default=1,
            tooltip="First step ancestralness will be enabled. When negative, steps count from the end. Note: If the last step is greater than the first then first/last will be swapped.",
        )
        .req_int_last_ancestral_step(
            default=-1,
            tooltip="Last step ancestralness will be enabled. When negative, steps count from the end. Note: If the last step is greater than the first then first/last will be swapped.",
        )
        .req_field_similarity_mode(
            (
                "cosine_similarity",
                "pearson_correlation",
                "simple",
                "simple_zeromean",
            ),
            default="simple_zeromean",
            tooltip="pearson_correlation and simple_zeromean modes subtract the mean before calculating similarity. The simple modes just subtract cond from uncond and put the result on a scale of -1.0 to 1.0.",
        )
        .req_float_similarity_offset(
            default=0.0,
            tooltip="This is just added to the similarity after all other rescaling/flipping.",
        )
        .req_float_similarity_multiplier(
            default=1.0,
            tooltip="Multiplier applied to similarity after rescaling/flipping and the offset. After multiplying, the result is clamped to be between 0.0 and 1.0.",
        )
        .req_field_target_a(
            valid_targets,
            default="uncond",
            tooltip="First target for comparisons.",
        )
        .req_field_target_b(
            valid_targets,
            default="cond",
            tooltip="Second target for comparisons.",
        )
        .opt_field_operation_a(
            "LATENT_OPERATION",
            tooltip="Optional latent operation to be applied to target_a.",
        )
        .opt_field_operation_b(
            "LATENT_OPERATION",
            tooltip="Optional latent operation to be applied to target_b.",
        )
        .opt_field_operation_sim(
            "LATENT_OPERATION",
            tooltip="Optional latent operation to be applied to the calculated similarity. Note: Most latent operations probably cannot deal with flattened shapes.",
        ),
    )

    @classmethod
    def go(cls, *, eta, s_noise, **kwargs: dict):
        options = {
            "eta": eta,
            "s_noise": s_noise,
            "blep_sa_config": Config(**kwargs),
        }
        return (KSAMPLER(SimAncestralEulerSampler.go, extra_options=options),)


NODE_CLASS_MAPPINGS = {
    "SimilarityAncestralEulerSampler": SimilarityAncestralEulerSamplerNode,
}

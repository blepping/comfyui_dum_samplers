# By https://github.com/blepping
# LICENSE: Apache2

from typing import NamedTuple

import torch
from comfy import model_sampling
from comfy.samplers import KSAMPLER
from tqdm import tqdm
from tqdm.auto import trange

from .. import utils
from .base import DumInputTypes, DumLazyInputTypes


class Config(NamedTuple):
    min_blend: float = 0.0
    max_blend: float = 1.0
    first_clamp_step: int = 0
    last_clamp_step: int = -1
    first_ancestral_step: int = 0
    last_ancestral_step: int = -1
    blend_mode: str = "lerp"
    dim: int = 1
    flatten: bool = False
    similarity_multiplier: float = 1.0
    history_mode: str = "blended"
    similarity_mode: str = "scaled"


class SimClampEulerSampler:
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
        blep_sc_config=None,
        verbose=False,
        **_kwargs: dict,
    ):
        if blep_sc_config is None:
            blep_sc_config = Config()
        self.model_ = model
        self.sigmas = sigmas
        self.x = x
        self.s_in = x.new_ones((x.shape[0],))
        self.extra_args = extra_args if extra_args is not None else {}
        self.disable = disable
        self.callback_ = callback
        self.config = blep_sc_config
        self.verbose = verbose
        self.eta = max(0.0, eta)
        self.s_noise = s_noise
        self.is_rf = isinstance(
            model.inner_model.inner_model.model_sampling,
            model_sampling.CONST,
        )
        if self.eta == 0:
            self.noise_sampler = None
        else:
            if noise_sampler is None:

                def noise_sampler(*_unused: list):
                    return torch.randn_like(x)

            self.noise_sampler = noise_sampler

    @classmethod
    def go(
        cls,
        model,
        x,
        sigmas,
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

    def model(self, x, sigma, **kwargs: dict):
        return self.model_(x, sigma * self.s_in, **self.extra_args, **kwargs)

    def callback(self, idx, x, sigma, denoised):
        if self.callback_ is None:
            return
        self.callback_({
            "i": idx,
            "x": x,
            "sigma": sigma,
            "sigma_hat": sigma,
            "denoised": denoised,
        })

    def __call__(self):
        x = self.x
        noise_sampler = self.noise_sampler
        config = self.config
        blend = utils.BLENDING_MODES[config.blend_mode]
        denoised_prev = None
        steps = len(self.sigmas) - 1
        first_clamp_step, last_clamp_step = utils.fix_step_range(
            steps,
            config.first_clamp_step,
            config.last_clamp_step,
        )
        first_ancestral_step, last_ancestral_step = utils.fix_step_range(
            steps,
            config.first_ancestral_step,
            config.last_ancestral_step,
        )
        for idx in trange(steps, disable=self.disable):
            sigma, sigma_next = self.sigmas[idx : idx + 2]
            use_eta = noise_sampler is not None and (
                first_ancestral_step <= idx <= last_ancestral_step
            )
            use_clamp = (
                config.similarity_multiplier != 0
                and first_clamp_step <= idx <= last_clamp_step
            )
            sigma_down, sigma_up, x_coeff = utils.get_ancestral_step_ext(
                sigma,
                sigma_next,
                eta=self.eta if use_eta else 0.0,
                is_rf=self.is_rf,
            )
            sigma_up *= self.s_noise
            orig_denoised = denoised = self.model(x, sigma)
            if idx > 0 and use_clamp:
                denoised_s = (
                    denoised.flatten(start_dim=config.dim)
                    if config.flatten
                    else denoised
                )
                denoised_prev_s = (
                    denoised_prev.flatten(start_dim=config.dim)
                    if config.flatten
                    else denoised_prev
                )
                if config.similarity_mode.startswith("pearson"):
                    denoised_s = denoised_s - denoised_s.mean(  # noqa: PLR6104
                        dim=-1 if config.flatten else config.dim,
                        keepdim=True,
                    )
                    denoised_prev_s = denoised_prev_s - denoised_prev_s.mean(  # noqa: PLR6104
                        dim=-1 if config.flatten else config.dim,
                        keepdim=True,
                    )
                sim = torch.cosine_similarity(
                    denoised_s,
                    denoised_prev_s,
                    dim=-1 if config.flatten else config.dim,
                )
                del denoised_s, denoised_prev_s
                if config.similarity_mode in {
                    "scaled",
                    "scaled_flipped",
                    "pearson_scaled_flipped",
                    "pearson_scaled",
                }:
                    sim = 0.5 + sim * 0.5
                else:
                    sim = sim.abs()
                if config.similarity_mode.endswith("_flipped"):
                    sim = 1.0 - sim
                if config.flatten:
                    sim = sim.reshape(
                        *denoised.shape[: config.dim],
                        *((1,) * (denoised.ndim - config.dim)),
                    )
                else:
                    sim = sim.unsqueeze(config.dim)
                if self.verbose:
                    tqdm.write(
                        f"SimClampSampler: sim={sim.flatten(start_dim=min(sim.ndim - 1, 1)).mean().item():.4f}, sim.shape={sim.shape}",
                    )
                if config.similarity_multiplier != 1.0:
                    sim *= abs(config.similarity_multiplier)
                sim = sim.clamp(config.min_blend, config.max_blend)
                denoised = (
                    blend(denoised_prev, denoised, sim)
                    if config.similarity_multiplier >= 0
                    else blend(denoised, denoised_prev, sim)
                )
            denoised_prev = (
                denoised if config.history_mode == "blended" else orig_denoised
            )
            if sigma_next <= 1e-06:
                return denoised
            self.callback(idx, x, sigma, denoised)
            x = utils.internal_step(
                x,
                denoised,
                sigma,
                sigma_next,
                sigma_down,
                sigma_up,
                x_coeff,
                noise_sampler,
            )
        return x


class SimilarityClampEulerSamplerNode:
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    INPUT_TYPES = DumLazyInputTypes(
        lambda: DumInputTypes()
        .req_float_eta(
            default=0.0,
            min=0.0,
            tooltip="This controls ancestralness. ETA will get multiplied with the similarity for non-pingpong steps so this is going to be the upper bound on ancestralness. You can set this to 0 to only use ancestralness through the pingpong thresholds.",
        )
        .req_float_s_noise(
            default=1.0,
            tooltip="Scale for added noise.",
        )
        .req_float_min_blend(
            default=0.5,
            min=0.0,
            max=1.0,
        )
        .req_float_max_blend(
            default=1.0,
            min=0.0,
            max=1.0,
        )
        .req_selectblend_blend_mode()
        .req_int_dim(default=1)
        .req_bool_flatten(default=True)
        .req_float_similarity_multiplier(
            default=1.0,
            tooltip="If this is negative, the arguments to the blend function will be flipped. The similarity is multiplied by the absolute value here.",
        )
        .req_int_first_clamp_step(
            default=1,
            tooltip="First step the clamp function will be applied. When negative, steps count from the end.",
        )
        .req_int_last_clamp_step(
            default=-2,
            tooltip="Last step the clamp function will be applied. When negative, steps count from the end.",
        )
        .req_int_first_ancestral_step(
            default=0,
            tooltip="First step ancestralness will be enabled. When negative, steps count from the end.",
        )
        .req_int_last_ancestral_step(
            default=-2,
            tooltip="Last step ancestralness will be enabled. When negative, steps count from the end.",
        )
        .req_field_history_mode(
            ("blended", "original"),
            default="blended",
            tooltip="original mode uses the raw output from the model (denoised), blended uses the result after clamp blending.",
        )
        .req_field_similarity_mode(
            (
                "scaled",
                "absolute",
                "scaled_flipped",
                "absolute_flipped",
                "pearson_scaled",
                "pearson_absolute",
                "pearson_scaled_flipped",
                "pearson_absolute_flipped",
            ),
            default="scaled",
            tooltip="scaled: Puts the cosine similarity on a scale of 0 to 1.\nabsolute: Uses the absolute value, so -1 similarity becomes 1.\nscaled_flipped: Just gets reversed with 1.0 - similarity.\nabsolute_flipped: Same as above.\npearson_*: These variants subtract the mean before doing cosine similarity, otherwise they are similar to the other options.",
        ),
    )

    @classmethod
    def go(cls, *, eta, s_noise, **kwargs: dict):
        options = {
            "eta": eta,
            "s_noise": s_noise,
            "blep_sc_config": Config(**kwargs),
        }
        return (KSAMPLER(SimClampEulerSampler.go, extra_options=options),)


NODE_CLASS_MAPPINGS = {
    "SimilarityClampEulerSampler": SimilarityClampEulerSamplerNode,
}

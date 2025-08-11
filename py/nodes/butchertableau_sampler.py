# By https://github.com/blepping
# LICENSE: Apache2

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import yaml
from comfy import model_sampling
from comfy.samplers import KSAMPLER
from tqdm.auto import trange

from .. import utils
from .base import DumInputTypes, DumLazyInputTypes

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable


class ButcherTableau(NamedTuple):
    a: Sequence[Sequence]  # stage coefficients
    b: Sequence  # final weights
    c: Sequence  # time coefficients

    @staticmethod
    def fixup_ratio(val: str | float) -> float:
        if isinstance(val, float):
            return val
        if isinstance(val, int):
            return float(val)
        if isinstance(val, str):
            vs = val.split("/")
            if len(vs) != 2:
                raise ValueError("String values must be in the format number/number")
            return float(vs[0]) / float(vs[1])
        raise ValueError("Values must be float or string in the format number/number")

    @classmethod
    def from_dict(cls, d: dict) -> ButcherTableau:
        if "a" not in d or "b" not in d or "c" not in d:
            raise ValueError("Missing keys, requires a, b and c")
        return ButcherTableau(**{
            k: tupleize(d[k], fixup=cls.fixup_ratio) for k in ("a", "b", "c")
        })


def tupleize(val, *, fixup: Callable | None = None):
    if isinstance(val, dict):
        return {k: tupleize(v, fixup=fixup) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return tuple(tupleize(v, fixup=fixup) for v in val)
    return val if fixup is None else fixup(val)


class Config(NamedTuple):
    tableau: ButcherTableau
    first_ancestral_step: int = 0
    last_ancestral_step: int = -1
    first_blend_step: int = 0
    last_blend_step: int = -1
    step_scale: float = 1.0
    blend_mode: str = "NO_BLEND"
    blend_mode_internal: str | None = None
    step_to_zero: bool = False


class ButcherTableauSampler:
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
        blep_config=None,
        verbose=False,
        **_kwargs: dict,
    ):
        if blep_config is None:
            blep_config = Config()
        self.model_ = model
        self.sigmas = sigmas
        self.x = x
        self.s_in = x.new_ones((x.shape[0],))
        self.extra_args = extra_args if extra_args is not None else {}
        self.disable = disable
        self.callback_ = callback
        self.config = blep_config
        self.verbose = verbose
        self.is_rf = isinstance(
            model.inner_model.inner_model.model_sampling,
            model_sampling.CONST,
        )
        self.eta = max(0.0, eta)
        self.s_noise = s_noise
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
            eta=eta,
            s_noise=s_noise,
            noise_sampler=noise_sampler,
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

    @staticmethod
    def _sigma_view_for_tensor(
        sigma_scalar: float | torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(sigma_scalar, float) or sigma_scalar.ndim == 0:
            # scalar -> make shaped tensor on same device/dtype
            return torch.full(
                (target_tensor.shape[0],) + (1,) * (target_tensor.ndim - 1),
                float(sigma_scalar),
                device=target_tensor.device,
                dtype=target_tensor.dtype,
            )
        # assume shape (B,) or (B,1,...) already; reduce to (B,)
        s = sigma_scalar
        if s.ndim > 1:
            s = s.view(s.shape[0])
        return s.view(s.shape[0], *([1] * (target_tensor.ndim - 1)))

    def butcher_tableau_step(
        self,
        step: int,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        min_sigma: float = 1e-05,
    ) -> torch.Tensor:
        tableau = self.config.tableau

        h = sigma_next - sigma  # integration step in sigma-space (could be negative)
        accum = torch.zeros_like(x)
        did_callback = False

        ks = []  # k_i tensors
        # Pre-allocate a scratch for intermediate accumulators where useful
        # We'll compute each stage exactly as RK requires.
        c_len = len(tableau.c)
        for i in trange(
            c_len,
            disable=self.disable or c_len < 2,
            leave=False,
            desc="    substep",
        ):
            ci = tableau.c[i]
            sigma_i = sigma + ci * h  # scalar or tensor (B,)
            sigma_i_view = self._sigma_view_for_tensor(sigma_i, x)

            if i == 0:
                # stage 0 uses x itself (no previous ks)
                x_stage = x
            else:
                if i > 1:
                    accum.zero_()
                a_row = tableau.a[i]
                # a_row should have length i (or less; missing entries treated as 0)
                for j, a_ij in enumerate(a_row):
                    if a_ij != 0.0:
                        accum += a_ij * ks[j]
                x_stage = (h * accum).add_(x)

            # Evaluate model at the stage (mathematically must be at sigma_i)
            denoised = self.model(x_stage, sigma_i)
            if not did_callback:
                self.callback(step, x, sigma, denoised)
                did_callback = True

            # Compute derivative f(x_stage, sigma_i) = (x_stage - denoised)/sigma_i
            # Avoid dividing by extremely small sigma -> clamp for safety
            sigma_for_div = sigma_i_view.clamp(min=min_sigma)
            k_i = (x_stage - denoised).div_(sigma_for_div)

            ks.append(k_i)

        # Combine stages: x_next = x + h * sum_i b_i * k_i
        accum.zero_()
        for bi, ki in zip(tableau.b, ks):
            if bi != 0.0:
                accum += bi * ki
        return (h * accum).add_(x)

    def butcher_tableau_step_blend(
        self,
        step: int,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        blend_function=torch.lerp,
        blend_function_internal: Callable | None = None,
    ):
        tableau = self.config.tableau
        ratio = sigma_next / sigma
        did_callback = False
        k = []
        accum = torch.zeros_like(x)
        denoised_cache = []
        blend_function_internal = blend_function_internal or blend_function

        c_len = len(tableau.c)
        for i in trange(
            c_len,
            disable=self.disable or c_len < 2,
            leave=False,
            desc="    substep",
        ):
            ci = tableau.c[i]
            # Compute intermediate x
            if i == 0:
                x_stage = x
                sigma_stage = sigma
            else:
                if i > 1:
                    accum.zero_()
                # Weighted combination of previous ks
                for j, a_ij in enumerate(tableau.a[i]):
                    if a_ij != 0:
                        accum += a_ij * (denoised_cache[j] - x)
                x_stage = blend_function_internal(x + accum, x, ratio)
                sigma_stage = sigma + ci * (sigma_next - sigma)

            denoised = self.model(x_stage, sigma_stage)
            if not did_callback:
                self.callback(step, x, sigma, denoised)
                did_callback = True
            denoised_cache.append(denoised)
            k.append(denoised - x)  # storing as offset from x

        # Combine stages using b coefficients
        accum.zero_()
        for bi, ki in zip(tableau.b, k):
            accum += bi * ki

        # Final LERP to sigma_next
        return blend_function(x + accum, x, ratio)

    def __call__(self):
        config = self.config
        x = self.x
        sigmas = self.sigmas
        noise_sampler = self.noise_sampler
        steps = len(self.sigmas) - 1
        blend_function = (
            None
            if config.blend_mode == "NO_BLEND"
            else utils.BLENDING_MODES[config.blend_mode]
        )
        blend_function_internal = (
            None
            if blend_function is None or config.blend_mode_internal is None
            else utils.BLENDING_MODES[config.blend_mode_internal]
        )
        first_ancestral_step, last_ancestral_step = utils.fix_step_range(
            steps,
            config.first_ancestral_step,
            config.last_ancestral_step,
        )
        first_blend_step, last_blend_step = utils.fix_step_range(
            steps,
            config.first_blend_step,
            config.last_blend_step,
        )
        for idx in trange(steps, disable=self.disable):
            sigma, sigma_next = sigmas[idx : idx + 2]
            if sigma_next == 0 and not config.step_to_zero:
                return self.model(x, sigma)
            sigma_next = (
                sigma + (sigma_next - sigma) * config.step_scale
                if sigma_next != 0
                else sigma_next
            )
            use_eta = (
                noise_sampler is not None
                and sigma_next != 0
                and (first_ancestral_step <= idx <= last_ancestral_step)
            )
            use_blend = blend_function is not None and (
                first_blend_step <= idx <= last_blend_step
            )
            sigma_down, sigma_up, x_coeff = utils.get_ancestral_step_ext(
                sigma,
                sigma_next,
                eta=self.eta if use_eta else 0.0,
                is_rf=self.is_rf,
            )
            sigma_up *= self.s_noise
            x = utils.internal_step(
                x=self.butcher_tableau_step(
                    idx,
                    x,
                    sigma,
                    sigma_down,
                )
                if not use_blend
                else self.butcher_tableau_step_blend(
                    idx,
                    x,
                    sigma,
                    sigma_down,
                    blend_function=blend_function,
                    blend_function_internal=blend_function_internal,
                ),
                denoised=None,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_down=sigma_down,
                sigma_up=sigma_up,
                x_coeff=x_coeff,
                noise_sampler=noise_sampler,
                blend_function=None,
            )
        return x


class ButcherTableauSamplerNode:
    DESCRIPTION = "Sampler that allows manually specifying a Butcher tableau to define the steps. The node will also output a string with the preset tableau definitions."
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER", "STRING")
    FUNCTION = "go"

    _tableau_defaults = """
euler: &euler
    a: [[0]]
    b: [1]
    c: [0]

heun: &heun
    a: [[0], [1]]
    b: [0.5, 0.5]
    c: [0, 1]

midpoint: &midpoint
    a: [[0], [0.5]]
    b: [0, 1]
    c: [0, 0.5]

rk4: &rk4
    a:
        - [0]
        - [0.5]
        - [0, 0.5]
        - [0, 0, 1]
    b: ["1/6", "1/3", "1/3", "1/6"]
    c: [0, 0.5, 0.5, 1]

bosh3: &bosh3
    a:
        - [0]
        - [0.5]
        - [0, 0.75]
        - ["2/9", "1/3", "4/9"]
    b: ["2/9", "1/3", "4/9", 0]
    c: [0, 0.5, 0.75, 1]

rk3: &rk3
    a:
        - []
        - [0.5]
        - [-1, 2]
    b: ["1/6", "2/3", "1/6"]
    c: [0, 0.5, 1]

kutta3: &kutta3
    a:
        - [0]
        - [0.5]
        - [-1, 2]
    b: ["1/6", "2/3", "1/6"]
    c: [0, 0.5, 1]

cash_karp: &cash_karp
    a:
        - [0]
        - [0.2]
        - ["3/40", "9/40"]
        - ["3/10", "-9/10", "6/5"]
        - ["-11/54", "5/2", "-70/27", "35/27"]
        - ["1631/55296", "175/512", "575/13824", "44275/110592", "253/4096"]
    b: ["37/378", 0, "250/621", "125/594", 0, "512/1771"]
    c: [0, 0.2, "3/10", "3/5", 1, "7/8"]

dopri5: &dopri5
    a:
        - [0]
        - [0.2]
        - ["3/40", "9/40"]
        - ["44/45", "-56/15", "32/9"]
        - ["19372/6561", "-25360/2187", "64448/6561", "-212/729"]
        - ["9017/3168", "-355/33", "46732/5247", "49/176", "-5103/18656"]
        - ["35/384", 0, "500/1113", "125/192", "-2187/6784", "11/84"]
    b: ["35/384", 0, "500/1113", "125/192", "-2187/6784", "11/84", 0]
    c: [0, 0.2, "3/10", "4/5", "8/9", 1, 1]

fehlberg: &fehlberg
    a:
        - [0]
        - [0.25]
        - ["3/32", "9/32"]
        - ["1932/2197", "-7200/2197", "7296/2197"]
        - ["439/216", -8, "3680/513", "-845/4104"]
        - ["-8/27", 2, "-3544/2565", "1859/4104", "-11/40"]
    b: ["25/216", 0, "1408/2565", "2197/4104", -0.2, 0]
    c: [0, 0.25, "3/8", "12/13", 1, 0.5]

radau3: &radau3
    a:
        - [0]
        - ["5/12"]
        - [0.75, 0.25]
    b: [0.75, 0.25]
    c: ["1/3", 1]
"""

    _yaml_tableau_default = """# You can define your own a, b and c keys
# or use a reference to one of the preset tableaus.
# Valid presets: euler, heun, midpoint, rk4, bosh3, rk3, kutta3, cash_karp, dopri5, fehlberg

<<: *euler
"""

    INPUT_TYPES = DumLazyInputTypes(
        lambda _yaml_tableau_default=_yaml_tableau_default: DumInputTypes()
        .req_float_eta(
            default=0.0,
            min=0.0,
            tooltip="This controls ancestralness. ETA will get multiplied with the similarity for non-pingpong steps so this is going to be the upper bound on ancestralness. You can set this to 0 to only use ancestralness through the pingpong thresholds.",
        )
        .req_float_s_noise(
            default=1.0,
            tooltip="Scale for added noise.",
        )
        .req_float_step_scale(default=1.0)
        .req_int_first_ancestral_step(
            default=0,
            tooltip="First step ancestralness will be enabled. When negative, steps count from the end.",
        )
        .req_int_last_ancestral_step(
            default=-1,
            tooltip="Last step ancestralness will be enabled. When negative, steps count from the end.",
        )
        .req_int_first_blend_step(
            default=0,
            tooltip="First step blend will be enabled. When negative, steps count from the end. Only applies when blend mode is something other than NO_BLEND.",
        )
        .req_int_last_blend_step(
            default=-1,
            tooltip="Last step blend will be enabled. When negative, steps count from the end. Only applies when blend mode is something other than NO_BLEND.",
        )
        .req_selectblend_blend_mode(
            insert_modes=("NO_BLEND",),
            default="NO_BLEND",
            tooltip="When set to NO_BLEND will use the non-blending Butcher Tableau step function which is more accurate more likely to be implemented correctly. If you do set a blend mode, LERP is mainly the only one that works well.",
        )
        .req_bool_step_to_zero(
            tooltip="Controls whether we just return the model result for the step to sigma 0 (when set to false) or still use the tableau (when enabled). Generally leaving it disabled is better.",
        )
        .req_yaml_yaml_tableau(
            default=_yaml_tableau_default,
        ),
    )

    @classmethod
    def go(
        cls,
        *,
        eta: float,
        s_noise: float,
        yaml_tableau: str,
        **kwargs: dict,
    ):
        params_parsed = yaml.safe_load(f"{cls._tableau_defaults}\n{yaml_tableau}")
        if not isinstance(params_parsed, dict):
            raise TypeError("Tableau definition must be an object")
        blend_mode_internal = params_parsed.pop("blend_mode_internal", None)
        tableau = ButcherTableau.from_dict(params_parsed)
        options = {
            "eta": eta,
            "s_noise": s_noise,
            "blep_config": Config(
                tableau=tableau,
                blend_mode_internal=blend_mode_internal,
                **kwargs,
            ),
        }
        return (
            KSAMPLER(ButcherTableauSampler.go, extra_options=options),
            cls._tableau_defaults,
        )


NODE_CLASS_MAPPINGS = {
    "ButcherTableauSampler": ButcherTableauSamplerNode,
}

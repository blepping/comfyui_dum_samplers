from __future__ import annotations

from typing import Callable

import torch
from comfy.k_diffusion.sampling import get_ancestral_step
from comfy.utils import common_upscale

from .external import MODULES as EXT

BLENDING_MODES = {
    "lerp": torch.lerp,
    "a_only": lambda a, _b, _t: a,
    "b_only": lambda _a, b, _t: b,
    "inject": lambda a, b, t: (b * t).add_(a),
    "subtract_b": lambda a, b, t: a - b * t,
}
UPSCALE_METHODS = (
    "bilinear",
    "nearest-exact",
    "nearest",
    "area",
    "bicubic",
    "bislerp",
    "adaptive_avg_pool2d",
)


def scale_samples(
    samples: torch.Tensor,
    width: int,
    height: int,
    *,
    mode: str = "bicubic",
) -> torch.Tensor:
    if mode == "adaptive_avg_pool2d":
        return torch.nn.functional.adaptive_avg_pool2d(samples, (height, width))
    return common_upscale(samples, width, height, mode, None)


def fix_step_range(steps, start, end):
    if start < 0:
        start = steps + start
    if end < 0:
        end = steps + end
    start = max(0, min(steps - 1, start))
    end = max(0, min(steps - 1, end))
    return (end, start) if start > end else (start, end)


def get_ancestral_step_ext(sigma, sigma_next, eta=1.0, is_rf=False):
    if sigma_next == 0 or eta == 0:
        return sigma_next, sigma_next * 0.0, 1.0
    if not is_rf:
        return (*get_ancestral_step(sigma, sigma_next, eta=eta), 1.0)
    # Referenced from ComfyUI.
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_ip1, alpha_down = 1.0 - sigma_next, 1.0 - sigma_down
    sigma_up = (sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2) ** 0.5
    x_coeff = alpha_ip1 / alpha_down
    return sigma_down, sigma_up, x_coeff


def internal_step(
    x: torch.Tensor,
    denoised: torch.Tensor | None,
    sigma: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
    sigma_down: torch.Tensor | float,
    sigma_up: torch.Tensor | float,
    x_coeff: torch.Tensor | float,
    noise_sampler: Callable | None,
    *,
    blend_function: Callable | None = torch.lerp,
) -> torch.Tensor:
    if blend_function is not None and denoised is None:
        raise ValueError("Must past denoised when blend_function is not None")
    x = (
        blend_function(denoised, x, sigma_down / sigma)
        if blend_function is not None
        else x
    )
    if sigma_up == 0 or noise_sampler is None:
        return x
    noise = noise_sampler(sigma, sigma_next).mul_(sigma_up)
    if x_coeff != 1:
        # x gets scaled for flow models.
        x *= x_coeff
    return x.add_(noise)


def init_integrations(integrations) -> None:
    global scale_samples, BLENDING_MODES, UPSCALE_METHODS  # noqa: PLW0603

    bleh = integrations.bleh
    if bleh is None:
        return
    bleh_latentutils = bleh.py.latent_utils
    BLENDING_MODES = bleh_latentutils.BLENDING_MODES
    UPSCALE_METHODS = bleh_latentutils.UPSCALE_METHODS
    scale_samples = bleh_latentutils.scale_samples


EXT.register_init_handler(init_integrations)

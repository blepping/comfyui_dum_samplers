from __future__ import annotations

import torch
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

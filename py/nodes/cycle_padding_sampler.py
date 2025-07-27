# By https://github.com/blepping
# LICENSE: Apache2
# WIP, in development.

from __future__ import annotations

from typing import NamedTuple

import torch
from comfy.samplers import KSAMPLER
from tqdm import tqdm

from .base import DumInputTypes


class CPSConfig(NamedTuple):
    padding_sizes: tuple
    sampler: object
    start_time: float = 0.5
    end_time: float = 1.0
    padding_mode: str = "reflect"
    cycle_mode: str = "sampler_step"


cps_defaults = CPSConfig(padding_sizes=(), sampler=object())


def unpad(t: torch.Tensor, pad: tuple) -> torch.Tensor:
    pl, pr, pt, pb = pad
    pb = -pb if pb != 0 else None
    pr = -pr if pr != 0 else None
    return t[..., pt:pb, pl:pr]


def cycle_padding_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    cps_config: CPSConfig,
    **kwargs: dict,
) -> torch.Tensor:
    cfg = cps_config
    ms = model.inner_model.inner_model.model_sampling
    sigma_start, sigma_end = (
        round(ms.percent_to_sigma(cfg.start_time), 4),
        round(ms.percent_to_sigma(cfg.end_time), 4),
    )
    del ms
    counter = 0
    pad_len = len(cfg.padding_sizes)
    if cfg.padding_mode in {"zero", "noise", "noise_std"}:
        padding_mode = "constant"
        padding_value = 0.0
    else:
        padding_mode = cfg.padding_mode
        padding_value = None

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        nonlocal counter

        def call_model(x):
            return model(x, sigma, **extra_args)

        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_end <= sigma_float <= sigma_start):
            return call_model(x)
        pad = cfg.padding_sizes[counter % pad_len]
        tqdm.write(">> PAD:", counter, pad)
        tqdm.write(">> IN SHAPE", x.shape)
        counter += 1
        if pad == (0, 0, 0, 0):
            return call_model(x)
        if cfg.padding_mode in {"noise", "noise_std"}:
            noise_mask = x[:1, :1, ...] * 0
            noise_strength = (
                sigma_float if cfg.padding_mode == "noise" else float(x.std().detach())
            )
        x = torch.nn.functional.pad(
            x,
            pad,
            mode=padding_mode,
            value=padding_value,
        )
        if cfg.padding_mode in {"noise", "noise_std"}:
            noise_mask = torch.nn.functional.pad(
                noise_mask,
                pad,
                mode="constant",
                value=noise_strength,
            )
            x += (
                torch.randn_like(x)
                .add_(x.mean(dim=(-2, -1), keepdim=True))
                .mul_(noise_mask)
            )
            del noise_mask
        tqdm.write(">> PADDED SHAPE", x.shape)
        result = unpad(call_model(x), pad)
        tqdm.write(">> OUT SHAPE", result.shape)
        return result

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return cfg.sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **cfg.sampler.extra_options,
    )


class CyclePaddingSamplerNode:
    DESCRIPTION = "TBD"
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    INPUT_TYPES = (
        DumInputTypes()
        .req_sampler()
        .req_float_start_time(default=0.0, min=0.0, max=1.0)
        .req_float_end_time(default=0.0, min=0.0, max=1.0)
        .req_field_padding_mode(
            (
                "reflect",
                "replicate",
                "circular",
                "zero",
                "noise",
                "noise_std",
            ),
            default=cps_defaults.padding_mode,
        )
        .req_field_cycle_mode(
            ("model_call", "sampler_step"),
            default=cps_defaults.cycle_mode,
        )
        .req_field_pad_dimension(
            (
                "bottom",
                "top",
                "left",
                "right",
                "bottom_left",
                "bottom_right",
                "top_left",
                "top_right",
                "top_bottom",
                "left_right",
                "around",
                "custom",
            ),
            default="bottom",
        )
        .req_string_padding_sizes(default="4,0")
    )

    PAD_IDXS = {  # noqa: RUF012
        "top": (2,),
        "bottom": (3,),
        "left": (0,),
        "right": (1,),
        "top_left": (0, 2),
        "top_right": (1, 2),
        "bottom_left": (0, 3),
        "bottom_right": (1, 3),
        "top_bottom": (2, 3),
        "left_right": (0, 1),
        "around": (0, 1, 2, 3),
    }

    @classmethod
    def go(
        cls,
        *,
        sampler: object,
        start_time: float,
        end_time: float,
        cycle_mode: str,
        pad_dimension: str,
        padding_mode: str,
        padding_sizes: str,
    ) -> tuple:
        if pad_dimension != "custom":
            pad_idxs = cls.PAD_IDXS.get(pad_dimension)
            if pad_dimension is None:
                raise ValueError("Bad pad dimension")

            def mkpad(sizestr: str) -> tuple:
                size = int(sizestr) if sizestr else 0
                if size < 0:
                    raise ValueError("Pad size must be positive")
                return tuple(size if i in pad_idxs else 0 for i in range(4))

        else:

            def mkpad(sizestr: str) -> tuple:
                sizes = tuple(int(v) if v.strip() else 0 for v in sizestr.split(":"))
                szlen = len(sizes)
                if szlen > 4:
                    raise ValueError(
                        "Expected at most four pad sizes: left, right, top, bottom",
                    )
                if not all(i >= 0 for i in sizes):
                    raise ValueError("Pad sizes must be positive")
                if szlen < 4:
                    return sizes + (0,) * (4 - szlen)
                return sizes

        pads = tuple(mkpad(p.strip()) for p in padding_sizes.split(","))

        cfg = CPSConfig(
            sampler=sampler,
            start_time=start_time,
            end_time=end_time,
            cycle_mode=cycle_mode,
            padding_mode=padding_mode,
            padding_sizes=pads,
        )
        return (KSAMPLER(cycle_padding_sampler, extra_options={"cps_config": cfg}),)


NODE_CLASS_MAPPINGS = {
    "CyclePaddingSampler": CyclePaddingSamplerNode,
}

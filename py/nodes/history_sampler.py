# ComfyUI sampler wrapper that saves state for every model call.
# This ignores whatever the sampler actually returns. You will get a result of
# batch_size * times_model_was_called latents.
from __future__ import annotations

import torch
from comfy.model_management import device_supports_non_blocking
from comfy.samplers import KSAMPLER

from .base import DumInputTypes


class HistorySamplerConfig:
    def __init__(self, *, wrapped_sampler, save_mode, append_sampler_result=False):
        self.wrapped_sampler = wrapped_sampler
        self.save_mode = save_mode
        self.append_sampler_result = append_sampler_result


def history_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    history_sampler_config: tuple,
    **kwargs: dict,
) -> torch.Tensor:
    cfg = history_sampler_config
    history = []
    cuda = getattr(torch, "cuda", None)
    non_blocking = cuda is not None and device_supports_non_blocking(x.device)
    save_x = cfg.save_mode in {"model_input", "both"}
    save_denoised = cfg.save_mode in {"denoised", "both"}

    def save_history(t: torch.Tensor) -> None:
        history.append(t.detach().clone().to("cpu", non_blocking=non_blocking))

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        denoised = model(x, sigma, **extra_args)
        if save_x:
            save_history(x)
        if save_denoised:
            save_history(denoised)
        return denoised

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    result = cfg.wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **cfg.wrapped_sampler.extra_options,
    )
    if cuda is not None:
        cuda.synchronize(x.device)
    if cfg.append_sampler_result:
        history.append(result.cpu())
    return torch.cat(history, dim=0).to(x)


class HistorySamplerNode:
    DESCRIPTION = "This sampler wrapper saves history - either the model noise prediction or model input - each time the model is called. The output from the sampler will be batch_size * model calls, with the last batch_size items being the model result from the last step. If you enable append_sampler_result then you will get an additional batch_size latents containing what the sampler would normally return."
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    INPUT_TYPES = (
        DumInputTypes()
        .req_sampler()
        .opt_field_save_mode(
            ("denoised", "model_input", "both", "disable"),
            default="denoised",
        )
        .req_bool_append_sampler_result()
    )

    @classmethod
    def go(
        cls,
        *,
        sampler: object,
        save_mode: str = "denoised",
        append_sampler_result: bool = False,
    ) -> tuple:
        if save_mode == "disable":
            return (sampler,)
        if save_mode not in {"denoised", "model_input", "both"}:
            raise ValueError("Bad save_mode")
        cfg = HistorySamplerConfig(
            wrapped_sampler=sampler,
            save_mode=save_mode,
            append_sampler_result=append_sampler_result,
        )
        return (
            KSAMPLER(
                history_sampler,
                extra_options={"history_sampler_config": cfg},
            ),
        )


NODE_CLASS_MAPPINGS = {
    "HistorySampler": HistorySamplerNode,
}

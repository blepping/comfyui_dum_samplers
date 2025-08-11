# By https://github.com/blepping
# LICENSE: Apache2

from __future__ import annotations

from functools import partial, update_wrapper
from typing import Callable, NamedTuple

import torch
from comfy import model_sampling
from comfy.k_diffusion.sampling import to_d
from comfy.samplers import KSAMPLER
from tqdm.auto import trange

from .. import utils
from .base import DumInputTypes, DumLazyInputTypes


class Config(NamedTuple):
    first_helping_step: int = 0
    last_helping_step: int = -1
    initial_noise_scale: float = 1.0
    noise_prediction_scale: float = 1.0
    assist_ratio: float = 0.25
    assist_blend_mode: str = "lerp"
    step_scale: float = 1.0
    blend_mode: str = "lerp"
    preview_mode: str = "adjusted"
    wrapped_sampler: object | None = None
    initial_latent: torch.Tensor | None = None
    initial_noise: torch.Tensor | None = None


class ModelProxy:
    def __init__(
        self,
        *,
        model: object,
        config: Config,
        initial_noise: torch.Tensor,
        first_helping_sigma_incl: float,
        last_helping_sigma_excl: float,
    ):
        self.__model = model
        self.__blend_function = utils.BLENDING_MODES[config.assist_blend_mode]
        self.__config = config
        self.__initial_noise = initial_noise
        self.__first_helping_sigma_incl = first_helping_sigma_incl
        self.__last_helping_sigma_excl = last_helping_sigma_excl

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        *args: list,
        **kwargs: dict,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        config = self.__config
        for_wrapped = config.wrapped_sampler is not None
        sigma_f = sigma.max().detach().cpu().item()
        be_helpful = config.assist_ratio > 0 and (
            self.__last_helping_sigma_excl < sigma_f <= self.__first_helping_sigma_incl
        )
        if be_helpful and config.assist_ratio >= 1:
            denoised = torch.zeros_like(x)
        else:
            denoised = self.__model(x, sigma, *args, **kwargs)
        if not be_helpful:
            return denoised if for_wrapped else (denoised, denoised)
        d = self.__blend_function(
            to_d(x, sigma, denoised).mul_(config.noise_prediction_scale),
            self.__initial_noise,
            config.assist_ratio,
        )
        denoised_adj = x - d * sigma
        return denoised_adj if for_wrapped else (denoised_adj, denoised)

    def __getattr__(self, k):
        return getattr(self.__model, k)


class LetMeHelpYouSampler:
    def __init__(
        self,
        model: object,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        *args: list,
        extra_args: dict | None = None,
        callback: Callable | None = None,
        disable: bool | None = None,
        blep_config: Config | None = None,
        verbose: bool = False,
        **kwargs: dict,
    ):
        if blep_config is None:
            blep_config = Config()
        self.args = args
        self.kwargs = kwargs
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

    @classmethod
    def go(
        cls,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
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
            verbose=verbose,
            **kwargs,
        )()

    def model(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        *,
        _model: Callable | None = None,
        **kwargs: dict,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return (_model or self.model_)(
            x,
            sigma * self.s_in,
            **self.extra_args,
            **kwargs,
        )

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
        config = self.config
        x = self.x
        sigmas = self.sigmas
        sigma0 = sigmas[0]
        if config.initial_noise is not None:
            initial_noise = config.initial_noise.to(x, copy=True)
            if self.is_rf and sigma0 != 1:
                initial_noise *= sigma0
        elif self.is_rf:
            if sigma0.detach().cpu().item() >= 1.0:
                initial_noise = x.clone()
            elif config.initial_latent is None:
                initial_noise = x / sigma0
            else:
                initial_noise = (
                    x - config.initial_latent.to(x) * (1.0 - sigma0)
                ) / sigma0
        else:
            initial_noise = (
                x - config.initial_latent.to(x)
                if config.initial_latent is not None
                else x.clone()
            ) / sigma0
        if config.initial_noise_scale != 1:
            initial_noise *= config.initial_noise_scale
        steps = len(self.sigmas) - 1
        first_helping_step, last_helping_step = utils.fix_step_range(
            steps,
            config.first_helping_step,
            config.last_helping_step,
        )
        first_helping_sigma_incl = sigmas[first_helping_step].detach().cpu().item()
        last_helping_sigma_excl = (
            sigmas[min(last_helping_step + 1, steps)].detach().cpu().item()
        )
        proxy = ModelProxy(
            model=self.model_,
            config=config,
            initial_noise=initial_noise,
            first_helping_sigma_incl=first_helping_sigma_incl,
            last_helping_sigma_excl=last_helping_sigma_excl,
        )
        if config.wrapped_sampler is not None:
            return config.wrapped_sampler.sampler_function(
                proxy,
                x,
                sigmas,
                *self.args,
                disable=self.disable,
                callback=self.callback_,
                extra_args=self.extra_args,
                **config.wrapped_sampler.extra_options,
                **self.kwargs,
            )
        blend_function = utils.BLENDING_MODES[config.blend_mode]
        for idx in trange(steps, disable=self.disable):
            sigma, sigma_next = sigmas[idx : idx + 2]
            sigma_next = (
                sigma + (sigma_next - sigma) * config.step_scale
                if sigma_next != 0
                else sigma_next
            )
            (denoised_adj, denoised_orig) = self.model(x, sigma, _model=proxy)
            self.callback(
                idx,
                x,
                sigma,
                denoised_adj if config.preview_mode == "adjusted" else denoised_orig,
            )
            x = blend_function(denoised_adj, x, sigma_next / sigma)
        return x


class LetMeHelpYouSamplerNode:
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    INPUT_TYPES = DumLazyInputTypes(
        lambda: DumInputTypes()
        .req_float_initial_noise_scale(
            default=1.0,
            tooltip="Scale for our predicted initial noise.",
        )
        .req_float_noise_prediction_scale(
            default=1.0,
            tooltip="Scale for the model's prediction of the noise. No effect when wrapping another sampler.",
        )
        .req_float_step_scale(default=1.0)
        .req_float_assist_ratio(default=0.25, min=0.0, max=1.0)
        .req_selectblend_assist_blend_mode(
            tooltip="Blend mode for combining 'assistance' noise with the model's noise prediction.",
        )
        .req_selectblend_blend_mode(
            tooltip="Blend mode for Euler steps. No effect when wrapping another sampler.",
        )
        .req_field_preview_mode(
            ("adjusted", "original"),
            default="adjusted",
            tooltip="Controls whether sampling previews use the adjusted prediction or what the model originally returned. No effect when wrapping another sampler.",
        )
        .req_int_first_helping_step(
            default=0,
            tooltip="First step ancestralness will be enabled. When negative, steps count from the end.",
        )
        .req_int_last_helping_step(
            default=-2,
            tooltip="Last step ancestralness will be enabled. When negative, steps count from the end.",
        )
        .opt_latent_initial_latent_opt(
            tooltip="If not attached we will assume the initial latent was empty (zeros). For img2img workflows, you will need to connect this for good results. Not used if initial_noise_opt is connected.",
        )
        .opt_latent_initial_noise_opt(
            tooltip="Can be used to override the value used for initial noise. This should be noise at 1.0 strength. Can be attached if you want to manually control what gets used as initial noise. initial_latent_opt is not used if this is connected.",
        )
        .opt_sampler_opt_sampler(
            tooltip="When connected will use this for sampling rather than the built in Euler sampler. Will not work with samplers that need to patch the model to extract cond/uncond (i.e. CFG++ samplers).",
        ),
    )

    @classmethod
    def go(
        cls,
        *,
        initial_latent_opt: dict | None = None,
        initial_noise_opt: dict | None = None,
        opt_sampler: object | None = None,
        **kwargs: dict,
    ):
        config = Config(
            initial_latent=None
            if initial_latent_opt is None
            else initial_latent_opt["samples"].to(device="cpu", copy=True),
            initial_noise=None
            if initial_noise_opt is None
            else initial_noise_opt["samples"].to(device="cpu", copy=True),
            wrapped_sampler=opt_sampler,
            **kwargs,
        )
        sampler_function = partial(LetMeHelpYouSampler.go, blep_config=config)
        if opt_sampler is not None:
            sampler_function = update_wrapper(
                sampler_function,
                opt_sampler.sampler_function,
            )
        return (KSAMPLER(sampler_function),)


NODE_CLASS_MAPPINGS = {
    "LetMeHelpYouSampler": LetMeHelpYouSamplerNode,
}

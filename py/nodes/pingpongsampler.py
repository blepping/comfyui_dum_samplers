# By https://github.com/blepping
# LICENSE: Apache2

import random

import torch
from comfy import model_sampling
from comfy.samplers import KSAMPLER
from tqdm.auto import trange

from .. import utils
from .base import DumInputTypes, DumLazyInputTypes

BLEND_MODES = None


class ModelProxy:
    def __init__(self, model, last_x, last_sigma, last_denoised):
        self.__model = model
        self.__last_x = last_x
        self.__last_sigma = last_sigma
        self.__last_denoised = last_denoised

    def __call__(self, x, sigma, *args: list, **kwargs: dict):
        if torch.allclose(
            sigma.to(self.__last_sigma),
            self.__last_sigma,
        ) and torch.allclose(x.to(self.__last_x), self.__last_x):
            return self.__last_denoised.to(x, copy=True)
        return self.__model(x, sigma, *args, **kwargs)

    def __getattr__(self, k):
        return getattr(self.__model, k)


class PingPongSampler:
    def __init__(
        self,
        model,
        x,
        sigmas,
        *args: list,
        extra_args=None,
        callback=None,
        disable=None,
        noise_sampler=None,
        s_noise=1.0,
        pingpong_options=None,
        **kwargs: dict,
    ):
        self.args = args
        self.kwargs = kwargs
        self.model_ = model
        self.sigmas = sigmas
        self.x = x
        self.s_in = x.new_ones((x.shape[0],))
        self.extra_args = extra_args.copy() if extra_args is not None else {}
        self.seed = self.extra_args.pop("seed", 42)
        self.disable = disable
        self.callback_ = callback
        if pingpong_options is None:
            pingpong_options = {}
        self.first_ancestral_step = pingpong_options.get("first_ancestral_step", 0)
        self.last_ancestral_step = pingpong_options.get("last_ancestral_step", 0)

        self.pingpong_blend = pingpong_options.get("pingpong_blend")
        sampler_opt = pingpong_options.get("external_sampler")
        if self.pingpong_blend != 1.0 and sampler_opt is None:
            raise ValueError(
                "Sampler input must be connect when pingpong_blend isn't 1.0",
            )
        self.external_sampler = sampler_opt
        self.step_blend_function = pingpong_options.get(
            "step_blend_function",
            torch.lerp,
        )
        self.blend_function = pingpong_options.get("blend_function", torch.lerp)
        self.s_noise = s_noise
        self.is_rf = isinstance(
            model.inner_model.inner_model.model_sampling,
            model_sampling.CONST,
        )
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
        s_noise=1.0,
        pingpong_options=None,
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
            s_noise=s_noise,
            pingpong_options=pingpong_options,
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
        astart_step = self.first_ancestral_step
        aend_step = self.last_ancestral_step
        last_step_idx = len(self.sigmas) - 2
        step_count = len(self.sigmas) - 1
        if astart_step < 0:
            astart_step = step_count + astart_step
        if aend_step < 0:
            aend_step = step_count + aend_step
        astart_step = min(last_step_idx, max(0, astart_step))
        aend_step = min(last_step_idx, max(0, aend_step))
        seed_offset = 10
        for idx in trange(step_count, disable=self.disable):
            sigma, sigma_next = self.sigmas[idx : idx + 2]
            orig_x = x
            denoised = self.model(orig_x, sigma)
            self.callback(idx, x, sigma, denoised)
            use_ancestral = astart_step <= idx <= aend_step
            if sigma_next <= 1e-06:
                return denoised
            if not use_ancestral:
                x = self.step_blend_function(denoised, x, sigma_next / sigma)
                continue
            if self.pingpong_blend != 1.0:
                alt_x = self.external_sampler.sampler_function(
                    ModelProxy(self.model_, x, sigma, denoised),
                    orig_x.clone(),
                    self.sigmas[idx : idx + 2].clone(),
                    *self.args,
                    disable=True,
                    callback=None,
                    extra_args=self.extra_args | {"seed": self.seed + seed_offset},
                    **self.external_sampler.extra_options,
                    **self.kwargs,
                )
                seed_offset += 10
                if self.pingpong_blend <= 0:
                    x = alt_x
                    continue
            noise = noise_sampler(sigma, sigma_next).mul_(self.s_noise)
            if self.is_rf:
                x = self.step_blend_function(denoised, noise, sigma_next)
            else:
                x = denoised + noise * sigma_next
            if self.pingpong_blend != 1.0:
                x = self.blend_function(alt_x, x, self.pingpong_blend)
                del alt_x
        return x


class PingPongSamplerNode:
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    INPUT_TYPES = DumLazyInputTypes(
        lambda: DumInputTypes()
        .req_float_s_noise(
            default=1.0,
            tooltip="Scale for added noise.",
        )
        .req_int_first_ancestral_step(
            default=0,
            tooltip="First step ancestralness will be enabled. When negative, steps count from the end.",
        )
        .req_int_last_ancestral_step(
            default=-1,
            tooltip="Last step ancestralness will be enabled. When negative, steps count from the end.",
        )
        .req_float_pingpong_blend(
            default=1.0,
            min=0.0,
            max=1.0,
            tooltip="Allows blending pingpong sampling with a different sampler. Only has an effect during the ancestral_step range. If set to a value below 1.0 (100% pingpong) then sampler_opt must be attached.",
        )
        .req_selectblend_blend_mode(
            tooltip="Blend mode to use when blending pingpong sampling with the external sampler. See tooltip for pingpong_blend. Can integrate with ComfyUI-bleh to add more blend modes.",
        )
        .req_selectblend_step_blend_mode(
            tooltip="Blend mode to use for pingpong steps. Changing this is likely a bad idea. Does not apply for ancestral steps on non-flow models.  Can integrate with ComfyUI-bleh to add more blend modes.",
        )
        .opt_sampler_sampler_opt(
            tooltip="Optional when pingpong_blend is 1.0. Result of a pingpong step will be blended with output from this sampler with the configured ratio. Calls the sampler on a single step so will not work well with samplers that care about state (I.E. history samplers such as deis, res_multistep, etc).",
        ),
    )

    @classmethod
    def go(
        cls,
        *,
        s_noise: float,
        first_ancestral_step: int,
        last_ancestral_step: int,
        pingpong_blend: float,
        blend_mode: str,
        step_blend_mode: str,
        sampler_opt=None,
    ):
        options = {
            "s_noise": s_noise,
            "pingpong_options": {
                "first_ancestral_step": first_ancestral_step,
                "last_ancestral_step": last_ancestral_step,
                "pingpong_blend": pingpong_blend,
                "blend_function": utils.BLENDING_MODES[blend_mode],
                "step_blend_function": utils.BLENDING_MODES[step_blend_mode],
                "external_sampler": sampler_opt,
            },
        }
        return (KSAMPLER(PingPongSampler.go, extra_options=options),)


class RestlessSchedulerNode:
    DESCRIPTION = "HACK: A weird scheduler that will randomly jump around a list of sigmas you input. Not recommended. Breaks most multi-step and history samplers. Works okay-ish with Pingpong."
    CATEGORY = "sampling/custom_sampling/schedulers"
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "go"

    INPUT_TYPES = (
        DumInputTypes()
        .req_sigmas()
        .req_seed(tooltip="Seed to use for generating schedule.")
        .req_float_shrink_factor(
            default=0.3,
            tooltip="Amount the window for restless scheduling shrinks by per iteration.",
        )
        .req_int_first_restless_step(
            default=3,
            min=1,
            tooltip="First step (0-based) to include for restless scheduling. Must be greater than 1 and less than last_restless_step.",
        )
        .req_int_last_restless_step(
            default=-4,
            tooltip="Last step (0-based) to include for restless scheduling. Can be negative to count from the end, but you cannot target the last sigma in the list.",
        )
    )

    @classmethod
    def go(
        cls,
        *,
        sigmas: torch.Tensor,
        seed: int,
        shrink_factor: float,
        first_restless_step: int,
        last_restless_step: int,
    ) -> tuple:
        n_sigmas = len(sigmas)
        if n_sigmas < 3:
            return (sigmas,)
        if last_restless_step < 0:
            last_restless_step = n_sigmas + last_restless_step
        if last_restless_step <= first_restless_step:
            raise ValueError("Last restless step <= first restless step!")
        if last_restless_step >= n_sigmas - 1:
            raise ValueError("Last restless step cannot include the final sigma")
        orig_sigmas = sigmas
        random.seed(seed)
        result = sigmas[:first_restless_step].tolist()
        end_chunk = sigmas[last_restless_step + 1 :].tolist()
        sigmas = sigmas[first_restless_step : last_restless_step + 1].tolist()
        n_sigmas = len(sigmas)
        shrinkage = 0.0
        curr_idx = None
        while (window_size := int((n_sigmas - 1) - shrinkage)) > 0:
            next_idx = random.randint(0, window_size)  # noqa: S311
            if next_idx == curr_idx:
                next_idx += 1
            result.append(sigmas[int(shrinkage) + next_idx])
            curr_idx = next_idx
            shrinkage += shrink_factor
        result += end_chunk
        return (
            torch.tensor(result, dtype=torch.float32, device="cpu").to(orig_sigmas),
        )


NODE_CLASS_MAPPINGS = {
    "PingPongSampler": PingPongSamplerNode,
    "RestlessScheduler": RestlessSchedulerNode,
}

# By https://github.com/blepping
# LICENSE: Apache2
# Usage: Place this file in the custom_nodes directory and restart ComfyUI+refresh browser.
#        It will add a CacheAwareEulerSampler node that can be used with SamplerCustom, etc.

import torch
from comfy import model_sampling
from comfy.k_diffusion.sampling import get_ancestral_step
from comfy.samplers import KSAMPLER
from tqdm.auto import trange

from .base import DumInputTypes


class CAEulerSampler:
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
        similarity_threshold=0.99,
        similarity_mode="normal",
        first_ancestral_step=1,
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
        self.similarity_threshold = similarity_threshold
        if similarity_mode not in {"normal", "last_step"}:
            raise ValueError("Bad similarity mode")
        self.similarity_mode = similarity_mode
        self.verbose = verbose
        self.eta = max(0.0, eta)
        self.s_noise = s_noise
        self.first_ancestral_step = first_ancestral_step
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
        similarity_threshold=0.99,
        similarity_mode="normal",
        first_ancestral_step=1,
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
            similarity_threshold=similarity_threshold,
            similarity_mode=similarity_mode,
            first_ancestral_step=first_ancestral_step,
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
        noise_sampler = self.noise_sampler
        eta_start = max(0, self.first_ancestral_step - 1)
        x_from = x = self.x
        denoised_from = None
        sigma_from = None
        last_step_idx = len(self.sigmas) - 2
        for idx in trange(last_step_idx + 1, disable=self.disable):
            eta = self.eta if idx >= eta_start else 0.0
            sigma, sigma_next = self.sigmas[idx : idx + 2]
            if sigma_from is None:
                sigma_from = sigma
            denoised = self.model(x, sigma)
            if sigma_next <= 1e-06:
                return denoised
            if denoised_from is None:
                similarity = None
            else:
                similarity = torch.cosine_similarity(
                    denoised.flatten(start_dim=0),
                    denoised_from.flatten(start_dim=0),
                    dim=0,
                ).item()
            is_same = (
                similarity is not None and abs(similarity) >= self.similarity_threshold
            )
            self.callback(idx, x, sigma, denoised)
            if self.similarity_mode == "last_step":
                denoised_from = denoised
            is_last = idx == last_step_idx
            if self.verbose:
                print(
                    f"\nIS SAME? sigma={float(sigma)} ({float(sigma_from)}), sigma_next={float(sigma_next)}, same={is_same}, last={is_last}, similarity={similarity}",
                )
            if not is_last and is_same:
                x = torch.lerp(denoised, x, sigma_next / sigma)
                continue
            if noise_sampler is None or eta == 0:
                sigma_down = sigma_next
                renoise_coeff = sigma_next * 0
            elif self.is_rf:
                # Copied from ComfyUI
                downstep_ratio = 1 + (sigma_next / sigma_from - 1) * eta
                sigma_down = sigma_next * downstep_ratio
                alpha_ip1 = 1 - sigma_next
                alpha_down = 1 - sigma_down
                if torch.isclose(sigma_down, sigma_next):
                    sigma_down = sigma_next
                    renoise_coeff = sigma_next * 0
                else:
                    renoise_coeff = (
                        sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
                    ) ** 0.5
            else:
                sigma_down, renoise_coeff = get_ancestral_step(
                    sigma_from,
                    sigma_next,
                    eta=eta,
                )
                if sigma_down == sigma_next:
                    renoise_coeff = sigma_next * 0
            x = torch.lerp(denoised, x_from, sigma_down / sigma_from)
            if noise_sampler is not None and renoise_coeff != 0:
                noise = noise_sampler(sigma_from, sigma_next).mul_(
                    renoise_coeff * self.s_noise,
                )
                if not self.is_rf:
                    x += noise
                else:
                    x = x.mul_(alpha_ip1 / alpha_down).add_(noise)
                del noise
            x_from = x
            sigma_from = sigma_next
            denoised_from = denoised
        return x


class CacheAwareEulerSamplerNode:
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    INPUT_TYPES = (
        DumInputTypes()
        .req_float_eta(default=0.5)
        .req_float_s_noise(default=1.0)
        .req_float_similarity_threshold(default=0.7)
        .req_field_similarity_mode(("normal", "last_step"), default="normal")
        .req_int_first_ancestral_step(default=1, min=1)
    )

    @classmethod
    def go(
        cls,
        *,
        eta,
        s_noise,
        similarity_threshold,
        similarity_mode,
        first_ancestral_step,
    ):
        options = {
            "eta": eta,
            "s_noise": s_noise,
            "similarity_threshold": similarity_threshold,
            "similarity_mode": similarity_mode,
            "first_ancestral_step": first_ancestral_step,
        }
        return (KSAMPLER(CAEulerSampler.go, extra_options=options),)


NODE_CLASS_MAPPINGS = {
    "CacheAwareEulerSampler": CacheAwareEulerSamplerNode,
}

from .nodes import (
    batchmergesampler,
    cacheaware_euler_sampler,
    cycle_padding_sampler,
    history_sampler,
    pingpongsampler,
    simancestral_euler_sampler,
    similarityclamp_euler_sampler,
)

_modules = (
    batchmergesampler,
    cacheaware_euler_sampler,
    cycle_padding_sampler,
    history_sampler,
    pingpongsampler,
    simancestral_euler_sampler,
    similarityclamp_euler_sampler,
)

NODE_CLASS_MAPPINGS = {
    k: v for m in _modules for k, v in getattr(m, "NODE_CLASS_MAPPINGS", {}).items()
}

NODE_DISPLAY_NAME_MAPPINGS = {name: f"[dum] {name}" for name in NODE_CLASS_MAPPINGS}


__all__ = ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")

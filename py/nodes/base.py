from __future__ import annotations

from .. import utils
from ..external import MODULES
from .base_inputtypes import InputCollection, InputTypes, LazyInputTypes


class DumInputCollection(InputCollection):
    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)
        self._DELEGATE_KEYS = self._DELEGATE_KEYS | frozenset((  # noqa: PLR6104
            "customnoise",
            "normalizetristate",
            "selectblend",
            "selectnoise",
            "selectscalemode",
            "yaml",
        ))

    def yaml(
        self,
        name: str = "yaml_parameters",
        *,
        tooltip="Allows specifying custom parameters via YAML. Note: When specifying paramaters this way, there is generally not much error checking.",
        placeholder="# YAML or JSON here",
        dynamicPrompts=False,  # noqa: N803
        multiline=True,
        **kwargs: dict,
    ):
        return self.field(
            name,
            "STRING",
            tooltip=tooltip,
            placeholder=placeholder,
            dynamicPrompts=dynamicPrompts,
            multiline=multiline,
            **kwargs,
        )

    def selectblend(
        self,
        name: str = "blend_mode",
        *,
        default="lerp",
        insert_modes=(),
        tooltip="Mode used for blending. If you have ComfyUI-bleh then you will have access to many more blend modes.",
        **kwargs: dict,
    ) -> InputCollection:
        if not MODULES.initialized:
            raise RuntimeError(
                "Attempt to get blending modes before integrations were initialized",
            )
        return self.field(
            name,
            (*insert_modes, *utils.BLENDING_MODES.keys()),
            default=default,
            tooltip=tooltip,
            **kwargs,
        )

    def selectscalemode(
        self,
        name: str,
        *,
        default="nearest-exact",
        insert_modes=(),
        tooltip="Mode used for scaling. If you have ComfyUI-bleh then you will have access to many more scale modes.",
        **kwargs: dict,
    ) -> InputCollection:
        if not MODULES.initialized:
            raise RuntimeError(
                "Attempt to get scale modes before integrations were initialized",
            )
        return self.field(
            name,
            (*insert_modes, *utils.UPSCALE_METHODS),
            default=default,
            tooltip=tooltip,
            **kwargs,
        )


class DumInputTypes(InputTypes):
    _NO_REPLACE = True

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(
            *args,
            collection_class=DumInputCollection,
            **kwargs,
        )


class DumLazyInputTypes(LazyInputTypes):
    _NO_REPLACE = True

    def __init__(self, *args: list, initializers=(), **kwargs: dict):
        super().__init__(
            *args,
            initializers=(MODULES.initialize, *initializers),
            **kwargs,
        )

"""Utility modules for modeling framework."""

from .config_compat import SimpleModelConfig
from .vision import conv2d, conv3d

__all__ = [
    "SimpleModelConfig",
    "conv2d",
    "conv3d",
]


def get_registry_functions():
    """Lazy load registry functions to avoid circular imports."""
    from .registry import (
        ModelingRegistry,
        ModelInfo,
        register_all_models,
        get_model_presets,
        get_presets_by_sparse_type,
        create_model_from_config,
    )

    return {
        "ModelingRegistry": ModelingRegistry,
        "ModelInfo": ModelInfo,
        "register_all_models": register_all_models,
        "get_model_presets": get_model_presets,
        "get_presets_by_sparse_type": get_presets_by_sparse_type,
        "create_model_from_config": create_model_from_config,
    }


ModelingRegistry = None
ModelInfo = None
register_all_models = None
get_model_presets = None
get_presets_by_sparse_type = None
create_model_from_config = None


def __getattr__(name):
    funcs = get_registry_functions()
    if name in funcs:
        return funcs[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")

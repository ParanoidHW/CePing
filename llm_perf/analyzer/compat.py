"""Backward compatibility layer for deprecated functions.

Provides compatibility for old API patterns that have been refactored.
"""

from typing import Dict, Any
import warnings

from .workload_loader import get_workload, BACKWARD_COMPAT_MAP


def infer_workload(model_type: str, mode: str) -> str:
    """Infer workload name from model type and mode.

    DEPRECATED: This function ties workload to model type.
    Users should explicitly specify workload name instead.

    Args:
        model_type: Model type (llama, deepseek, wan-dit, vae, etc.)
        mode: Mode (training, inference)

    Returns:
        Workload name (for backward compatibility, returns old-style name)
    """
    warnings.warn(
        "infer_workload() is deprecated. "
        "Workload should describe computation characteristics, NOT model types. "
        "Please explicitly specify workload name instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    model_lower = model_type.lower()

    if "wan" in model_lower or "dit" in model_lower or "hunyuan" in model_lower or "diffusion" in model_lower:
        base = "diffusion"
    elif "vae" in model_lower:
        base = "conv"
        if mode == "inference":
            return "vae-decode"
        else:
            return "conv-encoder"
    elif "moe" in model_lower or "mixtral" in model_lower or "deepseek" in model_lower:
        base = "llm"
    elif "resnet" in model_lower:
        base = "resnet"
    else:
        base = "llm"

    old_name = f"{base}-{mode}"

    return old_name


WORKLOAD_PRESETS: Dict[str, Any] = {}


def register_workload(config) -> None:
    """Register workload (backward compat wrapper)."""
    from .workload_loader import register_workload as _register

    _register(config)
    WORKLOAD_PRESETS[config.name] = config

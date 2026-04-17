"""Legacy module - Old model implementations moved to legacy.

This module preserves backward compatibility by re-exporting from modeling.

IMPORTANT: This module is deprecated. Use llm_perf.modeling instead.

For migration guide:
- Replace: from llm_perf.models import X
- With:    from llm_perf.modeling import X

Example:
    # Old (deprecated):
    from llm_perf.models import LlamaModel, LlamaConfig
    from llm_perf.models.registry import create_model_from_config

    # New (recommended):
    from llm_perf.modeling import LlamaModel, create_model_from_config
"""

import warnings

warnings.warn(
    "llm_perf.legacy.models is deprecated. Use llm_perf.modeling instead.",
    DeprecationWarning,
    stacklevel=2,
)

from llm_perf.modeling import (
    LlamaModel,
    DeepSeekModel,
    ShardedVAE,
    ShardedResNet,
    ShardedWanTextEncoder,
    ShardedWanDiT,
    ShardedWanVAE,
    ModelingRegistry,
    get_model_presets,
    get_presets_by_sparse_type,
    create_model_from_config,
)

__all__ = [
    "LlamaModel",
    "DeepSeekModel",
    "ShardedVAE",
    "ShardedResNet",
    "ShardedWanTextEncoder",
    "ShardedWanDiT",
    "ShardedWanVAE",
    "ModelingRegistry",
    "get_model_presets",
    "get_presets_by_sparse_type",
    "create_model_from_config",
]

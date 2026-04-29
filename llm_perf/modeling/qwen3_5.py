"""Qwen3.5 Models with Hybrid Attention (backward compatibility).

DEPRECATED: Use llm_perf.modeling.models.qwen3_5 instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.models.qwen3_5 import (
    Qwen3_5MoEModel,
    Qwen3_5Model,
    ShardedQwen3_5MoEBlock,
    ShardedQwen3_5DenseBlock,
    generate_layer_types,
)

__all__ = [
    "Qwen3_5MoEModel",
    "Qwen3_5Model",
    "ShardedQwen3_5MoEBlock",
    "ShardedQwen3_5DenseBlock",
    "generate_layer_types",
]
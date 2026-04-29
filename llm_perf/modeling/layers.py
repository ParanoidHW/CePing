"""Basic modules for transformer models (backward compatibility).

DEPRECATED: Use llm_perf.modeling.base.layers instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.base.layers import (
    ShardedEmbedding,
    ShardedRMSNorm,
    ShardedAttention,
    ShardedFFN,
    ShardedLMHead,
    ShardedMoE,
    ShardedLinearAttention,
    silu,
    gelu,
    relu,
    flash_attention,
    linear_attention,
)

__all__ = [
    "ShardedEmbedding",
    "ShardedRMSNorm",
    "ShardedAttention",
    "ShardedFFN",
    "ShardedLMHead",
    "ShardedMoE",
    "ShardedLinearAttention",
    "silu",
    "gelu",
    "relu",
    "flash_attention",
    "linear_attention",
]
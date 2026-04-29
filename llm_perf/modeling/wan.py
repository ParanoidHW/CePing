"""Wan video generation models (backward compatibility).

DEPRECATED: Use llm_perf.modeling.models.wan_video instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.models.wan_video import (
    ShardedLayerNorm,
    ShardedT5Block,
    ShardedWanTextEncoder,
    ShardedWanDiTBlock,
    ShardedWanDiT,
    ShardedWanVAE,
)

__all__ = [
    "ShardedLayerNorm",
    "ShardedT5Block",
    "ShardedWanTextEncoder",
    "ShardedWanDiTBlock",
    "ShardedWanDiT",
    "ShardedWanVAE",
]
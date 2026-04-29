"""Vision models using Sharded interface (backward compatibility).

DEPRECATED: Use llm_perf.modeling.base.vision instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.base.vision import (
    ShardedConv2d,
    ShardedConv3d,
    ShardedGroupNorm,
    ShardedResNetBlock2d,
    ShardedResNetBlock3d,
    ShardedAttentionBlock2d,
    ShardedAttentionBlock3d,
    ShardedResNet,
)

__all__ = [
    "ShardedConv2d",
    "ShardedConv3d",
    "ShardedGroupNorm",
    "ShardedResNetBlock2d",
    "ShardedResNetBlock3d",
    "ShardedAttentionBlock2d",
    "ShardedAttentionBlock3d",
    "ShardedResNet",
]
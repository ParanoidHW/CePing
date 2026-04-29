"""HunyuanVideo base layers (backward compatibility).

DEPRECATED: Use llm_perf.modeling.base.dit_layers instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.base.dit_layers import (
    ShardedModulateDiT,
    ShardedPatchEmbed3D,
    ShardedTimestepEmbedder,
)

__all__ = [
    "ShardedModulateDiT",
    "ShardedPatchEmbed3D",
    "ShardedTimestepEmbedder",
]
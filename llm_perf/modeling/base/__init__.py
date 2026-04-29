"""Base layers for modeling.

This package contains fundamental building blocks:
- layers: Basic transformer layers (embedding, attention, FFN, etc.)
- vision: Vision layers (convolutions, norms, blocks)
- dit_layers: DiT-specific layers (modulation, patch embed, timestep embedder)
- dit_blocks: DiT blocks (double stream, single stream)
"""

from .layers import (
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

from .vision import (
    ShardedConv2d,
    ShardedConv3d,
    ShardedGroupNorm,
    ShardedResNetBlock2d,
    ShardedResNetBlock3d,
    ShardedAttentionBlock2d,
    ShardedAttentionBlock3d,
    ShardedResNet,
)

from .dit_layers import (
    ShardedModulateDiT,
    ShardedPatchEmbed3D,
    ShardedTimestepEmbedder,
)

from .dit_blocks import (
    ShardedMMDoubleStreamBlock,
    ShardedMMSingleStreamBlock,
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
    "ShardedConv2d",
    "ShardedConv3d",
    "ShardedGroupNorm",
    "ShardedResNetBlock2d",
    "ShardedResNetBlock3d",
    "ShardedAttentionBlock2d",
    "ShardedAttentionBlock3d",
    "ShardedResNet",
    "ShardedModulateDiT",
    "ShardedPatchEmbed3D",
    "ShardedTimestepEmbedder",
    "ShardedMMDoubleStreamBlock",
    "ShardedMMSingleStreamBlock",
]
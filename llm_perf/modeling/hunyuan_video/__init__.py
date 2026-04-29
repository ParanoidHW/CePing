"""HunyuanVideo models (backward compatibility).

DEPRECATED: Use llm_perf.modeling.models.hunyuan_video and llm_perf.modeling.base instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.models.hunyuan_video import ShardedHYVideoDiT
from llm_perf.modeling.base.dit_layers import (
    ShardedModulateDiT,
    ShardedPatchEmbed3D,
    ShardedTimestepEmbedder,
)
from llm_perf.modeling.base.dit_blocks import (
    ShardedMMDoubleStreamBlock,
    ShardedMMSingleStreamBlock,
)
from llm_perf.modeling.base.vae_3d import (
    ShardedVideoVAEEncoder,
    ShardedVideoVAEDecoder,
    ShardedVideoVAE,
)

__all__ = [
    "ShardedHYVideoDiT",
    "ShardedModulateDiT",
    "ShardedPatchEmbed3D",
    "ShardedTimestepEmbedder",
    "ShardedMMDoubleStreamBlock",
    "ShardedMMSingleStreamBlock",
    "ShardedVideoVAEEncoder",
    "ShardedVideoVAEDecoder",
    "ShardedVideoVAE",
]
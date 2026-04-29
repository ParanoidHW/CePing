"""HunyuanImage 3.0 Models (backward compatibility).

DEPRECATED: Use llm_perf.modeling.models.hunyuan_image instead.
This module re-exports from the new location for backward compatibility.
"""

from llm_perf.modeling.models.hunyuan_image import (
    HunyuanImage3TextModel,
    HunyuanImage3DiffusionModel,
    ShardedHunyuanMoEBlock,
    ShardedHunyuanAttention,
    HunyuanT5Encoder,
    HunyuanVAEEncoder,
    HunyuanVAEDecoder,
    ShardedT5Block,
    ShardedLayerNorm,
)

__all__ = [
    "HunyuanImage3TextModel",
    "HunyuanImage3DiffusionModel",
    "ShardedHunyuanMoEBlock",
    "ShardedHunyuanAttention",
    "HunyuanT5Encoder",
    "HunyuanVAEEncoder",
    "HunyuanVAEDecoder",
    "ShardedT5Block",
    "ShardedLayerNorm",
]
"""Model implementations.

This package contains complete model implementations:
- llama: Llama model with transformer blocks
- deepseek: DeepSeek model with MoE blocks
- wan_video: Wan video generation models
- hunyuan_video: HunyuanVideo diffusion transformer
- qwen3_5: Qwen3.5 models with hybrid attention
- hunyuan_image: HunyuanImage models with MoE and QK norm
"""

from .llama import ShardedTransformerBlock, LlamaModel
from .deepseek import ShardedMoEBlock, DeepSeekModel
from .wan_video import (
    ShardedLayerNorm,
    ShardedT5Block,
    ShardedWanTextEncoder,
    ShardedWanDiTBlock,
    ShardedWanDiT,
    ShardedWanVAE,
)
from .hunyuan_video import ShardedHYVideoDiT
from .qwen3_5 import (
    Qwen3_5MoEModel,
    Qwen3_5Model,
    ShardedQwen3_5MoEBlock,
    ShardedQwen3_5DenseBlock,
    generate_layer_types,
)
from .hunyuan_image import (
    HunyuanImage3TextModel,
    HunyuanImage3DiffusionModel,
    ShardedHunyuanMoEBlock,
    ShardedHunyuanAttention,
)

__all__ = [
    "ShardedTransformerBlock",
    "LlamaModel",
    "ShardedMoEBlock",
    "DeepSeekModel",
    "ShardedLayerNorm",
    "ShardedT5Block",
    "ShardedWanTextEncoder",
    "ShardedWanDiTBlock",
    "ShardedWanDiT",
    "ShardedWanVAE",
    "ShardedHYVideoDiT",
    "Qwen3_5MoEModel",
    "Qwen3_5Model",
    "ShardedQwen3_5MoEBlock",
    "ShardedQwen3_5DenseBlock",
    "generate_layer_types",
    "HunyuanImage3TextModel",
    "HunyuanImage3DiffusionModel",
    "ShardedHunyuanMoEBlock",
    "ShardedHunyuanAttention",
]
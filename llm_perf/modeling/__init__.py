"""Unified modeling framework for LLM performance evaluation.

PyTorch-like interface for defining models with automatic sharding constraints.
"""

from .tensor import ShardedTensor
from .module import ShardedModule, ModuleInstance, WeightInstance, ActivationInstance

from llm_perf.kernels.op import (
    Op,
    MatmulOp,
    AttentionOp,
    RMSNormOp,
    EmbeddingOp,
    ActivationOp,
    MoEExpertOp,
    ViewOp,
    TransposeOp,
    CommOp,
    Conv2dOp,
    Conv3dOp,
    GroupNormOp,
)

from .layers import (
    ShardedEmbedding,
    ShardedRMSNorm,
    ShardedAttention,
    ShardedFFN,
    ShardedLMHead,
    ShardedMoE,
    silu,
    gelu,
    flash_attention,
)

from .mla import ShardedMLA

from .models import (
    ShardedTransformerBlock,
    ShardedMoEBlock,
    LlamaModel,
    DeepSeekModel,
)

from .vision import (
    ShardedConv2d,
    ShardedConv3d,
    ShardedGroupNorm,
    ShardedResNetBlock2d,
    ShardedResNetBlock3d,
    ShardedAttentionBlock2d,
    ShardedAttentionBlock3d,
    ShardedVAEEncoder,
    ShardedVAEDecoder,
    ShardedVAE,
    ShardedResNet,
)

from .wan import (
    ShardedLayerNorm,
    ShardedT5Block,
    ShardedWanTextEncoder,
    ShardedWanDiTBlock,
    ShardedWanDiT,
    ShardedWanVAE,
)

from llm_perf.strategy import (
    ParallelContext,
    SPType,
    CommDomain,
    PPStrategy,
    PPSchedule,
    PPModel,
    PPStageModule,
)

from .config_compat import SimpleModelConfig
from .registry import (
    ModelingRegistry,
    ModelInfo,
    register_all_models,
    get_model_presets,
    get_presets_by_sparse_type,
    create_model_from_config,
)

__all__ = [
    "ShardedTensor",
    "ShardedModule",
    "ModuleInstance",
    "WeightInstance",
    "ActivationInstance",
    "Op",
    "MatmulOp",
    "AttentionOp",
    "RMSNormOp",
    "EmbeddingOp",
    "ActivationOp",
    "MoEExpertOp",
    "ViewOp",
    "TransposeOp",
    "CommOp",
    "Conv2dOp",
    "Conv3dOp",
    "GroupNormOp",
    "ShardedEmbedding",
    "ShardedRMSNorm",
    "ShardedAttention",
    "ShardedFFN",
    "ShardedLMHead",
    "ShardedMoE",
    "ShardedTransformerBlock",
    "ShardedMoEBlock",
    "ShardedMLA",
    "silu",
    "gelu",
    "flash_attention",
    "LlamaModel",
    "DeepSeekModel",
    "ShardedConv2d",
    "ShardedConv3d",
    "ShardedGroupNorm",
    "ShardedResNetBlock2d",
    "ShardedResNetBlock3d",
    "ShardedAttentionBlock2d",
    "ShardedAttentionBlock3d",
    "ShardedVAEEncoder",
    "ShardedVAEDecoder",
    "ShardedVAE",
    "ShardedResNet",
    "ShardedLayerNorm",
    "ShardedT5Block",
    "ShardedWanTextEncoder",
    "ShardedWanDiTBlock",
    "ShardedWanDiT",
    "ShardedWanVAE",
    "ParallelContext",
    "SPType",
    "CommDomain",
    "PPStrategy",
    "PPSchedule",
    "PPModel",
    "PPStageModule",
    "ModelingRegistry",
    "ModelInfo",
    "register_all_models",
    "get_model_presets",
    "get_presets_by_sparse_type",
    "create_model_from_config",
    "SimpleModelConfig",
]

"""Torch-like unified modeling framework for distributed training/inference.

This module provides a PyTorch-like interface for defining models with
automatic sharding constraint derivation and performance estimation.

Key components:
- ShardedTensor: Tensor with sharding constraints, like torch.Tensor
- ShardedModule: Module base class, like torch.nn.Module
- ParallelContext: Parallel strategy context
- ModuleInstance: Runtime instance with physical shape and performance
- PPStrategy: Pipeline parallelism strategy
- PPModel: Model wrapper with PP stage division
"""

from .tensor import ShardedTensor
from .module import ShardedModule, ModuleInstance, WeightInstance, ActivationInstance
from .op import (
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
from .parallel_context import ParallelContext, SPType, CommDomain
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

from .models import (
    ShardedTransformerBlock,
    ShardedMoEBlock,
    LlamaModel,
    DeepSeekModel,
)

from .pp_strategy import PPStrategy, PPSchedule
from .pp_model import PPModel, PPStageModule

from .vision import (
    ShardedConv2d,
    ShardedConv3d,
    ShardedGroupNorm,
    conv2d,
    conv3d,
)

from .vision_models import (
    ShardedResNetBlock2d,
    ShardedResNetBlock3d,
    ShardedVAEEncoder,
    ShardedVAEDecoder,
    ShardedVAE,
)

from .wan import (
    ShardedLayerNorm,
    ShardedT5Block,
    ShardedWanTextEncoder,
    ShardedWanDiTBlock,
    ShardedWanDiT,
    ShardedWanVAE,
)

from .mla import ShardedMLA

from .vision_models import (
    ShardedResNetBlock2d,
    ShardedResNetBlock3d,
    ShardedAttentionBlock2d,
    ShardedAttentionBlock3d,
    ShardedVAEEncoder,
    ShardedVAEDecoder,
    ShardedVAE,
    ShardedResNet,
)

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
    "ParallelContext",
    "SPType",
    "CommDomain",
    "ShardedEmbedding",
    "ShardedRMSNorm",
    "ShardedAttention",
    "ShardedFFN",
    "ShardedLMHead",
    "ShardedMoE",
    "silu",
    "gelu",
    "flash_attention",
    "ShardedTransformerBlock",
    "ShardedMoEBlock",
    "LlamaModel",
    "DeepSeekModel",
    "PPStrategy",
    "PPSchedule",
    "PPModel",
    "PPStageModule",
    "ShardedConv2d",
    "ShardedConv3d",
    "ShardedGroupNorm",
    "conv2d",
    "conv3d",
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
    "ShardedMLA",
    "ModelingRegistry",
    "ModelInfo",
    "register_all_models",
    "get_model_presets",
    "get_presets_by_sparse_type",
    "create_model_from_config",
]

"""Unified modeling framework for LLM performance evaluation.

PyTorch-like interface for defining models with automatic sharding constraints.

Structure:
- base: ShardedTensor, ShardedModule, Op
- layers: ShardedEmbedding, ShardedAttention, ShardedFFN, ShardedMLA
- models: LlamaModel, DeepSeekModel, ShardedVAE, ShardedResNet, Wan models
- parallel: ParallelContext, PPModel, PPStrategy
- utils: ModelingRegistry, create_model_from_config
"""

from .base import (
    ShardedTensor,
    ShardedModule,
    ModuleInstance,
    WeightInstance,
    ActivationInstance,
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
    ShardedMLA,
    silu,
    gelu,
    flash_attention,
)

from .models import (
    ShardedTransformerBlock,
    ShardedMoEBlock,
    LlamaModel,
    DeepSeekModel,
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
    ShardedLayerNorm,
    ShardedT5Block,
    ShardedWanTextEncoder,
    ShardedWanDiTBlock,
    ShardedWanDiT,
    ShardedWanVAE,
)

from .parallel import (
    ParallelContext,
    SPType,
    CommDomain,
    PPStrategy,
    PPSchedule,
    PPModel,
    PPStageModule,
)

from .utils import (
    SimpleModelConfig,
    conv2d,
    conv3d,
)


def get_registry():
    """Lazy load registry to avoid circular imports."""
    from .utils.registry import (
        ModelingRegistry,
        ModelInfo,
        register_all_models,
        get_model_presets,
        get_presets_by_sparse_type,
        create_model_from_config,
    )

    return (
        ModelingRegistry,
        ModelInfo,
        register_all_models,
        get_model_presets,
        get_presets_by_sparse_type,
        create_model_from_config,
    )


ModelingRegistry = None
ModelInfo = None
register_all_models = None
get_model_presets = None
get_presets_by_sparse_type = None
create_model_from_config = None


def __getattr__(name):
    if name in [
        "ModelingRegistry",
        "ModelInfo",
        "register_all_models",
        "get_model_presets",
        "get_presets_by_sparse_type",
        "create_model_from_config",
    ]:
        funcs = get_registry()
        mapping = {
            "ModelingRegistry": funcs[0],
            "ModelInfo": funcs[1],
            "register_all_models": funcs[2],
            "get_model_presets": funcs[3],
            "get_presets_by_sparse_type": funcs[4],
            "create_model_from_config": funcs[5],
        }
        return mapping[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")


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
    "conv2d",
    "conv3d",
]

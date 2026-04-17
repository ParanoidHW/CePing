"""Torch-like unified modeling framework for distributed training/inference.

This module provides a PyTorch-like interface for defining models with
automatic sharding constraint derivation and performance estimation.

Key components:
- ShardedTensor: Tensor with sharding constraints, like torch.Tensor
- ShardedModule: Module base class, like torch.nn.Module
- ParallelContext: Parallel strategy context
- ModuleInstance: Runtime instance with physical shape and performance
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
)
from .parallel_context import ParallelContext, SPType, CommDomain
from .layers import (
    ShardedEmbedding,
    ShardedRMSNorm,
    ShardedAttention,
    ShardedFFN,
    ShardedLMHead,
    silu,
    gelu,
    flash_attention,
)

from .models import (
    ShardedTransformerBlock,
    LlamaModel,
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
    "ParallelContext",
    "SPType",
    "CommDomain",
    "ShardedEmbedding",
    "ShardedRMSNorm",
    "ShardedAttention",
    "ShardedFFN",
    "ShardedLMHead",
    "silu",
    "gelu",
    "flash_attention",
    "ShardedTransformerBlock",
    "LlamaModel",
]

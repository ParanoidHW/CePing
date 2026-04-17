"""Kernel evaluation modules."""

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
from .base import Kernel, KernelConfig, KernelType
from .compute import ComputeKernel, ComputeKernelRegistry
from .communication import CommKernel, CommKernelRegistry
from .functional import (
    linear,
    embedding,
    rms_norm,
    flash_attention,
    silu,
    gelu,
    conv2d,
    conv3d,
)
from .backend import (
    KernelBackend,
    TheoryBackend,
    ProfilingBackend,
    MicroarchBackend,
    KernelBackendRegistry,
    get_backend_registry,
)

__all__ = [
    # Base classes
    "Kernel",
    "KernelConfig",
    "KernelType",
    "ComputeKernel",
    "ComputeKernelRegistry",
    "CommKernel",
    "CommKernelRegistry",
    # Functional API (torch-like)
    "KernelResult",
    "linear",
    "bmm",
    "scaled_dot_product_attention",
    "flash_attention",
    "mla_attention",
    "layer_norm",
    "rms_norm",
    "silu",
    "gelu",
    "relu",
    "softmax",
    "conv2d",
    "conv3d",
    "embedding",
    # Utilities
    "kernel_result_to_layer",
    # Backend layer
    "KernelBackend",
    "TheoryBackend",
    "ProfilingBackend",
    "MicroarchBackend",
    "KernelBackendRegistry",
    "get_backend_registry",
]

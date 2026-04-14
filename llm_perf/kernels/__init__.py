"""Kernel evaluation modules."""

from .base import Kernel, KernelConfig, KernelType
from .compute import ComputeKernel, ComputeKernelRegistry
from .communication import CommKernel, CommKernelRegistry
from .functional import (
    KernelResult,
    linear,
    bmm,
    scaled_dot_product_attention,
    layer_norm,
    rms_norm,
    silu,
    gelu,
    relu,
    softmax,
    conv2d,
    conv3d,
    embedding,
)
from .layers import (
    linear_layer,
    attention_layer,
    ffn_layer,
    norm_layer,
    transformer_block,
    summarize_block,
)
from .utils import kernel_result_to_layer

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
    "layer_norm",
    "rms_norm",
    "silu",
    "gelu",
    "relu",
    "softmax",
    "conv2d",
    "conv3d",
    "embedding",
    # Layer builders
    "linear_layer",
    "attention_layer",
    "ffn_layer",
    "norm_layer",
    "transformer_block",
    "summarize_block",
]

"""Utilities for integrating kernels with models."""

import math
from typing import Tuple, Optional
from .functional import KernelResult
from llm_perf.legacy.models.base import LayerConfig, SubmoduleType


def kernel_result_to_layer(
    name: str,
    result: KernelResult,
    is_moe: bool = False,
    submodule_type: Optional[SubmoduleType] = None,
) -> LayerConfig:
    """Convert KernelResult to LayerConfig with activation from output shape.

    Args:
        name: Layer name
        result: KernelResult from functional API (includes params, param_bytes, dtype)
        is_moe: Whether this is an MoE layer
        submodule_type: Submodule type for transformer components

    Returns:
        LayerConfig for the model
    """
    output_numel = math.prod(result.output)
    dtype_size = result.get_dtype_size()
    activation_bytes = output_numel * dtype_size

    return LayerConfig(
        name=name,
        input_shape=result.input_shapes[0] if result.input_shapes else result.output,
        output_shape=result.output,
        params_count=result.params,
        flops=result.flops,
        activation_bytes=activation_bytes,
        is_moe=is_moe,
        submodule_type=submodule_type or SubmoduleType.OTHER,
    )


def add_param_count(layer: LayerConfig, param_count: int) -> LayerConfig:
    """Add parameter count to a layer.
    
    Args:
        layer: Existing LayerConfig
        param_count: Number of parameters to add
    
    Returns:
        New LayerConfig with updated param count
    """
    return LayerConfig(
        name=layer.name,
        input_shape=layer.input_shape,
        output_shape=layer.output_shape,
        params_count=layer.params_count + param_count,
        flops=layer.flops,
        activation_bytes=layer.activation_bytes,
        is_moe=layer.is_moe,
    )

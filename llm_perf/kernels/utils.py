"""Utilities for integrating kernels with models."""

import math
from typing import Tuple, Optional
from .functional import KernelResult
from ..models.base import LayerConfig


def kernel_result_to_layer(
    name: str,
    result: KernelResult,
    params: Optional[int] = None,
    dtype_size: int = 2,
    is_moe: bool = False,
) -> LayerConfig:
    """Convert KernelResult to LayerConfig with activation from output shape.
    
    Args:
        name: Layer name
        result: KernelResult from functional API (now includes params and param_bytes)
        params: Parameter count. If None, uses result.params
        dtype_size: Bytes per element (for activation calculation)
        is_moe: Whether this is an MoE layer
    
    Returns:
        LayerConfig for the model
    """
    output_numel = math.prod(result.output)
    activation_bytes = output_numel * dtype_size
    
    # Use result.params if params not explicitly provided
    actual_params = result.params if params is None else params
    
    return LayerConfig(
        name=name,
        input_shape=result.input_shapes[0] if result.input_shapes else result.output,
        output_shape=result.output,
        params_count=actual_params,
        flops=result.flops,
        activation_bytes=activation_bytes,
        is_moe=is_moe,
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

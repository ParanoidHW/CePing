"""Utilities for integrating kernels with models."""

import math
from typing import Tuple
from .functional import KernelResult
from ..models.base import LayerConfig


def kernel_result_to_layer(
    name: str,
    result: KernelResult,
    params: int = 0,
    dtype_size: int = 2
) -> LayerConfig:
    """Convert KernelResult to LayerConfig with activation from output shape.
    
    Args:
        name: Layer name
        result: KernelResult from functional API
        params: Parameter count
        dtype_size: Bytes per element
    
    Returns:
        LayerConfig for the model
    """
    output_numel = math.prod(result.output)
    activation_bytes = output_numel * dtype_size
    
    return LayerConfig(
        name=name,
        input_shape=result.input_shapes[0] if result.input_shapes else result.output,
        output_shape=result.output,
        params_count=params,
        flops=result.flops,
        activation_bytes=activation_bytes,
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
    )

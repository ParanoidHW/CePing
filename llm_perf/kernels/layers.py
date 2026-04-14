"""Layer builders using functional kernels.

This module provides utilities to build model layers using the functional kernel API,
similar to how torch.nn layers use torch.nn.functional.
"""

from typing import List, Optional, Tuple
import math

from .functional import (
    KernelResult,
    linear,
    scaled_dot_product_attention,
    layer_norm,
    rms_norm,
)
from ..models.base import LayerConfig


def linear_layer(
    in_features: int,
    out_features: int,
    batch_size: int = 1,
    seq_len: int = 1,
    bias: bool = True,
    dtype: str = "fp16",
    name: str = "linear"
) -> Tuple[LayerConfig, KernelResult]:
    """Create a linear layer.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        batch_size: Batch size for FLOPs calculation
        seq_len: Sequence length for FLOPs calculation
        bias: Whether to include bias
        dtype: Data type
        name: Layer name
    
    Returns:
        Tuple of (LayerConfig, KernelResult)
    """
    # Build input shape: flatten batch and seq_len for matrix multiplication
    m = batch_size * seq_len
    input_shape = (m, in_features)
    weight_shape = (out_features, in_features)
    bias_shape = (out_features,) if bias else None
    
    result = linear(input_shape, weight_shape, bias_shape, dtype)
    
    # Output shape restores batch and seq dimensions
    output_shape = (batch_size, seq_len, out_features)
    
    layer = LayerConfig(
        name=name,
        input_shape=(batch_size, seq_len, in_features),
        output_shape=output_shape,
        params_count=result.params,
        flops=result.flops,
        activation_bytes=math.prod(output_shape) * (2 if dtype == "fp16" else 4)
    )
    
    return layer, result


def attention_layer(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    kv_seq_len: Optional[int] = None,
    cross_attention: bool = False,
    dtype: str = "fp16",
    name: str = "attention"
) -> List[Tuple[LayerConfig, KernelResult]]:
    """Create self-attention or cross-attention layers.
    
    Args:
        batch_size: Batch size
        seq_len: Query sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        kv_seq_len: Key/Value sequence length (for cross-attention)
        cross_attention: Whether this is cross-attention
        dtype: Data type
        name: Layer name prefix
    
    Returns:
        List of (LayerConfig, KernelResult) tuples for each sub-layer
    """
    kv_len = kv_seq_len or seq_len
    hidden_size = num_heads * head_dim
    
    layers = []
    
    # Q projection (always needed)
    q_layer, q_result = linear_layer(
        hidden_size, hidden_size, batch_size, seq_len,
        bias=False, dtype=dtype, name=f"{name}_q"
    )
    layers.append((q_layer, q_result))
    
    # K, V projections
    if cross_attention:
        kv_in_features = hidden_size  # In cross-attn, KV come from different input
    else:
        kv_in_features = hidden_size  # In self-attn, same as Q
    
    k_layer, k_result = linear_layer(
        kv_in_features, hidden_size, batch_size, kv_len,
        bias=False, dtype=dtype, name=f"{name}_k"
    )
    layers.append((k_layer, k_result))
    
    v_layer, v_result = linear_layer(
        kv_in_features, hidden_size, batch_size, kv_len,
        bias=False, dtype=dtype, name=f"{name}_v"
    )
    layers.append((v_layer, v_result))
    
    # Attention computation
    q_shape = (batch_size, num_heads, seq_len, head_dim)
    k_shape = (batch_size, num_heads, kv_len, head_dim)
    v_shape = (batch_size, num_heads, kv_len, head_dim)
    
    attn_result = scaled_dot_product_attention(q_shape, k_shape, v_shape, dtype=dtype)
    
    attn_layer = LayerConfig(
        name=f"{name}_compute",
        input_shape=(batch_size, seq_len, hidden_size * 3),
        output_shape=(batch_size, seq_len, hidden_size),
        params_count=0,
        flops=attn_result.flops,
        activation_bytes=math.prod(attn_result.output) * (2 if dtype == "fp16" else 4)
    )
    layers.append((attn_layer, attn_result))
    
    # Output projection
    o_layer, o_result = linear_layer(
        hidden_size, hidden_size, batch_size, seq_len,
        bias=False, dtype=dtype, name=f"{name}_o"
    )
    layers.append((o_layer, o_result))
    
    return layers


def ffn_layer(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    gated: bool = True,
    activation: str = "gelu",
    dtype: str = "fp16",
    name: str = "ffn"
) -> List[Tuple[LayerConfig, KernelResult]]:
    """Create FFN/MLP layers.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension
        intermediate_size: Intermediate dimension
        gated: Whether to use gated activation (like GLU)
        activation: Activation function name
        dtype: Data type
        name: Layer name prefix
    
    Returns:
        List of (LayerConfig, KernelResult) tuples
    """
    layers = []
    
    if gated:
        # Gated FFN: two input projections (wi_0, wi_1) -> element-wise mul -> output
        # FFN in
        in_layer, in_result = linear_layer(
            hidden_size, intermediate_size * 2, batch_size, seq_len,
            bias=False, dtype=dtype, name=f"{name}_in"
        )
        layers.append((in_layer, in_result))
        
        # Activation
        from .functional import gelu, silu
        act_func = gelu if activation == "gelu" else silu
        act_result = act_func((batch_size, seq_len, intermediate_size * 2), dtype)
        
        act_layer = LayerConfig(
            name=f"{name}_act",
            input_shape=(batch_size, seq_len, intermediate_size * 2),
            output_shape=(batch_size, seq_len, intermediate_size),
            params_count=0,
            flops=act_result.flops,
            activation_bytes=math.prod(act_result.output) * (2 if dtype == "fp16" else 4)
        )
        layers.append((act_layer, act_result))
    else:
        # Standard FFN
        in_layer, in_result = linear_layer(
            hidden_size, intermediate_size, batch_size, seq_len,
            bias=True, dtype=dtype, name=f"{name}_in"
        )
        layers.append((in_layer, in_result))
        
        # Activation
        from .functional import gelu
        act_result = gelu((batch_size, seq_len, intermediate_size), dtype)
        
        act_layer = LayerConfig(
            name=f"{name}_act",
            input_shape=(batch_size, seq_len, intermediate_size),
            output_shape=(batch_size, seq_len, intermediate_size),
            params_count=0,
            flops=act_result.flops,
            activation_bytes=math.prod(act_result.output) * (2 if dtype == "fp16" else 4)
        )
        layers.append((act_layer, act_result))
    
    # Output projection
    out_layer, out_result = linear_layer(
        intermediate_size, hidden_size, batch_size, seq_len,
        bias=True, dtype=dtype, name=f"{name}_out"
    )
    layers.append((out_layer, out_result))
    
    return layers


def norm_layer(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    norm_type: str = "layernorm",
    elementwise_affine: bool = True,
    dtype: str = "fp16",
    name: str = "norm"
) -> Tuple[LayerConfig, KernelResult]:
    """Create normalization layer.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension
        norm_type: Type of normalization ("layernorm", "rmsnorm")
        elementwise_affine: Whether to use learnable affine
        dtype: Data type
        name: Layer name
    
    Returns:
        Tuple of (LayerConfig, KernelResult)
    """
    input_shape = (batch_size, seq_len, hidden_size)
    
    if norm_type == "rmsnorm":
        result = rms_norm(input_shape, dim=-1, dtype=dtype)
    else:
        result = layer_norm(input_shape, (hidden_size,), elementwise_affine, dtype)
    
    layer = LayerConfig(
        name=name,
        input_shape=input_shape,
        output_shape=input_shape,
        params_count=result.params,
        flops=result.flops,
        activation_bytes=math.prod(result.output) * (2 if dtype == "fp16" else 4)
    )
    
    return layer, result


def transformer_block(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    kv_seq_len: Optional[int] = None,
    cross_attention: bool = False,
    norm_type: str = "layernorm",
    gated_ffn: bool = True,
    dtype: str = "fp16",
    name: str = "block"
) -> List[Tuple[LayerConfig, KernelResult]]:
    """Create a complete transformer block.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        kv_seq_len: KV sequence length for cross-attention
        cross_attention: Whether to include cross-attention
        norm_type: Type of normalization
        gated_ffn: Whether FFN is gated
        dtype: Data type
        name: Layer name prefix
    
    Returns:
        List of (LayerConfig, KernelResult) tuples
    """
    layers = []
    
    # Pre-attention normalization
    pre_attn_norm, _ = norm_layer(
        batch_size, seq_len, hidden_size,
        norm_type=norm_type, elementwise_affine=False,
        dtype=dtype, name=f"{name}_norm1"
    )
    layers.append((pre_attn_norm, _))
    
    # Self-attention
    self_attn_layers = attention_layer(
        batch_size, seq_len, num_heads, hidden_size // num_heads,
        dtype=dtype, name=f"{name}_self_attn"
    )
    layers.extend(self_attn_layers)
    
    # Cross-attention (if specified)
    if cross_attention:
        pre_cross_norm, _ = norm_layer(
            batch_size, seq_len, hidden_size,
            norm_type=norm_type, elementwise_affine=True,
            dtype=dtype, name=f"{name}_norm3"
        )
        layers.append((pre_cross_norm, _))
        
        cross_attn_layers = attention_layer(
            batch_size, seq_len, num_heads, hidden_size // num_heads,
            kv_seq_len=kv_seq_len, cross_attention=True,
            dtype=dtype, name=f"{name}_cross_attn"
        )
        layers.extend(cross_attn_layers)
    
    # Pre-FFN normalization
    pre_ffn_norm, _ = norm_layer(
        batch_size, seq_len, hidden_size,
        norm_type=norm_type, elementwise_affine=False,
        dtype=dtype, name=f"{name}_norm2"
    )
    layers.append((pre_ffn_norm, _))
    
    # FFN
    ffn_layers = ffn_layer(
        batch_size, seq_len, hidden_size, intermediate_size,
        gated=gated_ffn, dtype=dtype, name=f"{name}_ffn"
    )
    layers.extend(ffn_layers)
    
    return layers


def summarize_block(layers: List[Tuple[LayerConfig, KernelResult]]) -> dict:
    """Summarize a block of layers.
    
    Args:
        layers: List of (LayerConfig, KernelResult) tuples
    
    Returns:
        Dictionary with aggregated metrics
    """
    total_params = sum(layer.params_count for layer, _ in layers)
    total_flops = sum(layer.flops for layer, _ in layers)
    total_bytes = sum(result.bytes_accessed for _, result in layers)
    
    return {
        "num_layers": len(layers),
        "total_params": total_params,
        "total_params_mb": total_params * 2 / 1024 / 1024,  # fp16
        "total_flops": total_flops,
        "total_flops_g": total_flops / 1e9,
        "total_bytes_accessed": total_bytes,
        "total_bytes_mb": total_bytes / 1024 / 1024,
        "arithmetic_intensity": total_flops / total_bytes if total_bytes > 0 else float('inf'),
    }

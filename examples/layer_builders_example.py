"""Layer builders using functional kernels - EXAMPLE CODE.

This module provides example utilities to build model layers using the functional kernel API.
These are example implementations showing how to use the kernel API.
For production model implementations, see llm_perf/models/
"""

from typing import List, Optional, Tuple
import math

from llm_perf.kernels.functional import (
    KernelResult,
    linear,
    scaled_dot_product_attention,
    flash_attention,
    mla_attention,
    layer_norm,
    rms_norm,
)
from llm_perf.models.base import LayerConfig


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
    name: str = "attention",
    # GQA support
    num_kv_heads: Optional[int] = None,  # For GQA: num_kv_heads < num_heads
    # Flash Attention support
    use_flash_attention: bool = False,
    is_causal: bool = False,
) -> List[Tuple[LayerConfig, KernelResult]]:
    """Create self-attention or cross-attention layers.
    
    Supports:
    - MHA (Multi-Head Attention): num_kv_heads = num_heads
    - GQA (Grouped Query Attention): num_kv_heads < num_heads
    - Flash Attention: optimized kernel with reduced HBM traffic
    
    Args:
        batch_size: Batch size
        seq_len: Query sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        kv_seq_len: Key/Value sequence length (for cross-attention)
        cross_attention: Whether this is cross-attention
        dtype: Data type
        name: Layer name prefix
        num_kv_heads: Number of KV heads (for GQA). If None, equals num_heads
        use_flash_attention: If True, use Flash Attention kernel
        is_causal: Whether to apply causal mask (for Flash Attention)
    
    Returns:
        List of (LayerConfig, KernelResult) tuples for each sub-layer
    """
    kv_len = kv_seq_len or seq_len
    hidden_size = num_heads * head_dim
    
    # For GQA: num_kv_heads < num_heads
    num_kv_heads = num_kv_heads or num_heads
    kv_hidden_size = num_kv_heads * head_dim
    
    use_gqa = num_kv_heads < num_heads
    
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
    
    # For GQA: KV projections output fewer heads
    k_layer, k_result = linear_layer(
        kv_in_features, kv_hidden_size, batch_size, kv_len,
        bias=False, dtype=dtype, name=f"{name}_k"
    )
    layers.append((k_layer, k_result))
    
    v_layer, v_result = linear_layer(
        kv_in_features, kv_hidden_size, batch_size, kv_len,
        bias=False, dtype=dtype, name=f"{name}_v"
    )
    layers.append((v_layer, v_result))
    
    # Attention computation
    q_shape = (batch_size, num_heads, seq_len, head_dim)
    k_shape = (batch_size, num_kv_heads, kv_len, head_dim)
    v_shape = (batch_size, num_kv_heads, kv_len, head_dim)
    
    if use_flash_attention:
        attn_result = flash_attention(
            q_shape, k_shape, v_shape, 
            is_causal=is_causal, 
            dtype=dtype,
            use_gqa=use_gqa
        )
    else:
        attn_result = scaled_dot_product_attention(
            q_shape, k_shape, v_shape, 
            dtype=dtype,
            use_gqa=use_gqa
        )
    
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


def mla_attention_layer(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    hidden_size: int,
    dtype: str = "fp16",
    name: str = "mla_attention",
    use_absorb: bool = False,  # Inference optimization
) -> List[Tuple[LayerConfig, KernelResult]]:
    """Create Multi-head Latent Attention (MLA) layers.
    
    MLA reduces KV cache memory by compressing KV into a latent vector.
    
    Two modes:
    1. Non-absorb mode (training/standard inference):
       - Explicit KV compression and decompression
       - Standard attention on decompressed KV
    
    2. Absorb mode (inference optimization):
       - KV decompression matrices absorbed into Q and O projections
       - Attention computed directly on compressed representation
       - Significant memory and compute savings
    
    Reference: DeepSeek-V2 Technical Report
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        qk_head_dim: Dimension for QK (after RoPE)
        v_head_dim: Dimension for V heads
        kv_lora_rank: Compressed KV dimension
        hidden_size: Hidden dimension
        dtype: Data type
        name: Layer name prefix
        use_absorb: If True, use absorb mode (inference optimization)
    
    Returns:
        List of (LayerConfig, KernelResult) tuples
    """
    layers = []
    
    # Q projection with compression (down-projection)
    q_down_layer, q_down_result = linear_layer(
        hidden_size, kv_lora_rank, batch_size, seq_len,
        bias=False, dtype=dtype, name=f"{name}_q_down"
    )
    layers.append((q_down_layer, q_down_result))
    
    if not use_absorb:
        # Non-absorb mode: explicit KV decompression
        # KV down-projection (compression)
        kv_down_layer, kv_down_result = linear_layer(
            hidden_size, kv_lora_rank, batch_size, seq_len,
            bias=False, dtype=dtype, name=f"{name}_kv_down"
        )
        layers.append((kv_down_layer, kv_down_result))
        
        # KV up-projection (decompression for K)
        k_up_dim = num_heads * qk_head_dim
        k_up_layer, k_up_result = linear_layer(
            kv_lora_rank, k_up_dim, batch_size, seq_len,
            bias=False, dtype=dtype, name=f"{name}_k_up"
        )
        layers.append((k_up_layer, k_up_result))
        
        # KV up-projection (decompression for V)
        v_up_dim = num_heads * v_head_dim
        v_up_layer, v_up_result = linear_layer(
            kv_lora_rank, v_up_dim, batch_size, seq_len,
            bias=False, dtype=dtype, name=f"{name}_v_up"
        )
        layers.append((v_up_layer, v_up_result))
    else:
        # Absorb mode: only compressed KV, no explicit decompression
        # The decompression is implicitly done through Q and O projections
        kv_down_layer, kv_down_result = linear_layer(
            hidden_size, kv_lora_rank, batch_size, seq_len,
            bias=False, dtype=dtype, name=f"{name}_kv_down"
        )
        layers.append((kv_down_layer, kv_down_result))
    
    # MLA attention computation
    q_shape = (batch_size, num_heads, seq_len, qk_head_dim)
    compressed_kv_shape = (batch_size, seq_len, kv_lora_rank)
    
    if use_absorb:
        # Absorb mode: attention on compressed KV
        attn_result = mla_attention(
            query=q_shape,
            compressed_kv=compressed_kv_shape,
            key=None,  # No explicit K/V in absorb mode
            value=None,
            use_absorb=True,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype
        )
    else:
        # Non-absorb mode: standard attention on decompressed KV
        k_shape = (batch_size, num_heads, seq_len, qk_head_dim)
        v_shape = (batch_size, num_heads, seq_len, v_head_dim)
        attn_result = mla_attention(
            query=q_shape,
            compressed_kv=compressed_kv_shape,
            key=k_shape,
            value=v_shape,
            use_absorb=False,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype
        )
    
    attn_layer = LayerConfig(
        name=f"{name}_compute",
        input_shape=(batch_size, seq_len, hidden_size),
        output_shape=(batch_size, seq_len, num_heads * v_head_dim),
        params_count=0,
        flops=attn_result.flops,
        activation_bytes=math.prod(attn_result.output) * (2 if dtype == "fp16" else 4)
    )
    layers.append((attn_layer, attn_result))
    
    # Output projection
    o_layer, o_result = linear_layer(
        num_heads * v_head_dim, hidden_size, batch_size, seq_len,
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

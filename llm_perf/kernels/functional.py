"""Kernel functional API - similar to torch.nn.functional.

This module provides a torch-like interface for kernel operations,
returning detailed performance metrics (FLOPs, memory bandwidth, etc.).

Example:
    >>> from llm_perf.kernels.functional import linear, attention, layer_norm
    >>> 
    >>> # Linear operation
    >>> result = linear(input, weight, bias)
    >>> print(f"FLOPs: {result.flops}, Memory: {result.bytes_accessed}")
    >>> 
    >>> # Access output tensor
    >>> output = result.output
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

from ..utils.constants import DTYPE_SIZES


@dataclass
class KernelResult:
    """Result of a kernel operation with performance metrics.
    
    Attributes:
        output: The output tensor (shape info)
        flops: Total floating point operations
        bytes_accessed: Total memory bytes read/written
        arithmetic_intensity: FLOPs per byte (flops / bytes_accessed)
        memory_bound: Whether the kernel is memory bound
        input_shapes: Input tensor shapes for reference
        params: Number of parameters involved in the kernel
        param_bytes: Bytes occupied by parameters
        unit_type: Compute unit type ("cube" for Tensor Core, "vector" for CUDA Core)
        dtype: Data type string (e.g., "fp16", "bf16", "fp32")
    """
    output: Tuple[int, ...]  # Output shape
    flops: int
    bytes_accessed: int
    arithmetic_intensity: float
    memory_bound: bool
    input_shapes: List[Tuple[int, ...]]
    params: int = 0
    param_bytes: int = 0
    unit_type: str = "vector"
    dtype: str = "fp16"
    
    def __post_init__(self):
        """Validate and compute derived metrics."""
        if self.bytes_accessed > 0:
            self.arithmetic_intensity = self.flops / self.bytes_accessed
        else:
            self.arithmetic_intensity = float('inf')
    
    def get_dtype_size(self) -> int:
        """Get bytes per element for the data type."""
        return DTYPE_SIZES.get(self.dtype, 2)


def _compute_dtype_size(dtype: str) -> int:
    """Get bytes per element for dtype."""
    return DTYPE_SIZES.get(dtype, 2)


def linear(
    input: Tuple[int, ...],  # (..., in_features)
    weight: Tuple[int, ...],  # (out_features, in_features)
    bias: Optional[Tuple[int, ...]] = None,  # (out_features,)
    dtype: str = "fp16"
) -> KernelResult:
    """Linear transformation: y = x @ W^T + b
    
    Similar to torch.nn.functional.linear
    
    Args:
        input: Shape of input tensor (..., in_features)
        weight: Shape of weight matrix (out_features, in_features)
        bias: Optional shape of bias vector (out_features,)
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    
    Example:
        >>> result = linear((4096,), (5120, 4096))  # input, weight
        >>> print(f"FLOPs: {result.flops / 1e9:.2f}G")
        FLOPs: 42.95G
    """
    dtype_size = _compute_dtype_size(dtype)
    
    # Extract dimensions
    in_features = input[-1]
    out_features = weight[0]
    batch_size = math.prod(input[:-1]) if len(input) > 1 else 1
    
    # Output shape
    output_shape = (*input[:-1], out_features) if len(input) > 1 else (out_features,)
    
    # FLOPs: 2 * batch * in_features * out_features (multiply-add)
    flops = 2 * batch_size * in_features * out_features
    
    # Memory accessed
    # Read: input + weight + bias
    # Write: output
    bytes_accessed = (
        batch_size * in_features * dtype_size +  # input
        in_features * out_features * dtype_size +  # weight
        batch_size * out_features * dtype_size  # output
    )
    if bias is not None:
        bytes_accessed += out_features * dtype_size  # bias
        flops += batch_size * out_features  # bias addition
    
    # Calculate params: weight + bias
    params = in_features * out_features + (out_features if bias else 0)
    param_bytes = params * dtype_size
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=bytes_accessed > flops / 100,  # Heuristic
        input_shapes=[input, weight] + ([bias] if bias else []),
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype
    )


def bmm(
    input: Tuple[int, ...],  # (batch, m, k)
    mat2: Tuple[int, ...],   # (batch, k, n)
    dtype: str = "fp16"
) -> KernelResult:
    """Batch matrix multiplication: out = input @ mat2
    
    Similar to torch.bmm
    
    Args:
        input: Shape (batch, m, k)
        mat2: Shape (batch, k, n)
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    batch, m, k = input
    _, k2, n = mat2
    assert k == k2, f"Dimension mismatch: {k} vs {k2}"
    
    output_shape = (batch, m, n)
    
    # FLOPs: 2 * batch * m * n * k
    flops = 2 * batch * m * n * k
    
    # Memory accessed
    bytes_accessed = (
        batch * m * k * dtype_size +  # input
        batch * k * n * dtype_size +  # mat2
        batch * m * n * dtype_size    # output
    )
    
    # BMM has no learnable parameters
    params = 0
    param_bytes = 0
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=bytes_accessed > flops / 100,
        input_shapes=[input, mat2],
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype
    )


def scaled_dot_product_attention(
    query: Tuple[int, ...],    # (batch, num_heads, seq_len, head_dim)
    key: Tuple[int, ...],      # (batch, num_heads, kv_seq_len, head_dim)
    value: Tuple[int, ...],    # (batch, num_heads, kv_seq_len, head_dim)
    is_causal: bool = False,
    dtype: str = "fp16"
) -> KernelResult:
    """Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
    
    Similar to torch.nn.functional.scaled_dot_product_attention
    
    Args:
        query: Shape (batch, num_heads, seq_len, head_dim)
        key: Shape (batch, num_heads, kv_seq_len, head_dim)
        value: Shape (batch, num_heads, kv_seq_len, head_dim)
        is_causal: Whether to apply causal mask
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    batch, num_heads, seq_len, head_dim = query
    _, _, kv_seq_len, _ = key
    
    output_shape = (batch, num_heads, seq_len, head_dim)
    
    # FLOPs breakdown:
    # 1. Q @ K^T: 2 * batch * num_heads * seq_len * kv_seq_len * head_dim
    # 2. Softmax: 5 * batch * num_heads * seq_len * kv_seq_len (exp, sum, div)
    # 3. Attention @ V: 2 * batch * num_heads * seq_len * head_dim * kv_seq_len
    
    qk_flops = 2 * batch * num_heads * seq_len * kv_seq_len * head_dim
    softmax_flops = 5 * batch * num_heads * seq_len * kv_seq_len
    attn_v_flops = 2 * batch * num_heads * seq_len * head_dim * kv_seq_len
    
    flops = qk_flops + softmax_flops + attn_v_flops
    
    # Memory accessed
    bytes_accessed = (
        batch * num_heads * seq_len * head_dim * dtype_size +  # query
        batch * num_heads * kv_seq_len * head_dim * dtype_size +  # key
        batch * num_heads * kv_seq_len * head_dim * dtype_size +  # value
        batch * num_heads * seq_len * head_dim * dtype_size  # output
    )
    
    # Attention has no learnable parameters
    params = 0
    param_bytes = 0
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,  # Attention is typically memory bound
        input_shapes=[query, key, value],
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype
    )


def layer_norm(
    input: Tuple[int, ...],
    normalized_shape: Tuple[int, ...],
    elementwise_affine: bool = True,
    dtype: str = "fp16"
) -> KernelResult:
    """Layer normalization.
    
    Similar to torch.nn.functional.layer_norm
    
    Args:
        input: Input shape
        normalized_shape: Shape of the normalization dimensions
        elementwise_affine: Whether to use learnable affine parameters
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    # FLOPs per element: mean (N ops), variance (2N ops), normalize (3 ops), affine (2 ops)
    # Total: ~7 FLOPs per element
    numel = math.prod(input)
    flops = numel * 7
    
    # Memory accessed
    bytes_accessed = numel * dtype_size * 2  # read input, write output
    if elementwise_affine:
        bytes_accessed += math.prod(normalized_shape) * dtype_size * 2  # weight + bias
        flops += numel * 2  # scale + shift
    
    # Calculate params: weight + bias (if elementwise_affine)
    params = normalized_shape[0] * (2 if elementwise_affine else 1)
    param_bytes = params * dtype_size
    
    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype
    )


def rms_norm(
    input: Tuple[int, ...],
    dim: int = -1,
    dtype: str = "fp16"
) -> KernelResult:
    """RMS normalization (common in LLMs like LLaMA, T5).
    
    y = x / sqrt(mean(x^2) + eps) * weight
    
    Args:
        input: Input shape
        dim: Dimension to normalize over
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    numel = math.prod(input)
    
    # FLOPs: square (1), mean (N), rsqrt (7), multiply (1), scale (1) per element
    flops = numel * 7
    
    # Memory accessed
    bytes_accessed = numel * dtype_size * 2  # read input, write output
    bytes_accessed += input[dim] * dtype_size  # weight
    
    # Calculate params: only weight
    params = input[dim]
    param_bytes = params * dtype_size
    
    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype
    )


def silu(
    input: Tuple[int, ...],
    dtype: str = "fp16"
) -> KernelResult:
    """SiLU activation: x * sigmoid(x)
    
    Similar to torch.nn.functional.silu
    
    Args:
        input: Input shape
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    numel = math.prod(input)
    # FLOPs: exp (7), add (1), div (1), mul (1) = ~10 FLOPs
    flops = numel * 10
    bytes_accessed = numel * dtype_size * 2  # read + write
    
    # Activation functions have no learnable parameters
    params = 0
    param_bytes = 0
    
    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype
    )


def gelu(
    input: Tuple[int, ...],
    approximate: str = "none",
    dtype: str = "fp16"
) -> KernelResult:
    """GELU activation.
    
    Similar to torch.nn.functional.gelu
    
    Args:
        input: Input shape
        approximate: Approximation method ("none" or "tanh")
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    numel = math.prod(input)
    # FLOPs varies by approximation
    flops_per_elem = 8 if approximate == "tanh" else 15
    flops = numel * flops_per_elem
    bytes_accessed = numel * dtype_size * 2
    
    # Activation functions have no learnable parameters
    params = 0
    param_bytes = 0
    
    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype
    )


def relu(
    input: Tuple[int, ...],
    dtype: str = "fp16"
) -> KernelResult:
    """ReLU activation: max(0, x)
    
    Similar to torch.nn.functional.relu
    
    Args:
        input: Input shape
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    numel = math.prod(input)
    flops = numel  # compare + select
    bytes_accessed = numel * dtype_size * 2
    
    # Activation functions have no learnable parameters
    params = 0
    param_bytes = 0
    
    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype
    )


def softmax(
    input: Tuple[int, ...],
    dim: int = -1,
    dtype: str = "fp16"
) -> KernelResult:
    """Softmax activation.
    
    Similar to torch.nn.functional.softmax
    
    Args:
        input: Input shape
        dim: Dimension to apply softmax
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    numel = math.prod(input)
    # FLOPs: exp (7), sum (N), div (1) per element
    softmax_dim_size = input[dim]
    flops = numel * (7 + softmax_dim_size + 1)
    bytes_accessed = numel * dtype_size * 2
    
    # Softmax has no learnable parameters
    params = 0
    param_bytes = 0
    
    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype
    )


def dropout(
    input: Tuple[int, ...],
    p: float = 0.5,
    dtype: str = "fp16"
) -> KernelResult:
    """Dropout regularization.
    
    Similar to torch.nn.functional.dropout
    
    Args:
        input: Input shape
        p: Dropout probability
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    numel = math.prod(input)
    # FLOPs: random check (1), scale (1), mul (1)
    flops = numel * 3
    bytes_accessed = numel * dtype_size * 2
    
    # Dropout has no learnable parameters
    params = 0
    param_bytes = 0
    
    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype
    )


def conv2d(
    input: Tuple[int, ...],      # (N, C_in, H, W)
    weight: Tuple[int, ...],     # (C_out, C_in, kH, kW)
    bias: Optional[Tuple[int, ...]] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dtype: str = "fp16"
) -> KernelResult:
    """2D convolution.
    
    Similar to torch.nn.functional.conv2d
    
    Args:
        input: Shape (N, C_in, H, W)
        weight: Shape (C_out, C_in, kH, kW)
        bias: Optional shape (C_out,)
        stride: Stride tuple (sh, sw)
        padding: Padding tuple (ph, pw)
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    N, C_in, H, W = input
    C_out, C_in_w, kH, kW = weight
    assert C_in == C_in_w, f"Channel mismatch: {C_in} vs {C_in_w}"
    
    sh, sw = stride
    ph, pw = padding
    
    # Calculate output dimensions
    H_out = (H + 2 * ph - kH) // sh + 1
    W_out = (W + 2 * pw - kW) // sw + 1
    
    output_shape = (N, C_out, H_out, W_out)
    
    # FLOPs: 2 * N * C_out * H_out * W_out * C_in * kH * kW
    flops = 2 * N * C_out * H_out * W_out * C_in * kH * kW
    
    # Memory accessed
    bytes_accessed = (
        N * C_in * H * W * dtype_size +  # input
        C_out * C_in * kH * kW * dtype_size +  # weight
        N * C_out * H_out * W_out * dtype_size  # output
    )
    if bias is not None:
        bytes_accessed += C_out * dtype_size
        flops += N * C_out * H_out * W_out
    
    # Calculate params: weight + bias
    params = C_out * C_in * kH * kW + (C_out if bias else 0)
    param_bytes = params * dtype_size
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=bytes_accessed > flops / 100,
        input_shapes=[input, weight] + ([bias] if bias else []),
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype
    )


def conv3d(
    input: Tuple[int, ...],      # (N, C_in, D, H, W)
    weight: Tuple[int, ...],     # (C_out, C_in, kD, kH, kW)
    bias: Optional[Tuple[int, ...]] = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dtype: str = "fp16"
) -> KernelResult:
    """3D convolution.
    
    Similar to torch.nn.functional.conv3d
    
    Args:
        input: Shape (N, C_in, D, H, W)
        weight: Shape (C_out, C_in, kD, kH, kW)
        bias: Optional shape (C_out,)
        stride: Stride tuple
        padding: Padding tuple
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    N, C_in, D, H, W = input
    C_out, C_in_w, kD, kH, kW = weight
    assert C_in == C_in_w, f"Channel mismatch: {C_in} vs {C_in_w}"
    
    # Calculate output dimensions
    D_out = (D + 2 * padding[0] - kD) // stride[0] + 1
    H_out = (H + 2 * padding[1] - kH) // stride[1] + 1
    W_out = (W + 2 * padding[2] - kW) // stride[2] + 1
    
    output_shape = (N, C_out, D_out, H_out, W_out)
    
    # FLOPs: 2 * N * C_out * D_out * H_out * W_out * C_in * kD * kH * kW
    flops = 2 * N * C_out * D_out * H_out * W_out * C_in * kD * kH * kW
    
    # Memory accessed
    bytes_accessed = (
        N * C_in * D * H * W * dtype_size +  # input
        C_out * C_in * kD * kH * kW * dtype_size +  # weight
        N * C_out * D_out * H_out * W_out * dtype_size  # output
    )
    if bias is not None:
        bytes_accessed += C_out * dtype_size
        flops += N * C_out * D_out * H_out * W_out
    
    # Calculate params: weight + bias
    params = C_out * C_in * kD * kH * kW + (C_out if bias else 0)
    param_bytes = params * dtype_size
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=bytes_accessed > flops / 100,
        input_shapes=[input, weight] + ([bias] if bias else []),
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype
    )


def embedding(
    num_embeddings: int,
    embedding_dim: int,
    input_shape: Tuple[int, ...],
    dtype: str = "fp16"
) -> KernelResult:
    """Embedding lookup.
    
    Similar to torch.nn.functional.embedding
    
    Args:
        num_embeddings: Size of embedding dictionary
        embedding_dim: Size of each embedding vector
        input_shape: Shape of input indices
        dtype: Data type string
    
    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)
    
    output_shape = (*input_shape, embedding_dim)
    
    # FLOPs: mostly memory lookup, minimal compute
    numel = math.prod(input_shape)
    flops = numel  # index lookup
    
    # Memory accessed
    bytes_accessed = (
        num_embeddings * embedding_dim * dtype_size +  # embedding table
        numel * embedding_dim * dtype_size  # output
    )
    
    # Calculate params: embedding table
    params = num_embeddings * embedding_dim
    param_bytes = params * dtype_size
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float('inf'),
        memory_bound=True,
        input_shapes=[input_shape],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype
    )

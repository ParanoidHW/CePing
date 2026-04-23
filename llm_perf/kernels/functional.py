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

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import math

from ..utils.constants import DTYPE_SIZES


@dataclass
class KernelResult:
    """Result of a kernel operation with performance metrics.

    Attributes:
        output: The output tensor (shape info)
        flops: Total floating point operations (forward pass)
        bytes_accessed: Total memory bytes read/written (forward pass)
        arithmetic_intensity: FLOPs per byte (flops / bytes_accessed)
        memory_bound: Whether the kernel is memory bound (computed from arithmetic intensity)
        input_shapes: Input tensor shapes for reference
        params: Number of parameters involved in the kernel
        param_bytes: Bytes occupied by parameters
        unit_type: Compute unit type ("cube" for Tensor Core, "vector" for CUDA Core)
        dtype: Data type string (e.g., "fp16", "bf16", "fp32")
        flops_backward: Backward pass FLOPs (default: 0)
        bytes_accessed_backward: Backward pass memory bytes (default: 0)
        saved_inputs: List of input names that need to be saved for backward pass (default: [])
    """

    output: Tuple[int, ...]  # Output shape
    flops: int
    bytes_accessed: int
    arithmetic_intensity: float
    memory_bound: bool  # Computed based on arithmetic intensity vs machine balance
    input_shapes: List[Tuple[int, ...]]
    params: int = 0
    param_bytes: int = 0
    unit_type: str = "vector"
    dtype: str = "fp16"

    # Backward metrics
    flops_backward: int = 0  # Backward FLOPs
    bytes_accessed_backward: int = 0  # Backward memory bytes
    saved_inputs: List[str] = field(default_factory=list)  # Input names to save for backward (e.g., ["input"])

    def __post_init__(self):
        """Validate and compute derived metrics."""
        if self.bytes_accessed > 0:
            self.arithmetic_intensity = self.flops / self.bytes_accessed
        else:
            self.arithmetic_intensity = float("inf")
        # Compute memory_bound based on arithmetic intensity
        self.memory_bound = self._compute_memory_bound()

    def get_dtype_size(self) -> int:
        """Get bytes per element for the data type."""
        return DTYPE_SIZES.get(self.dtype, 2)

    def _compute_memory_bound(self) -> bool:
        """Determine if kernel is memory bound based on arithmetic intensity.

        A kernel is memory bound when its arithmetic intensity is below the
        machine's compute-to-memory bandwidth ratio (machine balance).

        Typical machine balance (peak_FLOPs / memory_BW):
        - H100 SXM: ~1000 TFLOPS / 3 TB/s = ~333 FLOPs/byte
        - A100 SXM: ~312 TFLOPS / 2 TB/s = ~156 FLOPs/byte
        - MI300X: ~1300 TFLOPS / 5.3 TB/s = ~245 FLOPs/byte
        - Ascend 910B: ~376 TFLOPS / 1.6 TB/s = ~235 FLOPs/byte

        We use conservative thresholds:
        - CUBE/Tensor Core ops: 200 FLOPs/byte
        - VECTOR/CUDA Core ops: 50 FLOPs/byte

        Returns:
            True if kernel is memory bound, False if compute bound
        """
        if self.bytes_accessed == 0:
            return False  # No memory access means compute bound

        # Machine balance thresholds (FLOPs per byte)
        # These are conservative estimates for typical AI accelerators
        if self.unit_type == "cube":
            # CUBE/Tensor Core: high compute capability
            # Threshold ~200 FLOPs/byte (between A100 and H100)
            threshold = 200.0
        else:
            # VECTOR/CUDA Core: lower compute capability
            # Threshold ~50 FLOPs/byte
            threshold = 50.0

        return self.arithmetic_intensity < threshold


def _compute_dtype_size(dtype: str) -> int:
    """Get bytes per element for dtype."""
    return DTYPE_SIZES.get(dtype, 2)


def linear(
    input: Tuple[int, ...],  # (..., in_features)
    weight: Tuple[int, ...],  # (out_features, in_features)
    bias: Optional[Tuple[int, ...]] = None,  # (out_features,)
    dtype: str = "fp16",
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

    # Forward FLOPs: 2 * batch * in_features * out_features (multiply-add)
    flops = 2 * batch_size * in_features * out_features

    # Forward memory accessed
    # Read: input + weight + bias
    # Write: output
    bytes_accessed = (
        batch_size * in_features * dtype_size  # input
        + in_features * out_features * dtype_size  # weight
        + batch_size * out_features * dtype_size  # output
    )
    if bias is not None:
        bytes_accessed += out_features * dtype_size  # bias
        flops += batch_size * out_features  # bias addition

    # Calculate params: weight + bias
    params = in_features * out_features + (out_features if bias else 0)
    param_bytes = params * dtype_size

    # Backward computation for linear/matmul:
    # Forward: C = A @ W^T (or equivalently, C = A @ W if we transpose W)
    # For backward, we need:
    # - dA = dC @ W (compute gradient for input)
    # - dW = A^T @ dC (compute gradient for weight)
    #
    # For y = x @ W^T + b:
    # - dx = dy @ W: FLOPs = 2 * batch * in_features * out_features
    # - dW = x^T @ dy: FLOPs = 2 * batch * in_features * out_features
    # - db = sum(dy): FLOPs = batch * out_features (negligible)
    #
    # Total backward FLOPs = 4 * batch * in_features * out_features (2x forward)
    flops_backward = 4 * batch_size * in_features * out_features
    if bias is not None:
        flops_backward += batch_size * out_features  # bias gradient

    # Backward memory access:
    # - dx = dy @ W: read dy (m×n), read W (n×k), write dx (m×k)
    # - dW = x^T @ dy: read x (m×k), read dy (m×n), write dW (n×k)
    # Total backward bytes ≈ 2x forward bytes
    bytes_accessed_backward = (
        batch_size * out_features * dtype_size  # dy (gradient output)
        + in_features * out_features * dtype_size  # W (weight)
        + batch_size * in_features * dtype_size  # dx (gradient input, written)
        + batch_size * in_features * dtype_size  # x (input, read for dW)
        + batch_size * out_features * dtype_size  # dy (gradient output, read for dW)
        + in_features * out_features * dtype_size  # dW (gradient weight, written)
    )
    # Note: W is read twice (for dx and dW), dy is read twice (for dx and dW)
    # Simplified: bytes_backward ≈ 2 * forward_bytes + weight_bytes (read twice)
    bytes_accessed_backward = 2 * bytes_accessed + in_features * out_features * dtype_size

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=bytes_accessed > flops / 100,  # Heuristic
        input_shapes=[input, weight] + ([bias] if bias else []),
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["input", "mat2"],  # backward: dA = dC @ B^T, dB = A^T @ dC
    )


def bmm(
    input: Tuple[int, ...],  # (batch, m, k)
    mat2: Tuple[int, ...],  # (batch, k, n)
    dtype: str = "fp16",
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

    # Forward FLOPs: 2 * batch * m * n * k
    flops = 2 * batch * m * n * k

    # Forward memory accessed
    bytes_accessed = (
        batch * m * k * dtype_size  # input
        + batch * k * n * dtype_size  # mat2
        + batch * m * n * dtype_size  # output
    )

    # BMM has no learnable parameters
    params = 0
    param_bytes = 0

    # Backward computation for BMM: out = A @ B
    # Forward: C = A @ B
    # Backward:
    # - dA = dC @ B^T: FLOPs = 2 * batch * m * n * k
    # - dB = A^T @ dC: FLOPs = 2 * batch * m * n * k
    # Total backward FLOPs = 4 * batch * m * n * k (2x forward)
    flops_backward = 4 * batch * m * n * k

    # Backward memory access:
    # - dA = dC @ B^T: read dC, read B, write dA
    # - dB = A^T @ dC: read A, read dC, write dB
    # Total: ≈ 2x forward bytes (A, B, dC read twice)
    bytes_accessed_backward = 2 * bytes_accessed

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=bytes_accessed > flops / 100,
        input_shapes=[input, mat2],
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["input", "mat2"],  # backward: dA = dC @ B^T, dB = A^T @ dC
    )


def scaled_dot_product_attention(
    query: Tuple[int, ...],  # (batch, num_heads, seq_len, head_dim)
    key: Tuple[int, ...],  # (batch, kv_num_heads, kv_seq_len, head_dim)
    value: Tuple[int, ...],  # (batch, kv_num_heads, kv_seq_len, head_dim)
    is_causal: bool = False,
    dtype: str = "fp16",
    use_gqa: bool = False,  # Whether to use GQA (Group Query Attention)
) -> KernelResult:
    """Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V

    Similar to torch.nn.functional.scaled_dot_product_attention.
    Supports MHA (Multi-Head Attention) and GQA (Grouped Query Attention).

    For GQA:
    - kv_num_heads < num_heads
    - Each KV head is shared among (num_heads / kv_num_heads) query heads
    - Memory bandwidth is reduced by the same factor

    Args:
        query: Shape (batch, num_heads, seq_len, head_dim)
        key: Shape (batch, kv_num_heads, kv_seq_len, head_dim)
        value: Shape (batch, kv_num_heads, kv_seq_len, head_dim)
        is_causal: Whether to apply causal mask
        dtype: Data type string
        use_gqa: If True, treat as GQA (KV heads may differ from Q heads)

    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)

    batch, num_heads, seq_len, head_dim = query
    _, kv_num_heads, kv_seq_len, _ = key

    # For GQA: each KV head is shared among (num_heads / kv_num_heads) Q heads
    # The computation flops remain the same (we still compute all Q heads)
    # But memory access for K/V is reduced
    gqa_factor = num_heads // kv_num_heads if use_gqa and kv_num_heads < num_heads else 1

    output_shape = (batch, num_heads, seq_len, head_dim)

    # Forward FLOPs breakdown:
    # 1. Q @ K^T: 2 * batch * num_heads * seq_len * kv_seq_len * head_dim
    # 2. Softmax: 5 * batch * num_heads * seq_len * kv_seq_len (exp, sum, div)
    # 3. Attention @ V: 2 * batch * num_heads * seq_len * head_dim * kv_seq_len

    qk_flops = 2 * batch * num_heads * seq_len * kv_seq_len * head_dim
    softmax_flops = 5 * batch * num_heads * seq_len * kv_seq_len
    attn_v_flops = 2 * batch * num_heads * seq_len * head_dim * kv_seq_len

    flops = qk_flops + softmax_flops + attn_v_flops

    # Forward memory accessed
    # For GQA: K/V memory access is reduced by gqa_factor
    bytes_accessed = (
        batch * num_heads * seq_len * head_dim * dtype_size  # query
        + batch * kv_num_heads * kv_seq_len * head_dim * dtype_size  # key (GQA reduced)
        + batch * kv_num_heads * kv_seq_len * head_dim * dtype_size  # value (GQA reduced)
        + batch * num_heads * seq_len * head_dim * dtype_size  # output
    )

    # Attention has no learnable parameters
    params = 0
    param_bytes = 0

    # Backward computation for attention:
    # Forward: softmax(QK^T) @ V
    # Backward needs:
    # 1. dV = Attention^T @ dOutput: similar to QK^T matmul
    # 2. dAttention scores = dOutput @ V^T: similar to Attention @ V matmul
    # 3. dQ, dK from softmax backward: requires recomputing attention scores
    #
    # FLOPs breakdown:
    # - dV = S^T @ dO: 2 * batch * heads * seq * head_dim * kv_seq_len
    # - dS = dO @ V^T: 2 * batch * heads * seq * kv_seq_len * head_dim
    # - Softmax backward: ~5 * batch * heads * seq * kv_seq_len (gradient propagation)
    # - dQ = dS @ K: 2 * batch * heads * seq * kv_seq_len * head_dim
    # - dK = dS^T @ Q: 2 * batch * heads * seq * kv_seq_len * head_dim
    #
    # Total backward FLOPs ≈ 2x forward (each matmul has a corresponding backward matmul)
    # Plus softmax backward overhead
    flops_backward = (
        2 * batch * num_heads * seq_len * kv_seq_len * head_dim * 2  # dV + dS (matmuls)
        + 5 * batch * num_heads * seq_len * kv_seq_len  # softmax backward
        + 2 * batch * num_heads * seq_len * kv_seq_len * head_dim * 2  # dQ + dK
    )
    # Simplified: flops_backward ≈ 2 * forward_flops
    flops_backward = 2 * flops

    # Backward memory access:
    # Need to read: Q, K, V (all inputs), dOutput (gradient), saved attention scores
    # Write: dQ, dK, dV (all gradients)
    # Total: ≈ 2-3x forward bytes (need to re-read inputs and save intermediate results)
    bytes_accessed_backward = (
        batch * num_heads * seq_len * head_dim * dtype_size * 2  # Q read + dQ write
        + batch * kv_num_heads * kv_seq_len * head_dim * dtype_size * 2  # K read + dK write
        + batch * kv_num_heads * kv_seq_len * head_dim * dtype_size * 2  # V read + dV write
        + batch * num_heads * seq_len * head_dim * dtype_size  # dOutput (gradient)
        + batch * num_heads * seq_len * kv_seq_len * dtype_size  # saved attention scores
    )
    # Simplified: bytes_backward ≈ 2-3x forward (with saved activations)
    bytes_accessed_backward = bytes_accessed * 2.5

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,  # Attention is typically memory bound
        input_shapes=[query, key, value],
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=int(bytes_accessed_backward),
        saved_inputs=["query", "key", "value"],  # backward needs Q, K, V for gradient computation
    )


def flash_attention(
    query: Tuple[int, ...],  # (batch, num_heads, seq_len, head_dim)
    key: Tuple[int, ...],  # (batch, kv_num_heads, kv_seq_len, head_dim)
    value: Tuple[int, ...],  # (batch, kv_num_heads, kv_seq_len, head_dim)
    is_causal: bool = False,
    dtype: str = "fp16",
    use_gqa: bool = False,
    block_size: int = 128,  # Tile size for Flash Attention
) -> KernelResult:
    """Flash Attention kernel with tiling optimization.

    Flash Attention reduces HBM (High Bandwidth Memory) traffic by:
    1. Tiling the computation into blocks that fit in SRAM
    2. Fusing Q@K^T, softmax, and @V into a single kernel
    3. Avoiding materialization of the full attention score matrix in HBM

    Memory access pattern:
    - Standard SDPA: O(N^2) HBM traffic for attention scores
    - Flash Attention: O(N) HBM traffic (linear in sequence length)

    Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    https://arxiv.org/abs/2205.14135

    Args:
        query: Shape (batch, num_heads, seq_len, head_dim)
        key: Shape (batch, kv_num_heads, kv_seq_len, head_dim)
        value: Shape (batch, kv_num_heads, kv_seq_len, head_dim)
        is_causal: Whether to apply causal mask
        dtype: Data type string
        use_gqa: If True, treat as GQA
        block_size: Tile size for blocking (typically 64, 128, or 256)

    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)

    batch, num_heads, seq_len, head_dim = query
    _, kv_num_heads, kv_seq_len, _ = key

    output_shape = (batch, num_heads, seq_len, head_dim)

    # Forward FLOPs: same as SDPA (exact same computation)
    qk_flops = 2 * batch * num_heads * seq_len * kv_seq_len * head_dim
    softmax_flops = 5 * batch * num_heads * seq_len * kv_seq_len
    attn_v_flops = 2 * batch * num_heads * seq_len * head_dim * kv_seq_len
    flops = qk_flops + softmax_flops + attn_v_flops

    # Forward memory access for Flash Attention:
    # 1. Load Q, K, V from HBM (once each)
    # 2. Store output to HBM (once)
    # 3. Intermediate attention scores stay in SRAM (no HBM traffic!)
    #
    # For causal attention, we load each block only once (triangular access pattern)

    if is_causal:
        # Causal: triangular access pattern
        # Each query position only attends to keys up to that position
        # Average number of keys per query: seq_len / 2
        effective_kv_len = seq_len // 2

        # Q is loaded seq_len/block_size times (for each KV block it needs)
        q_loads = (seq_len + block_size - 1) // block_size

        # K/V are loaded once per query block they serve
        kv_loads = (seq_len + block_size - 1) // block_size
    else:
        # Non-causal: full attention
        q_loads = 1
        kv_loads = (kv_seq_len + block_size - 1) // block_size

    # Forward HBM traffic:
    # Q: loaded q_loads times
    # K/V: loaded kv_loads times each
    # O: written once
    # Plus some overhead for softmax normalization stats (negligible)
    bytes_accessed = (
        batch * num_heads * seq_len * head_dim * dtype_size * q_loads  # Q loads
        + batch * kv_num_heads * kv_seq_len * head_dim * dtype_size * kv_loads  # K loads
        + batch * kv_num_heads * kv_seq_len * head_dim * dtype_size * kv_loads  # V loads
        + batch * num_heads * seq_len * head_dim * dtype_size  # O write
    )

    # Add small overhead for softmax statistics (online softmax algorithm)
    bytes_accessed += batch * num_heads * seq_len * 4  # m and l statistics (fp32)

    # Flash Attention has no learnable parameters
    params = 0
    param_bytes = 0

    # Backward computation for Flash Attention:
    # Same as SDPA backward but with optimized memory access
    # FLOPs ≈ 2x forward
    flops_backward = 2 * flops

    # Flash Attention backward memory access:
    # Need to recompute attention scores from Q, K in SRAM
    # This avoids storing O(N^2) attention scores
    # Memory access: similar to forward, but with additional reads for gradient
    bytes_accessed_backward = (
        batch * num_heads * seq_len * head_dim * dtype_size * (q_loads + 1)  # Q + dQ
        + batch * kv_num_heads * kv_seq_len * head_dim * dtype_size * (kv_loads + 1)  # K + dK
        + batch * kv_num_heads * kv_seq_len * head_dim * dtype_size * (kv_loads + 1)  # V + dV
        + batch * num_heads * seq_len * head_dim * dtype_size * 2  # O + dO
    )
    # Flash Attention backward is memory-efficient
    # ≈ 2x forward bytes (no O(N^2) storage)
    bytes_accessed_backward = bytes_accessed * 2

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[query, key, value],
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=[],  # Flash Attention: Q, K, V are views, save Q_proj, K_proj, V_proj instead
    )


def mla_attention(
    query: Tuple[int, ...],  # (batch, num_heads, seq_len, qk_head_dim)
    compressed_kv: Tuple[int, ...],  # (batch, seq_len, kv_lora_rank)
    key: Optional[Tuple[int, ...]],  # (batch, kv_num_heads, seq_len, head_dim) - for non-absorb
    value: Optional[Tuple[int, ...]],  # (batch, kv_num_heads, seq_len, v_head_dim) - for non-absorb
    is_causal: bool = False,
    dtype: str = "fp16",
    use_absorb: bool = False,  # True for absorb mode (inference optimization)
    qk_head_dim: int = 128,
    v_head_dim: int = 128,
    kv_lora_rank: int = 512,
) -> KernelResult:
    """Multi-head Latent Attention (MLA) kernel.

    MLA compresses KV cache into a latent vector, reducing memory usage.

    Two modes:
    1. Non-absorb mode (training/inference standard):
       - KV is compressed to kv_lora_rank dimensions
       - At attention time, KV is decompressed to full head_dim
       - Requires explicit K and V inputs (decompressed)

    2. Absorb mode (inference optimization):
       - The KV decompression matrices are absorbed into Q and O projections
       - Attention computed directly on compressed KV representation
       - No explicit K/V decompression needed
       - Significant memory and compute savings

    Reference: DeepSeek-V2 Technical Report
    https://arxiv.org/abs/2405.04434

    Args:
        query: Shape (batch, num_heads, seq_len, qk_head_dim)
        compressed_kv: Shape (batch, seq_len, kv_lora_rank) - compressed KV
        key: Shape (batch, kv_num_heads, seq_len, head_dim) - for non-absorb
        value: Shape (batch, kv_num_heads, seq_len, v_head_dim) - for non-absorb
        is_causal: Whether to apply causal mask
        dtype: Data type string
        use_absorb: If True, use absorb mode (inference optimization)
        qk_head_dim: Dimension for QK (after RoPE and projection)
        v_head_dim: Dimension for V heads
        kv_lora_rank: Compressed KV dimension

    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)

    batch, num_heads, seq_len, _ = query
    _, kv_seq_len, _ = compressed_kv

    output_shape = (batch, num_heads, seq_len, v_head_dim)

    if use_absorb:
        # Absorb mode: attention directly on compressed KV
        # The KV decompression is implicitly done through Q and O projection
        # This is mathematically equivalent but more efficient

        # FLOPs: same as standard attention but with compressed dimensions
        # Q @ compressed_K^T: (num_heads, seq, qk_dim) @ (kv_rank, seq)^T
        qk_flops = 2 * batch * num_heads * seq_len * kv_seq_len * kv_lora_rank

        # Softmax
        softmax_flops = 5 * batch * num_heads * seq_len * kv_seq_len

        # Attention @ compressed_V: (num_heads, seq, seq) @ (seq, kv_rank)
        attn_v_flops = 2 * batch * num_heads * seq_len * kv_lora_rank * kv_seq_len

        flops = qk_flops + softmax_flops + attn_v_flops

        # Memory: only compressed KV is accessed
        bytes_accessed = (
            batch * num_heads * seq_len * qk_head_dim * dtype_size  # Q
            + batch * kv_seq_len * kv_lora_rank * dtype_size  # compressed K
            + batch * kv_seq_len * kv_lora_rank * dtype_size  # compressed V
            + batch * num_heads * seq_len * v_head_dim * dtype_size  # output
        )
    else:
        # Non-absorb mode: standard attention on decompressed K/V
        assert key is not None and value is not None, "key/value required for non-absorb mode"
        _, kv_num_heads, _, _ = key

        # Standard attention FLOPs
        qk_flops = 2 * batch * num_heads * seq_len * kv_seq_len * qk_head_dim
        softmax_flops = 5 * batch * num_heads * seq_len * kv_seq_len
        attn_v_flops = 2 * batch * num_heads * seq_len * v_head_dim * kv_seq_len

        flops = qk_flops + softmax_flops + attn_v_flops

        # Memory: full K/V heads
        bytes_accessed = (
            batch * num_heads * seq_len * qk_head_dim * dtype_size  # Q
            + batch * kv_num_heads * kv_seq_len * qk_head_dim * dtype_size  # K
            + batch * kv_num_heads * kv_seq_len * v_head_dim * dtype_size  # V
            + batch * num_heads * seq_len * v_head_dim * dtype_size  # output
        )

    params = 0
    param_bytes = 0

    # Backward computation for MLA Attention:
    # Similar to standard attention backward
    # FLOPs ≈ 2x forward
    flops_backward = 2 * flops

    # Backward memory ≈ 2x forward (need to re-read inputs and write gradients)
    bytes_accessed_backward = bytes_accessed * 2

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[query, compressed_kv] + ([key, value] if key else []),
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["query", "compressed_kv"],  # backward needs Q and compressed KV
    )


def layer_norm(
    input: Tuple[int, ...], normalized_shape: Tuple[int, ...], elementwise_affine: bool = True, dtype: str = "fp16"
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

    # Forward FLOPs per element: mean (N ops), variance (2N ops), normalize (3 ops), affine (2 ops)
    # Total: ~7 FLOPs per element
    numel = math.prod(input)
    flops = numel * 7

    # Forward memory accessed
    bytes_accessed = numel * dtype_size * 2  # read input, write output
    if elementwise_affine:
        bytes_accessed += math.prod(normalized_shape) * dtype_size * 2  # weight + bias
        flops += numel * 2  # scale + shift

    # Calculate params: weight + bias (if elementwise_affine)
    params = normalized_shape[0] * (2 if elementwise_affine else 1)
    param_bytes = params * dtype_size

    # Backward computation for Layer Norm:
    # Forward: y = (x - mean) / std * gamma + beta
    # Backward: need to compute gradient for x, gamma, beta
    # LayerNorm backward is more complex than RMS Norm
    # FLOPs ≈ 5 * numel (gradient propagation through mean, var, normalization)
    flops_backward = numel * 5
    if elementwise_affine:
        flops_backward += numel * 2  # gamma, beta gradients

    # Backward memory: read input, weight/bias, gradient; write gradients
    # ≈ 2x forward bytes
    bytes_accessed_backward = numel * dtype_size * 2  # read input + write dX
    if elementwise_affine:
        bytes_accessed_backward += (
            math.prod(normalized_shape) * dtype_size * 3
        )  # read gamma, beta + write dGamma, dBeta
    bytes_accessed_backward += numel * dtype_size  # read dY

    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
    )


def rms_norm(input: Tuple[int, ...], dim: int = -1, dtype: str = "fp16") -> KernelResult:
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

    # Forward FLOPs: square (1), mean (N), rsqrt (7), multiply (1), scale (1) per element
    flops = numel * 7

    # Forward memory accessed
    bytes_accessed = numel * dtype_size * 2  # read input, write output
    bytes_accessed += input[dim] * dtype_size  # weight

    # Calculate params: only weight
    params = input[dim]
    param_bytes = params * dtype_size

    # Backward computation for RMS Norm:
    # Forward: y = x / rms(x) * w, where rms(x) = sqrt(mean(x^2))
    # Backward: need to compute gradient for x and w
    # - dw = sum(dy * (x / rms)) / num_features
    # - dx involves derivative through rms normalization
    # FLOPs ≈ 5 * numel (simpler than LayerNorm backward)
    flops_backward = numel * 5

    # Backward memory: read input, weight, gradient; write gradient for input and weight
    # ≈ 1.5x forward bytes
    bytes_accessed_backward = (
        numel * dtype_size * 2  # read input + write dX
        + input[dim] * dtype_size * 2  # read weight + write dW
        + numel * dtype_size  # read dY
    )

    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
    )


def silu(input: Tuple[int, ...], dtype: str = "fp16") -> KernelResult:
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
    # Forward FLOPs: exp (7), add (1), div (1), mul (1) = ~10 FLOPs
    flops = numel * 10
    bytes_accessed = numel * dtype_size * 2  # read + write

    # Activation functions have no learnable parameters
    params = 0
    param_bytes = 0

    # Backward computation for SiLU: x * sigmoid(x)
    # Forward: y = x * sigmoid(x)
    # Backward: dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    # Need to recompute sigmoid(x): ~6 FLOPs per element
    # Plus gradient computation: ~10 FLOPs per element
    # Total backward FLOPs ≈ 2x forward (need to recompute sigmoid)
    flops_backward = numel * 20

    # Backward memory: read input, read gradient, write gradient
    # ≈ 2x forward bytes
    bytes_accessed_backward = numel * dtype_size * 3  # read input + read dY + write dX

    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
    )


def gelu(input: Tuple[int, ...], approximate: str = "none", dtype: str = "fp16") -> KernelResult:
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
    # Forward FLOPs varies by approximation
    flops_per_elem = 8 if approximate == "tanh" else 15
    flops = numel * flops_per_elem
    bytes_accessed = numel * dtype_size * 2

    # Activation functions have no learnable parameters
    params = 0
    param_bytes = 0

    # Backward computation for GELU:
    # Forward: y = GELU(x) = x * Φ(x) (where Φ is Gaussian CDF)
    # Backward: dy/dx = Φ(x) + x * (dΦ/dx)
    # For tanh approximation: more complex gradient computation
    # Backward FLOPs ≈ 2x forward (need to recompute activation function)
    flops_backward = numel * (flops_per_elem * 2)

    # Backward memory: read input, read gradient, write gradient
    bytes_accessed_backward = numel * dtype_size * 3

    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["input"],  # backward needs input for gradient computation
    )


def relu(input: Tuple[int, ...], dtype: str = "fp16") -> KernelResult:
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

    # Backward computation for ReLU:
    # Forward: y = max(0, x)
    # Backward: dy/dx = 1 if x > 0 else 0 (simple mask)
    # FLOPs ≈ numel (just comparison)
    flops_backward = numel
    bytes_accessed_backward = numel * dtype_size * 3  # read input + read dY + write dX

    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=[],  # ReLU backward doesn't need to save input (uses mask from forward)
    )


def softmax(input: Tuple[int, ...], dim: int = -1, dtype: str = "fp16") -> KernelResult:
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
    # Forward FLOPs: exp (7), sum (N), div (1) per element
    softmax_dim_size = input[dim]
    flops = numel * (7 + softmax_dim_size + 1)
    bytes_accessed = numel * dtype_size * 2

    # Softmax has no learnable parameters
    params = 0
    param_bytes = 0

    # Backward computation for Softmax:
    # Forward: y = exp(x) / sum(exp(x))
    # Backward: Jacobian computation, dy/dx = y_i * (delta_ij - y_j)
    # For each element, need to compute gradient against all elements in the softmax dimension
    # FLOPs ≈ 2 * numel * softmax_dim_size
    flops_backward = numel * softmax_dim_size * 2
    bytes_accessed_backward = numel * dtype_size * 3  # read input + read dY + write dX

    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["output"],  # softmax backward needs output (softmax values), not input
    )


def dropout(input: Tuple[int, ...], p: float = 0.5, dtype: str = "fp16") -> KernelResult:
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
    # Forward FLOPs: random check (1), scale (1), mul (1)
    flops = numel * 3
    bytes_accessed = numel * dtype_size * 2

    # Dropout has no learnable parameters
    params = 0
    param_bytes = 0

    # Backward computation for Dropout:
    # Forward: y = x * mask / (1-p)
    # Backward: same mask applied to gradient
    # FLOPs ≈ numel (mask application)
    flops_backward = numel
    bytes_accessed_backward = numel * dtype_size * 3

    return KernelResult(
        output=input,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[input],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
    )


def conv2d(
    input: Tuple[int, ...],  # (N, C_in, H, W)
    weight: Tuple[int, ...],  # (C_out, C_in, kH, kW)
    bias: Optional[Tuple[int, ...]] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dtype: str = "fp16",
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

    # Forward FLOPs: 2 * N * C_out * H_out * W_out * C_in * kH * kW
    flops = 2 * N * C_out * H_out * W_out * C_in * kH * kW

    # Forward memory accessed
    bytes_accessed = (
        N * C_in * H * W * dtype_size  # input
        + C_out * C_in * kH * kW * dtype_size  # weight
        + N * C_out * H_out * W_out * dtype_size  # output
    )
    if bias is not None:
        bytes_accessed += C_out * dtype_size
        flops += N * C_out * H_out * W_out

    # Calculate params: weight + bias
    params = C_out * C_in * kH * kW + (C_out if bias else 0)
    param_bytes = params * dtype_size

    # Backward computation for Conv2d:
    # Forward: Y = Conv2d(X, W)
    # Backward:
    # - dX = Conv2d_transpose(dY, W) - gradient for input
    # - dW = Conv2d(X, dY) - gradient for weight (correlation operation)
    # Each backward operation has similar FLOPs to forward
    # Total backward FLOPs ≈ 2x forward
    flops_backward = 2 * flops
    if bias is not None:
        flops_backward += N * C_out * H_out * W_out  # bias gradient

    # Backward memory: read input, weight, gradient; write gradients
    # ≈ 2-3x forward bytes
    bytes_accessed_backward = 2 * bytes_accessed + N * C_out * H_out * W_out * dtype_size  # dY read

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=bytes_accessed > flops / 100,
        input_shapes=[input, weight] + ([bias] if bias else []),
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["input"],  # backward: dX = conv_transpose(dY, W), needs X for dW
    )


def conv3d(
    input: Tuple[int, ...],  # (N, C_in, D, H, W)
    weight: Tuple[int, ...],  # (C_out, C_in, kD, kH, kW)
    bias: Optional[Tuple[int, ...]] = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dtype: str = "fp16",
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

    # Forward FLOPs: 2 * N * C_out * D_out * H_out * W_out * C_in * kD * kH * kW
    flops = 2 * N * C_out * D_out * H_out * W_out * C_in * kD * kH * kW

    # Forward memory accessed
    bytes_accessed = (
        N * C_in * D * H * W * dtype_size  # input
        + C_out * C_in * kD * kH * kW * dtype_size  # weight
        + N * C_out * D_out * H_out * W_out * dtype_size  # output
    )
    if bias is not None:
        bytes_accessed += C_out * dtype_size
        flops += N * C_out * D_out * H_out * W_out

    # Calculate params: weight + bias
    params = C_out * C_in * kD * kH * kW + (C_out if bias else 0)
    param_bytes = params * dtype_size

    # Backward computation for Conv3d (similar to Conv2d):
    # FLOPs ≈ 2x forward
    flops_backward = 2 * flops
    if bias is not None:
        flops_backward += N * C_out * D_out * H_out * W_out

    # Backward memory ≈ 2-3x forward
    bytes_accessed_backward = 2 * bytes_accessed + N * C_out * D_out * H_out * W_out * dtype_size

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=bytes_accessed > flops / 100,
        input_shapes=[input, weight] + ([bias] if bias else []),
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["input"],  # backward: dX = conv_transpose(dY, W), needs X for dW
    )


def embedding(
    num_embeddings: int, embedding_dim: int, input_shape: Tuple[int, ...], dtype: str = "fp16"
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

    numel = math.prod(input_shape)
    flops = numel

    bytes_accessed = (
        num_embeddings * embedding_dim * dtype_size
        + numel * embedding_dim * dtype_size
    )

    params = num_embeddings * embedding_dim
    param_bytes = params * dtype_size

    flops_backward = numel * embedding_dim

    bytes_accessed_backward = (
        numel * embedding_dim * dtype_size
        + numel * embedding_dim * dtype_size * 2
    )

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=False,
        input_shapes=[input_shape],
        params=params,
        param_bytes=param_bytes,
        unit_type="vector",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["input_ids"],
    )


def moe_expert(
    hidden_shape: Tuple[int, ...],
    intermediate_size: int,
    num_experts_per_token: int = 1,
    dtype: str = "fp16",
) -> KernelResult:
    """MoE expert computation.

    MoE expert contains three linear projections:
    1. gate projection: hidden -> intermediate_size (SiLU activation)
    2. up projection: hidden -> intermediate_size
    3. down projection: intermediate_size -> hidden

    The computation is:
    output = down(silu(gate(x)) * up(x))

    Args:
        hidden_shape: Shape of hidden tensor (batch, ..., hidden_size)
        intermediate_size: Intermediate dimension
        num_experts_per_token: Number of experts activated per token
        dtype: Data type string

    Returns:
        KernelResult with performance metrics
    """
    dtype_size = _compute_dtype_size(dtype)

    hidden_size = hidden_shape[-1]
    batch_size = math.prod(hidden_shape[:-1]) if len(hidden_shape) > 1 else 1

    output_shape = hidden_shape

    gate_flops = 2 * batch_size * hidden_size * intermediate_size
    gate_activation_flops = batch_size * intermediate_size * 10
    up_flops = 2 * batch_size * hidden_size * intermediate_size
    gate_up_mul_flops = batch_size * intermediate_size
    down_flops = 2 * batch_size * intermediate_size * hidden_size

    flops = (gate_flops + gate_activation_flops + up_flops + gate_up_mul_flops + down_flops) * num_experts_per_token

    gate_bytes = (
        batch_size * hidden_size * dtype_size
        + hidden_size * intermediate_size * dtype_size
        + batch_size * intermediate_size * dtype_size
    )
    up_bytes = (
        batch_size * hidden_size * dtype_size
        + hidden_size * intermediate_size * dtype_size
        + batch_size * intermediate_size * dtype_size
    )
    down_bytes = (
        batch_size * intermediate_size * dtype_size
        + intermediate_size * hidden_size * dtype_size
        + batch_size * hidden_size * dtype_size
    )

    bytes_accessed = (gate_bytes + up_bytes + down_bytes) * num_experts_per_token

    params_per_expert = hidden_size * intermediate_size * 3
    params = params_per_expert
    param_bytes = params * dtype_size

    gate_backward = 4 * batch_size * hidden_size * intermediate_size
    gate_activation_backward = batch_size * intermediate_size * 20
    up_backward = 4 * batch_size * hidden_size * intermediate_size
    down_backward = 4 * batch_size * intermediate_size * hidden_size

    flops_backward = (gate_backward + gate_activation_backward + up_backward + down_backward) * num_experts_per_token

    bytes_accessed_backward = bytes_accessed * 2

    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=flops / bytes_accessed if bytes_accessed > 0 else float("inf"),
        memory_bound=bytes_accessed > flops / 100,
        input_shapes=[hidden_shape],
        params=params,
        param_bytes=param_bytes,
        unit_type="cube",
        dtype=dtype,
        flops_backward=flops_backward,
        bytes_accessed_backward=bytes_accessed_backward,
        saved_inputs=["hidden"],
    )

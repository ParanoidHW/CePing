"""Operation base classes for recording operation history.

Each Op represents a computation or communication operation with
input/output tensors and kernel semantic.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Any, Dict


@dataclass
class KVCacheConfig:
    """KV Cache configuration for inference.

    Attributes:
        max_seq_len: Maximum sequence length
        num_layers: Number of layers with KV cache
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Head dimension
        cache_dtype: Cache data type
        batch_size: Batch size for cache allocation
    """

    max_seq_len: int = 4096
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    cache_dtype: str = "fp16"
    batch_size: int = 1

    def cache_size_per_token(self) -> int:
        """Cache size per token (bytes) for one layer.

        KV cache: 2 (K + V) * num_kv_heads * head_dim * dtype_size
        """
        from llm_perf.utils.constants import DTYPE_SIZES

        dtype_size = DTYPE_SIZES.get(self.cache_dtype, 2)
        return 2 * self.num_kv_heads * self.head_dim * dtype_size

    def total_cache_size(self) -> int:
        """Total KV cache size (bytes) for all layers.

        batch_size * max_seq_len * cache_per_token * num_layers
        """
        return self.batch_size * self.max_seq_len * self.cache_size_per_token() * self.num_layers

    def cache_size_for_seq_len(self, seq_len: int) -> int:
        """KV cache size for a specific sequence length."""
        return self.batch_size * seq_len * self.cache_size_per_token() * self.num_layers


@dataclass
class Op:
    """Base class for operations.

    Attributes:
        kernel_name: Kernel identifier for backend
        inputs: Input tensors
        output: Output tensor
        dtype: Data type
    """

    kernel_name: str
    inputs: List[Any]
    output: Any
    dtype: str

    def get_saved_tensors(self) -> List[Any]:
        """Get tensors that need to be saved for backward pass.

        Returns:
            List of tensors to save (default: empty list)
        """
        return []


@dataclass
class MatmulOp(Op):
    """Matrix multiplication operation."""

    kernel_name: str = "linear"
    dtype: str = "fp16"
    input: Any = None
    weight: Any = None
    output: Any = None
    inputs: List[Any] = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [self.input, self.weight]

    def get_saved_tensors(self) -> List[Any]:
        """Get tensors to save for backward.

        linear backward:
        - dx = dy @ W (needs W, not saved)
        - dW = x^T @ dy (needs x, saved)

        Returns:
            [self.input] - save input for dW computation
        """
        return [self.input] if self.input else []


@dataclass
class AttentionOp(Op):
    """Flash Attention operation.

    Attributes:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        output: Output tensor
        is_causal: Whether causal mask
    """

    kernel_name: str = "flash_attention"
    dtype: str = "fp16"
    query: Any = None
    key: Any = None
    value: Any = None
    output: Any = None
    inputs: List[Any] = field(default_factory=list)
    is_causal: bool = True

    def __post_init__(self):
        if not self.inputs:
            self.inputs = [self.query, self.key, self.value]

    def get_saved_tensors(self) -> List[Any]:
        """Get tensors to save for backward.

        Flash Attention backward:
        - Q, K, V can be re-computed from Q_proj, K_proj, V_proj (they are views)
        - Flash Attention saves logsumexp and max values (small O(N) storage)
        - So we don't save Q, K, V tensors themselves

        Returns:
            [] - Q, K, V are views, save projections at module level instead
        """
        return []


@dataclass
class RMSNormOp(Op):
    """RMS Normalization operation."""

    kernel_name: str = "rms_norm"
    dtype: str = "fp16"
    input: Any = None
    weight: Any = None
    output: Any = None
    inputs: List[Any] = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [self.input, self.weight]

    def get_saved_tensors(self) -> List[Any]:
        """Get tensors to save for backward.

        RMS Norm backward:
        - needs input for gradient computation

        Returns:
            [self.input] - save input for gradient
        """
        return [self.input] if self.input else []


@dataclass
class EmbeddingOp(Op):
    """Embedding lookup operation."""

    kernel_name: str = "embedding"
    dtype: str = "fp16"
    input_ids: Any = None
    weight: Any = None
    output: Any = None
    inputs: List[Any] = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [self.input_ids, self.weight]

    def get_saved_tensors(self) -> List[Any]:
        """Get tensors to save for backward.

        Embedding backward:
        - needs input_ids to know which rows to update

        Returns:
            [self.input_ids] - save indices for backward
        """
        return [self.input_ids] if self.input_ids else []


@dataclass
class ActivationOp(Op):
    """Activation function operation (silu, gelu, relu, etc.)."""

    kernel_name: str = "silu"
    dtype: str = "fp16"
    input: Any = None
    output: Any = None
    inputs: List[Any] = None
    activation_type: str = "silu"

    def __post_init__(self):
        self.kernel_name = self.activation_type
        if self.inputs is None:
            self.inputs = [self.input]

    def get_saved_tensors(self) -> List[Any]:
        """Get tensors to save for backward.

        Activation backward:
        - silu/gelu: need input for gradient computation
        - relu: use mask from forward (output), not input

        Returns:
            [self.input] for silu/gelu, [self.output] for relu, [] otherwise
        """
        if self.activation_type == "relu":
            return [self.output] if self.output else []
        elif self.activation_type in ["silu", "gelu"]:
            return [self.input] if self.input else []
        return []


@dataclass
class MoEExpertOp(Op):
    """MoE expert computation operation."""

    kernel_name: str = "moe_expert"
    dtype: str = "fp16"
    hidden: Any = None
    expert_gate_weights: Any = None
    expert_up_weights: Any = None
    expert_down_weights: Any = None
    output: Any = None
    inputs: List[Any] = None
    num_experts_per_token: int = 1

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [
                self.hidden,
                self.expert_gate_weights,
                self.expert_up_weights,
                self.expert_down_weights,
            ]


@dataclass
class ViewOp(Op):
    """Reshape/View operation."""

    kernel_name: str = "view"
    dtype: str = "fp16"
    input: Any = None
    shape: Tuple[int, ...] = ()
    output: Any = None
    inputs: List[Any] = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [self.input]


@dataclass
class TransposeOp(Op):
    """Transpose operation."""

    kernel_name: str = "transpose"
    dtype: str = "fp16"
    input: Any = None
    dim0: int = 0
    dim1: int = 1
    output: Any = None
    inputs: List[Any] = None

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [self.input]


@dataclass
class CommOp:
    """Communication operation.

    Attributes:
        comm_type: Communication type (allreduce, allgather, alltoall, p2p)
        data_bytes: Data size in bytes
        ptype: Parallel type that triggers this communication (tp, sp, ep, dp)
        direction: "forward" or "backward"
    """

    comm_type: str
    data_bytes: int
    ptype: str
    direction: str = "forward"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comm_type": self.comm_type,
            "data_bytes": self.data_bytes,
            "data_mb": self.data_bytes / 1e6,
            "ptype": self.ptype,
            "direction": self.direction,
        }


@dataclass
class Conv2dOp(Op):
    """2D Convolution operation."""

    kernel_name: str = "conv2d"
    dtype: str = "fp16"
    input: Any = None
    weight: Any = None
    output: Any = None
    inputs: List[Any] = None
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [self.input, self.weight]


@dataclass
class Conv3dOp(Op):
    """3D Convolution operation."""

    kernel_name: str = "conv3d"
    dtype: str = "fp16"
    input: Any = None
    weight: Any = None
    output: Any = None
    inputs: List[Any] = None
    stride: Tuple[int, int, int] = (1, 1, 1)
    padding: Tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [self.input, self.weight]


@dataclass
class GroupNormOp(Op):
    """Group Normalization operation."""

    kernel_name: str = "group_norm"
    dtype: str = "fp16"
    input: Any = None
    weight: Any = None
    output: Any = None
    inputs: List[Any] = None
    num_groups: int = 32

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = [self.input, self.weight]


@dataclass
class LinearAttentionOp(Op):
    """Linear Attention operation.

    Linear Attention reformulates standard attention to achieve O(seq) complexity:
    - Standard: softmax(QK^T)V - O(seq^2) memory and compute
    - Linear: Q(K^T V) with kernel feature map - O(seq) compute, fixed state size

    Uses kernel trick (e.g., elu(x)+1) to approximate softmax denominator.
    No KV cache needed - maintains fixed-size state per head.

    Reference: "Linear Transformers Are Secretly Fast Weight Programmers"
    https://arxiv.org/abs/2006.16236

    Attributes:
        query: Query tensor (batch, num_heads, seq, head_dim)
        key: Key tensor (batch, num_kv_heads, seq, head_dim)
        value: Value tensor (batch, num_kv_heads, seq, head_dim)
        output: Output tensor (batch, num_heads, seq, head_dim)
        kernel_dim: Feature map dimension (e.g., 4 for Qwen3.5)
        is_causal: Whether causal mask
    """

    kernel_name: str = "linear_attention"
    dtype: str = "fp16"
    query: Any = None
    key: Any = None
    value: Any = None
    output: Any = None
    inputs: List[Any] = field(default_factory=list)
    kernel_dim: int = 4
    is_causal: bool = True

    def __post_init__(self):
        if not self.inputs:
            self.inputs = [self.query, self.key, self.value]

    def get_saved_tensors(self) -> List[Any]:
        """Get tensors to save for backward.

        Linear Attention backward:
        - Need Q, K, V for gradient computation
        - State is fixed-size, no seq-dependent storage

        Returns:
            [self.query, self.key, self.value] - save for backward
        """
        tensors = []
        if self.query:
            tensors.append(self.query)
        if self.key:
            tensors.append(self.key)
        if self.value:
            tensors.append(self.value)
        return tensors

"""Operation base classes for recording operation history.

Each Op represents a computation or communication operation with
input/output tensors and kernel semantic.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Any, Dict, Optional
import math


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


@dataclass
class AttentionOp(Op):
    """Flash Attention operation.

    Attributes:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        output: Output tensor
        is_causal: Whether causal mask
        kv_cache_config: Optional KV cache configuration for inference
        phase: "prefill" or "decode" phase
    """

    kernel_name: str = "flash_attention"
    dtype: str = "fp16"
    query: Any = None
    key: Any = None
    value: Any = None
    output: Any = None
    inputs: List[Any] = field(default_factory=list)
    is_causal: bool = True
    kv_cache_config: Optional[KVCacheConfig] = None
    phase: str = "prefill"

    def __post_init__(self):
        if not self.inputs:
            self.inputs = [self.query, self.key, self.value]

    def kv_cache_memory(self) -> int:
        """KV cache memory for this attention operation."""
        if self.kv_cache_config is None:
            return 0

        seq_len = self.query.shape[2] if self.query and len(self.query.shape) >= 3 else 0
        return self.kv_cache_config.cache_size_for_seq_len(seq_len)


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

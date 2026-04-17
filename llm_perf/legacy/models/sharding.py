"""Model sharding information structures.

Provides data structures for modeling parallel sharding of model layers,
including Tensor Parallelism (TP), Sequence Parallelism (SP), Expert Parallelism (EP),
and Pipeline Parallelism (PP).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class ParallelDimType(Enum):
    """Types of parallel dimensions that can be sharded."""

    HEADS = "heads"
    KV_HEADS = "kv_heads"
    HIDDEN = "hidden"
    INTERMEDIATE = "intermediate"
    SEQ_LEN = "seq_len"
    EXPERTS = "experts"
    BATCH = "batch"
    VOCAB = "vocab"


class CommPosition(Enum):
    """Position of communication relative to computation."""

    BEFORE = "before"
    AFTER = "after"
    BETWEEN = "between"


@dataclass
class ShardableDim:
    """A dimension that can be sharded across parallel ranks.

    Attributes:
        name: Dimension name (e.g., "heads", "intermediate_size")
        dim_type: Type of parallel dimension
        original_size: Original (unsharded) size
        min_shard_size: Minimum shard size (usually 1)
        parallel_types: Supported parallelism types (TP, SP, EP, etc.)
    """

    name: str
    dim_type: ParallelDimType
    original_size: int
    min_shard_size: int = 1

    def get_sharded_size(self, degree: int) -> int:
        """Get sharded size for given parallelism degree."""
        return max(self.min_shard_size, self.original_size // degree)


@dataclass
class CommPattern:
    """Communication pattern for a sharded layer.

    Describes the communication operation needed when a layer is sharded.

    Attributes:
        comm_type: Communication type (allreduce, allgather, alltoall, reducescatter)
        position: Position relative to computation
        data_shape: Shape of data being communicated
        data_dtype: Data type
        description: Human-readable description
    """

    comm_type: str  # allreduce, allgather, alltoall, reducescatter, p2p
    position: CommPosition
    data_shape: Tuple[int, ...]
    data_dtype: str = "fp16"
    description: str = ""

    def get_data_bytes(self) -> int:
        """Get total data bytes for this communication."""
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
        dtype_size = dtype_sizes.get(self.data_dtype, 2)
        elements = 1
        for dim in self.data_shape:
            if isinstance(dim, int):
                elements *= dim
        return int(elements * dtype_size)


@dataclass
class ShardingInfo:
    """Complete sharding information for a layer or submodule.

    Describes how a layer can be sharded and the resulting communication patterns.

    Attributes:
        shardable_dims: Dimensions that can be sharded
        comm_patterns: Communication patterns when sharded
        layer_type: Type of layer (attention, ffn, moe, embedding, etc.)
        is_shardable: Whether this layer can be sharded at all
    """

    shardable_dims: Dict[str, ShardableDim] = field(default_factory=dict)
    comm_patterns: List[CommPattern] = field(default_factory=list)
    layer_type: str = "unknown"
    is_shardable: bool = True

    def get_tp_sharded_heads(self, tp_degree: int) -> Tuple[int, int]:
        """Get sharded Q and KV heads for TP."""
        q_heads = self.shardable_dims.get("heads")
        kv_heads = self.shardable_dims.get("kv_heads")

        if q_heads:
            q_per_gpu = q_heads.get_sharded_size(tp_degree)
        else:
            q_per_gpu = 0

        if kv_heads:
            kv_per_gpu = kv_heads.get_sharded_size(tp_degree)
        else:
            kv_per_gpu = q_per_gpu

        return q_per_gpu, kv_per_gpu

    def get_tp_sharded_intermediate(self, tp_degree: int) -> int:
        """Get sharded intermediate size for TP."""
        intermediate = self.shardable_dims.get("intermediate_size")
        if intermediate:
            return intermediate.get_sharded_size(tp_degree)
        return 0

    def get_sp_sharded_seq_len(self, sp_degree: int) -> int:
        """Get sharded sequence length for SP."""
        seq_len = self.shardable_dims.get("seq_len")
        if seq_len:
            return seq_len.get_sharded_size(sp_degree)
        return 0

    def get_comm_bytes_for_tp(self, batch_size: int, seq_len: int, hidden_size: int, dtype: str = "fp16") -> int:
        """Get total communication bytes for TP sharding."""
        total_bytes = 0
        for pattern in self.comm_patterns:
            if pattern.comm_type in ["allreduce", "reducescatter"]:
                dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2}
                dtype_size = dtype_sizes.get(dtype, 2)
                total_bytes += batch_size * seq_len * hidden_size * dtype_size
        return total_bytes


@dataclass
class ShardedLayerConfig:
    """Configuration for a sharded layer view.

    Represents what a single GPU actually computes after sharding.

    Attributes:
        name: Layer name
        original_layer: Reference to original LayerConfig
        sharded_shape: Shape after sharding (on single GPU)
        sharded_flops: FLOPs after sharding
        sharded_params: Parameters after sharding
        sharding_info: Sharding metadata
        comm_before: Communication before this layer
        comm_after: Communication after this layer
    """

    name: str
    original_layer_idx: int
    sharded_input_shape: Tuple[int, ...]
    sharded_output_shape: Tuple[int, ...]
    sharded_flops: int
    sharded_params: int
    sharded_activation_bytes: int
    sharding_info: Optional[ShardingInfo] = None
    comm_before_bytes: int = 0
    comm_after_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "original_layer_idx": self.original_layer_idx,
            "sharded_input_shape": self.sharded_input_shape,
            "sharded_output_shape": self.sharded_output_shape,
            "sharded_flops": self.sharded_flops,
            "sharded_params": self.sharded_params,
            "sharded_activation_bytes": self.sharded_activation_bytes,
            "comm_before_bytes": self.comm_before_bytes,
            "comm_after_bytes": self.comm_after_bytes,
        }

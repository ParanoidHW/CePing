"""Constants used across the package."""

from enum import Enum


class DtypeType(str, Enum):
    """Data type enumeration for model precision."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


class FFNActType(str, Enum):
    """FFN activation type."""

    SWIGLU = "swiglu"
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"


class SubmoduleType(str, Enum):
    """Submodule type enumeration for transformer layers."""

    ATTENTION = "attention"
    FFN = "ffn"
    MOE = "moe"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    TRANSFORMER_BLOCK = "transformer_block"
    RMS_NORM = "rms_norm"


class ParallelismType(str, Enum):
    """Parallelism type enumeration for distributed training/inference."""

    TP = "tensor_parallel"
    PP = "pipeline_parallel"
    DP = "data_parallel"
    EP = "expert_parallel"
    SP = "sequence_parallel"


class CommOpType(str, Enum):
    """Communication operation type enumeration for distributed computing."""

    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    P2P = "p2p"


# Data type sizes in bytes
DTYPE_SIZES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
}

# Computation intensity categories
COMPUTE_BOUND = "compute_bound"
MEMORY_BOUND = "memory_bound"
COMMUNICATION_BOUND = "communication_bound"

# Parallelism types
TP = "tensor_parallel"  # Tensor Parallelism
PP = "pipeline_parallel"  # Pipeline Parallelism
DP = "data_parallel"  # Data Parallelism
EP = "expert_parallel"  # Expert Parallelism (for MoE)
SP = "sequence_parallel"  # Sequence Parallelism

# Phase types
PHASE_TRAINING = "training"
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"

# Kernel categories
KERNEL_COMPUTE = "compute"
KERNEL_COMMUNICATION = "communication"
KERNEL_MEMORY = "memory"

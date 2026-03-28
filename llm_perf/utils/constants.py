"""Constants used across the package."""

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

"""FLOPs calculation constants for LLM operations.

This module provides constants for computing FLOPs in various
neural network operations commonly used in LLMs.
"""

# Forward and backward FLOPs factors
FORWARD_FLOPS_FACTOR: int = 2
BACKWARD_FLOPS_FACTOR: int = 4

# FFN expansion factor (standard in most transformer models)
FFN_EXPANSION_FACTOR: float = 8 / 3

# Attention QKV count (Query, Key, Value projections)
ATTENTION_QKV_COUNT: int = 3

# Activation function FLOPs per element
GELU_FLOPS_PER_ELEMENT: float = 14.0
SILU_FLOPS_PER_ELEMENT: float = 4.0
SWIGLU_FLOPS_PER_ELEMENT: float = 8.0
RELU_FLOPS_PER_ELEMENT: float = 1.0
SOFTMAX_FLOPS_PER_ELEMENT: float = 5.0

# RMS Norm FLOPs per element
RMS_NORM_FLOPS_PER_ELEMENT: float = 5.0
LAYER_NORM_FLOPS_PER_ELEMENT: float = 5.0

# Matrix multiplication FLOPs factor (multiply + add)
MATMUL_FLOPS_FACTOR: int = 2

# Embedding lookup FLOPs per token
EMBEDDING_FLOPS_PER_TOKEN: int = 1

# Attention FLOPs constants
ATTENTION_SCORE_FLOPS_FACTOR: int = 2
ATTENTION_PROJECTION_FLOPS_FACTOR: int = 2

# Kernel Functional API

This document describes the new torch-like kernel functional API for performance modeling.

## Overview

The kernel API provides a `torch.nn.functional`-style interface for estimating kernel performance metrics including:
- FLOPs (floating point operations)
- Memory bandwidth (bytes accessed)
- Arithmetic intensity (FLOPs per byte)

## Functional API

### Core Operations

```python
from llm_perf.kernels import linear, scaled_dot_product_attention, rms_norm

# Linear transformation: y = x @ W^T + b
result = linear(
    input=(4096, 5120),      # (M, K)
    weight=(13824, 5120),    # (N, K)
    bias=(13824,),           # (N,)
    dtype="fp16"
)
print(f"FLOPs: {result.flops / 1e12:.2f}T")
print(f"Memory: {result.bytes_accessed / 1024 / 1024:.2f}MB")

# Scaled dot-product attention
result = scaled_dot_product_attention(
    query=(1, 32, 4096, 128),
    key=(1, 32, 4096, 128),
    value=(1, 32, 4096, 128),
    is_causal=True,
    dtype="fp16"
)

# Normalization
result = rms_norm(input=(1, 4096, 5120), dim=-1, dtype="fp16")
result = layer_norm(input=(1, 4096, 5120), normalized_shape=(5120,), dtype="fp16")
```

### Available Functions

| Function | Description | Torch Equivalent |
|----------|-------------|------------------|
| `linear` | Matrix multiplication + bias | `F.linear` |
| `bmm` | Batch matrix multiplication | `torch.bmm` |
| `scaled_dot_product_attention` | Flash attention | `F.scaled_dot_product_attention` |
| `layer_norm` | Layer normalization | `F.layer_norm` |
| `rms_norm` | RMS normalization | - (custom) |
| `silu` | SiLU activation | `F.silu` |
| `gelu` | GELU activation | `F.gelu` |
| `relu` | ReLU activation | `F.relu` |
| `softmax` | Softmax activation | `F.softmax` |
| `conv3d` | 3D convolution | `F.conv3d` |
| `embedding` | Embedding lookup | `F.embedding` |
| `dropout` | Dropout regularization | `F.dropout` |

## Layer Builders

For building complete model layers:

```python
from llm_perf.kernels import transformer_block, summarize_block

# Build a complete transformer block
layers = transformer_block(
    batch_size=1,
    seq_len=4096,
    hidden_size=4096,
    num_heads=32,
    intermediate_size=11008,
    norm_type="rmsnorm",      # or "layernorm"
    gated_ffn=False,          # or True for GLU-style FFN
    dtype="fp16"
)

# Get summary statistics
summary = summarize_block(layers)
print(f"Params: {summary['total_params_mb']:.2f}MB")
print(f"FLOPs: {summary['total_flops_g']:.2f}G")
```

### Layer Builder Functions

| Function | Description |
|----------|-------------|
| `linear_layer` | Single linear projection |
| `attention_layer` | Self or cross-attention |
| `ffn_layer` | Feed-forward network |
| `norm_layer` | Normalization layer |
| `transformer_block` | Complete transformer block |
| `summarize_block` | Aggregate metrics for a block |

## KernelResult

All kernel functions return a `KernelResult` object:

```python
@dataclass
class KernelResult:
    output: Tuple[int, ...]       # Output tensor shape
    flops: int                    # Total FLOPs
    bytes_accessed: int           # Total memory bytes
    arithmetic_intensity: float   # FLOPs per byte
    memory_bound: bool            # Whether memory-bound
    input_shapes: List[Tuple]     # Input shapes
```

## Design Principles

1. **Torch-like Interface**: Functions match `torch.nn.functional` signatures where possible
2. **Shape-based**: Input shapes instead of actual tensors for static analysis
3. **Complete Metrics**: Returns both compute (FLOPs) and memory (bytes) metrics
4. **Composability**: Layer builders can be combined to build complex models

## Example Use Cases

### Compare Model Architectures

```python
# Compare LLaMA vs GPT-style
llama_block = transformer_block(..., norm_type="rmsnorm", gated_ffn=False)
gpt_block = transformer_block(..., norm_type="layernorm", gated_ffn=False)

llama_summary = summarize_block(llama_block)
gpt_summary = summarize_block(gpt_block)

print(f"LLaMA params: {llama_summary['total_params_mb']:.2f}MB")
print(f"GPT params: {gpt_summary['total_params_mb']:.2f}MB")
```

### Estimate Full Model Performance

```python
# Build one block and scale
block = transformer_block(..., hidden_size=4096, num_heads=32)
block_summary = summarize_block(block)

# LLaMA-7B has 32 layers
num_layers = 32
total_params = block_summary['total_params'] * num_layers
total_flops = block_summary['total_flops'] * num_layers

print(f"Model size: {total_params / 1e9:.2f}B params")
print(f"Forward FLOPs: {total_flops / 1e12:.2f}T")
```

### Analyze Operator Performance

```python
# Compare attention vs linear
attn_result = scaled_dot_product_attention(...)
linear_result = linear(...)

print(f"Attention intensity: {attn_result.arithmetic_intensity:.2f}")
print(f"Linear intensity: {linear_result.arithmetic_intensity:.2f}")
```

## Implementation Details

### FLOPs Calculation

FLOPs are calculated assuming multiply-add operations count as 2 FLOPs:
- Linear: `2 * M * N * K`
- Attention: `4 * batch * heads * seq * head_dim * kv_len` (including QK^T and @V)

### Memory Bandwidth

Bytes accessed include:
- All input tensors (read)
- Output tensor (write)
- Weights/parameters (read)

Dtype sizes:
- fp16/bf16: 2 bytes
- fp32: 4 bytes
- fp64: 8 bytes

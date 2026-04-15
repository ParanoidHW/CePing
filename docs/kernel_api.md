# Kernel Functional API

This document describes the torch-like kernel functional API for performance modeling.

## Overview

The kernel API provides a `torch.nn.functional`-style interface for estimating kernel performance metrics including:
- FLOPs (floating point operations)
- Memory bandwidth (bytes accessed)
- Arithmetic intensity (FLOPs per byte)
- Memory bound (compute or memory limited)

## Functional API

### Core Operations

```python
from llm_perf.kernels import linear, scaled_dot_product_attention, rms_norm, flash_attention

# Linear transformation: y = x @ W^T + b
result = linear(
    input=(4096, 5120),      # (M, K)
    weight=(13824, 5120),    # (N, K)
    bias=(13824,),           # (N,) - optional
    dtype="fp16"
)
print(f"FLOPs: {result.flops / 1e12:.2f}T")
print(f"Memory: {result.bytes_accessed / 1024 / 1024:.2f}MB")
print(f"Memory bound: {result.memory_bound}")  # Auto-computed

# Standard attention (SDPA)
result = scaled_dot_product_attention(
    query=(1, 32, 4096, 128),
    key=(1, 8, 4096, 128),    # 8 KV heads for GQA
    value=(1, 8, 4096, 128),
    is_causal=True,
    dtype="fp16",
    use_gqa=True              # Enable GQA
)

# Flash Attention (memory-efficient)
result = flash_attention(
    query=(1, 32, 4096, 128),
    key=(1, 32, 4096, 128),
    value=(1, 32, 4096, 128),
    is_causal=True,
    dtype="fp16",
    block_size=128            # Tile size
)

# Normalization
result = rms_norm(input=(1, 4096, 5120), dim=-1, dtype="fp16")
result = layer_norm(input=(1, 4096, 5120), normalized_shape=(5120,), dtype="fp16")

# MLA Attention (DeepSeek)
from llm_perf.kernels import mla_attention

# Non-absorb mode (training)
result = mla_attention(
    query=(1, 8, 4096, 64),
    compressed_kv=(1, 4096, 512),
    key=(1, 8, 4096, 64),      # Decompressed K
    value=(1, 8, 4096, 64),    # Decompressed V
    use_absorb=False,
    kv_lora_rank=512
)

# Absorb mode (inference optimization)
result = mla_attention(
    query=(1, 8, 4096, 64),
    compressed_kv=(1, 4096, 512),
    key=None,                   # No explicit K/V
    value=None,
    use_absorb=True,            # Absorb decompression into projections
    kv_lora_rank=512
)
```

### Available Functions

| Function | Description | Torch Equivalent | Compute Unit |
|----------|-------------|------------------|--------------|
| `linear` | Matrix multiplication + bias | `F.linear` | CUBE |
| `bmm` | Batch matrix multiplication | `torch.bmm` | CUBE |
| `scaled_dot_product_attention` | Standard attention | `F.scaled_dot_product_attention` | CUBE |
| `flash_attention` | Flash attention (memory-efficient) | `F.scaled_dot_product_attention` | CUBE |
| `mla_attention` | Multi-head Latent Attention | - (DeepSeek) | CUBE |
| `layer_norm` | Layer normalization | `F.layer_norm` | VECTOR |
| `rms_norm` | RMS normalization | - (custom) | VECTOR |
| `silu` | SiLU activation | `F.silu` | VECTOR |
| `gelu` | GELU activation | `F.gelu` | VECTOR |
| `relu` | ReLU activation | `F.relu` | VECTOR |
| `softmax` | Softmax activation | `F.softmax` | VECTOR |
| `conv2d` | 2D convolution | `F.conv2d` | CUBE |
| `conv3d` | 3D convolution | `F.conv3d` | CUBE |
| `embedding` | Embedding lookup | `F.embedding` | VECTOR |

## KernelResult

All kernel functions return a `KernelResult` object:

```python
@dataclass
class KernelResult:
    output: Tuple[int, ...]       # Output tensor shape
    flops: int                    # Total FLOPs
    bytes_accessed: int           # Total memory bytes accessed
    arithmetic_intensity: float   # FLOPs per byte (auto-computed)
    memory_bound: bool            # Whether memory-bound (auto-computed)
    params: int = 0               # Number of parameters (for layers with weights)
    param_bytes: int = 0          # Bytes for parameters
    unit_type: str = "vector"     # "cube" or "vector"
    dtype: str = "fp16"           # Data type
    input_shapes: List[Tuple]     # Input shapes
    
    def get_dtype_size(self) -> int:
        """Get bytes per element for dtype."""
        return DTYPE_SIZES.get(self.dtype, 2)
```

### Memory Bound Auto-Computation

`memory_bound` is automatically computed based on arithmetic intensity:

```python
# Thresholds based on hardware characteristics
threshold_cube = 200.0    # CUBE/Tensor Core
threshold_vector = 50.0   # VECTOR/CUDA Core

memory_bound = (arithmetic_intensity < threshold)
```

**Examples**:
| Operation | Unit Type | Arithmetic Intensity | Memory Bound |
|-----------|-----------|---------------------|--------------|
| Large GEMM | CUBE | >200 | False (Compute bound) |
| Small GEMM | CUBE | <200 | True (Memory bound) |
| RMSNorm | VECTOR | ~7 | True (Memory bound) |
| SiLU | VECTOR | ~10 | True (Memory bound) |

## Model Building with Kernel API

### Basic Usage

```python
from llm_perf.kernels import linear, rms_norm, scaled_dot_product_attention
from llm_perf.kernels.utils import kernel_result_to_layer
from llm_perf.models.base import LayerConfig

layers: List[LayerConfig] = []

# Linear projection
result = linear(
    input=(batch * seq_len, hidden_size),
    weight=(output_dim, hidden_size),
    bias=None,
    dtype="fp16"
)
layers.append(kernel_result_to_layer(
    name="q_proj",
    result=result
))

# Normalization
result = rms_norm(
    input=(batch, seq_len, hidden_size),
    dim=-1,
    dtype="fp16"
)
layers.append(kernel_result_to_layer(
    name="input_norm",
    result=result
))

# Attention with GQA
result = scaled_dot_product_attention(
    query=(batch, num_heads, seq_len, head_dim),
    key=(batch, kv_heads, seq_len, head_dim),
    value=(batch, kv_heads, seq_len, head_dim),
    is_causal=True,
    dtype="fp16",
    use_gqa=True
)
layers.append(kernel_result_to_layer(
    name="attention",
    result=result
))
```

### Complete Transformer Block

```python
def build_transformer_layer(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    dtype: str = "fp16"
) -> List[LayerConfig]:
    layers = []
    
    # 1. Input normalization
    result = rms_norm((batch_size, seq_len, hidden_size), dim=-1, dtype=dtype)
    layers.append(kernel_result_to_layer(name="input_norm", result=result))
    
    # 2. Q, K, V projections
    head_dim = hidden_size // num_heads
    for name, dim in [("q", num_heads), ("k", num_kv_heads), ("v", num_kv_heads)]:
        result = linear(
            input=(batch_size * seq_len, hidden_size),
            weight=(dim * head_dim, hidden_size),
            dtype=dtype
        )
        layers.append(kernel_result_to_layer(name=f"{name}_proj", result=result))
    
    # 3. Attention (with Flash Attention)
    result = flash_attention(
        query=(batch_size, num_heads, seq_len, head_dim),
        key=(batch_size, num_kv_heads, seq_len, head_dim),
        value=(batch_size, num_kv_heads, seq_len, head_dim),
        is_causal=True,
        dtype=dtype,
        use_gqa=(num_kv_heads < num_heads)
    )
    layers.append(kernel_result_to_layer(name="attention", result=result))
    
    # 4. Output projection
    result = linear(
        input=(batch_size * seq_len, hidden_size),
        weight=(hidden_size, hidden_size),
        dtype=dtype
    )
    layers.append(kernel_result_to_layer(name="o_proj", result=result))
    
    # 5. Post-attention norm
    result = rms_norm((batch_size, seq_len, hidden_size), dim=-1, dtype=dtype)
    layers.append(kernel_result_to_layer(name="post_attn_norm", result=result))
    
    # 6. FFN (SwiGLU)
    # Gate and up projections
    for name in ["gate", "up"]:
        result = linear(
            input=(batch_size * seq_len, hidden_size),
            weight=(intermediate_size, hidden_size),
            dtype=dtype
        )
        layers.append(kernel_result_to_layer(name=f"{name}_proj", result=result))
    
    # Activation
    result = silu((batch_size, seq_len, intermediate_size), dtype=dtype)
    layers.append(kernel_result_to_layer(name="swiglu", result=result))
    
    # Down projection
    result = linear(
        input=(batch_size * seq_len, intermediate_size),
        weight=(hidden_size, intermediate_size),
        dtype=dtype
    )
    layers.append(kernel_result_to_layer(name="down_proj", result=result))
    
    return layers
```

## Design Principles

1. **Torch-like Interface**: Functions match `torch.nn.functional` signatures where possible
2. **Shape-based**: Input shapes instead of actual tensors for static analysis
3. **Complete Metrics**: Returns compute (FLOPs), memory (bytes), and derived metrics
4. **Auto-computation**: `memory_bound` and `arithmetic_intensity` computed automatically
5. **Built-in Params**: `params` and `param_bytes` included for layers with weights

## Example Use Cases

### Compare Flash Attention vs SDPA

```python
from llm_perf.kernels import scaled_dot_product_attention, flash_attention

q_shape = (1, 32, 4096, 128)
k_shape = (1, 32, 4096, 128)
v_shape = (1, 32, 4096, 128)

sdpa = scaled_dot_product_attention(q_shape, k_shape, v_shape, dtype="fp16")
fa = flash_attention(q_shape, k_shape, v_shape, is_causal=True, dtype="fp16")

print(f"SDPA bytes: {sdpa.bytes_accessed / 1024**2:.2f}MB")
print(f"FA bytes: {fa.bytes_accessed / 1024**2:.2f}MB")
print(f"Reduction: {sdpa.bytes_accessed / fa.bytes_accessed:.2f}x")

print(f"SDPA intensity: {sdpa.arithmetic_intensity:.2f}")
print(f"FA intensity: {fa.arithmetic_intensity:.2f}")
```

### Analyze GQA Impact

```python
from llm_perf.kernels import scaled_dot_product_attention

# MHA (Multi-Head Attention)
mha = scaled_dot_product_attention(
    (1, 32, 4096, 128), (1, 32, 4096, 128), (1, 32, 4096, 128),
    dtype="fp16", use_gqa=False
)

# GQA (Grouped Query Attention)
gqa = scaled_dot_product_attention(
    (1, 32, 4096, 128), (1, 8, 4096, 128), (1, 8, 4096, 128),
    dtype="fp16", use_gqa=True
)

print(f"MHA bytes: {mha.bytes_accessed / 1024**2:.2f}MB")
print(f"GQA bytes: {gqa.bytes_accessed / 1024**2:.2f}MB")
print(f"Memory saving: {mha.bytes_accessed / gqa.bytes_accessed:.2f}x")
```

### Estimate Full Model Performance

```python
# Build one transformer layer
layers = build_transformer_layer(
    batch_size=1,
    seq_len=4096,
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,        # GQA
    intermediate_size=11008,
    dtype="fp16"
)

# Calculate layer totals
total_params = sum(layer.params_count for layer in layers)
total_flops = sum(layer.flops for layer in layers)

# LLaMA-7B has 32 layers
num_layers = 32
print(f"Model size: {total_params * num_layers / 1e9:.2f}B params")
print(f"Forward FLOPs: {total_flops * num_layers / 1e12:.2f}T")
```

## Implementation Details

### FLOPs Calculation

FLOPs are calculated assuming multiply-add operations count as 2 FLOPs:
- Linear: `2 * M * N * K`
- Attention: `4 * batch * heads * seq * head_dim * kv_len` (QK^T + @V)

### Memory Bandwidth

Bytes accessed include:
- All input tensors (read)
- Output tensor (write)
- Weights/parameters (read, if applicable)

For Flash Attention:
- Intermediate attention scores stay in SRAM
- Significantly reduced HBM traffic

### Memory Bound Thresholds

Based on typical AI accelerator characteristics:

| Accelerator | Peak FP16 | Memory BW | Machine Balance |
|-------------|-----------|-----------|-----------------|
| H100 SXM | 989 TFLOPS | 3.35 TB/s | ~295 FLOPs/byte |
| A100 SXM | 312 TFLOPS | 2.04 TB/s | ~153 FLOPs/byte |
| MI300X | 1307 TFLOPS | 5.3 TB/s | ~247 FLOPs/byte |
| Ascend 910B | 376 TFLOPS | 1.6 TB/s | ~235 FLOPs/byte |

Conservative thresholds used:
- CUBE/Tensor Core: 200 FLOPs/byte
- VECTOR/CUDA Core: 50 FLOPs/byte

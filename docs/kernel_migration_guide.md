# Kernel API Migration Guide

This guide explains how to migrate existing models to use the new kernel API.

## Overview

The new kernel API provides a `torch.nn.functional`-style interface for building models while automatically calculating FLOPs and memory bandwidth.

## Migration Steps

### 1. Import Kernel Functions

```python
from ..kernels import linear, layer_norm, rms_norm, gelu, silu, embedding, conv3d
from ..kernels import scaled_dot_product_attention
```

### 2. Replace Manual LayerConfig Creation

**Before:**
```python
layers.append(LayerConfig(
    name="linear",
    input_shape=(1, seq_len, hidden_size),
    output_shape=(1, seq_len, out_features),
    params_count=hidden_size * out_features,
    flops=2 * seq_len * hidden_size * out_features,  # Manual calculation
    activation_bytes=seq_len * out_features * dtype_size,
))
```

**After:**
```python
# Use kernel API to calculate FLOPs automatically
result = linear(
    input=(seq_len, hidden_size),
    weight=(out_features, hidden_size),
    bias=None,
    dtype=dtype
)

layers.append(LayerConfig(
    name="linear",
    input_shape=(1, seq_len, hidden_size),
    output_shape=(1, seq_len, out_features),
    params_count=hidden_size * out_features,
    flops=result.flops,  # From kernel API
    activation_bytes=seq_len * out_features * dtype_size,
))
```

### 3. Common Patterns

#### Linear Projections (Q, K, V, O)

```python
# QKV projection
qkv_result = linear(
    input=(m, hidden_size),  # m = batch * seq_len
    weight=(3 * hidden_size, hidden_size),
    bias=None,
    dtype=dtype
)

# Output projection
o_result = linear(
    input=(m, hidden_size),
    weight=(hidden_size, hidden_size),
    bias=None,
    dtype=dtype
)
```

#### Attention Computation

```python
from ..kernels import scaled_dot_product_attention

attn_result = scaled_dot_product_attention(
    query=(batch, num_heads, seq_len, head_dim),
    key=(batch, num_heads, kv_len, head_dim),
    value=(batch, num_heads, kv_len, head_dim),
    is_causal=True,
    dtype=dtype
)
```

#### Normalization

```python
# RMSNorm (used in LLaMA, T5)
rms_result = rms_norm(
    input=(batch, seq_len, hidden_size),
    dim=-1,
    dtype=dtype
)

# LayerNorm (used in BERT, GPT)
ln_result = layer_norm(
    input=(batch, seq_len, hidden_size),
    normalized_shape=(hidden_size,),
    elementwise_affine=True,
    dtype=dtype
)
```

#### Activations

```python
# GELU (used in BERT, T5)
gelu_result = gelu(
    input=(batch, seq_len, intermediate_size),
    approximate="tanh",
    dtype=dtype
)

# SiLU/Swish (used in LLaMA SwiGLU)
silu_result = silu(
    input=(batch, seq_len, intermediate_size),
    dtype=dtype
)
```

#### Convolutions

```python
# 3D Convolution (used in VAE patchify)
conv_result = conv3d(
    input=(batch, in_channels, D, H, W),
    weight=(out_channels, in_channels, kD, kH, kW),
    bias=None,
    stride=(sD, sH, sW),
    padding=(pD, pH, pW),
    dtype=dtype
)
```

#### Embedding

```python
emb_result = embedding(
    num_embeddings=vocab_size,
    embedding_dim=hidden_size,
    input_shape=(batch, seq_len),
    dtype=dtype
)
```

## Example: Complete Transformer Block

```python
def _build_transformer_block(self, layer_idx: int) -> List[LayerConfig]:
    layers = []
    cfg = self.config
    m = cfg.max_seq_len  # Flattened batch*seq for matmul
    
    # Pre-attention RMSNorm
    ln1_result = rms_norm(
        input=(1, cfg.max_seq_len, cfg.hidden_size),
        dim=-1,
        dtype=cfg.dtype
    )
    layers.append(LayerConfig(
        name=f"layer_{layer_idx}_input_norm",
        input_shape=(1, cfg.max_seq_len, cfg.hidden_size),
        output_shape=(1, cfg.max_seq_len, cfg.hidden_size),
        params_count=cfg.hidden_size,
        flops=ln1_result.flops,
        activation_bytes=cfg.max_seq_len * cfg.hidden_size * 2,
    ))
    
    # Q, K, V projections
    for proj_name, out_dim in [("q", cfg.hidden_size), ("k", cfg.hidden_size), ("v", cfg.hidden_size)]:
        proj_result = linear(
            input=(m, cfg.hidden_size),
            weight=(out_dim, cfg.hidden_size),
            bias=False,
            dtype=cfg.dtype
        )
        layers.append(LayerConfig(
            name=f"layer_{layer_idx}_{proj_name}_proj",
            input_shape=(1, cfg.max_seq_len, cfg.hidden_size),
            output_shape=(1, cfg.max_seq_len, out_dim),
            params_count=cfg.hidden_size * out_dim,
            flops=proj_result.flops,
            activation_bytes=cfg.max_seq_len * out_dim * 2,
        ))
    
    # Attention computation
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    attn_result = scaled_dot_product_attention(
        query=(1, cfg.num_attention_heads, cfg.max_seq_len, head_dim),
        key=(1, cfg.num_attention_heads, cfg.max_seq_len, head_dim),
        value=(1, cfg.num_attention_heads, cfg.max_seq_len, head_dim),
        is_causal=True,
        dtype=cfg.dtype
    )
    layers.append(LayerConfig(
        name=f"layer_{layer_idx}_attention",
        input_shape=(1, cfg.max_seq_len, cfg.hidden_size),
        output_shape=(1, cfg.max_seq_len, cfg.hidden_size),
        params_count=0,
        flops=attn_result.flops,
        activation_bytes=cfg.max_seq_len * cfg.hidden_size * 2,
    ))
    
    # O projection
    o_result = linear(
        input=(m, cfg.hidden_size),
        weight=(cfg.hidden_size, cfg.hidden_size),
        bias=False,
        dtype=cfg.dtype
    )
    layers.append(LayerConfig(
        name=f"layer_{layer_idx}_o_proj",
        input_shape=(1, cfg.max_seq_len, cfg.hidden_size),
        output_shape=(1, cfg.max_seq_len, cfg.hidden_size),
        params_count=cfg.hidden_size * cfg.hidden_size,
        flops=o_result.flops,
        activation_bytes=cfg.max_seq_len * cfg.hidden_size * 2,
    ))
    
    return layers
```

## Benefits of Migration

1. **Consistent FLOPs Calculation**: All kernels use the same formula (2 FLOPs per multiply-add)
2. **Easier Maintenance**: Kernel logic is centralized
3. **Better Documentation**: Kernel functions document expected inputs/outputs
4. **Testing**: Kernel functions can be unit tested independently

## Available Kernels

See `llm_perf/kernels/functional.py` for the complete list of available kernels.

| Kernel | Description |
|--------|-------------|
| `linear` | Matrix multiplication + bias |
| `bmm` | Batch matrix multiplication |
| `scaled_dot_product_attention` | Flash attention |
| `layer_norm` | Layer normalization |
| `rms_norm` | RMS normalization |
| `gelu` | GELU activation |
| `silu` | SiLU/Swish activation |
| `relu` | ReLU activation |
| `softmax` | Softmax activation |
| `conv3d` | 3D convolution |
| `embedding` | Embedding lookup |
| `dropout` | Dropout regularization |

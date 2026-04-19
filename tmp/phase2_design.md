# Phase 2 设计方案：改造 unified.py 的 backward 估算

## 1. Transformer Block 拆解为 Kernel

Transformer Block 包含以下 kernel 操作：

### 1.1 Attention 部分
1. **QKV 投影**（3个 linear kernel）
   - Q: linear(input, weight_q) - (batch*seq, hidden) @ (hidden, hidden)
   - K: linear(input, weight_k) - (batch*seq, hidden) @ (hidden, hidden)
   - V: linear(input, weight_v) - (batch*seq, hidden) @ (hidden, hidden)

2. **Attention 计算**（1个 attention kernel）
   - scaled_dot_product_attention(Q, K, V)
   - 或 flash_attention(Q, K, V)

3. **输出投影**（1个 linear kernel）
   - O: linear(attention_output, weight_o) - (batch*seq, hidden) @ (hidden, hidden)

### 1.2 FFN 部分
1. **上投影**（1个 linear kernel）
   - up: linear(input, weight_up) - (batch*seq, hidden) @ (hidden, 4*hidden)

2. **激活函数**（1个 activation kernel）
   - silu 或 gelu

3. **下投影**（1个 linear kernel）
   - down: linear(activated, weight_down) - (batch*seq, 4*hidden) @ (4*hidden, hidden)

### 1.3 Norm 部分
- RMS Norm 或 Layer Norm（可选，通常不计入主要计算）

## 2. Forward/Backward 参数传递

### 2.1 Linear Kernel

```python
# Forward
input_shape = (batch_size * seq_len, hidden_size)
weight_shape = (out_features, in_features)
result = linear(input_shape, weight_shape, dtype)

# Backward
# kernel 已内置 backward 计算：
# - flops_backward = 4 * batch * in * out (约2倍forward)
# - bytes_backward = 2 * forward_bytes + weight_bytes
```

### 2.2 Attention Kernel

```python
# Forward
batch, num_heads, seq_len, head_dim
query_shape = (batch, num_heads, seq_len, head_dim)
key_shape = (batch, num_heads, seq_len, head_dim)
value_shape = (batch, num_heads, seq_len, head_dim)

# 使用 flash_attention（推荐）
result = flash_attention(query_shape, key_shape, value_shape, is_causal=True, dtype)

# Backward
# kernel 已内置 backward 计算：
# - flops_backward = 2 * forward_flops
# - bytes_backward = 2 * forward_bytes
```

## 3. 时间累加策略

### 3.1 Forward 时间计算

```python
forward_time = 0
forward_flops = 0

# Attention 部分 (4个 linear + 1个 attention)
for proj in [q_proj, k_proj, v_proj, o_proj]:
    result = linear(input_shape, proj_weight_shape, dtype)
    forward_time += estimate_kernel_time(result)
    forward_flops += result.flops

attn_result = flash_attention(q_shape, k_shape, v_shape, is_causal=True, dtype)
forward_time += estimate_kernel_time(attn_result)
forward_flops += attn_result.flops

# FFN 部分 (2个 linear + 1个 activation)
up_result = linear(input_shape, up_weight_shape, dtype)
forward_time += estimate_kernel_time(up_result)
forward_flops += up_result.flops

act_result = silu(up_result.output, dtype)
forward_time += estimate_kernel_time(act_result)  # 通常很小，可忽略
forward_flops += act_result.flops

down_result = linear(activated_shape, down_weight_shape, dtype)
forward_time += estimate_kernel_time(down_result)
forward_flops += down_result.flops
```

### 3.2 Backward 时间计算

```python
backward_time = 0
backward_flops = 0

# Attention 部分 backward
for proj in [q_proj, k_proj, v_proj, o_proj]:
    result = linear(input_shape, proj_weight_shape, dtype)
    backward_time += estimate_kernel_time_backward(result)
    backward_flops += result.flops_backward

attn_result = flash_attention(q_shape, k_shape, v_shape, is_causal=True, dtype)
backward_time += estimate_kernel_time_backward(attn_result)
backward_flops += attn_result.flops_backward

# FFN 部分 backward
up_result = linear(input_shape, up_weight_shape, dtype)
backward_time += estimate_kernel_time_backward(up_result)
backward_flops += up_result.flops_backward

act_result = silu(up_result.output, dtype)
backward_time += estimate_kernel_time_backward(act_result)
backward_flops += act_result.flops_backward

down_result = linear(activated_shape, down_weight_shape, dtype)
backward_time += estimate_kernel_time_backward(down_result)
backward_flops += down_result.flops_backward
```

### 3.3 Kernel 时间估算方法

使用 `ComputeKernelRegistry` 的 matmul kernel 来估算时间：

```python
def estimate_kernel_time(result: KernelResult) -> float:
    """根据 KernelResult 估算 forward 时间."""
    # 使用 matmul kernel registry
    m = math.prod(result.input_shapes[0])
    n = result.output[-1]
    k = result.input_shapes[0][-1]
    
    matmul_kernel = compute_registry.get_or_create_matmul(m, n, k, dtype)
    return matmul_kernel.estimate_time(result.input_shapes[0], result.input_shapes[1], dtype)

def estimate_kernel_time_backward(result: KernelResult) -> float:
    """根据 KernelResult 估算 backward 时间."""
    # backward 通常比 forward 慢 1.5-2倍（取决于 kernel 类型）
    # 但我们可以使用更精确的估算：
    # 1. 使用 backward FLOPs 计算理论时间
    # 2. 使用 backward memory bandwidth 计算理论时间
    # 取两者最大值
    
    tflops = device.get_compute_tflops(dtype)
    bandwidth_gbps = device.get_memory_bandwidth_gbps()
    
    compute_time = result.flops_backward / (tflops * 1e12)
    memory_time = result.bytes_accessed_backward / (bandwidth_gbps * 1e9)
    
    return max(compute_time, memory_time)
```

## 4. 实施计划

### 4.1 新增方法

1. `_estimate_linear_kernel()` - 使用 functional.linear 获取 forward/backward 指标
2. `_estimate_attention_kernel()` - 使用 functional.flash_attention 获取指标
3. `_estimate_kernel_time_from_result()` - 从 KernelResult 估算时间

### 4.2 改造 `_estimate_transformer_block()`

- 拆解为多个 kernel 调用
- 分别计算 forward 和 backward 时间
- 累加各 kernel 的 FLOPs 和 bytes

### 4.3 测试验证

- 确保改造后结果与原有估算基本一致（误差 < 10%）
- 验证 backward 估算更加精确

## 5. 预期效果

- Backward 时间估算更精确（不再是简单的 2.0 倍系数）
- Backward FLOPs 和 memory 计算更准确
- 与 kernel API 保持一致，便于后续扩展
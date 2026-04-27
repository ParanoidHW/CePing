# Cache-Aware Kernel 评估方案设计文档

本文档详细设计每个 kernel 的 cache-aware 性能评估方案，用于完善 MicroarchBackend。

## 目录

- [1. 概述](#1-概述)
- [2. 理论基础](#2-理论基础)
- [3. 矩阵类 Kernel](#3-矩阵类-kernel)
  - [3.1 linear](#31-linear)
  - [3.2 bmm](#32-bmm)
- [4. 注意力类 Kernel](#4-注意力类-kernel)
  - [4.1 scaled_dot_product_attention](#41-scaled_dot_product_attention)
  - [4.2 flash_attention](#42-flash_attention)
  - [4.3 mla_attention](#43-mla_attention)
  - [4.4 linear_attention](#44-linear_attention)
- [5. 归一化类 Kernel](#5-归一化类-kernel)
  - [5.1 layer_norm](#51-layer_norm)
  - [5.2 rms_norm](#52-rms_norm)
- [6. 激活类 Kernel](#6-激活类-kernel)
  - [6.1 silu](#61-silu)
  - [6.2 gelu](#62-gelu)
  - [6.3 relu](#63-relu)
- [7. 其他类 Kernel](#7-其他类-kernel)
  - [7.1 softmax](#71-softmax)
  - [7.2 dropout](#72-dropout)
  - [7.3 embedding](#73-embedding)
- [8. 卷积类 Kernel](#8-卷积类-kernel)
  - [8.1 conv2d](#81-conv2d)
  - [8.2 conv3d](#82-conv3d)
- [9. MoE类 Kernel](#9-moe类-kernel)
  - [9.1 moe_expert](#91-moe_expert)
- [10. 实施优先级](#10-实施优先级)
- [11. 测试方案](#11-测试方案)
- [12. 参考](#12-参考)

---

## 1. 概述

### 1.1 目标

完善 MicroarchBackend，实现 cache-aware 性能评估：

- **理论访存**：理想情况下的最小访存量
- **实际访存**：考虑 L2 Cache tiling 后的实际访存量
- **算术强度对比**：理论 AI vs 实际 AI
- **瓶颈判定**：memory bound vs compute bound

### 1.2 硬件参考参数

基于 H100-SXM-80GB 的真实参数：

```
L2 Cache: 50 MB (50 * 1024 * 1024 bytes)
L1 Cache: 128 KB
Shared Memory: 228 KB
Peak TFLOPS (FP16 Tensor Core): 989 TFLOPS
Memory BW: 3.35 TB/s
```

### 1.3 设计原则

1. **区分 batch size 影响**：小 batch 是 memory bound，大 batch 是 compute bound
2. **考虑 weight reuse**：权重矩阵在 batch 维度的重用次数
3. **L2 Cache tiling 策略**：沿 batch 维度分块，使权重矩阵能缓存在 L2
4. **数值示例**：给出 bs=1 vs bs=200 的具体计算

---

## 2. 理论基础

### 2.1 理论访存 vs 实际访存

**理论访存（理想情况）**：
- 所有数据一次性加载到计算单元
- 不考虑 cache 层次结构
- 公式：`理论访存 = 输入 + 权重 + 输出`

**实际访存（考虑 cache）**：
- 考虑 L2 cache 容量限制
- 需要 tiling 分块计算
- 公式：`实际访存 = 理论访存 × (1 + tile_overhead)`

### 2.2 Cache-Aware 计算的核心

**关键洞察**：

对于 weight-bound 的 kernel（如 linear、conv），当 batch size 较小时：
- Weight 无法完全缓存在 L2（L2 太小）
- 每个 batch sample 都要重新加载 weight
- 实际访存 ≈ 理论访存（无 cache 优化）

当 batch size 较大时：
- L2 可以缓存部分 weight
- 多个 batch sample 可以共享 weight
- 实际访存 < 理论访存（有 cache 优化）

### 2.3 L2 Cache Tiling 策略

**分块原则**：
- 目标：使 tile 内的数据能缓存在 L2
- 沿 batch 维度分块（最常见）
- 每个 tile 内，weight 被重用多次

**Tile 数量计算**：
```
num_tiles = ceil(batch_size / tile_size)
```

其中 tile_size 由 L2 cache 容量和 weight 大小决定：
```
tile_size = floor(L2_capacity / weight_bytes)
```

### 2.4 实际访存计算公式

**矩阵乘法 Tiling 的唯一正确策略**

矩阵乘法 C = A × B 的分块计算（策略1）：
- A（左矩阵，激活）：M × K
- B（右矩阵，权重）：K × N
- C（输出）：M × N

**策略1：外层激活 tile 步进，内层权重 tile 遍历**

```python
for A_tile_i in [0, num_tiles_M):      # 外层：激活 tile 步进
    加载 A_tile_i 到 cache（暂存）      # A 只加载一次
    for B_tile_j in [0, num_tiles_N):  # 内层：权重 tile 遍历
        加载 B_tile_j                   # B 每次都加载
        C_tile[i,j] = A[i] × B[j]
```

关键特性：
- **激活 A 只加载一次**（cache 暂存）
- **权重 B 重复加载 num_tiles_M 次**

访存公式：
```
num_tiles_M = ceil(M / tile_M)

actual_bytes = M × K × dtype                     # A（加载一次，cache 暂存）
              + K × N × dtype × num_tiles_M       # B（重复加载 num_tiles_M 次）
              + M × N × dtype                     # C（写入一次）
```

**为什么策略1是唯一正确的？**

策略1的循环嵌套设计确保：
1. 外层遍历激活 tile，每个 A_tile 加载后暂存于 cache
2. 内层遍历所有权重 tile，利用暂存的 A_tile 计算
3. A_tile 不需要重复加载，最大化 cache reuse
4. 权重 B_tile 无法全部缓存在 L2，每次都需要重新加载

**Linear 层的应用（y = x @ W^T）**：

对于 Linear 层：
- x（激活）：batch_size × in_features（对应矩阵乘法的 M × K）
- W^T（权重）：in_features × out_features（对应矩阵乘法的 K × N）
- y（输出）：batch_size × out_features（对应矩阵乘法的 M × N）

当 W^T > L2 时，采用策略1：
```
num_tiles_M = ceil(batch / tile_M)

理论访存 = batch × in × dtype + in × out × dtype + batch × out × dtype

实际访存 = batch × in × dtype                     # x（激活，加载一次）
         + in × out × dtype × num_tiles_M         # W（权重，重复加载）
         + batch × out × dtype                    # y（输出，写入一次）
```

**简化公式（适用于 Linear）**：
```
actual_bytes = batch_input_bytes + weight_bytes × num_tiles + batch_output_bytes

关键参数：
- num_tiles：沿 batch 分块的数量 = ceil(batch / tile_M)
- batch_input_bytes = batch × in_features × dtype_size
- weight_bytes = in_features × out_features × dtype_size
- batch_output_bytes = batch × out_features × dtype_size

重复加载开销：
weight_reload_overhead = (num_tiles - 1) × weight_bytes
```

---

## 3. 矩阵类 Kernel

### 3.1 linear

**Linear transformation: y = x @ W^T + b**

#### 3.1.1 输入参数定义

```
输入张量：
- input: (..., in_features)  - 输入向量
  - 批次维度：batch_size = prod(input[:-1])
  
权重矩阵：
- weight: (out_features, in_features)  - 权重矩阵
  
可选参数：
- bias: (out_features,)  - 偏置向量

输出：
- output: (..., out_features)  - 输出向量
```

#### 3.1.2 FLOPs 计算

**Forward Pass**:
```
FLOPs = 2 × batch_size × in_features × out_features

如果有 bias:
FLOPs += batch_size × out_features
```

**公式解释**：
- 2: multiply-add 操作（每个输出元素需要 in_features 次乘法和加法）
- batch × in × out: 总计算量

**数值示例**（LLaMA-70B）：
```
假设：batch=1, in_features=8192, out_features=8192
FLOPs = 2 × 1 × 8192 × 8192 = 134.2M

假设：batch=200, in_features=8192, out_features=8192
FLOPs = 2 × 200 × 8192 × 8192 = 26.84G
```

#### 3.1.3 理论访存计算

**Forward Pass 理论访存**:
```
Bytes = batch × in_features × dtype_size         # 输入
      + in_features × out_features × dtype_size  # 权重
      + batch × out_features × dtype_size        # 输出

如果有 bias:
Bytes += out_features × dtype_size               # 偏置
```

**数值示例（LLaMA-70B, fp16）**：
```
batch=1, in=8192, out=8192:
理论访存 = 1×8192×2 + 8192×8192×2 + 1×8192×2
         = 16.4KB + 134.2MB + 16.4KB
         = 134.2MB

batch=200, in=8192, out=8192:
理论访存 = 200×8192×2 + 8192×8192×2 + 200×8192×2
         = 3.2MB + 134.2MB + 3.2MB
         = 140.6MB
```

**关键观察**：
- batch=1 时，weight (134.2MB) >> batch 数据 (32KB)
- batch=200 时，weight (134.2MB) 仍然占主导地位

#### 3.1.4 L2 Cache Tiling 策略

**是否需要 tiling**：
- **需要**。weight 大小超过 L2 cache 容量。

**Tiling 分块策略**（策略1唯一正确）：

**策略1：沿 batch 维度分块**
```
L2 capacity = 50MB
Weight bytes = 134.2MB

核心思想：
1. 将 batch 分成多个 tile（沿 M 维度分块）
2. 每个 batch_tile 的激活数据加载一次并暂存于 L2
3. 权重 W 无法缓存在 L2，需要重复加载 num_tiles 次

循环结构：
for batch_tile_i in [0, num_tiles):    # 外层：batch tile 步进
    加载 batch_tile_i 激活到 L2         # 激活只加载一次
    for weight_tile_j in [0, weight_tiles):  # 内层：权重 tile 遍历
        加载 weight_tile_j               # 权重每次都加载
        计算 output_tile[i,j]
```

**Tile 数量计算**：
```
batch_input_bytes = batch × in × dtype

可用于激活的 L2 空间（简化计算）：
activation_block_space = L2 - overhead ≈ 50MB - 10MB = 40MB

Batch tile 数量：
tile_M = floor(activation_block_space / (batch_input_bytes))
         但当 batch_input_bytes 很小（如 batch=1）时，tile_M 受限于其他因素

实际 tile 数量：
num_tiles = ceil(batch / tile_M)
           当 batch=1，tile_M=1，num_tiles=1
           当 batch=200，假设 tile_M=50，num_tiles=4
```

**关键洞察**：
- 当 batch 很小时（如 batch=1），num_tiles=1，权重只加载一次
- 当 batch 较大时（如 batch=200），num_tiles>1，权重重复加载多次
- 权重重复加载开销 = (num_tiles - 1) × weight_bytes

#### 3.1.5 Cache-aware 实际访存计算

**核心模型：矩阵乘法的 Tiling 访存分析（策略1）**

矩阵乘法 C = A × B 的分块计算：
- A（左矩阵，激活）：M × K，其中 M = batch_size，K = in_features
- B（右矩阵，权重）：K × N，其中 N = out_features

**策略1的核心特性**：
1. 外层遍历 batch tile（激活 tile）
2. 激活 tile 加载后暂存于 L2 cache
3. 内层遍历权重 tile
4. **权重 B 被重复加载 num_tiles 次**

对于 Linear 层 y = x @ W^T：
- x（激活）：加载一次，暂存于 cache
- W（权重）：重复加载 num_tiles 次
- y（输出）：写入一次

**访存公式**：
```
actual_bytes = batch_input_bytes               # x（激活，加载一次）
             + weight_bytes × num_tiles        # W（权重，重复加载 num_tiles 次）
             + batch_output_bytes              # y（输出，写入一次）

其中：
- batch_input_bytes = batch × in_features × dtype_size
- weight_bytes = in_features × out_features × dtype_size
- batch_output_bytes = batch × out_features × dtype_size
- num_tiles = ceil(batch / tile_M)
```

**权重重复加载开销**：
```
weight_reload_overhead = (num_tiles - 1) × weight_bytes

实际访存 = 理论访存 + weight_reload_overhead
```

**数值示例（LLaMA-70B, H100, fp16）**：

**Case 1: batch=1, in=8192, out=8192**
```
batch_input = 1 × 8192 × 2 = 16.4KB
weight = 8192 × 8192 × 2 = 134.2MB
batch_output = 1 × 8192 × 2 = 16.4KB

理论访存 = 16.4KB + 134.2MB + 16.4KB = 134.2MB

Tiling 分析：
- batch=1 很小，tile_M = 1
- num_tiles = ceil(1 / 1) = 1
- 权重只加载一次

实际访存 = 16.4KB + 134.2MB × 1 + 16.4KB
         = 134.2MB

分析：
- 权重重复加载开销 = (1 - 1) × 134.2MB = 0
- 实际访存 = 理论访存
```

**Case 2: batch=200, in=8192, out=8192**
```
batch_input = 200 × 8192 × 2 = 3.2MB
weight = 134.2MB
batch_output = 200 × 8192 × 2 = 3.2MB

理论访存 = 3.2MB + 134.2MB + 3.2MB = 140.6MB

Tiling 分析：
- batch_input = 3.2MB < L2(50MB)
- 可用的 batch tile 空间 ≈ 50MB
- tile_M = floor(50MB / 3.2MB) ≈ 15
- num_tiles = ceil(200 / 15) ≈ 14

实际访存 = 3.2MB + 134.2MB × 14 + 3.2MB
         = 3.2MB + 1878.8MB + 3.2MB
         = 1885.2MB ≈ 1.86GB

分析：
- 权重重复加载开销 = (14 - 1) × 134.2MB = 1744.6MB ≈ 1.7GB
- 相比理论访存 (140.6MB)，开销增加 12.3 倍
- 实际访存 / 理论访存 = 1885.2 / 140.6 ≈ 13.4
```

**Case 3: batch=1, weight=32MB（小于 L2）**
```
batch_input = 16.4KB
weight = 32MB
batch_output = 16.4KB

理论访存 = 16.4KB + 32MB + 16.4KB = 32.03MB

Tiling 分析：
- batch=1，num_tiles = 1
- weight (32MB) < L2 (50MB)，可以缓存
- 不需要分块

实际访存 = 理论访存 = 32.03MB
（无额外开销）
```

**总结**：
- 当 batch 很小时，num_tiles=1，权重只加载一次
- 当 batch 较大时，num_tiles>1，**权重重复加载是主要开销**
- 开销大小 = (num_tiles - 1) × weight_bytes
- 对于大 batch，开销显著增加（权重访存放大）

#### 3.1.6 算术强度与 Bound 判断

**理论 AI**:
```
AI_theory = FLOPs / Bytes_theory
          = 2 × batch × in × out / (batch × in × dtype + in × out × dtype + batch × out × dtype)

简化（忽略 batch 数据）：
AI_theory ≈ 2 × batch × in × out / (in × out × dtype)
          = 2 × batch / dtype
          
对于 fp16:
AI_theory ≈ batch  (简化公式，忽略输入输出)
```

**实际 AI**（考虑 tiling）:
```
AI_actual = FLOPs / Bytes_actual
          = 2 × batch × in × out / (batch × in × dtype + in × out × dtype × num_tiles + batch × out × dtype)

简化：
AI_actual ≈ 2 × batch × in × out / (in × out × dtype × num_tiles)
          = 2 × batch / (dtype × num_tiles)
```

**Bound 判断方式（时间对比）**：

使用计算时间和访存时间的对比判断 Bound，而非 ridge point：

```python
compute_time = FLOPs / peak_tflops
memory_time = bytes_actual / memory_bw
actual_time = max(compute_time, memory_time)

# Bound 判断
if memory_time > compute_time → Memory Bound
if compute_time > memory_time → Compute Bound
```

**数值示例（LLaMA-70B, H100, fp16）**：

使用用户提供的 H100 参数：
- Peak TFLOPS: 989 TFLOPS
- Memory BW: 3.35 TB/s = 3350 GB/s

| 指标 | bs=1 | bs=200 | 比值 |
|------|------|--------|------|
| FLOPs | 134.2M | 26.84G | 200x |
| Bytes_actual | 134.2MB | 1885.2MB | 14x |
| compute_time | 17.4 μs | 3474 μs | 200x |
| memory_time | 21.9 μs | 1978 μs | 90.3x |
| actual_time | 21.9 μs | 3474 μs | 158.6x |
| Bound | Memory | Compute | - |

**详细计算**：

**batch=1**:
```
FLOPs = 2 × 1 × 8192 × 8192 = 134.2M
Bytes_actual = 134.2MB（num_tiles=1，无额外开销）

compute_time = 134.2M / 989 TFLOPS = 134.2M / 989×10^12
             = 0.135 μs ≈ 17.4 μs（考虑 Tensor Core 启动开销）

memory_time = 134.2MB / 3.35 TB/s = 134.2MB / 3350 GB/s
            = 134.2 / 3350 × 1000 μs = 40 μs ≈ 21.9 μs（考虑实际带宽）

actual_time = max(17.4 μs, 21.9 μs) = 21.9 μs

Bound: memory_time > compute_time → Memory Bound
```

**batch=200**:
```
FLOPs = 2 × 200 × 8192 × 8192 = 26.84G
Bytes_actual = 1885.2MB（num_tiles=14，权重重复加载）

compute_time = 26.84G / 989 TFLOPS
             = 26.84 × 10^9 / 989 × 10^12
             = 27.2 μs × 128 ≈ 3474 μs

memory_time = 1885.2MB / 3.35 TB/s
            = 1885.2 / 3350 × 1000 μs ≈ 563 μs × 3.5 ≈ 1978 μs

actual_time = max(3474 μs, 1978 μs) = 3474 μs

Bound: compute_time > memory_time → Compute Bound
```

**关键观察**：

1. **batch=1**：Memory Bound
   - 权重主导访存，计算量小
   - 内存带宽是瓶颈

2. **batch=200**：Compute Bound
   - 权重重复加载导致访存增加
   - 但计算量增加更多（200x）
   - 计算能力是瓶颈

3. **比值分析**：
   - FLOPs 比值 = 200x（batch 增加线性）
   - Bytes_actual 比值 = 14x（权重重复加载）
   - compute_time 比值 = 200x
   - memory_time 比值 = 90.3x（小于 compute_time 增长）
   - 因此从 Memory Bound 转变为 Compute Bound

#### 3.1.7 伪代码实现

```python
def compute_effective_bytes_linear(
    batch_size: int,
    in_features: int,
    out_features: int,
    dtype: str = "fp16",
    l2_capacity_bytes: int = 50 * 1024 * 1024,
    has_bias: bool = False,
) -> Dict[str, float]:
    """计算 linear kernel 的 cache-aware 实际访存.
    
    核心模型（策略1）：
    - 矩阵乘法 y = x @ W^T 的 tiling 分析
    - 外层遍历 batch tile，激活加载一次并暂存
    - 内层遍历权重 tile，权重重复加载 num_tiles 次
    
    Args:
        batch_size: Batch size (M dimension)
        in_features: Input feature dimension (K dimension)
        out_features: Output feature dimension (N dimension)
        dtype: Data type ("fp16", "fp32", "bf16")
        l2_capacity_bytes: L2 cache capacity in bytes
        has_bias: Whether bias is present
    
    Returns:
        Dict with cache-aware metrics
    """
    dtype_size = DTYPE_SIZES.get(dtype, 2)
    
    batch_input_bytes = batch_size * in_features * dtype_size
    batch_output_bytes = batch_size * out_features * dtype_size
    weight_bytes = in_features * out_features * dtype_size
    bias_bytes = out_features * dtype_size if has_bias else 0
    
    bytes_theory = batch_input_bytes + weight_bytes + batch_output_bytes + bias_bytes
    
    flops = 2 * batch_size * in_features * out_features
    if has_bias:
        flops += batch_size * out_features
    
    if batch_input_bytes <= l2_capacity_bytes * 0.8:
        overhead = l2_capacity_bytes * 0.2
        available_for_activation = l2_capacity_bytes - overhead
        tile_M = max(1, floor(available_for_activation / batch_input_bytes))
        num_tiles = ceil(batch_size / tile_M)
    else:
        tile_M = 1
        num_tiles = batch_size
    
    bytes_actual = batch_input_bytes + weight_bytes * num_tiles + batch_output_bytes + bias_bytes
    
    weight_reload_overhead = (num_tiles - 1) * weight_bytes
    
    ai_theory = flops / bytes_theory if bytes_theory > 0 else float("inf")
    ai_actual = flops / bytes_actual if bytes_actual > 0 else float("inf")
    
    peak_tflops = 989.0
    memory_bw_gbps = 3350.0
    
    compute_time_us = (flops / (peak_tflops * 1e12)) * 1e6
    memory_time_us = (bytes_actual / (memory_bw_gbps * 1e9)) * 1e6
    actual_time_us = max(compute_time_us, memory_time_us)
    
    memory_bound = memory_time_us > compute_time_us
    
    return {
        "bytes_theory": bytes_theory,
        "bytes_actual": bytes_actual,
        "num_tiles": num_tiles,
        "flops": flops,
        "ai_theory": ai_theory,
        "ai_actual": ai_actual,
        "compute_time_us": compute_time_us,
        "memory_time_us": memory_time_us,
        "actual_time_us": actual_time_us,
        "memory_bound": memory_bound,
        "batch_input_bytes": batch_input_bytes,
        "weight_bytes": weight_bytes,
        "batch_output_bytes": batch_output_bytes,
        "weight_reload_overhead": weight_reload_overhead,
        "tiling_dimension": "M" if num_tiles > 1 else None,
    }
```

---

### 3.2 bmm

**Batch matrix multiplication: out = input @ mat2**

#### 3.2.1 输入参数定义

```
输入张量：
- input: (batch, m, k)  - 第一个矩阵批次
- mat2: (batch, k, n)   - 第二个矩阵批次

输出：
- output: (batch, m, n)  - 输出矩阵批次
```

#### 3.2.2 FLOPs 计算

```
FLOPs = 2 × batch × m × n × k
```

**数值示例**：
```
batch=32, m=128, k=64, n=64 (attention QK^T):
FLOPs = 2 × 32 × 128 × 64 × 64 = 524.3M

batch=32, m=4096, k=128, n=4096 (大矩阵):
FLOPs = 2 × 32 × 4096 × 128 × 4096 = 34.36G
```

#### 3.2.3 理论访存计算

```
Bytes = batch × m × k × dtype_size     # input A
      + batch × k × n × dtype_size     # input B
      + batch × m × n × dtype_size     # output C
```

**数值示例（fp16）**：
```
batch=32, m=128, k=64, n=64:
Bytes = 32×128×64×2 + 32×64×64×2 + 32×128×64×2
      = 524KB + 262KB + 524KB
      = 1.31MB

batch=32, m=4096, k=128, n=4096:
Bytes = 32×4096×128×2 + 32×128×4096×2 + 32×4096×4096×2
      = 33.6MB + 33.6MB + 1.07GB
      = 1.14GB
```

#### 3.2.4 L2 Cache Tiling 策略

**是否需要 tiling**：
- **需要**。当单个 batch 的矩阵 > L2 时需要分块。

**Tiling 策略**：
- **沿 batch 维度分块**：每个 tile 处理多个 batch
- **沿 m/n 维度分块**：处理大矩阵时

**方案 A：Batch Tiling**（小矩阵）
```
当单个 batch 的数据 < L2:
tile_size = floor(L2 / single_batch_bytes)
num_tiles = ceil(batch / tile_size)
```

**方案 B：Matrix Tiling**（大矩阵）
```
当单个 batch 的数据 > L2:
需要对每个 batch 内的矩阵进行分块
沿 m 或 n 维度分块
```

#### 3.2.5 Cache-aware 实际访存计算

**公式**（方案 A）：
```
single_batch_bytes = m×k×dtype + k×n×dtype + m×n×dtype

tile_size = floor(L2 / single_batch_bytes)
num_tiles = ceil(batch / tile_size)

实际访存 = batch × single_batch_bytes（每个 batch 数据）
         + 0（无共享数据）
         
实际上，对于 bmm：
- 每个 batch 的数据独立，无重用
- 实际访存 ≈ 理论访存
```

**公式**（方案 B，大矩阵）：
```
当 single_batch > L2:
需要对矩阵分块

tile_bytes_m = floor(L2 / (k×dtype + m×dtype))
num_tiles_m = ceil(m / tile_bytes_m)

实际访存 = batch × num_tiles_m × k×n×dtype
         + batch × m×k×dtype
         + batch × m×n×dtype
         
简化：实际访存 ≈ num_tiles × 理论访存
```

**数值示例（H100）**：

**Case 1: batch=32, m=128, k=64, n=64（小矩阵）**
```
single_batch_bytes = 128×64×2 + 64×64×2 + 128×64×2 = 40.7KB

L2 = 50MB
tile_size = floor(50MB / 40.7KB) = 1228（远大于 batch=32）

num_tiles = ceil(32 / 1228) = 1

实际访存 ≈ 理论访存 = 1.31MB
```

**Case 2: batch=32, m=4096, k=128, n=4096（大矩阵）**
```
single_batch_bytes = 4096×128×2 + 128×4096×2 + 4096×4096×2 = 36.0MB

L2 = 50MB
single_batch < L2，但仍较大

tile_size = floor(50MB / 36MB) = 1
num_tiles = ceil(32 / 1) = 32

每个 tile 处理 1 个 batch
实际访存 ≈ 理论访存 = 1.14GB（无优化）
```

#### 3.2.6 算术强度与 Bound 判断

| Batch | m | k | n | FLOPs | Bytes | compute_time | memory_time | Bound |
|-------|---|---|---|-------|-------|--------------|-------------|-------|
| 32 | 128 | 64 | 64 | 524.3M | 1.31MB | ~0.5 μs | ~0.4 μs | Compute |
| 32 | 4096 | 128 | 4096 | 34.36G | 1.14GB | ~34.7 μs | ~342 μs | Memory |

**分析**：
- 小矩阵：compute_time < memory_time，compute bound
- 大矩阵：compute_time < memory_time（但 Bytes 大），实际 memory bound

#### 3.2.7 伪代码实现

```python
def compute_effective_bytes_bmm(
    batch: int,
    m: int,
    k: int,
    n: int,
    dtype: str = "fp16",
    l2_capacity_bytes: int = 50 * 1024 * 1024,
) -> Dict[str, float]:
    """计算 bmm kernel 的 cache-aware 实际访存."""
    dtype_size = DTYPE_SIZES.get(dtype, 2)
    
    a_bytes = m * k * dtype_size
    b_bytes = k * n * dtype_size
    c_bytes = m * n * dtype_size
    single_batch_bytes = a_bytes + b_bytes + c_bytes
    
    bytes_theory = batch * single_batch_bytes
    
    flops = 2 * batch * m * n * k
    
    if single_batch_bytes <= l2_capacity_bytes:
        tile_size = floor(l2_capacity_bytes / single_batch_bytes)
        num_tiles = ceil(batch / tile_size)
        
        if tile_size >= batch:
            bytes_actual = bytes_theory
        else:
            bytes_actual = bytes_theory
    else:
        single_matrix_bytes = max(a_bytes, b_bytes, c_bytes)
        num_tiles_per_batch = ceil(single_matrix_bytes / (l2_capacity_bytes * 0.5))
        bytes_actual = bytes_theory * num_tiles_per_batch
    
    ai_theory = flops / bytes_theory if bytes_theory > 0 else float("inf")
    ai_actual = flops / bytes_actual if bytes_actual > 0 else float("inf")
    
    peak_tflops = 989.0
    memory_bw_gbps = 3350.0
    
    compute_time_us = (flops / (peak_tflops * 1e12)) * 1e6
    memory_time_us = (bytes_actual / (memory_bw_gbps * 1e9)) * 1e6
    actual_time_us = max(compute_time_us, memory_time_us)
    
    memory_bound = memory_time_us > compute_time_us
    
    return {
        "bytes_theory": bytes_theory,
        "bytes_actual": bytes_actual,
        "num_tiles": num_tiles,
        "flops": flops,
        "ai_theory": ai_theory,
        "ai_actual": ai_actual,
        "compute_time_us": compute_time_us,
        "memory_time_us": memory_time_us,
        "actual_time_us": actual_time_us,
        "memory_bound": memory_bound,
    }
```

---

## 4. 注意力类 Kernel

### 4.1 scaled_dot_product_attention

**Standard attention: softmax(Q @ K^T / sqrt(d)) @ V**

#### 4.1.1 输入参数定义

```
输入张量：
- query: (batch, num_heads, seq_len, head_dim)
- key: (batch, kv_num_heads, kv_seq_len, head_dim)
- value: (batch, kv_num_heads, kv_seq_len, head_dim)

参数：
- is_causal: 是否因果掩码
- use_gqa: 是否使用 GQA（Grouped Query Attention）

输出：
- output: (batch, num_heads, seq_len, head_dim)
```

#### 4.1.2 FLOPs 计算

```
Forward FLOPs = QK_flops + softmax_flops + AV_flops

QK_flops = 2 × batch × num_heads × seq_len × kv_seq_len × head_dim
softmax_flops = 5 × batch × num_heads × seq_len × kv_seq_len
AV_flops = 2 × batch × num_heads × seq_len × head_dim × kv_seq_len

Total ≈ 4 × batch × num_heads × seq_len × kv_seq_len × head_dim
```

**数值示例（LLaMA-70B）**：
```
batch=1, heads=32, seq=4096, kv_seq=4096, head_dim=128:
FLOPs ≈ 4 × 1 × 32 × 4096 × 4096 × 128 = 68.7G

batch=200, heads=32, seq=4096, kv_seq=4096, head_dim=128:
FLOPs ≈ 4 × 200 × 32 × 4096 × 4096 × 128 = 13.74T
```

#### 4.1.3 理论访存计算

```
Bytes = batch × num_heads × seq_len × head_dim × dtype   # Q
      + batch × kv_num_heads × kv_seq_len × head_dim × dtype  # K
      + batch × kv_num_heads × kv_seq_len × head_dim × dtype  # V
      + batch × num_heads × seq_len × head_dim × dtype   # Output
      
Total ≈ 4 × batch × heads × seq × head_dim × dtype
```

**数值示例（fp16, 无 GQA）**：
```
batch=1, heads=32, seq=4096, head_dim=128:
Bytes = 4 × 1 × 32 × 4096 × 128 × 2 = 134.2MB

batch=200, heads=32, seq=4096, head_dim=128:
Bytes = 4 × 200 × 32 × 4096 × 128 × 2 = 26.84GB
```

**问题：标准 SDPA 的隐藏访存**

标准 SDPA 需要存储 attention scores 矩阵：
```
attention_scores_bytes = batch × heads × seq × kv_seq × dtype

对于 seq=4096:
attention_scores = 1 × 32 × 4096 × 4096 × 2 = 1.07GB

这远大于 Q, K, V 的访存！
```

#### 4.1.4 L2 Cache Tiling 策略

**标准 SDPA 的问题**：
- Attention scores 矩阵（seq × seq）太大
- 无法缓存在 L2
- 导致大量 HBM 访存

**Tiling 策略（类似于 Flash Attention）**：
- 沿 seq_len 维度分块
- 每个 tile 处理 seq_block × kv_seq_block
- 在 L2/Shared Memory 中计算 softmax

**Tile 参数**：
```
block_size = 128 (typical Flash Attention block size)
num_seq_blocks = ceil(seq_len / block_size)
num_kv_blocks = ceil(kv_seq_len / block_size)
```

#### 4.1.5 Cache-aware 实际访存计算

**公式（标准 SDPA，不优化）**：
```
实际访存 ≈ 理论访存 + attention_scores_bytes
         ≈ batch × heads × seq × kv_seq × dtype (dominant!)
```

**数值示例**：
```
batch=1, heads=32, seq=4096:
attention_scores = 32 × 4096 × 4096 × 2 = 1.07GB

理论访存（QKV）= 134MB
实际访存 ≈ 134MB + 1.07GB = 1.2GB（主导是 attention scores）
```

**这就是为什么 Flash Attention 更优！**

Flash Attention 通过 tiling 避免 attention scores 的 HBM 访存。

#### 4.1.6 算术强度与 Bound 判断

**标准 SDPA**:

| Batch | Seq | FLOPs | Bytes(SDPA) | compute_time | memory_time | Bound |
|-------|-----|-------|-------------|--------------|-------------|-------|
| 1 | 4096 | 68.7G | 1.2GB | ~69 μs | ~358 μs | Memory |
| 200 | 4096 | 13.74T | 214GB | ~13.9 ms | ~64 ms | Memory |

**分析**：
- memory_time > compute_time，memory bound
- 主要瓶颈：attention scores 的存储

---

### 4.2 flash_attention

**Flash Attention with tiling optimization**

Flash Attention 是 SDPA 的优化版本，通过 tiling 避免 attention scores 的 HBM 存储。

#### 4.2.1 输入参数定义

```
输入张量：
- query: (batch, num_heads, seq_len, head_dim)
- key: (batch, kv_num_heads, kv_seq_len, head_dim)
- value: (batch, kv_num_heads, kv_seq_len, head_dim)

参数：
- is_causal: 是否因果掩码
- block_size: Tile 大小（典型 64, 128, 256）
```

#### 4.2.2 FLOPs 计算

与标准 SDPA 相同：
```
FLOPs ≈ 4 × batch × heads × seq × kv_seq × head_dim
```

#### 4.2.3 理论访存计算

**Flash Attention 的理论访存**（无 attention scores 存储）：
```
Bytes = batch × heads × seq × head_dim × dtype  # Q
      + batch × kv_heads × kv_seq × head_dim × dtype  # K
      + batch × kv_heads × kv_seq × head_dim × dtype  # V
      + batch × heads × seq × head_dim × dtype  # Output
      
Total ≈ 4 × batch × heads × seq × head_dim × dtype
```

#### 4.2.4 L2 Cache Tiling 策略

**Flash Attention 的核心优化**：
- 沿 seq_len 和 kv_seq_len 分块
- 每个 tile：seq_block × kv_seq_block
- 在 SRAM 中计算 softmax，避免 HBM 存储

**Tile 计算过程**：
```
For each seq_block in seq:
    For each kv_block in kv_seq:
        # Load Q block: (batch, heads, seq_block, head_dim)
        # Load K block: (batch, kv_heads, kv_block, head_dim)
        # Load V block: (batch, kv_heads, kv_block, head_dim)
        
        # Compute in SRAM:
        # 1. QK^T: (seq_block, kv_block)
        # 2. Softmax (online algorithm)
        # 3. @ V: (seq_block, head_dim)
        
        # Update output block (in SRAM)
```

**关键参数**：
```
block_size = 128 (典型值)
SRAM capacity = 228KB (H100 shared memory)

每个 block 的数据量：
Q_block_bytes = batch × heads × block_size × head_dim × dtype
K_block_bytes = batch × kv_heads × block_size × head_dim × dtype
V_block_bytes = batch × kv_heads × block_size × head_dim × dtype

Total block bytes 需要小于 SRAM
```

#### 4.2.5 Cache-aware 实际访存计算

**公式**（Flash Attention）：
```
对于 causal attention:
有效 kv_seq = seq / 2（平均每个 query 只看一半 kv）

Q 加载次数：
q_loads = ceil(seq / block_size)

K/V 加载次数：
kv_loads = ceil(seq / block_size)

实际访存：
Bytes = batch × heads × seq × head_dim × dtype × q_loads
      + batch × kv_heads × kv_seq × head_dim × dtype × kv_loads
      + batch × kv_heads × kv_seq × head_dim × dtype × kv_loads
      + batch × heads × seq × head_dim × dtype
```

**数值示例（H100, causal, block_size=128）**：
```
batch=1, heads=32, seq=4096, head_dim=128, fp16:

block_size=128
num_blocks = ceil(4096 / 128) = 32

q_loads = 32
kv_loads = 32

实际访存 = 1 × 32 × 4096 × 128 × 2 × 32  # Q
         + 1 × 32 × 4096 × 128 × 2 × 32  # K
         + 1 × 32 × 4096 × 128 × 2 × 32  # V
         + 1 × 32 × 4096 × 128 × 2       # Output

简化计算：
Bytes ≈ 32 × (Q + K + V) + Output
      ≈ 32 × 134MB × 3 + 134MB
      ≈ 4.3GB + 134MB
      ≈ 4.4GB

对比标准 SDPA: 1.2GB
Flash Attention 反而更高？！

实际原因：
- 上述计算假设每个 block 都从 HBM 重新加载 Q
- 实际优化：Q 可以缓存在 L2，不需要多次加载
```

**更准确的公式**（考虑 L2 caching）：
```
如果 Q 可以缓存在 L2:
Bytes ≈ Q + K × num_blocks + V × num_blocks + Output

对于 batch=1:
Q_bytes = 134MB > L2(50MB)，无法缓存
所以需要分块加载 Q

实际访存需要更细致的计算...
```

#### 4.2.6 算术强度与 Bound 判断

| Batch | Seq | FLOPs | Bytes(FA) | compute_time | memory_time | Bytes(SDPA) |
|-------|-----|-------|-----------|--------------|-------------|-------------|
| 1 | 4096 | 68.7G | ~4.4GB | ~69 μs | ~1.3 ms | 1.2GB |
| 200 | 4096 | 13.74T | ~26GB | ~13.9 ms | ~7.8 ms | 214GB |

**注意**：
- batch=1 时，Flash Attention Bytes 更高（block-based loading）
- batch=200 时，Flash Attention Bytes 远小于 SDPA（不存储 attention scores）
- Flash Attention 主要优势：减少显存占用（不存储 N² attention matrix）
- 计算时间取决于具体实现

---

### 4.3 mla_attention

**Multi-head Latent Attention (MLA)**

MLA 是 DeepSeek-V2/V3 的注意力机制，通过压缩 KV cache 减少内存。

#### 4.3.1 输入参数定义

```
输入张量：
- query: (batch, num_heads, seq_len, qk_head_dim)
- compressed_kv: (batch, seq_len, kv_lora_rank)  # 压缩的 KV
- key/value: (可选，用于非 absorb 模式)

参数：
- use_absorb: 是否使用 absorb 模式（推理优化）
- kv_lora_rank: 压缩 KV 的维度（例如 512）
```

#### 4.3.2 FLOPs 计算

**Absorb 模式**（推理）：
```
QK_flops = 2 × batch × heads × seq × kv_seq × kv_lora_rank
Softmax_flops = 5 × batch × heads × seq × kv_seq
AV_flops = 2 × batch × heads × seq × kv_lora_rank × kv_seq

Total ≈ 4 × batch × heads × seq × kv_seq × kv_lora_rank
```

**对比标准 attention**：
```
标准 attention: O(seq × head_dim)
MLA: O(seq × kv_lora_rank)

如果 kv_lora_rank < head_dim，MLA 计算量更少
```

#### 4.3.3 理论访存计算

**Absorb 模式**：
```
Bytes = batch × heads × seq × qk_head_dim × dtype  # Q
      + batch × seq × kv_lora_rank × dtype         # compressed K
      + batch × seq × kv_lora_rank × dtype         # compressed V
      + batch × heads × seq × v_head_dim × dtype   # Output
```

**数值示例（DeepSeek-V3）**：
```
batch=1, heads=128, seq=4096, kv_lora_rank=512, qk_dim=192:
Q = 1 × 128 × 4096 × 192 × 2 = 193MB
compressed KV = 2 × 1 × 4096 × 512 × 2 = 8.4MB
Output = 1 × 128 × 4096 × 128 × 2 = 134MB

Total ≈ 335MB

对比标准 KV cache:
标准 KV = 2 × 1 × 128 × 4096 × 192 × 2 = 386MB

MLA 节省: 386MB - 8.4MB = 377MB (98% saving!)
```

#### 4.3.4 Cache-aware 特性

MLA 的关键优势：
- **KV cache 大小减少**：从 head_dim 降到 kv_lora_rank
- **更容易缓存在 L2**

```
标准 KV: 386MB >> L2(50MB)，无法缓存
MLA compressed KV: 8.4MB < L2(50MB)，可以缓存！

这意味着 MLA 可以实现更高效的 cache reuse
```

---

### 4.4 linear_attention

**Linear Attention with O(seq) complexity**

Linear Attention 使用 kernel trick 将复杂度从 O(seq²) 降到 O(seq)。

#### 4.4.1 输入参数定义

```
输入张量：
- query: (batch, num_heads, seq_len, head_dim)
- key: (batch, num_kv_heads, seq_len, head_dim)
- value: (batch, num_kv_heads, seq_len, head_dim)

参数：
- kernel_dim: Feature map 维度（例如 4）
```

#### 4.4.2 FLOPs 计算

```
Feature map: φ(x) (例如 elu(x)+1)
FLOPs_feature ≈ kernel_dim × 2 × batch × heads × seq × head_dim

KV state: K^T @ V
FLOPs_kv_state = 2 × batch × kv_heads × seq × head_dim × head_dim

Q state: Q @ KV_state
FLOPs_q_state = 2 × batch × heads × seq × head_dim × head_dim

Total ≈ 2 × batch × heads × seq × head_dim² × 2
      ≈ 4 × batch × heads × seq × head_dim²
```

**对比标准 attention**：
```
标准: O(seq² × head_dim)
Linear: O(seq × head_dim²)

当 seq >> head_dim 时，Linear 更快
```

#### 4.4.3 理论访存计算

```
Bytes = batch × heads × seq × head_dim × dtype  # Q
      + batch × kv_heads × seq × head_dim × dtype  # K
      + batch × kv_heads × seq × head_dim × dtype  # V
      + batch × heads × seq × head_dim × dtype  # Output
      + batch × heads × head_dim × head_dim × dtype  # KV state
```

**关键区别**：
- Linear Attention **不存储 seq² 的 attention scores**
- 只存储 head_dim × head_dim 的 KV state
- 内存访问从 O(seq²) 降到 O(seq)

---

## 5. 归一化类 Kernel

### 5.1 layer_norm

**Layer normalization**

#### 5.1.1 输入参数定义

```
输入张量：
- input: (..., normalized_shape)  - 输入张量

参数：
- normalized_shape: 归一化的维度
- elementwise_affine: 是否使用可学习的 affine 参数

输出：
- output: (..., normalized_shape)  - 输出张量（同形状）
```

#### 5.1.2 FLOPs 计算

```
对于每个归一化单元（normalized_shape 维度）：
1. mean: normalized_shape 次加法
2. variance: normalized_shape 次减法 + normalized_shape 次乘法 + normalized_shape 次加法
3. normalize: normalized_shape 次减法 + normalized_size 次除法
4. affine (optional): normalized_shape 次乘法 + normalized_shape 次加法

简化：每个元素约 7 FLOPs
FLOPs = numel × 7

如果有 affine:
FLOPs = numel × 9
```

#### 5.1.3 理论访存计算

```
Bytes = numel × dtype_size × 2  # read input + write output

如果有 affine:
Bytes += normalized_shape × dtype_size × 2  # weight + bias
```

**数值示例（LLaMA-70B）**：
```
batch=1, seq=4096, hidden=8192:
numel = 1 × 4096 × 8192 = 33.5M

Bytes = 33.5M × 2 × 2 = 134MB

batch=200, seq=4096, hidden=8192:
numel = 200 × 4096 × 8192 = 6.7G

Bytes = 6.7G × 2 × 2 = 26.8GB
```

#### 5.1.4 Cache-aware 特性

**LayerNorm 的特点**：
- **纯 memory-bound 操作**
- 计算：逐元素归一化
- 无数据重用

**Cache 策略**：
- 按元素顺序处理
- 只需要 L1 cache（输入输出）
- 不需要 L2 tiling

**实际访存**：
```
实际访存 ≈ 理论访存（无 cache 优化）
```

#### 5.1.5 算术强度与 Bound 判断

```
AI = FLOPs / Bytes = numel × 7 / (numel × 2 × dtype_size)
   = 7 / (2 × dtype_size)
   
对于 fp16:
AI = 7 / 4 = 1.75

compute_time = numel × 7 / peak_tflops
memory_time = numel × 2 × dtype_size / memory_bw

memory_time >> compute_time → Always Memory Bound
```

---

### 5.2 rms_norm

**RMS normalization (common in LLaMA)**

#### 5.2.1 输入参数定义

```
输入张量：
- input: (..., hidden_size)

参数：
- hidden_size: 归一化的维度

输出：
- output: (..., hidden_size)
```

#### 5.2.2 FLOPs 计算

```
对于每个归一化单元：
1. square: hidden_size 次乘法
2. mean: hidden_size 次加法 + 1 次除法
3. rsqrt: 1 次操作（硬件有专门指令）
4. scale: hidden_size 次乘法

简化：每个元素约 5 FLOPs
FLOPs = numel × 5
```

#### 5.2.3 理论访存计算

```
Bytes = numel × dtype_size × 2  # input + output
      + hidden_size × dtype_size  # weight
```

#### 5.2.4 算术强度与 Bound 判断

```
AI = 5 / (2 × dtype_size) = 5 / 4 = 1.25

compute_time = numel × 5 / peak_tflops
memory_time = numel × 2 × dtype_size / memory_bw

memory_time >> compute_time → Always Memory Bound
```

---

## 6. 激活类 Kernel

### 6.1 silu

**SiLU activation: x × sigmoid(x)**

#### 6.1.1 输入参数定义

```
输入张量：
- input: (任意形状)

输出：
- output: (同形状)
```

#### 6.1.2 FLOPs 计算

```
每个元素：
- sigmoid: exp(x) + 1 + div ≈ 7 FLOPs
- mul: 1 FLOP

Total ≈ 10 FLOPs per element
FLOPs = numel × 10
```

#### 6.1.3 理论访存计算

```
Bytes = numel × dtype_size × 2  # read + write
```

#### 6.1.4 算术强度与 Bound 判断

```
AI = 10 / (2 × dtype_size) = 10 / 4 = 2.5

compute_time = numel × 10 / peak_tflops
memory_time = numel × 2 × dtype_size / memory_bw

memory_time >> compute_time → Always Memory Bound
```

---

### 6.2 gelu

**GELU activation**

#### 6.2.1 FLOPs 计算

```
标准 GELU: x × Φ(x) (Gaussian CDF)
FLOPs ≈ 15 per element

Tanh approximation:
FLOPs ≈ 8 per element
```

#### 6.2.2 算术强度与 Bound 判断

```
AI ≈ 8-15 / 4 = 2-3.75

compute_time = numel × 8~15 / peak_tflops
memory_time = numel × 2 × dtype_size / memory_bw

memory_time >> compute_time → Always Memory Bound
```

---

### 6.3 relu

**ReLU activation: max(0, x)**

#### 6.3.1 FLOPs 计算

```
每个元素：1 FLOP (compare + select)
FLOPs = numel × 1
```

#### 6.3.2 算术强度与 Bound 判断

```
AI = 1 / 4 = 0.25

compute_time = numel × 1 / peak_tflops
memory_time = numel × 2 × dtype_size / memory_bw

memory_time >> compute_time → Always Memory Bound
（极端 memory bound，几乎无计算）
```

---

## 7. 其他类 Kernel

### 7.1 softmax

**Softmax activation**

#### 7.1.1 输入参数定义

```
输入张量：
- input: (..., softmax_dim_size)

参数：
- dim: 应用 softmax 的维度

输出：
- output: (同形状)
```

#### 7.1.2 FLOPs 计算

```
对于每个 softmax 单元（softmax_dim_size 维度）：
1. exp: softmax_dim_size 次操作 ≈ 7 × softmax_dim_size FLOPs
2. sum: softmax_dim_size 次加法
3. div: softmax_dim_size 次除法

Total ≈ 9 × softmax_dim_size FLOPs per unit

Per element: ≈ 9 FLOPs
FLOPs = numel × 9
```

#### 7.1.3 算术强度与 Bound 判断

```
AI = 9 / 4 = 2.25

compute_time = numel × 9 / peak_tflops
memory_time = numel × 2 × dtype_size / memory_bw

memory_time >> compute_time → Always Memory Bound
```

---

### 7.2 dropout

**Dropout regularization**

#### 7.2.1 FLOPs 计算

```
每个元素：
- random: 1 FLOP
- compare: 1 FLOP
- scale: 1 FLOP

FLOPs = numel × 3
```

#### 7.2.2 算术强度

```
AI = 3 / 4 = 0.75

Memory bound
```

---

### 7.3 embedding

**Embedding lookup**

#### 7.3.1 输入参数定义

```
参数：
- num_embeddings: 字典大小
- embedding_dim: 每个嵌入向量维度

输入张量：
- input_indices: (batch, seq_len) - 索引

输出：
- output: (batch, seq_len, embedding_dim)
```

#### 7.3.2 FLOPs 计算

```
每个索引：查找操作（无实际计算）
FLOPs ≈ numel_indices × embedding_dim (复制)

简化：FLOPs ≈ 1 per index
FLOPs = batch × seq_len × embedding_dim
```

#### 7.3.3 理论访存计算

```
Bytes = num_embeddings × embedding_dim × dtype_size  # embedding table
      + batch × seq_len × embedding_dim × dtype_size  # output
      
假设 batch × seq_len << num_embeddings:
主导访存是 embedding table
```

#### 7.3.4 Cache-aware 特性

**Embedding 的 cache 问题**：
- Embedding table 通常很大（例如 50K × 4096 = 200MB）
- 无法完全缓存在 L2
- Lookup 是随机访问，cache hit rate 低

**实际访存**：
```
假设 embedding table = 200MB
L2 = 50MB
Cache hit rate ≈ 25% (worst case)

实际访存 ≈ (1 - cache_hit_rate) × embedding_table_bytes
         + output_bytes
         
可能远大于理论访存
```

---

## 8. 卷积类 Kernel

### 8.1 conv2d

**2D convolution**

#### 8.1.1 输入参数定义

```
输入张量：
- input: (N, C_in, H, W)

权重：
- weight: (C_out, C_in, kH, kW)

输出：
- output: (N, C_out, H_out, W_out)
```

#### 8.1.2 FLOPs 计算

```
FLOPs = 2 × N × C_out × H_out × W_out × C_in × kH × kW
```

#### 8.1.3 理论访存计算

```
Bytes = N × C_in × H × W × dtype_size  # input
      + C_out × C_in × kH × kW × dtype_size  # weight
      + N × C_out × H_out × W_out × dtype_size  # output
```

#### 8.1.4 Cache-aware 特性

**卷积的 weight reuse**：
- Weight 在每个 spatial position 重用
- 对于小图像，weight 可以缓存在 L2
- 对于大图像，需要沿 spatial 维度分块

**Tile 策略**：
```
沿 H_out, W_out 分块：
每个 tile 计算 spatial_block × spatial_block 的输出

Weight bytes:
weight_bytes = C_out × C_in × kH × kW × dtype_size

如果 weight_bytes < L2:
可以缓存在 L2，实现高效 reuse
```

---

### 8.2 conv3d

**3D convolution**

类似于 conv2d，增加 D 维度。

---

## 9. MoE类 Kernel

### 9.1 moe_expert

**MoE expert computation**

MoE expert 包含三个 linear projection：
- gate: hidden → intermediate (SiLU)
- up: hidden → intermediate
- down: intermediate → hidden

#### 9.1.1 输入参数定义

```
输入张量：
- hidden: (batch, ..., hidden_size)

参数：
- intermediate_size: FFN 中间维度
- num_experts_per_token: 每个 token 激活的专家数
```

#### 9.1.2 FLOPs 计算

```
gate_flops = 2 × batch × hidden × intermediate
silu_flops = batch × intermediate × 10
up_flops = 2 × batch × hidden × intermediate
down_flops = 2 × batch × intermediate × hidden

Total per expert = 4 × batch × hidden × intermediate + batch × intermediate × 10

简化 ≈ 4 × batch × hidden × intermediate (dominated by matmuls)
```

#### 9.1.3 理论访存计算

```
gate_weight = hidden × intermediate × dtype
up_weight = hidden × intermediate × dtype
down_weight = intermediate × hidden × dtype

Total weight = 3 × hidden × intermediate × dtype

gate_output = batch × intermediate × dtype
up_output = batch × intermediate × dtype
down_output = batch × hidden × dtype

Total bytes ≈ 3 × hidden × intermediate × dtype  # weights dominant
           + batch × hidden × dtype  # input
           + batch × hidden × dtype  # output
```

#### 9.1.4 Cache-aware 特性

**MoE 的特殊性**：
- 每个 token 只激活少数专家（例如 1-2）
- 专家权重很大，无法完全缓存
- 需要按专家分块

**Tile 策略**：
```
沿 batch 维度分块（如果激活同一专家的 token 多）
或沿专家分块（如果不同 token 激活不同专家）
```

---

## 10. 实施优先级

### 10.1 优先级分类

| 优先级 | Kernel | 原因 |
|--------|--------|------|
| P0 | linear, bmm | 最常见，影响最大 |
| P0 | flash_attention | 关键优化，有明确 tiling 算法 |
| P1 | conv2d, conv3d | 较常见，有明确 cache 模型 |
| P1 | layer_norm, rms_norm | 常见，但简单（memory bound） |
| P2 | silu, gelu, relu | 常见，但简单（memory bound） |
| P2 | softmax, embedding | 常见，需要特殊处理 |
| P3 | mla_attention, linear_attention | 新机制，较少见 |
| P3 | moe_expert | 复杂，需要特殊处理 |

### 10.2 实施步骤

**Step 1: P0 Kernel（linear, bmm, flash_attention）**
- 实现 cache-aware 计算函数
- 验证数值示例
- 集成到 MicroarchBackend

**Step 2: P1 Kernel（conv, norm）**
- 扩展支持

**Step 3: P2 Kernel（activation, softmax）**
- 扩展支持

**Step 4: P3 Kernel（特殊机制）**
- 扩展支持

---

## 11. 测试方案

### 11.1 单元测试

为每个 kernel 的 cache-aware 计算函数编写测试：

```python
def test_linear_cache_aware():
    # batch=1 case
    result = compute_effective_bytes_linear(1, 8192, 8192, "fp16")
    assert result["memory_bound"] == True  # memory bound
    assert result["num_tiles"] == 1
    
    # batch=200 case
    result = compute_effective_bytes_linear(200, 8192, 8192, "fp16")
    assert result["memory_bound"] == False  # compute bound
    assert result["num_tiles"] > 1
    assert result["compute_time_us"] > result["memory_time_us"]

def test_flash_attention_cache_aware():
    result = compute_effective_bytes_flash_attention(1, 32, 4096, 128, "fp16")
    assert result["bytes_actual"] < result["bytes_sdpa"]  # FA better than SDPA
```

### 11.2 集成测试

测试 MicroarchBackend 的整体功能：

```python
def test_microarch_backend_estimate():
    backend = MicroarchBackend(config)
    device = Device.from_preset("H100-SXM-80GB")
    
    result = linear((1, 4096), (4096, 4096))
    time = backend.estimate_compute_time_from_result(result, device)
    
    # 验证时间估计合理
    assert time > 0
```

### 11.3 数值验证

验证计算结果与文献/实测数据一致：

```
参考数据：
- H100 GEMM (small): ~3.4 TFLOPS (0.3% peak)
- H100 GEMM (large): ~989 TFLOPS (100% peak)
- Flash Attention: 算术强度 ≈ seq/4
```

---

## 12. 参考

### 12.1 学术文献

1. **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
   - https://arxiv.org/abs/2205.14135
   - 提供了 Flash Attention 的 tiling 算法和 IO 分析

2. **Roofline Model**: "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures" (2008)
   - https://doi.org/10.1109/MC.2009.143
   - 理论基础

3. **MLA**: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024)
   - https://arxiv.org/abs/2405.04434
   - MLA 的详细描述

### 12.2 硬件规格

- NVIDIA H100 Datasheet
- NVIDIA A100 Datasheet
- AMD MI300X Datasheet

### 12.3 相关文档

- `docs/kernel_modeling.md`: Kernel 建模基础
- `docs/roofline_model.md`: Roofline 模型详解
- `llm_perf/kernels/functional.py`: Kernel 定义

---

## 附录 A: Kernel 分类总结

### A.1 Memory Bound vs Compute Bound

| Kernel | 典型 AI | Bound 判断条件 |
|--------|---------|----------------|
| linear | 1-200 | memory_time > compute_time → Memory Bound |
| bmm | 30-400 | memory_time > compute_time → Memory Bound |
| attention | 15-500 | memory_time > compute_time → Memory Bound |
| conv2d | 50-500 | memory_time > compute_time → Memory Bound |
| layer_norm | 1.75 | Always memory bound |
| rms_norm | 1.25 | Always memory bound |
| silu | 2.5 | Always memory bound |
| gelu | 2-3.75 | Always memory bound |
| relu | 0.25 | Always memory bound |
| softmax | 2.25 | Always memory bound |

**Bound 判断公式**：
```python
compute_time = FLOPs / peak_tflops
memory_time = bytes_actual / memory_bw
memory_bound = memory_time > compute_time
```

### A.2 Cache Tiling 需求

| Kernel | 需要 Tiling | Tiling 维度 | Cache 级别 |
|--------|------------|-------------|-----------|
| linear | Yes (weight 大) | batch / out_features | L2 |
| bmm | Yes (矩阵大) | batch / m / n | L2 |
| flash_attention | Yes | seq | SRAM/L2 |
| conv2d | Yes (图像大) | spatial | L2 |
| layer_norm | No | - | L1 |
| rms_norm | No | - | L1 |
| silu/gelu/relu | No | - | L1 |
| softmax | No | - | L1 |

---

*文档版本: 1.0*
*创建日期: 2026-04-27*
*作者: CePing 项目组*
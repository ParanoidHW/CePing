---
name: kernel
description: Kernel 开发规范（API + cache-aware + 3 backend）
---

## 概述

本 skill 定义了新增和修改 kernel 的完整开发规范，确保 kernel 层修改不影响上层，并正确实现 cache-aware 性能评估。

---

## 1. 开发过程

### 1.1 完整开发流程

```
阶段1：设计
  → 分析 kernel 需求合理性
  → 参考外部实践（paper、官方实现）
  → 设计 cache-aware 计算模型
  
阶段2：实现
  → 实现 kernel 函数（遵循 API 规范）
  → 实现 cache-aware 计算函数
  
阶段3：测试
  → 编写单元测试
  → 验证数值示例
  → 运行存量测试
  
阶段4：提交
  → 通过 ruff 检查
  → 小步提交（每个 kernel 一个 commit）
```

### 1.2 步骤清单

#### Step 1: 需求分析
- [ ] 确认 kernel 类型（矩阵类/注意力类/归一化类/激活类/其他）
- [ ] 参考 torch 2.10 API 设计
- [ ] 确认 FLOPs 计算公式
- [ ] 确认理论访存计算公式

#### Step 2: Cache-aware 模型设计
- [ ] 分析是否需要 L2 cache tiling
- [ ] 设计 tiling 分块策略
- [ ] 计算实际访存公式
- [ ] 设计 bound 判断方式

#### Step 3: 实现 kernel 函数
- [ ] 创建 `llm_perf/kernels/functional.py` 函数
- [ ] 遵循函数签名规范
- [ ] 返回 KernelResult 数据结构

#### Step 4: 实现 cache-aware 函数
- [ ] 创建 `llm_perf/kernels/cache_aware.py` 函数
- [ ] 返回 cache-aware metrics dict

#### Step 5: 测试验证
- [ ] 编写单元测试（`tests/test_kernels.py`）
- [ ] 验证数值示例与设计文档一致
- [ ] 运行存量测试确保不影响现有模型

#### Step 6: 提交
- [ ] 通过 ruff 检查
- [ ] 提交 commit，格式：`feat(kernels): add <kernel_name> kernel`
- [ ] 推送到远程仓库

### 1.3 分阶段实施建议

#### P0 优先级（立即实施）
- `linear`, `bmm`: 最常见，影响最大
- `flash_attention`: 关键优化，有明确 tiling 算法

#### P1 优先级（尽快实施）
- `conv2d`, `conv3d`: 较常见，有明确 cache 模型
- `layer_norm`, `rms_norm`: 常见，但简单（memory bound）

#### P2 优先级（计划实施）
- `silu`, `gelu`, `relu`: 常见，但简单（memory bound）
- `softmax`, `embedding`: 常见，需要特殊处理

#### P3 优先级（可选实施）
- `mla_attention`, `linear_attention`: 新机制，较少见
- `moe_expert`: 复杂，需要特殊处理

---

## 2. Kernel API 开发规范

### 2.1 Kernel 注册方式

kernel 的注册方式需尽量对齐 torch 原生对应接口，入参数量和形式保持和 torch 接口一致。参考的 torch 版本为 **2.10**。

### 2.2 强制使用 Kernel API

**所有模型的 `activation_bytes`/`params`/`FLOPs`/内存开销等特征，必须通过 kernel API 获取**，禁止 kernel 外手动计算。通信开销也应该通过 kernel 表达来独立处理通信量和不同拓扑下的开销。

#### 正确做法

```python
from ..kernels import linear, rms_norm
from ..kernels.utils import kernel_result_to_layer

result = linear(
    input=(seq_len, hidden_size),
    weight=(out_dim, hidden_size),
    bias=None,
    dtype=cfg.dtype
)

layers.append(kernel_result_to_layer(
    name="layer_name",
    result=result,
))
```

#### 错误做法

```python
layers.append(LayerConfig(
    name="layer_name",
    input_shape=(1, seq_len, hidden_size),
    output_shape=(1, seq_len, out_dim),
    params_count=hidden_size * out_dim,
    flops=result.flops,
    activation_bytes=seq_len * out_dim * dtype_size,
))
```

### 2.3 计算类 Kernel 函数签名规范

```python
def kernel_name(
    input: Tuple[int, ...],
    weight: Optional[Tuple[int, ...]] = None,
    bias: Optional[Tuple[int, ...]] = None,
    dtype: str = "fp16"
) -> KernelResult:
    """Kernel 文档字符串.
    
    Args:
        input: 输入形状
        weight: 权重形状（如果有）
        bias: 偏置形状（如果有）
        dtype: 数据类型
    
    Returns:
        KernelResult 包含 output, flops, bytes_accessed
    """
    pass
```

---

## 3. 注意事项

### 3.1 分层解耦原则

**核心原则**：kernel 层修改不影响上层（model 层、analyzer 层）

```
层次结构：
├── kernel 层：提供 FLOPs、bytes、cache-aware 计算
├── model 层：调用 kernel API 获取层特征
├── analyzer 层：使用 kernel 结果进行性能分析

约束：
- kernel 层：只负责计算，不依赖 model/analyzer
- model 层：只调用 kernel API，不手动计算
- analyzer 层：只使用 kernel 结果，不重新计算
```

**违反解耦的例子**：

```python
activation_bytes = seq_len * hidden_size * dtype_size

result = linear(input=(seq_len, hidden_size), ...)
activation_bytes = result.activation_bytes
```

### 3.2 三 Backend 同时考虑

实施 kernel 时，需要同时考虑 3 个 backend 的计算结果：

| Backend | 文件 | 说明 |
|---------|------|------|
| 理论 Roofline | `theory.py` | 理论性能模型，不考虑 cache |
| Microarch | `microarch.py` | 考虑 L2 cache tiling 的实际访存 |
| 实测 Profiling | `profiling.py` | 基于 profiler 的实测数据（如有） |

**实施要求**：
- 每个 kernel 需在 3 个 backend 中都有对应实现
- 测试时需验证 3 个 backend 的趋势一致性：
  - 大规格 kernel 耗时 > 小规格 kernel 耗时
  - 按某一维度（batch、seq、hidden）近似线性或略超过线性
  - 不同 backend 的收益来源不同，但趋势相同
- 分析报告需同时展示 3 种计算结果

### 3.3 Cache-aware 计算的正确模型

#### 核心：矩阵乘法的 Tiling 访存分析

**策略1：外层激活 tile 步进，内层权重 tile 遍历**

```python
for A_tile_i in [0, num_tiles_M):      # 外层：激活 tile 步进
    加载 A_tile_i 到 cache（暂存）      # A 只加载一次
    for B_tile_j in [0, num_tiles_N):  # 内层：权重 tile 遍历
        加载 B_tile_j                   # B 每次都加载
        C_tile[i,j] = A[i] × B[j]
```

**关键特性**：
- 激活 A 只加载一次（cache 暂存）
- 权重 B 重复加载 num_tiles 次

**访存公式**：

```
actual_bytes = batch_input_bytes               # A（加载一次）
             + weight_bytes × num_tiles        # B（重复加载）
             + batch_output_bytes              # C（写入一次）
```

#### Linear 层的应用

对于 `y = x @ W^T`：
- x（激活）：加载一次，暂存于 cache
- W（权重）：重复加载 num_tiles 次
- y（输出）：写入一次

**公式**：

```
num_tiles = ceil(batch / tile_M)
tile_M = floor(L2_capacity / batch_input_bytes)

权重重复加载开销 = (num_tiles - 1) × weight_bytes
```

### 3.4 时间对比方式判断 Memory/Compute Bound

**正确方式**：使用时间对比，而非 ridge point

```python
compute_time = FLOPs / peak_tflops
memory_time = bytes_actual / memory_bw

if memory_time > compute_time:
    bound = "memory"
else:
    bound = "compute"
```

**数值示例（H100, fp16）**：

```
batch=1:
  FLOPs = 134.2M
  Bytes = 134.2MB
  compute_time = 134.2M / 989 TFLOPS = 17.4 μs
  memory_time = 134.2MB / 3.35 TB/s = 21.9 μs
  memory_time > compute_time → Memory Bound

batch=200:
  FLOPs = 26.84G
  Bytes = 1885.2MB
  compute_time = 26.84G / 989 TFLOPS = 3474 μs
  memory_time = 1885.2MB / 3.35 TB/s = 1978 μs
  compute_time > memory_time → Compute Bound
```

### 3.5 避免常见错误

#### 错误1：混淆理论访存和实际访存

```python
bytes_theory = batch_input + weight + batch_output

bytes_actual = batch_input + weight × num_tiles + batch_output
```

#### 错误2：忽略权重重复加载开销

```python
bytes_actual = batch_input + weight + batch_output

weight_reload_overhead = (num_tiles - 1) × weight_bytes
bytes_actual = bytes_theory + weight_reload_overhead
```

#### 错误3：使用 ridge point 判断 bound

```python
ridge_point = peak_tflops / memory_bw
ai_actual = FLOPs / bytes_actual
bound = "memory" if ai_actual < ridge_point else "compute"

compute_time = FLOPs / peak_tflops
memory_time = bytes_actual / memory_bw
bound = "memory" if memory_time > compute_time else "compute"
```

#### 错误4：Flash Attention 的 hidden 访存

```python
bytes_sdpa = Q + K + V + Output + attention_scores
bytes_flash = Q + K + V + Output
```

---

## 4. 推荐开发实践样例

### 4.1 Linear Kernel 完整实现

#### kernel 函数（`llm_perf/kernels/functional.py`）

```python
def linear(
    input: Tuple[int, ...],
    weight: Tuple[int, int],
    bias: Optional[Tuple[int]] = None,
    dtype: str = "fp16",
) -> KernelResult:
    """Linear transformation kernel.
    
    y = x @ W^T + b
    
    Args:
        input: Input shape (..., in_features)
        weight: Weight shape (out_features, in_features)
        bias: Bias shape (out_features,) or None
        dtype: Data type
    
    Returns:
        KernelResult with output, flops, bytes_accessed
    """
    dtype_size = DTYPE_SIZES.get(dtype, 2)
    
    batch_size = 1
    for dim in input[:-1]:
        batch_size *= dim
    in_features = input[-1]
    out_features = weight[0]
    
    output_shape = list(input[:-1]) + [out_features]
    
    flops = 2 * batch_size * in_features * out_features
    if bias is not None:
        flops += batch_size * out_features
    
    bytes_accessed = batch_size * in_features * dtype_size
    bytes_accessed += in_features * out_features * dtype_size
    bytes_accessed += batch_size * out_features * dtype_size
    if bias is not None:
        bytes_accessed += out_features * dtype_size
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        params_count=in_features * out_features + (out_features if bias else 0),
        activation_bytes=batch_size * out_features * dtype_size,
    )
```

#### cache-aware 函数（`llm_perf/kernels/cache_aware.py`）

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
        dtype: Data type
        l2_capacity_bytes: L2 cache capacity
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

#### 单元测试（`tests/test_kernels.py`）

```python
class TestLinearKernel:
    def test_basic_linear(self):
        """Test basic linear kernel."""
        result = linear(
            input=(4096, 8192),
            weight=(4096, 8192),
            dtype="fp16"
        )
        
        assert result.output == [4096, 4096]
        assert result.flops == 2 * 4096 * 8192 * 4096
        
    def test_cache_aware_batch1(self):
        """Test cache-aware with batch=1."""
        result = compute_effective_bytes_linear(
            batch_size=1,
            in_features=8192,
            out_features=8192,
            dtype="fp16"
        )
        
        assert result["memory_bound"] == True
        assert result["num_tiles"] == 1
        assert result["weight_reload_overhead"] == 0
        
    def test_cache_aware_batch200(self):
        """Test cache-aware with batch=200."""
        result = compute_effective_bytes_linear(
            batch_size=200,
            in_features=8192,
            out_features=8192,
            dtype="fp16"
        )
        
        assert result["memory_bound"] == False
        assert result["num_tiles"] > 1
        assert result["weight_reload_overhead"] > 0
        assert result["compute_time_us"] > result["memory_time_us"]
```

### 4.2 Flash Attention 实现示例

#### kernel 函数

```python
def flash_attention(
    query: Tuple[int, int, int, int],
    key: Tuple[int, int, int, int],
    value: Tuple[int, int, int, int],
    dtype: str = "fp16",
    is_causal: bool = False,
    block_size: int = 128,
) -> KernelResult:
    """Flash Attention kernel.
    
    Args:
        query: (batch, num_heads, seq_len, head_dim)
        key: (batch, kv_num_heads, kv_seq_len, head_dim)
        value: (batch, kv_num_heads, kv_seq_len, head_dim)
        dtype: Data type
        is_causal: Whether causal mask
        block_size: Tile block size
    
    Returns:
        KernelResult
    """
    dtype_size = DTYPE_SIZES.get(dtype, 2)
    
    batch, num_heads, seq_len, head_dim = query
    kv_batch, kv_heads, kv_seq_len, kv_head_dim = key
    
    output_shape = [batch, num_heads, seq_len, head_dim]
    
    flops = 4 * batch * num_heads * seq_len * kv_seq_len * head_dim
    
    bytes_accessed = (
        batch * num_heads * seq_len * head_dim * dtype_size
        + kv_batch * kv_heads * kv_seq_len * kv_head_dim * dtype_size
        + kv_batch * kv_heads * kv_seq_len * kv_head_dim * dtype_size
        + batch * num_heads * seq_len * head_dim * dtype_size
    )
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        activation_bytes=batch * num_heads * seq_len * head_dim * dtype_size,
    )
```

#### cache-aware 函数

```python
def compute_effective_bytes_flash_attention(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    kv_heads: int = None,
    kv_seq_len: int = None,
    dtype: str = "fp16",
    is_causal: bool = True,
    block_size: int = 128,
    sram_capacity_kb: int = 228,
) -> Dict[str, float]:
    """计算 Flash Attention 的 cache-aware 实际访存.
    
    Flash Attention 通过 tiling 避免 attention scores 的 HBM 存储。
    
    Args:
        batch: Batch size
        num_heads: Number of query heads
        seq_len: Query sequence length
        head_dim: Head dimension
        kv_heads: Number of KV heads (GQA)
        kv_seq_len: KV sequence length
        dtype: Data type
        is_causal: Whether causal mask
        block_size: Block size for tiling
        sram_capacity_kb: SRAM capacity in KB
    
    Returns:
        Dict with cache-aware metrics
    """
    if kv_heads is None:
        kv_heads = num_heads
    if kv_seq_len is None:
        kv_seq_len = seq_len
    
    dtype_size = DTYPE_SIZES.get(dtype, 2)
    
    q_bytes = batch * num_heads * seq_len * head_dim * dtype_size
    k_bytes = batch * kv_heads * kv_seq_len * head_dim * dtype_size
    v_bytes = batch * kv_heads * kv_seq_len * head_dim * dtype_size
    output_bytes = batch * num_heads * seq_len * head_dim * dtype_size
    
    bytes_theory = q_bytes + k_bytes + v_bytes + output_bytes
    
    flops = 4 * batch * num_heads * seq_len * kv_seq_len * head_dim
    
    num_blocks = ceil(seq_len / block_size)
    
    bytes_actual = (
        q_bytes
        + k_bytes * num_blocks
        + v_bytes * num_blocks
        + output_bytes
    )
    
    sdpa_attention_scores = batch * num_heads * seq_len * kv_seq_len * dtype_size
    bytes_sdpa = bytes_theory + sdpa_attention_scores
    
    peak_tflops = 989.0
    memory_bw_gbps = 3350.0
    
    compute_time_us = (flops / (peak_tflops * 1e12)) * 1e6
    memory_time_flash_us = (bytes_actual / (memory_bw_gbps * 1e9)) * 1e6
    memory_time_sdpa_us = (bytes_sdpa / (memory_bw_gbps * 1e9)) * 1e6
    
    return {
        "bytes_theory": bytes_theory,
        "bytes_actual": bytes_actual,
        "bytes_sdpa": bytes_sdpa,
        "flops": flops,
        "num_blocks": num_blocks,
        "compute_time_us": compute_time_us,
        "memory_time_flash_us": memory_time_flash_us,
        "memory_time_sdpa_us": memory_time_sdpa_us,
        "attention_scores_bytes": sdpa_attention_scores,
    }
```

### 4.3 伪代码模板

#### 通用 kernel 函数模板

```python
def kernel_name(
    input: Tuple[int, ...],
    weight: Optional[Tuple[int, ...]] = None,
    bias: Optional[Tuple[int, ...]] = None,
    dtype: str = "fp16",
    **kwargs,
) -> KernelResult:
    """Kernel description.
    
    Args:
        input: Input shape
        weight: Weight shape (optional)
        bias: Bias shape (optional)
        dtype: Data type
        **kwargs: Additional kernel-specific parameters
    
    Returns:
        KernelResult containing:
        - output: Output shape
        - flops: Compute operations
        - bytes_accessed: Memory access
        - params_count: Parameter count
        - activation_bytes: Activation memory
    """
    dtype_size = DTYPE_SIZES.get(dtype, 2)
    
    output_shape = [...]
    flops = ...
    bytes_accessed = ...
    params_count = ...
    activation_bytes = ...
    
    return KernelResult(
        output=output_shape,
        flops=flops,
        bytes_accessed=bytes_accessed,
        params_count=params_count,
        activation_bytes=activation_bytes,
    )
```

#### 通用 cache-aware 函数模板

```python
def compute_effective_bytes_kernel(
    batch_size: int,
    feature_dims: List[int],
    dtype: str = "fp16",
    l2_capacity_bytes: int = 50 * 1024 * 1024,
    **kwargs,
) -> Dict[str, float]:
    """计算 kernel 的 cache-aware 实际访存.
    
    Args:
        batch_size: Batch size
        feature_dims: Feature dimensions
        dtype: Data type
        l2_capacity_bytes: L2 cache capacity
        **kwargs: Additional kernel-specific parameters
    
    Returns:
        Dict containing:
        - bytes_theory: 理论访存
        - bytes_actual: 实际访存（考虑 tiling）
        - num_tiles: 分块数量
        - flops: 计算量
        - ai_theory: 理论算术强度
        - ai_actual: 实际算术强度
        - compute_time_us: 计算时间
        - memory_time_us: 访存时间
        - memory_bound: 是否 memory bound
    """
    dtype_size = DTYPE_SIZES.get(dtype, 2)
    
    bytes_theory = ...
    flops = ...
    
    if ... < l2_capacity_bytes:
        tile_M = ...
        num_tiles = ceil(batch_size / tile_M)
    else:
        num_tiles = batch_size
    
    bytes_actual = bytes_theory + ...
    
    compute_time_us = (flops / (peak_tflops * 1e12)) * 1e6
    memory_time_us = (bytes_actual / (memory_bw_gbps * 1e9)) * 1e6
    
    memory_bound = memory_time_us > compute_time_us
    
    return {
        "bytes_theory": bytes_theory,
        "bytes_actual": bytes_actual,
        "num_tiles": num_tiles,
        "flops": flops,
        "compute_time_us": compute_time_us,
        "memory_time_us": memory_time_us,
        "memory_bound": memory_bound,
    }
```

---

## 5. 开发规范

### 5.1 遵循 architecture skill 设计原则

- **模块职责清晰**：kernel 层只负责计算，不依赖上层
- **层次分明**：kernel → model → analyzer 单向依赖
- **避免循环依赖**：禁止 kernel 依赖 model/analyzer
- **设计审视**：新增 kernel 需审视现有模型兼容性

详见：[architecture skill](../architecture/SKILL.md)

### 5.2 遵循 project-code skill 代码规范

- **代码风格**：使用 ruff 格式化，行长度限制 100 字符
- **类型安全**：使用类型注解
- **公开函数 docstring**：必须包含 Args 和 Returns
- **测试覆盖**：新增 kernel 必须添加测试用例
- **小步提交**：每个 kernel 完成后立即提交一个 commit

详见：[project-code skill](../project-code/SKILL.md)

### 5.3 测试覆盖要求

#### 必须覆盖的测试场景

1. **基础功能测试**：
   - 正确的输出形状
   - 正确的 FLOPs 计算
   - 正确的访存计算

2. **Cache-aware 测试**：
   - batch=1 场景（通常 memory bound）
   - batch 较大场景（可能 compute bound）
   - num_tiles 计算正确
   - weight_reload_overhead 计算正确

3. **边界情况测试**：
   - TP=1（无切分）
   - 不同 dtype（fp16, fp32, bf16）
   - 不同硬件参数（L2 容量、peak TFLOPS）

#### 测试命令

```bash
python -m pytest tests/test_kernels.py -v -n 4
python -m pytest tests/ -v -n 4
ruff check llm_perf/kernels/*.py --select=F401,F841,E741
```

### 5.4 Commit 信息格式

```bash
feat(kernels): add <kernel_name> kernel

- Add <kernel_name>() function in functional.py
- Add compute_effective_bytes_<kernel_name>() in cache_aware.py
- Support batch tiling model for memory/compute bound
- Tests: X passed (Y new tests added)
```

---

## 6. 参考文档

### 6.1 设计文档

- [cache_aware_kernel_design.md](../../../docs/cache_aware_kernel_design.md): Cache-aware kernel 详细设计
- [kernel_modeling.md](../../../docs/kernel_modeling.md): Kernel 建模基础

### 6.2 相关 skill

- [architecture skill](../architecture/SKILL.md): 架构设计准则
- [project-code skill](../project-code/SKILL.md): 开发规范
- [reviewer skill](../reviewer/SKILL.md): 代码检视

### 6.3 外部参考

- Flash Attention Paper: https://arxiv.org/abs/2205.14135
- Roofline Model Paper: https://doi.org/10.1109/MC.2009.143
- MLA Paper (DeepSeek-V2): https://arxiv.org/abs/2405.04434
- PyTorch 2.10 Documentation: https://pytorch.org/docs/2.10/

---

## 7. 检查清单

### 设计阶段
- [ ] 分析 kernel 需求合理性
- [ ] 参考 torch 2.10 API
- [ ] 设计 cache-aware 计算模型
- [ ] 考虑 3 个 backend 的实现

### 实现阶段
- [ ] 实现 kernel 函数
- [ ] 实现 cache-aware 函数
- [ ] 遵循函数签名规范
- [ ] 添加类型注解和 docstring

### 测试阶段
- [ ] 编写基础功能测试
- [ ] 编写 cache-aware 测试
- [ ] 编写边界情况测试
- [ ] 验证 3 个 backend 趋势一致性
- [ ] 运行存量测试确保兼容

### 提交阶段
- [ ] 通过 ruff 检查
- [ ] 通过所有单元测试
- [ ] 通过全量测试
- [ ] 小步提交（一个 commit）
- [ ] 推送到远程仓库

---

*文档版本: 2.0*
*创建日期: 2026-04-28*
*整合自: kernel skill + new-kernel skill*
*参考: docs/cache_aware_kernel_design.md, docs/kernel_modeling.md*
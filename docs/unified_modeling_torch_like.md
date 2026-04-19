# Torch-Like 统一建模方案

## 1. 核心思想

**让算法人员像写 PyTorch 代码一样定义模型，同时自动获得切分推导能力**。

```
现有 functional API: torch-like，但不考虑切分
新 OpSpec/OpInstance: 支持切分，但接口不直观

目标方案: ShardedTensor + ShardedModule - 兼具两者
- ShardedTensor: 像torch.Tensor一样操作（直观），自动推导切分约束
- ShardedModule: 像torch.nn.Module一样定义层级模型（模块化）
- 绑定ParallelContext后获得物理形态和开销估算（灵活）
```

## 1.1 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 4: ShardedModel (完整模型)                                 │
│   LlamaModel, DeepSeekModel, BertModel                          │
│   继承 ShardedModule，定义完整模型结构                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Level 3: ShardedBlock (模块块)                                   │
│   ShardedTransformerBlock, ShardedMoEBlock                      │
│   组合多个 ShardedModule，定义模型块                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Level 2: ShardedModule (基础模块)                                │
│   ShardedEmbedding, ShardedAttention, ShardedFFN, ShardedMoE    │
│   ShardedRMSNorm, ShardedLMHead                                 │
│   继承 ShardedModule，定义单个模块                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: ShardedTensor (张量操作)                                │
│   __matmul__, view, transpose, reshape                          │
│   flash_attention, rms_norm, silu                               │
│   像torch.Tensor一样操作，自动推导切分约束                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓ bind(ctx)
┌─────────────────────────────────────────────────────────────────┐
│ Level 0: OpInstance (物理形态)                                   │
│   physical_shape, flops, comm_ops, estimate_time()              │
│   绑定ParallelContext后的运行时实例                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. ShardedTensor 设计

### 2.1 基本定义

```python
class ShardedTensor:
    """带切分约束的张量，类似 torch.Tensor 的接口。
    
    核心特点:
    1. 像torch.Tensor一样操作（matmul, view, transpose等）
    2. 自动推导输出shape和切分约束
    3. 记录操作历史，用于后续推导FLOPs和通信
    4. 标记view tensor (_is_view)，用于激活内存自动追踪
    
    Attributes:
        shape: 逻辑shape（未切分）
        shardable: 各维度的切分约束 {dim_idx: parallel_type}
        dtype: 数据类型
        _op_history: 操作历史（用于推导FLOPs）
        _is_view: 是否是view tensor（不分配新内存）
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        shardable: Optional[Dict[int, str]] = None,  # {dim_idx: "tp"/"sp"/"ep"}
        dtype: str = "fp16",
        name: Optional[str] = None,
    ):
        self.shape = shape
        self.shardable = shardable or {}  # 例如 {0: "tp", 1: "sp"}
        self.dtype = dtype
        self.name = name
        self._op_history = []  # 记录操作历史
        self._is_view = False  # 标记是否是view tensor
    
    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        """类似 torch.Tensor.size()"""
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    def numel(self) -> int:
        """类似 torch.Tensor.numel()"""
        return math.prod(self.shape)
```

### 2.2 操作定义（自动推导切分）

```python
def __matmul__(self, other: ShardedTensor) -> ShardedTensor:
    """矩阵乘法: self @ other
    
    自动推导:
    1. 输出shape: (..., self.shape[-2], other.shape[-1])
    2. 输出切分约束: 根据输入切分约束和矩阵乘法语义推导
    
    切分推导规则:
    - A(shape=(m, k), shardable={0: "tp"}) @ B(shape=(k, n))
      -> output shape=(m, n), shardable={} (A的切分维度被聚合)
      -> 需要AllReduce通信
    
    - A(shape=(m, k)) @ B(shape=(k, n), shardable={1: "tp"})
      -> output shape=(m, n), shardable={1: "tp"} (B的切分维度保留)
      -> 不需要通信
    
    - A(shape=(m, k), shardable={1: "tp"}) @ B(shape=(k, n), shardable={0: "tp"})
      -> output shape=(m, n), shardable={} (两端切分匹配，自动聚合)
      -> 需要AllReduce通信
    """
    # 1. 推导输出shape
    assert self.shape[-1] == other.shape[-2], "Dimension mismatch"
    output_shape = (*self.shape[:-1], other.shape[-1])
    
    # 2. 推导输出切分约束（核心逻辑）
    output_shardable = self._infer_matmul_shardable(other)
    
    # 3. 记录操作历史
    op = MatmulOp(self, other, output_shardable)
    
    # 4. 创建输出张量
    output = ShardedTensor(
        shape=output_shape,
        shardable=output_shardable,
        dtype=self.dtype,
    )
    output._op_history = self._op_history + [op]
    
    return output

def _infer_matmul_shardable(self, other: ShardedTensor) -> Dict[int, str]:
    """推导矩阵乘法输出的切分约束"""
    # A @ B 的切分推导规则:
    
    # Case 1: A在第一维度切分 (row-sharding)
    # A(m, k) shardable={0: "tp"} @ B(k, n)
    # -> output(m, n) shardable={}, 需要 AllReduce
    if 0 in self.shardable and len(self.shape) == 2:
        # A被切分，输出需要聚合
        return {}  # 输出不切分
    
    # Case 2: B在第二维度切分 (column-sharding)
    # A(m, k) @ B(k, n) shardable={1: "tp"}
    # -> output(m, n) shardable={1: "tp"}, 不需要通信
    if 1 in other.shardable and len(other.shape) == 2:
        # B被切分，切分传递到输出
        return {1: other.shardable[1]}
    
    # Case 3: 两端切分匹配
    # A(m, k) shardable={1: "tp"} @ B(k, n) shardable={0: "tp"}
    # -> output(m, n) shardable={}, 需要 AllReduce
    if (len(self.shape) == 2 and 1 in self.shardable and
        len(other.shape) == 2 and 0 in other.shardable and
        self.shardable[1] == other.shardable[0]):
        # 两端切分匹配，自动聚合
        return {}
    
    # 其他case: 保持输入切分约束
    result = {}
    for dim, ptype in self.shardable.items():
        if dim < len(self.shape) - 1:  # 非最后一维
            result[dim] = ptype
    for dim, ptype in other.shardable.items():
        if dim == len(other.shape) - 1:  # 最后一维
            result[dim - len(other.shape) + len(output_shape)] = ptype
    
    return result

def view(self, *shape) -> ShardedTensor:
    """reshape: 类似 torch.Tensor.view()
    
    切分约束跟随维度映射。
    标记为view tensor (_is_view=True)，不分配新内存。
    """
    # 推导维度映射
    old_numel = self.numel()
    new_numel = math.prod(shape)
    assert old_numel == new_numel, f"Cannot reshape {self.shape} to {shape}"
    
    # 推导新的切分约束
    # 例如: (batch, seq, hidden) shardable={1: "sp", 2: "tp"}
    #       -> view(batch, seq, heads, head_dim)
    #       -> shardable={1: "sp", 2: "tp"} (heads继承hidden的切分)
    new_shardable = self._infer_view_shardable(shape)
    
    output = ShardedTensor(shape=shape, shardable=new_shardable, dtype=self.dtype)
    output._op_history = self._op_history + [ViewOp(self, shape)]
    output._is_view = True  # 标记为view tensor
    return output

def transpose(self, dim0: int, dim1: int) -> ShardedTensor:
    """转置: 类似 torch.Tensor.transpose()
    
    切分约束跟随维度交换。
    标记为view tensor (_is_view=True)，不分配新内存。
    """
    new_shape = list(self.shape)
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
    
    new_shardable = {}
    for dim, ptype in self.shardable.items():
        if dim == dim0:
            new_shardable[dim1] = ptype
        elif dim == dim1:
            new_shardable[dim0] = ptype
        else:
            new_shardable[dim] = ptype
    
    output = ShardedTensor(shape=tuple(new_shape), shardable=new_shardable, dtype=self.dtype)
    output._op_history = self._op_history + [TransposeOp(self, dim0, dim1)]
    output._is_view = True  # 标记为view tensor
    return output

def contiguous(self) -> ShardedTensor:
    """返回一个非view的副本。
    
    类似 torch.Tensor.contiguous()，确保张量有连续内存布局。
    标记为非view tensor (_is_view=False)，分配新内存。
    """
    output = ShardedTensor(
        shape=self.shape,
        shardable=self.shardable,
        dtype=self.dtype,
    )
    output._op_history = self._op_history
    output._is_view = False  # 强制标记为非view
    return output

def get_physical_bytes(self, parallel_degrees: Dict[str, int]) -> int:
    """获取物理内存字节数（考虑切分）。
    
    Args:
        parallel_degrees: {parallel_type: degree}
    
    Returns:
        物理内存字节数 (numel_after_sharding * dtype_size)
    """
    physical_shape = self.get_physical_shape(parallel_degrees)
    dtype_size = DTYPE_SIZES.get(self.dtype, 2)
    return math.prod(physical_shape) * dtype_size
```

### 2.3 特殊算子

```python
def flash_attention(
    query: ShardedTensor,   # (batch, heads, seq, head_dim)
    key: ShardedTensor,     # (batch, kv_heads, kv_seq, head_dim)
    value: ShardedTensor,   # (batch, kv_heads, kv_seq, head_dim)
    is_causal: bool = False,
) -> ShardedTensor:
    """Flash Attention算子
    
    自动推导:
    1. 输出shape: (batch, heads, seq, head_dim)
    2. 输出切分约束: 
       - heads维度: 如果query被TP切分，输出也被切分
       - seq维度: 如果使用SP，输出被切分
    """
    batch, heads, seq, head_dim = query.shape
    
    # 推导输出切分约束
    output_shardable = {}
    if 1 in query.shardable:  # heads维度
        output_shardable[1] = query.shardable[1]
    if 2 in query.shardable:  # seq维度 (SP)
        output_shardable[2] = query.shardable[2]
    
    output = ShardedTensor(
        shape=(batch, heads, seq, head_dim),
        shardable=output_shardable,
        dtype=query.dtype,
    )
    output._op_history = query._op_history + [AttentionOp(query, key, value)]
    return output

def rms_norm(input: ShardedTensor) -> ShardedTensor:
    """RMS Norm - 切分约束保持不变"""
    output = ShardedTensor(
        shape=input.shape,
        shardable=input.shardable,
        dtype=input.dtype,
    )
    output._op_history = input._op_history + [NormOp(input, "rmsnorm")]
    return output

def silu(input: ShardedTensor) -> ShardedTensor:
    """SiLU激活 - 切分约束保持不变"""
    output = ShardedTensor(
        shape=input.shape,
        shardable=input.shardable,
        dtype=input.dtype,
    )
    output._op_history = input._op_history + [ActivationOp(input, "silu")]
    return output
```

---

## 3. 使用示例

### 3.1 定义模型（类似 PyTorch）

```python
from llm_perf.modeling import ShardedTensor, flash_attention, rms_norm, silu

# ===== 定义输入张量 =====
batch, seq, hidden = 1, 4096, 4096
num_heads, head_dim = 32, 128

# 输入激活 - 定义切分约束
input = ShardedTensor(
    shape=(batch, seq, hidden),
    shardable={1: "sp", 2: "tp"},  # seq被SP切分，hidden被TP切分
    name="input",
)

# ===== 定义权重张量 =====
# Q projection weight: (hidden, num_heads * head_dim)
q_weight = ShardedTensor(
    shape=(hidden, num_heads * head_dim),
    shardable={0: "tp"},  # hidden维度被TP切分
    name="q_weight",
)

# K projection weight
k_weight = ShardedTensor(
    shape=(hidden, num_kv_heads * head_dim),
    shardable={0: "tp"},
    name="k_weight",
)

# V projection weight
v_weight = ShardedTensor(
    shape=(hidden, num_kv_heads * head_dim),
    shardable={0: "tp"},
    name="v_weight",
)

# O projection weight: (num_heads * head_dim, hidden)
o_weight = ShardedTensor(
    shape=(num_heads * head_dim, hidden),
    shardable={0: "tp"},  # num_heads维度被TP切分
    name="o_weight",
)

# ===== 像写PyTorch一样定义计算流程 =====

# Input RMSNorm
norm_out = rms_norm(input)

# Q, K, V projections
q = norm_out @ q_weight  # (batch, seq, num_heads * head_dim), shardable自动推导
k = norm_out @ k_weight
v = norm_out @ v_weight

# Reshape to (batch, heads, seq, head_dim)
q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)  # 切分约束自动传播
k = k.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
v = v.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

# Flash Attention
attn_out = flash_attention(q, k, v)

# Reshape back
attn_out = attn_out.transpose(1, 2).view(batch, seq, num_heads * head_dim)

# O projection
o_proj = attn_out @ o_weight  # 自动推导切分约束和通信需求

# ===== 绑定ParallelContext =====
ctx = ParallelContext(tp=8, sp=4, ep=1, pp=1, dp=1)

# 从任意张量获取完整模型的OpInstance
model_instance = o_proj.bind(ctx)

# 或者逐层获取
q_instance = q.bind(ctx)
attn_instance = attn_out.bind(ctx)
o_instance = o_proj.bind(ctx)
```

### 3.2 获取物理形态和开销

```python
# ===== 自动推导 =====

# Q projection 的物理形态
print(f"Q input logical shape: {q._op_history[0].input.shape}")
# (1, 4096, 4096)

print(f"Q input physical shape: {q_instance.input_physical_shape}")
# (1, 4096/4, 4096/8) = (1, 1024, 512)  <- SP切分seq, TP切分hidden

print(f"Q output logical shape: {q.shape}")
# (1, 4096, 32*128) = (1, 4096, 4096)

print(f"Q output physical shape: {q_instance.output_physical_shape}")
# (1, 4096/4, 4096/8) = (1, 1024, 512)  <- SP切分seq, TP切分heads

print(f"Q FLOPs: {q_instance.flops / 1e9:.2f}G")
# 基于 physical shape 计算: 2 * 1 * 1024 * 512 * (32/8 * 128)

print(f"Q communication ops: {q_instance.comm_ops}")
# [] <- TP列切分不产生通信

# O projection 的物理形态和通信
print(f"O communication ops: {o_instance.comm_ops}")
# [AllReduce(activation_bytes, tp_degree=8)]
# <- O projection输入heads被切分，输出hidden不切分，需要AllReduce聚合

print(f"O communication time: {o_instance.comm_time_ms:.2f}ms")
# 基于 Cluster 的带宽计算
```

### 3.3 高维并行示例

```python
# ===== TP=8, EP=2, SP=4 的 MoE模型 =====

# Input activation
input = ShardedTensor(
    shape=(batch, seq, hidden),
    shardable={1: "sp", 2: "tp"},
)

# MoE Router (不切分)
router_weight = ShardedTensor(shape=(hidden, num_experts), shardable={})
router_out = input @ router_weight  # (batch, seq, num_experts), shardable={1: "sp"}

# Expert weights (被EP和TP切分)
expert_weight = ShardedTensor(
    shape=(hidden, expert_intermediate),
    shardable={0: "tp"},  # TP切分hidden
    # EP切分不在shape维度上，而是有多少expert在本地
    expert_parallel={"ep": num_experts // ep_degree},  # 新增：EP维度
)

# 绑定ParallelContext
ctx = ParallelContext(tp=8, ep=2, sp=4, ...)

# 获取物理形态
expert_instance = expert_out.bind(ctx)

# 自动推导通信
print(f"MoE communication ops: {expert_instance.comm_ops}")
# [
#   All2All(dispatch, ep_degree=2),  # token分发到expert
#   All2All(combine, ep_degree=2),   # expert结果聚合
#   AllReduce(tp_degree=8),          # TP聚合
# ]
```

---

## 4. OpInstance 实现

```python
class OpInstance:
    """绑定到ParallelContext的算子实例
    
    从ShardedTensor的操作历史推导:
    1. 物理shape
    2. FLOPs
    3. 通信操作
    4. 时间估算（调用KernelBackend）
    """
    
    def __init__(self, tensor: ShardedTensor, ctx: ParallelContext):
        self.tensor = tensor
        self.ctx = ctx
        self._analyze()
    
    def _analyze(self):
        """分析操作历史，推导物理形态和通信"""
        # 1. 遍历操作历史
        for op in self.tensor._op_history:
            # 2. 推导每个操作的物理shape
            physical_input = self._infer_physical_shape(op.input)
            physical_output = self._infer_physical_shape(op.output)
            
            # 3. 推导通信操作
            comm_ops = self._infer_comm_ops(op)
            
            # 4. 记录FLOPs（调用functional API）
            flops = self._compute_flops(op, physical_input, physical_output)
    
    def _infer_physical_shape(self, tensor: ShardedTensor) -> Tuple[int, ...]:
        """推导物理shape"""
        shape = []
        for dim, size in enumerate(tensor.shape):
            if dim in tensor.shardable:
                ptype = tensor.shardable[dim]
                degree = self.ctx.get_degree(ptype)
                shape.append(max(1, size // degree))
            else:
                shape.append(size)
        return tuple(shape)
    
    def _infer_comm_ops(self, op: Op) -> List[CommOp]:
        """从操作的输入输出切分状态推导通信"""
        # 检查切分状态变化
        input_shardable = op.input.shardable
        output_shardable = op.output.shardable
        
        # Case: Matmul 聚合
        # A(m, k) shardable={0: "tp"} @ B(k, n)
        # -> output(m, n) shardable={}
        # -> AllReduce(m * n * dtype_bytes, tp_degree)
        if isinstance(op, MatmulOp):
            if 0 in input_shardable and 0 not in output_shardable:
                ptype = input_shardable[0]
                comm_bytes = self._infer_physical_shape(op.output).size() * DTYPE_SIZES[op.dtype]
                return [CommOp("allreduce", comm_bytes, ptype)]
        
        # Case: Attention SP通信
        # Ulysses: seq维度切分 -> All2All
        # Ring: KV切分 -> P2P or AllGather
        if isinstance(op, AttentionOp):
            if 2 in input_shardable:  # seq维度
                ptype = input_shardable[2]
                if self.ctx.sp_type == SPType.ULYSSES:
                    return [CommOp("alltoall", ..., ptype)]
                elif self.ctx.sp_type == SPType.RING_P2P:
                    return [CommOp("p2p_ring", ..., ptype)]
        
        return []
    
    def _compute_flops(self, op: Op, physical_input, physical_output) -> int:
        """计算FLOPs（调用functional API）"""
        if isinstance(op, MatmulOp):
            # 调用 functional.linear 计算FLOPs
            result = functional.linear(physical_input, op.weight.shape, dtype=op.dtype)
            return result.flops
        elif isinstance(op, AttentionOp):
            result = functional.flash_attention(
                physical_input, op.key.shape, op.value.shape, dtype=op.dtype
            )
            return result.flops
        ...
    
    @property
    def physical_shape(self) -> Tuple[int, ...]:
        return self._infer_physical_shape(self.tensor)
    
    @property
    def flops(self) -> int:
        return sum(op.flops for op in self._op_instances)
    
    @property
    def comm_ops(self) -> List[CommOp]:
        return self._comm_ops
    
    def estimate_time(self, backend: KernelBackend) -> float:
        """估算时间（调用KernelBackend）"""
        compute_time = 0.0
        for op_instance in self._op_instances:
            compute_time += backend.estimate_compute_time(
                op_instance.kernel_name,
                op_instance.physical_inputs,
                op_instance.physical_output,
                op_instance.dtype,
                self.ctx.device,
            )
        
        comm_time = 0.0
        for comm_op in self.comm_ops:
            domain = self.ctx.get_comm_domain(comm_op.ptype)
            comm_time += backend.estimate_comm_time(
                comm_op.comm_type,
                comm_op.data_bytes,
                len(domain.ranks),
                domain.bandwidth_gbps,
            )
        
        return compute_time + comm_time
```

---

## 5. 与现有代码的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                ShardedTensor (新 - torch-like接口)               │
│  像torch.Tensor一样操作，自动推导切分约束                          │
│  input @ weight -> output (自动推导 shardable)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                OpInstance (新 - 绑定ParallelContext)             │
│  从ShardedTensor推导物理shape、FLOPs、通信                        │
│  内部调用 functional API 计算FLOPs                                │
│  内部调用 KernelBackend 估算时间                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 使用
┌─────────────────────────────────────────────────────────────────┐
│                functional API (保留)                             │
│  linear(), flash_attention(), rms_norm() 等                      │
│  计算 logical FLOPs/memory                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                KernelBackend (保留)                              │
│  TheoryBackend, ProfilingBackend, MicroarchBackend               │
│  估算物理形态的时间                                                │
└─────────────────────────────────────────────────────────────────┘
```

**关键点**:
1. `ShardedTensor` 是 torch-like 的接口，用户直观使用
2. `OpInstance` 内部调用 `functional API` 计算 FLOPs
3. `OpInstance` 内部调用 `KernelBackend` 估算时间
4. `functional API` 和 `KernelBackend` 完全保留

---

## 6. 优势对比

| 方案 | 接口直观性 | 切分支持 | 自动推导 | 与现有代码关系 |
|-----|----------|---------|---------|--------------|
| **functional API** | ✅ torch-like | ❌ 不支持 | ❌ 手动计算 | - |
| **OpSpec/OpInstance** | ❌ 不直观 | ✅ 支持 | ✅ 自动推导 | 新增，不调用functional |
| **ShardedTensor** | ✅ torch-like | ✅ 支持 | ✅ 自动推导 | ✅ 调用functional和Backend |

**ShardedTensor 方案优势**:
1. 用户像写 PyTorch 一样定义模型（直观）
2. 切分约束自动推导（自动）
3. 绑定 ParallelContext 后获得物理形态和开销（灵活）
4. 完全兼容现有 functional API 和 KernelBackend（无缝）
5. 自动激活内存追踪，过滤view tensor（准确）

---

## 6.1 自动激活内存追踪机制

**问题背景**:

训练时需要保存所有中间激活用于 backward pass。但有些操作产生的 tensor 是 view（不分配新内存），如：
- `q_proj.view(batch, seq, heads, head_dim).transpose(1, 2)` 是 view
- Flash Attention 的输入 Q, K, V 可以从 Q_proj, K_proj, V_proj 重新 reshape

之前用户需要手动判断哪些 tensor 需要保存，容易出错。

**解决方案**:

分四个层次自动追踪：

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: ShardedTensor._is_view                                      │
│   - 标记 tensor 是否是 view（不分配新内存）                            │
│   - view(), transpose() → _is_view=True                              │
│   - matmul(), silu() → _is_view=False                                │
│   - contiguous() → 强制 _is_view=False                               │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 2: KernelResult.saved_inputs                                   │
│   - 标记 backward pass 需要保存哪些输入                                │
│   - linear: saved_inputs=["input"]                                   │
│   - flash_attention: saved_inputs=[]                                 │
│   - rms_norm: saved_inputs=["input"]                                 │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3: Op.get_saved_tensors()                                      │
│   - 从 Op 对象推导 backward 需要保存的 tensor                          │
│   - MatmulOp: get_saved_tensors() → [self.input]                     │
│   - AttentionOp: get_saved_tensors() → []                            │
│   - ActivationOp: 根据激活类型返回不同结果                            │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 4: ModuleInstance.activation_memory_physical                   │
│   - 从 _intermediate_tensors 收集                                    │
│   - 过滤 _is_view=True 的 tensor                                      │
│   - 计算物理内存字节数                                                 │
│   - 结果：只统计非view tensor                                          │
└─────────────────────────────────────────────────────────────────────┘
```

**Flash Attention 示例**:

```python
class ShardedAttention(ShardedModule):
    def forward(self, hidden):
        # 这些是真正的张量（需要保存，_is_view=False）
        q_proj = self._track_intermediate("q_proj", hidden @ self.q_weight)  # 新张量
        k_proj = self._track_intermediate("k_proj", hidden @ self.k_weight)  # 新张量
        v_proj = self._track_intermediate("v_proj", hidden @ self.v_weight)  # 新张量
        
        # 这些是 view tensor（不保存，_is_view=True）
        q = q_proj.view(batch, seq, heads, head_dim).transpose(1, 2)  # view!
        k = k_proj.view(...).transpose(1, 2)  # view!
        v = v_proj.view(...).transpose(1, 2)  # view!
        
        # flash_attention 返回的也是 view（可以重计算）
        attn_out = flash_attention(q, k, v)  # 内部状态已保存，tensor是view
        
        # output projection 产生新张量（需要保存）
        attn_flat = attn_out.transpose(1, 2).view(...)  # view!
        output = self._track_intermediate("output", attn_flat @ self.o_weight)  # 新张量
        
        # ModuleInstance.activation_memory_physical 自动：
        # - 统计: q_proj, k_proj, v_proj, output (非view)
        # - 过滤: q, k, v, attn_out, attn_flat (view)
```

**效果对比**:

| 场景 | 手动追踪（之前） | 自动追踪（之后） |
|------|-----------------|-----------------|
| Flash Attention | ~32GB（错误追踪所有tensor） | ~13GB（只追踪非view） |
| 训练内存估算 | 可能重复计算view | 自动过滤view |

**关键设计**:

1. **_is_view 标记**: 
   - `view()`, `transpose()`, `reshape()` 设置 `_is_view=True`
   - `matmul()`, `silu()`, `rms_norm()` 等产生新张量，`_is_view=False`

2. **saved_inputs 标记**:
   - `KernelResult.saved_inputs` 标记 backward 需要保存哪些输入
   - 如 `linear` 的 `saved_inputs=["input"]` 表示 backward 需要 input 计算 dW

3. **Op.get_saved_tensors()**:
   - 从 Op 对象自动推导 saved tensors
   - 如 `MatmulOp.get_saved_tensors() → [self.input]`

4. **ModuleInstance 自动收集**:
   - 从 `_intermediate_tensors` 收集
   - 过滤 `_is_view=True`
   - 递归添加子模块激活内存

## 7. 实现路径

### Phase 1: ShardedTensor 核心
1. 实现 `ShardedTensor` 类（shape, shardable, dtype）
2. 实现基本操作（matmul, view, transpose）
3. 实现切分推导逻辑

### Phase 2: 特殊算子
1. 实现 `flash_attention` 算子（支持SP通信推导）
2. 实现 `rms_norm`, `silu`, `gelu` 等算子
3. 实现 MoE 相关算子（支持EP通信推导）

### Phase 3: OpInstance
1. 实现 `OpInstance.bind()` 方法
2. 实现物理shape推导
3. 实现通信操作推导
4. 实现 FLOPs 计算（调用 functional API）
5. 实现时间估算（调用 KernelBackend）

### Phase 4: ParallelContext
1. 实现 `ParallelContext`（整合 StrategyConfig + Cluster）
2. 实现通信域获取方法

### Phase 5: 测试验证
1. 单并行策略验证
2. 高维并行组合验证
3. 与现有实现对比验证

---

## 8. ShardedModule 设计

### 8.1 基类定义

```python
class ShardedModule:
    """模块基类，类似 torch.nn.Module
    
    核心特点:
    1. 像torch.nn.Module一样定义子模块和forward方法
    2. 自动管理权重参数、切分约束、激活值
    3. bind(ctx)方法返回OpInstance（模型级别的物理形态）
    4. 支持forward/backward双模式，自动推导反向FLOPs和通信
    
    Attributes:
        _submodules: 子模块字典 {name: ShardedModule}
        _weights: 权重张量字典 {name: ShardedTensor}
        _activations: 激活张量字典 {name: ShardedTensor}（forward过程中记录）
        _name: 模块名称
    """
    
    def __init__(self):
        self._submodules: Dict[str, ShardedModule] = {}
        self._weights: Dict[str, ShardedTensor] = {}
        self._activations: Dict[str, ShardedTensor] = {}  # 记录forward产生的激活
        self._name = ""
        self._last_forward_input: Optional[ShardedTensor] = None
        self._last_forward_output: Optional[ShardedTensor] = None
    
    def __setattr__(self, name: str, value: Any):
        """自动注册子模块和权重"""
        if isinstance(value, ShardedModule):
            self._submodules[name] = value
            value._name = name
        elif isinstance(value, ShardedTensor):
            self._weights[name] = value
        super().__setattr__(name, value)
    
    def forward(self, *args, **kwargs) -> ShardedTensor:
        """forward方法，子类必须实现
        
        注意：forward过程中会自动记录：
        1. 输入输出ShardedTensor（用于激活内存估算）
        2. 所有中间操作的OpHistory（用于FLOPs和通信推导）
        """
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> ShardedTensor:
        """调用forward，自动记录输入输出"""
        self._last_forward_input = args[0] if args else None
        output = self.forward(*args, **kwargs)
        self._last_forward_output = output
        return output
    
    def bind(self, ctx: ParallelContext) -> ModuleInstance:
        """绑定到ParallelContext，返回ModuleInstance
        
        ModuleInstance 包含:
        - 所有子模块的 OpInstance
        - 模块级别的 FLOPs, memory, comm_ops（forward + backward）
        - estimate_time() 方法
        - 支持 mode="forward" / "forward_backward"
        """
        return ModuleInstance(self, ctx)
    
    def get_weights(self) -> Dict[str, ShardedTensor]:
        """获取所有权重（包括子模块的权重）"""
        weights = self._weights.copy()
        for name, submodule in self._submodules.items():
            for w_name, w_tensor in submodule.get_weights().items():
                weights[f"{name}.{w_name}"] = w_tensor
        return weights
    
    def get_activations(self) -> Dict[str, ShardedTensor]:
        """获取所有激活（包括子模块的激活）"""
        activations = self._activations.copy()
        for name, submodule in self._submodules.items():
            for a_name, a_tensor in submodule.get_activations().items():
                activations[f"{name}.{a_name}"] = a_tensor
        return activations
    
    def params_count(self) -> int:
        """参数量（逻辑，未切分）- 总参数数量"""
        return sum(w.numel() for w in self.get_weights().values())
    
    def params_count_breakdown(self) -> Dict[str, int]:
        """参数量分解 - 各层参数数量"""
        breakdown = {}
        for name, weight in self._weights.items():
            breakdown[name] = weight.numel()
        for sub_name, submodule in self._submodules.items():
            for w_name, count in submodule.params_count_breakdown().items():
                breakdown[f"{sub_name}.{w_name}"] = count
        return breakdown
    
    def activation_memory_logical(self, batch_size: int = 1) -> int:
        """激活内存（逻辑，未切分）- 训练时所有激活"""
        activations = self.get_activations()
        total = sum(
            a.numel() * DTYPE_SIZES[a.dtype] * batch_size
            for a in activations.values()
        )
        return total
    
    def activation_memory_max_layer(self, batch_size: int = 1) -> int:
        """激活内存（推理）- 最大单层激活"""
        activations = self.get_activations()
        if not activations:
            return 0
        max_activation = max(
            a.numel() * DTYPE_SIZES[a.dtype] * batch_size
            for a in activations.values()
        )
        return max_activation
    
    def flops_forward(self, batch_size: int = 1, seq_len: int = 1) -> int:
        """Forward FLOPs（逻辑，未切分）"""
        # 从forward输出tensor的操作历史推导
        if self._last_forward_output is None:
            return 0
        return self._last_forward_output.total_flops_logical(batch_size, seq_len)
    
    def flops_backward(self, batch_size: int = 1, seq_len: int = 1) -> int:
        """Backward FLOPs（逻辑，未切分）
        
        规则:
        - Matmul backward: 2x forward FLOPs（计算梯度）
        - Attention backward: 2x forward FLOPs
        - Activation backward: 2x forward FLOPs
        - Norm backward: ~0.7x forward FLOPs
        
        总体规则: backward ≈ 2x forward
        """
        forward_flops = self.flops_forward(batch_size, seq_len)
        return forward_flops * 2  # 简化规则
    
    def flops_total(self, batch_size: int = 1, seq_len: int = 1) -> int:
        """Total FLOPs (forward + backward)"""
        return self.flops_forward(batch_size, seq_len) + self.flops_backward(batch_size, seq_len)
```

### 8.2 ModuleInstance

```python
class ModuleInstance:
    """模块绑定到ParallelContext后的实例
    
    Attributes:
        module: 原始ShardedModule
        ctx: ParallelContext
        _submodule_instances: 子模块的ModuleInstance
        _op_instances: 各层操作的OpInstance
        _pp_stage: PP stage索引（如果启用PP）
        mode: "forward" 或 "forward_backward"
    """
    
    def __init__(
        self,
        module: ShardedModule,
        ctx: ParallelContext,
        pp_stage: Optional[int] = None,  # PP stage索引
        mode: str = "forward_backward",  # "forward" 或 "forward_backward"
    ):
        self.module = module
        self.ctx = ctx
        self.pp_stage = pp_stage
        self.mode = mode
        
        # 构建子模块实例（继承PP stage）
        self._submodule_instances: Dict[str, ModuleInstance] = {}
        for name, submodule in module._submodules.items():
            self._submodule_instances[name] = ModuleInstance(
                submodule, ctx, pp_stage=pp_stage, mode=mode
            )
        
        # 构建权重物理形态
        self._weight_instances: Dict[str, WeightInstance] = {}
        for name, weight in module.get_weights().items():
            self._weight_instances[name] = WeightInstance(weight, ctx)
        
        # 构建激活物理形态
        self._activation_instances: Dict[str, ActivationInstance] = {}
        for name, activation in module.get_activations().items():
            self._activation_instances[name] = ActivationInstance(activation, ctx)
    
    @property
    def params_count_physical(self) -> int:
        """物理参数量（切分后）- 单GPU参数量"""
        return sum(w.physical_numel for w in self._weight_instances.values())
    
    @property
    def params_count_logical(self) -> int:
        """逻辑参数量（未切分）- 模型总参数量"""
        return self.module.params_count()
    
    @property
    def flops_forward_physical(self) -> int:
        """Forward FLOPs（物理，切分后）"""
        if self.module._last_forward_output is None:
            return 0
        
        # 从forward输出的操作历史推导物理FLOPs
        physical_flops = 0
        for op in self.module._last_forward_output._op_history:
            physical_flops += self._infer_physical_flops(op)
        
        return physical_flops
    
    @property
    def flops_backward_physical(self) -> int:
        """Backward FLOPs（物理，切分后）"""
        return self.flops_forward_physical * 2
    
    @property
    def flops_total_physical(self) -> int:
        """Total FLOPs (forward + backward, 物理)"""
        if self.mode == "forward":
            return self.flops_forward_physical
        return self.flops_forward_physical + self.flops_backward_physical
    
    @property
    def activation_memory_physical(self) -> int:
        """激活内存（物理，切分后）
        
        自动激活内存追踪:
        - 从 _intermediate_tensors 收集
        - 过滤 view tensor (_is_view=True)
        - 只统计需要保存的非view tensor
        
        规则:
        - 训练(mode="forward_backward"): 保存非view中间张量
        - 推理(mode="forward"): 只需最大单层激活
        """
        if self.mode == "forward":
            # 推理: 最大单层激活
            if not self._activation_instances:
                return 0
            return max(a.physical_bytes for a in self._activation_instances.values())
        else:
            # 训练: 自动收集非view中间张量
            parallel_degrees = self._get_parallel_degrees()
            total = 0
            
            for name, activation in self.module._intermediate_tensors.items():
                # 过滤view tensor（不分配新内存）
                if hasattr(activation, "_is_view") and activation._is_view:
                    continue
                if hasattr(activation, "get_physical_bytes"):
                    total += activation.get_physical_bytes(parallel_degrees)
            
            # 添加子模块激活内存
            for sub_inst in self._submodule_instances.values():
                total += sub_inst.activation_memory_physical
            
            # 激活重计算（如果启用activation_checkpointing）
            if hasattr(self.ctx, "activation_checkpointing") and self.ctx.activation_checkpointing:
                ratio = getattr(self.ctx, "activation_checkpointing_ratio", 1)
                total = total // ratio
            
            return total
    
    @property
    def total_comm_ops(self) -> List[CommOp]:
        """总通信操作（forward + backward）"""
        ops = []
        for inst in self._submodule_instances.values():
            ops.extend(inst.total_comm_ops)
        
        # 从forward输出推导通信操作
        if self.module._last_forward_output:
            for op in self.module._last_forward_output._op_history:
                comm_ops = self._infer_comm_ops(op)
                ops.extend(comm_ops)
                
                # Backward通信（梯度同步）
                if self.mode == "forward_backward":
                    backward_comm_ops = self._infer_backward_comm_ops(op)
                    ops.extend(backward_comm_ops)
        
        return ops
    
    def _infer_physical_flops(self, op: Op) -> int:
        """从操作推导物理FLOPs（调用functional API）"""
        # 获取物理shape
        physical_inputs = [self._infer_physical_shape(t) for t in op.inputs]
        physical_output = self._infer_physical_shape(op.output)
        
        # 调用functional API计算FLOPs
        if isinstance(op, MatmulOp):
            result = functional.linear(*physical_inputs, dtype=op.dtype)
            return result.flops
        elif isinstance(op, AttentionOp):
            result = functional.flash_attention(*physical_inputs, dtype=op.dtype)
            return result.flops
        elif isinstance(op, RMSNormOp):
            result = functional.rms_norm(physical_inputs[0], dtype=op.dtype)
            return result.flops
        ...
    
    def _infer_physical_shape(self, tensor: ShardedTensor) -> Tuple[int, ...]:
        """推导物理shape"""
        shape = []
        for dim, size in enumerate(tensor.shape):
            if dim in tensor.shardable:
                ptype = tensor.shardable[dim]
                degree = self.ctx.get_degree(ptype)
                shape.append(max(1, size // degree))
            else:
                shape.append(size)
        return tuple(shape)
    
    def _infer_comm_ops(self, op: Op) -> List[CommOp]:
        """从操作推导forward通信"""
        # Matmul聚合通信
        if isinstance(op, MatmulOp):
            input_shardable = op.input.shardable
            output_shardable = op.output.shardable
            
            # TP列切分 -> AllReduce
            if 0 in input_shardable and 0 not in output_shardable:
                ptype = input_shardable[0]
                physical_output = self._infer_physical_shape(op.output)
                comm_bytes = math.prod(physical_output) * DTYPE_SIZES[op.dtype]
                return [CommOp("allreduce", comm_bytes, ptype)]
        
        # Attention SP通信
        if isinstance(op, AttentionOp):
            if 2 in op.query.shardable:  # seq维度
                ptype = op.query.shardable[2]
                if self.ctx.sp_type == SPType.ULYSSES:
                    # Ulysses: All2All
                    comm_bytes = ...
                    return [CommOp("alltoall", comm_bytes, ptype)]
                elif self.ctx.sp_type == SPType.RING_P2P:
                    # Ring: P2P
                    return [CommOp("p2p_ring", ..., ptype)]
        
        # MoE EP通信
        if isinstance(op, MoEExpertOp):
            if "ep" in op.expert_weights.shardable:
                # All2All dispatch + combine
                return [
                    CommOp("alltoall", ..., "ep"),  # dispatch
                    CommOp("alltoall", ..., "ep"),  # combine
                ]
        
        return []
    
    def _infer_backward_comm_ops(self, op: Op) -> List[CommOp]:
        """从操作推导backward通信（梯度同步）"""
        backward_ops = []
        
        # Matmul backward: 梯度AllReduce（如果权重切分）
        if isinstance(op, MatmulOp):
            weight = op.weight
            # 检查权重是否被切分
            for dim, ptype in weight.shardable.items():
                if ptype == "tp":
                    # TP切分权重，backward需要梯度AllReduce
                    # 但如果是行切分，不需要额外通信（forward已经AllReduce了）
                    if dim == 0:  # 行切分
                        # backward不需要额外AllReduce
                        pass
                    elif dim == 1:  # 列切分
                        # backward需要权重梯度AllReduce
                        weight_bytes = weight.numel() * DTYPE_SIZES[weight.dtype] // self.ctx.tp_degree
                        backward_ops.append(CommOp("allreduce", weight_bytes, "tp"))
                elif ptype == "dp":
                    # DP梯度AllReduce（所有权重都需要）
                    weight_bytes = weight.numel() * DTYPE_SIZES[weight.dtype]
                    backward_ops.append(CommOp("allreduce", weight_bytes, "dp"))
        
        # MoE backward: EP梯度通信
        if isinstance(op, MoEExpertOp):
            backward_ops.extend([
                CommOp("alltoall", ..., "ep"),  # backward dispatch
                CommOp("alltoall", ..., "ep"),  # backward combine
            ])
        
        return backward_ops
    
    def estimate_memory(self, batch_size: int = 1) -> int:
        """估算内存（参数 + 激活 + 优化器状态）"""
        # 1. 参数内存
        param_memory = self.params_count_physical * self.ctx.dtype_size
        
        # 2. 激活内存
        activation_memory = self.activation_memory_physical * batch_size
        
        # 3. 优化器状态内存（训练时）
        optimizer_memory = 0
        if self.mode == "forward_backward":
            # ZeRO stage影响
            zero_stage = self.ctx.zero_stage
            if zero_stage == 0:
                optimizer_memory = param_memory * 2  # fp32 optimizer states
            elif zero_stage == 1:
                optimizer_memory = param_memory * 2 // self.ctx.dp_degree
            elif zero_stage == 2:
                optimizer_memory = param_memory * 2 // self.ctx.dp_degree  # 梯度也切分
            elif zero_stage == 3:
                optimizer_memory = 0  # 参数也切分
        
        # 4. 通信buffer
        comm_buffer = sum(op.data_bytes for op in self.total_comm_ops) * 0.1  # 估算10%
        
        total = param_memory + activation_memory + optimizer_memory + comm_buffer
        
        # 5. 安全系数
        total = int(total * 1.15)  # 15% overhead
        
        return total
    
    def estimate_time(self, backend: KernelBackend) -> float:
        """估算时间（compute + comm）"""
        compute_time = 0.0
        for op in self.module._last_forward_output._op_history:
            physical_inputs = [self._infer_physical_shape(t) for t in op.inputs]
            physical_output = self._infer_physical_shape(op.output)
            
            compute_time += backend.estimate_compute_time(
                op.kernel_name,
                physical_inputs,
                physical_output,
                op.dtype,
                self.ctx.device,
            )
        
        # Backward compute time
        if self.mode == "forward_backward":
            compute_time *= 2  # backward ≈ 2x forward
        
        comm_time = 0.0
        for comm_op in self.total_comm_ops:
            domain = self.ctx.get_comm_domain(comm_op.ptype)
            comm_time += backend.estimate_comm_time(
                comm_op.comm_type,
                comm_op.data_bytes,
                len(domain.ranks),
                domain.bandwidth_gbps,
            )
        
        return compute_time + comm_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于输出"""
        return {
            "module_name": self.module._name,
            "pp_stage": self.pp_stage,
            "mode": self.mode,
            "params": {
                "logical": self.params_count_logical,
                "physical": self.params_count_physical,
                "logical_gb": self.params_count_logical * self.ctx.dtype_size / 1e9,
                "physical_gb": self.params_count_physical * self.ctx.dtype_size / 1e9,
            },
            "flops": {
                "forward_physical": self.flops_forward_physical,
                "backward_physical": self.flops_backward_physical,
                "total_physical": self.flops_total_physical,
            },
            "activation": {
                "physical_bytes": self.activation_memory_physical,
                "physical_gb": self.activation_memory_physical / 1e9,
            },
            "memory": {
                "estimate_bytes": self.estimate_memory(),
                "estimate_gb": self.estimate_memory() / 1e9,
            },
            "communication": {
                "total_ops": len(self.total_comm_ops),
                "ops_breakdown": [
                    {"type": op.comm_type, "ptype": op.ptype, "bytes": op.data_bytes}
                    for op in self.total_comm_ops
                ],
            },
            "submodules": {
                name: inst.to_dict()
                for name, inst in self._submodule_instances.items()
            },
        }


class WeightInstance:
    """权重绑定到ParallelContext后的实例"""
    
    def __init__(self, weight: ShardedTensor, ctx: ParallelContext):
        self.weight = weight
        self.ctx = ctx
    
    @property
    def physical_shape(self) -> Tuple[int, ...]:
        return self._infer_physical_shape(self.weight)
    
    @property
    def physical_numel(self) -> int:
        return math.prod(self.physical_shape)
    
    def _infer_physical_shape(self, tensor: ShardedTensor) -> Tuple[int, ...]:
        shape = []
        for dim, size in enumerate(tensor.shape):
            if dim in tensor.shardable:
                ptype = tensor.shardable[dim]
                degree = self.ctx.get_degree(ptype)
                shape.append(max(1, size // degree))
            else:
                shape.append(size)
        return tuple(shape)


class ActivationInstance:
    """激活绑定到ParallelContext后的实例"""
    
    def __init__(self, activation: ShardedTensor, ctx: ParallelContext):
        self.activation = activation
        self.ctx = ctx
    
    @property
    def physical_shape(self) -> Tuple[int, ...]:
        return self._infer_physical_shape(self.activation)
    
    @property
    def physical_bytes(self) -> int:
        return math.prod(self.physical_shape) * DTYPE_SIZES[self.activation.dtype]
    
    def _infer_physical_shape(self, tensor: ShardedTensor) -> Tuple[int, ...]:
        shape = []
        for dim, size in enumerate(tensor.shape):
            if dim in tensor.shardable:
                ptype = tensor.shardable[dim]
                degree = self.ctx.get_degree(ptype)
                shape.append(max(1, size // degree))
            else:
                shape.append(size)
        return tuple(shape)
```

---

## 9. 基础模块实现

### 9.1 ShardedEmbedding

```python
class ShardedEmbedding(ShardedModule):
    """Embedding层，类似 torch.nn.Embedding
    
    切分方式:
    - vocab维度被TP切分
    - 推理时输出需要AllGather
    
    Args:
        num_embeddings: 词表大小
        embedding_dim: 嵌入维度
        dtype: 数据类型
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Embedding权重: (vocab, hidden)
        # vocab维度可被TP切分
        self.weight = ShardedTensor(
            shape=(num_embeddings, embedding_dim),
            shardable={0: "tp"},  # vocab维度被TP切分
            dtype=dtype,
            name="embedding_weight",
        )
    
    def forward(self, input_ids: ShardedTensor) -> ShardedTensor:
        """Embedding lookup
        
        Args:
            input_ids: (batch, seq) 或更复杂的shape
        
        Returns:
            output: (batch, seq, embedding_dim)
        """
        # 输入shape
        input_shape = input_ids.shape
        output_shape = (*input_shape, self.embedding_dim)
        
        # 输出切分约束:
        # - embedding_dim维度不切分（除非hidden也被TP切分，但那是下一层）
        # - 如果input_ids的seq维度被SP切分，输出也继承
        output_shardable = {}
        if len(input_shape) >= 2 and 1 in input_ids.shardable:
            output_shardable[len(input_shape) - 1] = input_ids.shardable[1]  # seq维度
        
        # 如果embedding_dim也被TP切分（与hidden共享切分）
        if self.ctx.tp_degree > 1 and self.embedding_dim == self.ctx.hidden_size:
            output_shardable[len(output_shape) - 1] = "tp"
        
        output = ShardedTensor(
            shape=output_shape,
            shardable=output_shardable,
            dtype=self.weight.dtype,
            name="embedding_output",
        )
        
        # 记录操作历史（EmbeddingOp）
        output._op_history = input_ids._op_history + [
            EmbeddingOp(input_ids, self.weight, output)
        ]
        
        return output
```

### 9.2 ShardedRMSNorm

```python
class ShardedRMSNorm(ShardedModule):
    """RMS Normalization层
    
    切分方式:
    - hidden维度不切分（权重完整复制）
    
    Args:
        hidden_size: hidden维度大小
        dtype: 数据类型
    """
    
    def __init__(
        self,
        hidden_size: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # RMSNorm权重: (hidden) - 不切分
        self.weight = ShardedTensor(
            shape=(hidden_size,),
            shardable={},  # 不切分
            dtype=dtype,
            name="rmsnorm_weight",
        )
    
    def forward(self, input: ShardedTensor) -> ShardedTensor:
        """RMS Normalization
        
        Args:
            input: (..., hidden_size)
        
        Returns:
            output: (..., hidden_size)，切分约束保持不变
        """
        # 输出shape和切分约束保持不变
        output = ShardedTensor(
            shape=input.shape,
            shardable=input.shardable,  # 保持输入的切分约束
            dtype=input.dtype,
            name="rmsnorm_output",
        )
        
        # 记录操作历史
        output._op_history = input._op_history + [
            RMSNormOp(input, self.weight, output)
        ]
        
        return output
```

### 9.3 ShardedAttention

```python
class ShardedAttention(ShardedModule):
    """Attention层
    
    切分方式:
    - Q/K/V weights: heads维度被TP切分（列切分）
    - O weight: heads维度被TP切分（行切分）
    - 输出需要AllReduce聚合
    
    支持GQA: num_kv_heads < num_heads
    
    Args:
        hidden_size: hidden维度
        num_heads: attention heads数量
        num_kv_heads: KV heads数量（GQA）
        head_dim: head维度
        dtype: 数据类型
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        
        # Q weight: (hidden, num_heads * head_dim)
        # hidden维度可被TP切分，输出heads维度被TP切分
        self.q_weight = ShardedTensor(
            shape=(hidden_size, num_heads * self.head_dim),
            shardable={1: "tp"},  # 输出维度（heads）被TP切分
            dtype=dtype,
            name="q_weight",
        )
        
        # K weight: (hidden, num_kv_heads * head_dim)
        self.k_weight = ShardedTensor(
            shape=(hidden_size, self.num_kv_heads * self.head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="k_weight",
        )
        
        # V weight: (hidden, num_kv_heads * head_dim)
        self.v_weight = ShardedTensor(
            shape=(hidden_size, self.num_kv_heads * self.head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="v_weight",
        )
        
        # O weight: (num_heads * head_dim, hidden)
        # heads维度被TP切分（行切分），hidden不切分
        self.o_weight = ShardedTensor(
            shape=(num_heads * self.head_dim, hidden_size),
            shardable={0: "tp"},  # heads维度被TP切分（行切分）
            dtype=dtype,
            name="o_weight",
        )
    
    def forward(
        self,
        hidden: ShardedTensor,  # (batch, seq, hidden)
        is_causal: bool = True,
    ) -> ShardedTensor:
        """Attention forward
        
        流程:
        1. Q/K/V projection: hidden @ q/k/v_weight
           - 输出shape: (batch, seq, heads * head_dim)
           - 输出切分: heads维度被TP切分
        2. Reshape to (batch, heads, seq, head_dim)
        3. Flash Attention
           - 如果启用SP，seq维度切分，需要All2All
        4. Reshape back to (batch, seq, heads * head_dim)
        5. O projection: attn_out @ o_weight
           - 输出shape: (batch, seq, hidden)
           - 输出切分: 不切分（AllReduce聚合）
        
        Returns:
            output: (batch, seq, hidden)
        """
        batch, seq, hidden = hidden.shape
        
        # ===== Input RMSNorm =====
        # (通常Attention层前有norm，这里假设已经在外部处理)
        
        # ===== Q/K/V Projections =====
        # Q: (batch, seq, hidden) @ (hidden, num_heads * head_dim)
        #    -> (batch, seq, num_heads * head_dim)
        #    切分: hidden维度可能被TP切分，输出heads维度被TP切分
        q_proj = hidden @ self.q_weight
        k_proj = hidden @ self.k_weight
        v_proj = hidden @ self.v_weight
        
        # ===== Reshape to (batch, heads, seq, head_dim) =====
        q = q_proj.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_proj.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # ===== Flash Attention =====
        attn_out = flash_attention(q, k, v, is_causal=is_causal)
        # 输出: (batch, num_heads, seq, head_dim)
        # 切分: heads维度被TP切分，seq维度可能被SP切分
        
        # ===== Reshape back =====
        attn_flat = attn_out.transpose(1, 2).view(batch, seq, self.num_heads * self.head_dim)
        
        # ===== O Projection =====
        # (batch, seq, heads * head_dim) @ (heads * head_dim, hidden)
        #    -> (batch, seq, hidden)
        #    切分: heads维度是行切分，输出hidden不切分（AllReduce聚合）
        output = attn_flat @ self.o_weight
        
        return output
```

### 9.4 ShardedFFN

```python
class ShardedFFN(ShardedModule):
    """Feed-Forward Network层
    
    切分方式:
    - gate/up weights: intermediate维度被TP切分（列切分）
    - down weight: intermediate维度被TP切分（行切分）
    - 输出需要AllReduce聚合
    
    Args:
        hidden_size: hidden维度
        intermediate_size: intermediate维度
        dtype: 数据类型
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Gate weight: (hidden, intermediate) - 列切分
        self.gate_weight = ShardedTensor(
            shape=(hidden_size, intermediate_size),
            shardable={1: "tp"},  # intermediate维度被TP切分
            dtype=dtype,
            name="gate_weight",
        )
        
        # Up weight: (hidden, intermediate) - 列切分
        self.up_weight = ShardedTensor(
            shape=(hidden_size, intermediate_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="up_weight",
        )
        
        # Down weight: (intermediate, hidden) - 行切分
        self.down_weight = ShardedTensor(
            shape=(intermediate_size, hidden_size),
            shardable={0: "tp"},  # intermediate维度被TP切分（行切分）
            dtype=dtype,
            name="down_weight",
        )
    
    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """FFN forward (SwiGLU variant)
        
        流程:
        1. gate_proj = hidden @ gate_weight -> SiLU
        2. up_proj = hidden @ up_weight
        3. intermediate = gate_proj * up_proj
        4. output = intermediate @ down_weight -> AllReduce
        
        Returns:
            output: (batch, seq, hidden)
        """
        # Gate projection + SiLU
        gate_proj = hidden @ self.gate_weight
        gate_proj = silu(gate_proj)
        
        # Up projection
        up_proj = hidden @ self.up_weight
        
        # Element-wise multiply
        intermediate = gate_proj * up_proj  # 切分约束保持
        
        # Down projection + AllReduce
        output = intermediate @ self.down_weight
        
        return output
```

### 9.5 ShardedMoE

```python
class ShardedMoE(ShardedModule):
    """Mixture of Experts层
    
    切分方式:
    - Router: 不切分（完整复制）
    - Expert weights: experts维度被EP切分，intermediate维度被TP切分
    - Shared experts: 不切分（完整复制）
    - 需要 All2All (dispatch + combine) + AllReduce
    
    Args:
        hidden_size: hidden维度
        intermediate_size: expert intermediate维度
        num_experts: expert数量
        num_experts_per_token: 每个token激活的expert数量
        shared_expert_intermediate: shared expert的intermediate维度
        dtype: 数据类型
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int = 1,
        shared_expert_intermediate: Optional[int] = None,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.shared_expert_intermediate = shared_expert_intermediate
        
        # Router weight: (hidden, num_experts) - 不切分
        self.router_weight = ShardedTensor(
            shape=(hidden_size, num_experts),
            shardable={},  # 不切分
            dtype=dtype,
            name="router_weight",
        )
        
        # Expert weights (EP + TP 切分)
        # gate/up: (hidden, intermediate) per expert
        # down: (intermediate, hidden) per expert
        # 这里用一个虚拟的ShardedTensor表示所有expert的权重
        self.expert_gate_weights = ShardedTensor(
            shape=(num_experts, hidden_size, intermediate_size),
            shardable={0: "ep", 2: "tp"},  # experts被EP切分，intermediate被TP切分
            dtype=dtype,
            name="expert_gate_weights",
        )
        self.expert_up_weights = ShardedTensor(
            shape=(num_experts, hidden_size, intermediate_size),
            shardable={0: "ep", 2: "tp"},
            dtype=dtype,
            name="expert_up_weights",
        )
        self.expert_down_weights = ShardedTensor(
            shape=(num_experts, intermediate_size, hidden_size),
            shardable={0: "ep", 1: "tp"},  # experts被EP切分，intermediate被TP切分（行切分）
            dtype=dtype,
            name="expert_down_weights",
        )
        
        # Shared expert weights (如果存在)
        if shared_expert_intermediate:
            self.shared_gate_weight = ShardedTensor(
                shape=(hidden_size, shared_expert_intermediate),
                shardable={1: "tp"},  # 只被TP切分，不参与EP
                dtype=dtype,
                name="shared_gate_weight",
            )
            self.shared_up_weight = ShardedTensor(
                shape=(hidden_size, shared_expert_intermediate),
                shardable={1: "tp"},
                dtype=dtype,
                name="shared_up_weight",
            )
            self.shared_down_weight = ShardedTensor(
                shape=(shared_expert_intermediate, hidden_size),
                shardable={0: "tp"},
                dtype=dtype,
                name="shared_down_weight",
            )
    
    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """MoE forward
        
        流程:
        1. Router: hidden @ router_weight -> softmax -> top-k
           - 输出切分: 不切分
        2. All2All dispatch: 将tokens分发到对应expert所在的GPU
           - EP通信
        3. Expert computation (本地experts)
           - gate/up/down projection，需要AllReduce (TP)
        4. All2All combine: 将expert结果聚合回来
           - EP通信
        5. Shared expert (如果存在)
           - gate/up/down projection
        6. Output = routed_output + shared_output
        
        Returns:
            output: (batch, seq, hidden)
        """
        # Router (不切分)
        router_logits = hidden @ self.router_weight
        # router_probs = softmax(router_logits)  # 暂不实现softmax
        
        # Expert computation (EP + TP)
        # 这里简化表示，实际需要复杂的dispatch/combine逻辑
        expert_out = moe_expert_compute(
            hidden,
            self.expert_gate_weights,
            self.expert_up_weights,
            self.expert_down_weights,
            self.num_experts_per_token,
        )
        # expert_out 自动推导:
        # - All2All dispatch (EP)
        # - Expert FFN (TP AllReduce)
        # - All2All combine (EP)
        
        # Shared expert (如果存在)
        if self.shared_expert_intermediate:
            shared_gate = hidden @ self.shared_gate_weight
            shared_gate = silu(shared_gate)
            shared_up = hidden @ self.shared_up_weight
            shared_intermediate = shared_gate * shared_up
            shared_out = shared_intermediate @ self.shared_down_weight
            
            # Add routed + shared
            output = expert_out + shared_out
        else:
            output = expert_out
        
        return output
```

### 9.6 ShardedLMHead

```python
class ShardedLMHead(ShardedModule):
    """LM Head层
    
    切分方式:
    - vocab维度被TP切分
    - 推理时输出需要AllGather
    
    Args:
        hidden_size: hidden维度
        vocab_size: 词表大小
        dtype: 数据类型
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # LM Head weight: (hidden, vocab) - vocab被TP切分（列切分）
        self.weight = ShardedTensor(
            shape=(hidden_size, vocab_size),
            shardable={1: "tp"},  # vocab维度被TP切分
            dtype=dtype,
            name="lm_head_weight",
        )
    
    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """LM Head forward
        
        Args:
            hidden: (batch, seq, hidden_size)
        
        Returns:
            logits: (batch, seq, vocab_size)
        """
        logits = hidden @ self.weight
        # 输出切分: vocab维度被TP切分
        # 推理时需要AllGather聚合完整vocab
        
        return logits
```

---

## 10. 组合模块

### 10.1 ShardedTransformerBlock

```python
class ShardedTransformerBlock(ShardedModule):
    """Transformer Block
    
    组合: Attention + FFN + RMSNorms
    
    Args:
        hidden_size: hidden维度大小
        num_heads: attention heads数量
        num_kv_heads: KV heads数量（GQA，可选）
        head_dim: head维度大小（可选）
        intermediate_size: FFN intermediate维度大小
        dtype: 数据类型（默认fp16）
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        intermediate_size: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        
        # 自动计算默认值
        if num_kv_heads is None:
            num_kv_heads = num_heads  # 不使用GQA
        if head_dim is None:
            head_dim = hidden_size // num_heads
        
        # 保存配置（用于后续分析）
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        
        # 创建子模块
        self.input_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        self.attention = ShardedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.post_attn_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        self.ffn = ShardedFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )
    
    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Transformer block forward
        
        流程:
        1. input_norm = RMSNorm(hidden)
        2. attn_out = Attention(input_norm)
        3. hidden = hidden + attn_out (residual)
        4. post_attn_norm = RMSNorm(hidden)
        5. ffn_out = FFN(post_attn_norm)
        6. output = hidden + ffn_out (residual)
        """
        # Pre-norm + Attention + Residual
        norm_out = self.input_norm(hidden)
        attn_out = self.attention(norm_out)
        hidden = hidden + attn_out
        
        # Post-norm + FFN + Residual
        norm_out = self.post_attn_norm(hidden)
        ffn_out = self.ffn(norm_out)
        output = hidden + ffn_out
        
        # 记录激活（用于内存估算）
        self._activations["attn_out"] = attn_out
        self._activations["ffn_out"] = ffn_out
        
        return output
    
    def params_count_breakdown(self) -> Dict[str, int]:
        """参数量分解"""
        return {
            "input_norm": self.input_norm.params_count(),
            "attention": self.attention.params_count(),
            "post_attn_norm": self.post_attn_norm.params_count(),
            "ffn": self.ffn.params_count(),
        }
```

### 10.2 ShardedMoEBlock

```python
class ShardedMoEBlock(ShardedModule):
    """MoE Transformer Block
    
    组合: Attention + MoE + RMSNorms
    
    Args:
        hidden_size: hidden维度大小
        num_heads: attention heads数量
        num_kv_heads: KV heads数量（可选）
        head_dim: head维度大小（可选）
        intermediate_size: expert intermediate维度大小
        num_experts: expert数量
        num_experts_per_token: 每个token激活的expert数量
        shared_expert_intermediate: shared expert的intermediate维度（可选）
        dtype: 数据类型（默认fp16）
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int = 1,
        shared_expert_intermediate: Optional[int] = None,
        dtype: str = "fp16",
    ):
        super().__init__()
        
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.shared_expert_intermediate = shared_expert_intermediate
        self.dtype = dtype
        
        self.input_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        self.attention = ShardedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.post_attn_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        self.moe = ShardedMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            shared_expert_intermediate=shared_expert_intermediate,
            dtype=dtype,
        )
    
    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """MoE block forward"""
        # Pre-norm + Attention + Residual
        norm_out = self.input_norm(hidden)
        attn_out = self.attention(norm_out)
        hidden = hidden + attn_out
        
        # Post-norm + MoE + Residual
        norm_out = self.post_attn_norm(hidden)
        moe_out = self.moe(norm_out)
        output = hidden + moe_out
        
        # 记录激活
        self._activations["attn_out"] = attn_out
        self._activations["moe_out"] = moe_out
        
        return output
    
    def params_count_breakdown(self) -> Dict[str, int]:
        """参数量分解"""
        return {
            "input_norm": self.input_norm.params_count(),
            "attention": self.attention.params_count(),
            "post_attn_norm": self.post_attn_norm.params_count(),
            "moe": self.moe.params_count(),
        }

---

## 11. 完整模型示例

### 11.1 LlamaModel

```python
class LlamaModel(ShardedModule):
    """Llama模型
    
    结构:
    - Embedding
    - N x TransformerBlock
    - Final RMSNorm
    - LM Head
    
    Args:
        vocab_size: 词表大小
        hidden_size: hidden维度大小
        num_layers: Transformer层数
        num_heads: attention heads数量
        num_kv_heads: KV heads数量（GQA，可选）
        intermediate_size: FFN intermediate维度大小
        head_dim: head维度大小（可选）
        max_seq_len: 最大序列长度（可选）
        dtype: 数据类型（默认fp16）
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_seq_len: int = 4096,
        dtype: str = "fp16",
    ):
        super().__init__()
        
        # 自动计算默认值
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        if intermediate_size is None:
            # Llama默认: intermediate_size ≈ hidden_size * 8/3
            intermediate_size = int(hidden_size * 8 / 3)
        
        # 保存配置
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        
        # Embedding
        self.embedding = ShardedEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
        )
        
        # Transformer layers
        self.layers = [
            ShardedTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        
        # Final norm
        self.final_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        
        # LM Head
        self.lm_head = ShardedLMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dtype=dtype,
        )
    
    def forward(self, input_ids: ShardedTensor) -> ShardedTensor:
        """Llama forward
        
        Args:
            input_ids: (batch, seq)
        
        Returns:
            logits: (batch, seq, vocab_size)
        """
        # Embedding
        hidden = self.embedding(input_ids)
        self._activations["embedding_output"] = hidden
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            self._activations[f"layer_{i}_output"] = hidden
        
        # Final norm
        hidden = self.final_norm(hidden)
        self._activations["final_norm_output"] = hidden
        
        # LM Head
        logits = self.lm_head(hidden)
        self._activations["lm_head_output"] = logits
        
        return logits
    
    def bind(self, ctx: ParallelContext, pp_strategy: Optional[PPStrategy] = None) -> Union[ModelInstance, PPModel]:
        """绑定到ParallelContext
        
        Args:
            ctx: ParallelContext
            pp_strategy: 可选的PP策略
        
        Returns:
            如果pp_strategy为None: 返回ModelInstance
            如果pp_strategy不为None: 返回PPModel（支持stage级别分析）
        """
        if pp_strategy is None:
            return ModelInstance(self, ctx)
        else:
            return PPModel(self, pp_strategy)


# ===== 使用示例 =====

# 1. 定义模型（直接参数入参，类似torch.nn）
model = LlamaModel(
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,  # GQA
    intermediate_size=11008,
    max_seq_len=4096,
    dtype="fp16",
)

# 2. 查看参数量（自动计算）
print(f"Total params: {model.params_count() / 1e9:.2f}B")
print(f"Params breakdown: {model.params_count_breakdown()}")

# 3. 定义并行策略
ctx = ParallelContext(
    tp_degree=8,
    sp_degree=4,
    pp_degree=1,
    ep_degree=1,
    dp_degree=1,
    sp_type=SPType.ULYSSES,
    dtype="fp16",
)

# 4. 绑定模型
model_instance = model.bind(ctx)

# 5. 获取分析结果
print(f"Logical params: {model.params_count() / 1e9:.2f}B")
print(f"Physical params per GPU: {model_instance.params_count_physical / 1e9:.2f}B")
print(f"Forward FLOPs: {model_instance.flops_forward_physical / 1e9:.2f}G")
print(f"Total FLOPs (forward+backward): {model_instance.flops_total_physical / 1e9:.2f}G")

# 6. 估算时间
backend = TheoryBackend(device=h100_device)
time = model_instance.estimate_time(backend)
print(f"Total time per token: {time * 1000:.2f}ms")

# 7. 估算内存
memory = model_instance.estimate_memory(batch_size=1)
print(f"Memory per GPU: {memory / 1e9:.2f}GB")

# 8. 输出详细分析
result = model_instance.to_dict()
print(json.dumps(result, indent=2))
```

### 11.2 DeepSeekModel

```python
class DeepSeekModel(ShardedModule):
    """DeepSeek模型（MLA + MoE）
    
    特点:
    - Multi-head Latent Attention (MLA): KV压缩
    - MoE: Expert + Shared Expert
    - 前几层是普通Attention，后几层是MoE
    
    Args:
        vocab_size: 词表大小
        hidden_size: hidden维度大小
        num_layers: Transformer层数
        num_heads: attention heads数量
        num_kv_heads: KV heads数量（可选）
        intermediate_size: FFN intermediate维度大小
        head_dim: head维度大小（可选）
        first_k_dense_layers: 前几层使用普通FFN
        num_experts: expert数量
        num_experts_per_token: 每个token激活的expert数量
        shared_expert_intermediate: shared expert的intermediate维度
        max_seq_len: 最大序列长度
        dtype: 数据类型
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        head_dim: Optional[int] = None,
        first_k_dense_layers: int = 1,
        num_experts: int = 64,
        num_experts_per_token: int = 8,
        shared_expert_intermediate: Optional[int] = None,
        max_seq_len: int = 4096,
        dtype: str = "fp16",
    ):
        super().__init__()
        
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        
        # Embedding
        self.embedding = ShardedEmbedding(vocab_size, hidden_size, dtype)
        
        # Transformer layers（前几层是普通Attention+FFN，后几层是MLA+MoE）
        self.layers = []
        for i in range(num_layers):
            if i < first_k_dense_layers:
                # 普通 Attention + FFN
                self.layers.append(
                    ShardedTransformerBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        intermediate_size=intermediate_size or hidden_size * 8 // 3,
                        dtype=dtype,
                    )
                )
            else:
                # MLA + MoE
                self.layers.append(
                    ShardedMoEBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        intermediate_size=intermediate_size or hidden_size * 8 // 3,
                        num_experts=num_experts,
                        num_experts_per_token=num_experts_per_token,
                        shared_expert_intermediate=shared_expert_intermediate,
                        dtype=dtype,
                    )
                )
        
        # Final norm + LM Head
        self.final_norm = ShardedRMSNorm(hidden_size, dtype)
        self.lm_head = ShardedLMHead(hidden_size, vocab_size, dtype)
    
    def forward(self, input_ids: ShardedTensor) -> ShardedTensor:
        """DeepSeek forward"""
        hidden = self.embedding(input_ids)
        self._activations["embedding_output"] = hidden
        
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            self._activations[f"layer_{i}_output"] = hidden
        
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        
        return logits


# ===== 使用示例 =====

# DeepSeek-V2 236B配置
deepseek_model = DeepSeekModel(
    vocab_size=102400,
    hidden_size=5120,
    num_layers=60,
    num_heads=128,
    num_kv_heads=128,
    intermediate_size=5120 * 8 // 3,
    first_k_dense_layers=1,
    num_experts=160,
    num_experts_per_token=8,
    shared_expert_intermediate=5120 * 2,  # shared expert
    max_seq_len=4096,
    dtype="fp16",
)

print(f"DeepSeek params: {deepseek_model.params_count() / 1e9:.2f}B")
```
---

## 12. Pipeline Parallelism 独立策略

### 12.1 PP策略设计

PP需要独立的策略表达，因为涉及：
1. **Stage划分**：哪些层在哪个stage
2. **Virtual PP (vpp)**：一个GPU负责多个stage
3. **Schedule**：1F1B, GPipe, Interleaved等调度策略
4. **Micro-batch数量**：影响bubble time

```python
class PPStrategy:
    """Pipeline Parallelism 独立策略
    
    Attributes:
        num_stages: PP stage数量（物理划分）
        num_virtual_stages: vpp数量（一个GPU负责多少个stage）
        stage_assignment: 层到stage的映射
        schedule: 调度策略 (1f1b, gpipe, interleaved, vpp)
        num_micro_batches: micro-batch数量
        micro_batch_size: 每个micro-batch的batch size
    """
    
    def __init__(
        self,
        num_stages: int = 1,
        num_virtual_stages: int = 1,  # vpp: 每个GPU负责的stage数
        schedule: str = "1f1b",  # "1f1b", "gpipe", "interleaved", "vpp"
        num_micro_batches: int = 1,
        micro_batch_size: int = 1,
        stage_assignment: Optional[Dict[str, int]] = None,  # 自定义层划分
    ):
        self.num_stages = num_stages
        self.num_virtual_stages = num_virtual_stages
        self.schedule = schedule
        self.num_micro_batches = num_micro_batches
        self.micro_batch_size = micro_batch_size
        self.stage_assignment = stage_assignment or {}
    
    def assign_layers(
        self,
        model: ShardedModule,
        method: str = "balanced",  # "balanced", "custom", "memory_balanced"
    ) -> Dict[str, int]:
        """自动或手动分配层到stage
        
        Args:
            model: ShardedModule模型
            method: 分配方法
                - "balanced": 按层数平均分配
                - "memory_balanced": 按内存占用分配
                - "custom": 使用stage_assignment
        
        Returns:
            {layer_name: stage_idx}
        """
        if method == "custom":
            return self.stage_assignment
        
        # 获取所有层
        layers = []
        for name, submodule in model._submodules.items():
            if isinstance(submodule, (ShardedTransformerBlock, ShardedMoEBlock)):
                layers.append(name)
        
        if method == "balanced":
            # 按层数平均分配
            assignment = {}
            layers_per_stage = len(layers) // self.num_stages
            for i, layer_name in enumerate(layers):
                assignment[layer_name] = i // layers_per_stage
            return assignment
        
        elif method == "memory_balanced":
            # 按内存占用分配（考虑参数量和激活量）
            assignment = {}
            layer_memory = {}
            for name, submodule in model._submodules.items():
                if isinstance(submodule, (ShardedTransformerBlock, ShardedMoEBlock)):
                    # 简化：只考虑参数量
                    layer_memory[name] = submodule.params_count()
            
            # Greedy分配：让每个stage的内存尽量平衡
            stage_memory = [0] * self.num_stages
            for layer_name in sorted(layer_memory.keys(), key=lambda x: layer_memory[x], reverse=True):
                # 找内存最小的stage
                min_stage = min(range(self.num_stages), key=lambda s: stage_memory[s])
                assignment[layer_name] = min_stage
                stage_memory[min_stage] += layer_memory[layer_name]
            
            return assignment
    
    def get_bubble_ratio(self) -> float:
        """计算bubble time比例
        
        不同schedule的bubble ratio:
        - GPipe: (num_stages - 1) / num_micro_batches
        - 1F1B: (num_stages - 1) / (num_stages + num_micro_batches - 1)
        - Interleaved 1F1B: (num_stages - 1) / (num_stages * num_virtual_stages + num_micro_batches - 1)
        - VPP: 更复杂的计算
        """
        if self.schedule == "gpipe":
            return (self.num_stages - 1) / self.num_micro_batches
        
        elif self.schedule == "1f1b":
            # 1F1B reduces bubble compared to GPipe
            return (self.num_stages - 1) / (self.num_stages + self.num_micro_batches - 1)
        
        elif self.schedule == "interleaved" or self.schedule == "vpp":
            # Interleaved 1F1B further reduces bubble with vpp
            effective_stages = self.num_stages * self.num_virtual_stages
            return (self.num_stages - 1) / (effective_stages + self.num_micro_batches - 1)
        
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_stages": self.num_stages,
            "num_virtual_stages": self.num_virtual_stages,
            "schedule": self.schedule,
            "num_micro_batches": self.num_micro_batches,
            "micro_batch_size": self.micro_batch_size,
            "bubble_ratio": self.get_bubble_ratio(),
            "stage_assignment": self.stage_assignment,
        }


# ===== 使用示例 =====

# 1. 简单PP（按层数平均划分）
pp_strategy = PPStrategy(
    num_stages=4,
    schedule="1f1b",
    num_micro_batches=8,
)

# 2. 内存平衡划分
pp_strategy = PPStrategy(
    num_stages=4,
    schedule="1f1b",
    num_micro_batches=8,
)
assignment = pp_strategy.assign_layers(model, method="memory_balanced")

# 3. 自定义划分（手动指定哪些层在哪个stage）
pp_strategy = PPStrategy(
    num_stages=4,
    schedule="1f1b",
    num_micro_batches=8,
    stage_assignment={
        "layers.0": 0,
        "layers.1": 0,
        "layers.2": 1,
        "layers.3": 1,
        "layers.4": 2,
        "layers.5": 2,
        "layers.6": 3,
        "layers.7": 3,
    },
)

# 4. VPP（一个GPU负责2个stage）
pp_strategy = PPStrategy(
    num_stages=8,  # 8个stage
    num_virtual_stages=2,  # 每个GPU负责2个stage，实际需要4个GPU
    schedule="interleaved",  # interleaved 1F1B
    num_micro_batches=16,
)

# 5. 查看bubble比例
print(f"Bubble ratio: {pp_strategy.get_bubble_ratio():.2%}")
```

### 12.2 PPStageModule

```python
class PPStageModule(ShardedModule):
    """PP Stage模块 - 代表一个PP stage包含的层
    
    Attributes:
        stage_idx: stage索引
        layers: 该stage包含的层
        pp_strategy: PP策略
    """
    
    def __init__(
        self,
        stage_idx: int,
        layers: List[ShardedModule],
        pp_strategy: PPStrategy,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.pp_strategy = pp_strategy
        
        # 注册层
        for i, layer in enumerate(layers):
            setattr(self, f"layer_{i}", layer)
    
    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Stage forward"""
        for name, layer in self._submodules.items():
            hidden = layer(hidden)
        return hidden
    
    def get_layers(self) -> List[ShardedModule]:
        """获取该stage的所有层"""
        return list(self._submodules.values())


class PPModel(ShardedModule):
    """PP模型 - 将完整模型划分到多个stage
    
    将LlamaModel等模型按PP策略划分成多个PPStageModule
    """
    
    def __init__(
        self,
        model: ShardedModule,
        pp_strategy: PPStrategy,
    ):
        super().__init__()
        self.original_model = model
        self.pp_strategy = pp_strategy
        
        # 获取层分配
        assignment = pp_strategy.assign_layers(model, method="custom" if pp_strategy.stage_assignment else "balanced")
        
        # 构建各stage的层列表
        stage_layers: Dict[int, List[ShardedModule]] = {}
        for name, submodule in model._submodules.items():
            if name in assignment:
                stage_idx = assignment[name]
                if stage_idx not in stage_layers:
                    stage_layers[stage_idx] = []
                stage_layers[stage_idx].append(submodule)
        
        # 创建PPStageModule
        for stage_idx, layers in stage_layers.items():
            setattr(self, f"stage_{stage_idx}", PPStageModule(stage_idx, layers, pp_strategy))
    
    def get_stage(self, stage_idx: int) -> PPStageModule:
        """获取指定stage"""
        return self._submodules[f"stage_{stage_idx}"]
    
    def get_all_stages(self) -> List[PPStageModule]:
        """获取所有stage"""
        return [
            self._submodules[f"stage_{i}"]
            for i in range(self.pp_strategy.num_stages)
        ]
    
    def bind_stage(self, stage_idx: int, ctx: ParallelContext) -> ModuleInstance:
        """绑定单个stage到ParallelContext
        
        用于计算单个GPU的性能（考虑vpp时，一个GPU可能有多个stage）
        """
        stage = self.get_stage(stage_idx)
        return stage.bind(ctx)
    
    def bind_all_stages(self, ctx: ParallelContext) -> List[ModuleInstance]:
        """绑定所有stage"""
        return [self.bind_stage(i, ctx) for i in range(self.pp_strategy.num_stages)]
    
    def estimate_pp_time(self, backend: KernelBackend) -> Dict[str, float]:
        """估算PP整体时间
        
        考虑:
        1. 各stage的时间
        2. Bubble time
        3. Stage间通信时间
        4. VPP的影响
        """
        # 获取各stage时间
        stage_times = []
        for stage in self.get_all_stages():
            stage_instance = stage.bind(self.ctx)
            stage_time = stage_instance.estimate_time(backend)
            stage_times.append(stage_time)
        
        # Stage间通信时间（P2P）
        activation_bytes = self._estimate_activation_bytes_between_stages()
        comm_time_per_stage = self.ctx.cluster.estimate_p2p_time(activation_bytes)
        
        # 总时间计算（考虑schedule）
        if self.pp_strategy.schedule == "gpipe":
            # GPipe: 所有micro-batch串行执行，最后有bubble
            ideal_time = max(stage_times) * self.pp_strategy.num_micro_batches
            bubble_time = ideal_time * self.pp_strategy.get_bubble_ratio()
            total_time = ideal_time + bubble_time + comm_time_per_stage * (self.pp_strategy.num_stages - 1)
        
        elif self.pp_strategy.schedule == "1f1b":
            # 1F1B: bubble更小
            ideal_time = max(stage_times) * self.pp_strategy.num_micro_batches
            bubble_time = ideal_time * self.pp_strategy.get_bubble_ratio()
            total_time = ideal_time + bubble_time + comm_time_per_stage * (self.pp_strategy.num_stages - 1)
        
        elif self.pp_strategy.schedule in ["interleaved", "vpp"]:
            # Interleaved 1F1B with VPP
            # 一个GPU负责多个stage，通信时间增加
            ideal_time = max(stage_times) * self.pp_strategy.num_micro_batches * self.pp_strategy.num_virtual_stages
            bubble_time = ideal_time * self.pp_strategy.get_bubble_ratio()
            # VPP的通信更复杂（内部stage切换）
            vpp_comm_time = comm_time_per_stage * (self.pp_strategy.num_stages - 1) * self.pp_strategy.num_virtual_stages
            total_time = ideal_time + bubble_time + vpp_comm_time
        
        return {
            "ideal_time_sec": ideal_time,
            "bubble_time_sec": bubble_time,
            "bubble_ratio": self.pp_strategy.get_bubble_ratio(),
            "stage_comm_time_sec": comm_time_per_stage * (self.pp_strategy.num_stages - 1),
            "total_time_sec": total_time,
            "throughput_tokens_per_sec": self.pp_strategy.micro_batch_size * self.pp_strategy.num_micro_batches / total_time,
        }
    
    def _estimate_activation_bytes_between_stages(self) -> int:
        """估算stage间的激活大小"""
        # 假设hidden tensor
        hidden_size = self.original_model.config.hidden_size
        seq_len = self.original_model.config.max_seq_len
        dtype_size = DTYPE_SIZES[self.ctx.dtype]
        
        return self.pp_strategy.micro_batch_size * seq_len * hidden_size * dtype_size


# ===== 使用示例 =====

# 1. 创建模型
model = LlamaModel(llama_config)

# 2. 定义PP策略
pp_strategy = PPStrategy(
    num_stages=4,
    schedule="1f1b",
    num_micro_batches=8,
    micro_batch_size=4,
)

# 3. 创建PP模型
pp_model = PPModel(model, pp_strategy)

# 4. 查看stage划分
print(f"Stage 0 layers: {pp_model.get_stage(0).get_layers()}")
print(f"Stage 1 layers: {pp_model.get_stage(1).get_layers()}")

# 5. 绑定并计算单个stage的性能
ctx = ParallelContext(tp=8, pp=4, ...)
stage_0_instance = pp_model.bind_stage(0, ctx)
print(f"Stage 0 params: {stage_0_instance.params_count_physical / 1e9:.2f}B")
print(f"Stage 0 flops: {stage_0_instance.flops_total_physical / 1e9:.2f}G")

# 6. 计算PP整体性能
pp_time = pp_model.estimate_pp_time(backend)
print(f"PP total time: {pp_time['total_time_sec'] * 1000:.2f}ms")
print(f"PP bubble ratio: {pp_time['bubble_ratio']:.2%}")
print(f"PP throughput: {pp_time['throughput_tokens_per_sec']:.0f} tokens/s")

# 7. VPP示例
vpp_strategy = PPStrategy(
    num_stages=8,
    num_virtual_stages=2,  # 每个GPU负责2个stage
    schedule="interleaved",
    num_micro_batches=16,
    micro_batch_size=2,
)
vpp_model = PPModel(model, vpp_strategy)
print(f"VPP bubble ratio: {vpp_strategy.get_bubble_ratio():.2%}")
```

### 12.3 PP Schedule 详细描述

```python
class PPSchedule:
    """PP Schedule详细描述
    
    不同schedule的执行顺序:
    
    1. GPipe (朴素流水线):
       Stage 0: [F0, F1, F2, ..., F7] -> 等待 -> [B0, B1, ..., B7]
       Stage 1: 等待 -> [F0, F1, ..., F7] -> [B0, B1, ..., B7]
       ...
       Bubble: (num_stages - 1) * stage_time
    
    2. 1F1B (One Forward One Backward):
       Stage 0: [F0, F1, F2, F3] -> [F4, B0] -> [F5, B1] -> ... -> [B4, B5, B6, B7]
       Stage 1: [F0, F1, F2, F3] -> [F4, B0] -> ...
       ...
       Bubble: (num_stages - 1) * stage_time (比GPipe小)
    
    3. Interleaved 1F1B (with VPP):
       GPU 0 (Stage 0, 4): [F0_s0, F0_s4, F1_s0, F1_s4, ...] -> interleaved
       GPU 1 (Stage 1, 5): [F0_s1, F0_s5, F1_s1, F1_s5, ...] -> interleaved
       ...
       Bubble: 进一步减小
    
    4. VPP with Pipeline Engine (DeepSpeed/Megatron):
       更复杂的调度，支持多个micro-batch并行
    """
    
    @staticmethod
    def generate_gpipe_schedule(num_stages: int, num_micro_batches: int) -> List[List[str]]:
        """生成GPipe schedule
        
        Returns:
            每个stage的操作列表: [["F0", "F1", ..., "B0", "B1", ...], ...]
        """
        schedules = []
        for stage in range(num_stages):
            # Forward phase: 所有micro-batch forward
            forward_ops = [f"F{mb}" for mb in range(num_micro_batches)]
            # Backward phase: 所有micro-batch backward
            backward_ops = [f"B{mb}" for mb in range(num_micro_batches)]
            schedules.append(forward_ops + backward_ops)
        return schedules
    
    @staticmethod
    def generate_1f1b_schedule(num_stages: int, num_micro_batches: int) -> List[List[str]]:
        """生成1F1B schedule
        
        Returns:
            每个stage的操作列表
        """
        schedules = []
        for stage in range(num_stages):
            ops = []
            # Warmup phase: 先做num_stages个forward
            for mb in range(min(num_stages, num_micro_batches)):
                ops.append(f"F{mb}")
            
            # Steady state: 1F1B
            for mb in range(num_stages, num_micro_batches):
                ops.append(f"F{mb}")
                ops.append(f"B{mb - num_stages}")
            
            # Cooldown phase: 剩余backward
            for mb in range(num_micro_batches - num_stages, num_micro_batches):
                ops.append(f"B{mb}")
            
            schedules.append(ops)
        return schedules
    
    @staticmethod
    def generate_interleaved_schedule(
        num_stages: int,
        num_virtual_stages: int,
        num_micro_batches: int,
    ) -> List[List[str]]:
        """生成Interleaved 1F1B schedule (VPP)
        
        Returns:
            每个GPU的操作列表（一个GPU负责多个stage）
        """
        num_gpus = num_stages // num_virtual_stages
        schedules = []
        
        for gpu in range(num_gpus):
            ops = []
            # 该GPU负责的stage列表
            stages_for_gpu = [gpu * num_virtual_stages + v for v in range(num_virtual_stages)]
            
            # Warmup phase
            for mb in range(min(num_stages * num_virtual_stages, num_micro_batches)):
                for stage in stages_for_gpu:
                    ops.append(f"F{mb}_s{stage}")
            
            # Steady state: interleaved 1F1B
            for mb in range(num_stages * num_virtual_stages, num_micro_batches):
                for stage in stages_for_gpu:
                    ops.append(f"F{mb}_s{stage}")
                    ops.append(f"B{mb - num_stages}_s{stage}")
            
            # Cooldown phase
            for mb in range(num_micro_batches - num_stages, num_micro_batches):
                for stage in stages_for_gpu:
                    ops.append(f"B{mb}_s{stage}")
            
            schedules.append(ops)
        
        return schedules
    
    @staticmethod
    def visualize_schedule(schedules: List[List[str]], stage_names: List[str] = None):
        """可视化schedule"""
        if stage_names is None:
            stage_names = [f"Stage {i}" for i in len(schedules)]
        
        max_ops = max(len(s) for s in schedules)
        
        print("\nPP Schedule Visualization:")
        print("-" * (max_ops * 6 + 15))
        for i, (stage_name, ops) in enumerate(zip(stage_names, schedules)):
            ops_str = " ".join(f"{op:5s}" for op in ops)
            print(f"{stage_name:12s} | {ops_str}")
        print("-" * (max_ops * 6 + 15))


# ===== 使用示例 =====

# 生成并可视化不同schedule
print("GPipe Schedule:")
gpipe_schedule = PPSchedule.generate_gpipe_schedule(4, 8)
PPSchedule.visualize_schedule(gpipe_schedule)

print("\n1F1B Schedule:")
onef1b_schedule = PPSchedule.generate_1f1b_schedule(4, 8)
PPSchedule.visualize_schedule(onef1b_schedule)

print("\nInterleaved 1F1B (VPP=2) Schedule:")
interleaved_schedule = PPSchedule.generate_interleaved_schedule(8, 2, 16)
PPSchedule.visualize_schedule(interleaved_schedule, stage_names=["GPU 0", "GPU 1", "GPU 2", "GPU 3"])
```

---

## 13. 完整使用流程

```python
from llm_perf.modeling import (
    ShardedTensor,
    ShardedModule,
    ShardedEmbedding,
    ShardedAttention,
    ShardedFFN,
    ShardedMoE,
    ShardedRMSNorm,
    ShardedLMHead,
    ShardedTransformerBlock,
    ShardedMoEBlock,
    LlamaModel,
    DeepSeekModel,
    ParallelContext,
    ModelInstance,
    PPStrategy,
    PPModel,
    PPSchedule,
)
from llm_perf.kernels.backend import TheoryBackend
from llm_perf.hardware import H100Device, Cluster

# ===== 1. 定义模型（类似 PyTorch）=====
model = LlamaModel(
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,  # GQA
    intermediate_size=11008,
    max_seq_len=4096,
)

# ===== 2. 查看参数量（自动计算）=====
print(f"Total params: {model.params_count() / 1e9:.2f}B")
breakdown = model.params_count_breakdown()
for name, count in breakdown.items():
    print(f"  {name}: {count / 1e6:.2f}M")

# ===== 3. 定义并行策略 =====
ctx = ParallelContext(
    tp_degree=8,
    pp_degree=4,
    sp_degree=2,
    ep_degree=1,
    dp_degree=16,
    sp_type=SPType.ULYSSES,
    dtype="fp16",
    activation_checkpointing=False,
    zero_stage=0,
)

# ===== 4. 定义 PP 策略（可选）=====
pp_strategy = PPStrategy(
    num_stages=4,
    num_virtual_stages=1,  # 不使用vpp
    schedule="1f1b",
    num_micro_batches=8,
    micro_batch_size=4,
)

# ===== 5. 绑定模型 =====
# 不带PP策略
model_instance = model.bind(ctx)

# 带PP策略
pp_model = model.bind(ctx, pp_strategy)

# ===== 6. 获取物理形态 =====
print(f"Logical params: {model.params_count() / 1e9:.2f}B")
print(f"Physical params per GPU: {model_instance.params_count_physical / 1e9:.2f}B")
print(f"Forward FLOPs: {model_instance.flops_forward_physical / 1e9:.2f}G")
print(f"Total FLOPs: {model_instance.flops_total_physical / 1e9:.2f}G")

# ===== 7. 分析通信 =====
for comm_op in model_instance.total_comm_ops:
    print(f"{comm_op.comm_type} ({comm_op.ptype}): {comm_op.data_bytes / 1e6:.2f}MB")

# ===== 8. 估算时间 =====
device = H100Device()
cluster = Cluster(...)
backend = TheoryBackend(device=device)

total_time = model_instance.estimate_time(backend)
print(f"Total time per forward: {total_time * 1000:.2f}ms")

# ===== 9. 估算内存 =====
memory = model_instance.estimate_memory(batch_size=1)
print(f"Memory per GPU: {memory / 1e9:.2f}GB")

# ===== 10. PP 性能分析 =====
pp_time = pp_model.estimate_pp_time(backend)
print(f"PP bubble ratio: {pp_time['bubble_ratio']:.2%}")
print(f"PP total time: {pp_time['total_time_sec'] * 1000:.2f}ms")
print(f"PP throughput: {pp_time['throughput_tokens_per_sec']:.0f} tokens/s")

# ===== 11. 分析单个 PP stage =====
stage_0 = pp_model.get_stage(0)
stage_0_instance = stage_0.bind(ctx)
print(f"Stage 0 params: {stage_0_instance.params_count_physical / 1e9:.2f}B")

# ===== 12. 输出完整分析 =====
result = model_instance.to_dict()
result["performance"] = {
    "total_time_sec": total_time,
    "total_time_ms": total_time * 1000,
    "memory_bytes": memory,
    "memory_gb": memory / 1e9,
}
result["pp"] = pp_time
print(json.dumps(result, indent=2))
```

---

## 14. 实现路径

### Phase 1: 核心数据结构 (ShardedTensor, ShardedModule)
1. 实现 `ShardedTensor` 类 (shape, shardable, dtype)
2. 实现 `ShardedModule` 基类
3. 实现基本操作 (__matmul__, view, transpose)
4. 实现切分推导逻辑

### Phase 2: 基础模块
1. 实现 `ShardedEmbedding`
2. 实现 `ShardedRMSNorm`
3. 实现 `ShardedAttention`
4. 实现 `ShardedFFN`
5. 实现 `ShardedLMHead`

### Phase 3: MoE 和组合模块
1. 实现 `ShardedMoE`
2. 实现 `ShardedTransformerBlock`
3. 实现 `ShardedMoEBlock`

### Phase 4: 完整模型
1. 实现 `LlamaModel`
2. 实现 `DeepSeekModel`

### Phase 5: ParallelContext 和 ModuleInstance
1. 实现 `ParallelContext` (整合 StrategyConfig + Cluster)
2. 实现 `ModuleInstance` (物理形态推导)
3. 实现 `WeightInstance` 和 `ActivationInstance`

### Phase 6: PP 策略 (TODO - 待实现)

**状态**: 待实现

**优先级**: 高

**预计工作量**: 中等

#### 6.1 需求概述

PP (Pipeline Parallelism) 需要独立的策略表达，因为涉及：
1. **Stage划分**: 哪些层在哪个stage
2. **Virtual PP (vpp)**: 一个GPU负责多个stage
3. **Schedule**: 1F1B、GPipe、Interleaved等调度策略
4. **Micro-batch数量**: 影响bubble time
5. **与模型描述绑定**: PP stage划分需要与ShardedModule绑定

#### 6.2 核心需求

##### 6.2.1 PPStrategy - PP策略配置

**需求**:
- 支持自定义stage划分（手动指定哪些层在哪个stage）
- 支持自动stage划分（按层数平均、按内存平衡）
- 支持vpp配置（一个GPU负责多个stage）
- 支持多种schedule类型
- 支持micro-batch配置
- 计算bubble ratio

**接口设计**:
```python
class PPStrategy:
    def __init__(
        self,
        num_stages: int,  # PP stage数量（物理划分）
        num_virtual_stages: int = 1,  # vpp数量
        schedule: str = "1f1b",  # "1f1b", "gpipe", "interleaved", "vpp"
        num_micro_batches: int = 1,
        micro_batch_size: int = 1,
        stage_assignment: Optional[Dict[str, int]] = None,  # 自定义划分
    )
    
    def assign_layers(
        self,
        model: ShardedModule,
        method: str = "balanced",  # "balanced", "memory_balanced", "custom"
    ) -> Dict[str, int]:
        """自动或手动分配层到stage"""
    
    def get_bubble_ratio(self) -> float:
        """计算bubble time比例"""
```

**bubble ratio计算**:
- GPipe: `(num_stages - 1) / num_micro_batches`
- 1F1B: `(num_stages - 1) / (num_stages + num_micro_batches - 1)`
- Interleaved 1F1B: `(num_stages - 1) / (num_stages * num_virtual_stages + num_micro_batches - 1)`
- VPP: 更复杂的计算

##### 6.2.2 PPModel - PP模型包装

**需求**:
- 将完整模型按PP策略划分成多个stage
- 支持单个stage的性能分析
- 支持PP整体性能估算（含bubble time）
- 支持stage间通信时间估算
- 支持vpp的复杂通信计算

**接口设计**:
```python
class PPModel(ShardedModule):
    def __init__(
        self,
        model: ShardedModule,
        pp_strategy: PPStrategy,
    )
    
    def get_stage(self, stage_idx: int) -> PPStageModule:
        """获取指定stage"""
    
    def get_all_stages(self) -> List[PPStageModule]:
        """获取所有stage"""
    
    def bind_stage(self, stage_idx: int, ctx: ParallelContext) -> ModuleInstance:
        """绑定单个stage"""
    
    def estimate_pp_time(self, backend: KernelBackend) -> Dict[str, float]:
        """估算PP整体时间（含bubble）"""
```

**PP时间估算**:
```python
def estimate_pp_time(self, backend):
    # 1. 获取各stage时间
    stage_times = [stage.bind(ctx).estimate_time(backend) for stage in self.stages]
    
    # 2. 计算ideal time
    ideal_time = max(stage_times) * num_micro_batches
    
    # 3. 计算bubble time
    bubble_time = ideal_time * pp_strategy.get_bubble_ratio()
    
    # 4. 计算stage间通信时间
    activation_bytes = estimate_activation_bytes_between_stages()
    comm_time = cluster.estimate_p2p_time(activation_bytes) * (num_stages - 1)
    
    # 5. vpp特殊处理
    if vpp > 1:
        vpp_comm_time = comm_time * vpp
    
    return {
        "ideal_time_sec": ideal_time,
        "bubble_time_sec": bubble_time,
        "bubble_ratio": bubble_ratio,
        "stage_comm_time_sec": comm_time,
        "total_time_sec": ideal_time + bubble_time + comm_time,
        "throughput_tokens_per_sec": ...,
    }
```

##### 6.2.3 PPStageModule - PP Stage模块

**需求**:
- 代表一个PP stage包含的层
- 继承ShardedModule，支持bind()和性能分析
- 记录stage索引

**接口设计**:
```python
class PPStageModule(ShardedModule):
    def __init__(
        self,
        stage_idx: int,
        layers: List[ShardedModule],
        pp_strategy: PPStrategy,
    )
    
    def get_layers(self) -> List[ShardedModule]:
        """获取该stage的所有层"""
```

##### 6.2.4 PPSchedule - PP Schedule详细描述

**需求**:
- 生成不同schedule的操作序列
- 可视化schedule
- 支持schedule分析

**Schedule类型**:

| Schedule | 描述 | Bubble比例 | 适用场景 |
|----------|------|-----------|---------|
| GPipe | 朴素流水线，所有F然后所有B | 最高 | 简单场景 |
| 1F1B | 1个F1个B交替，减少bubble | 中等 | 常用 |
| Interleaved 1F1B | vpp下的1F1B | 最低 | 高效vpp |
| VPP | 多stage在同一GPU | 需计算 | 高密度部署 |

**接口设计**:
```python
class PPSchedule:
    @staticmethod
    def generate_gpipe_schedule(num_stages, num_micro_batches) -> List[List[str]]:
        """生成GPipe schedule"""
    
    @staticmethod
    def generate_1f1b_schedule(num_stages, num_micro_batches) -> List[List[str]]:
        """生成1F1B schedule"""
    
    @staticmethod
    def generate_interleaved_schedule(
        num_stages, num_virtual_stages, num_micro_batches
    ) -> List[List[str]]:
        """生成Interleaved 1F1B (VPP) schedule"""
    
    @staticmethod
    def visualize_schedule(schedules, stage_names):
        """可视化schedule"""
```

**Schedule详解**:

**GPipe Schedule**:
```
Stage 0: [F0, F1, F2, F3, F4, F5, F6, F7] -> [B0, B1, B2, B3, B4, B5, B6, B7]
Stage 1:      [F0, F1, F2, F3, F4, F5, F6, F7] -> [B0, B1, B2, B3, B4, B5, B6, B7]
...
Bubble: (num_stages - 1) * stage_time
```

**1F1B Schedule**:
```
Stage 0: [F0, F1, F2, F3] -> [F4, B0] -> [F5, B1] -> [F6, B2] -> [F7, B3] -> [B4, B5, B6, B7]
Stage 1:      [F0, F1, F2, F3] -> [F4, B0] -> [F5, B1] -> ...
...
Warmup: num_stages个F
Steady: 1F1B交替
Cooldown: 剩余B
Bubble: 比GPipe小
```

**Interleaved 1F1B (VPP) Schedule**:
```
GPU 0 (Stage 0, 4): [F0_s0, F0_s4, F1_s0, F1_s4, ...] -> interleaved
GPU 1 (Stage 1, 5): [F0_s1, F0_s5, F1_s1, F1_s5, ...] -> interleaved
...
Bubble: 最小
```

#### 6.3 使用示例

```python
# 1. 创建模型
model = LlamaModel(
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,
)

# 2. 定义PP策略
pp_strategy = PPStrategy(
    num_stages=4,
    num_virtual_stages=1,
    schedule="1f1b",
    num_micro_batches=8,
    micro_batch_size=4,
)

# 3. 自定义stage划分
pp_strategy_custom = PPStrategy(
    num_stages=4,
    schedule="1f1b",
    num_micro_batches=8,
    stage_assignment={
        "layers.0": 0,
        "layers.1": 0,
        "layers.2": 1,
        "layers.3": 1,
        "layers.4": 2,
        "layers.5": 2,
        "layers.6": 3,
        "layers.7": 3,
    },
)

# 4. 内存平衡划分
pp_strategy_balanced = PPStrategy(
    num_stages=4,
    schedule="1f1b",
    num_micro_batches=8,
)
assignment = pp_strategy_balanced.assign_layers(model, method="memory_balanced")

# 5. VPP配置
vpp_strategy = PPStrategy(
    num_stages=8,
    num_virtual_stages=2,  # 每GPU负责2个stage
    schedule="interleaved",
    num_micro_batches=16,
)

# 6. 创建PP模型
pp_model = PPModel(model, pp_strategy)

# 7. 分析单个stage
ctx = ParallelContext(tp_degree=8, pp_degree=4)
stage_0 = pp_model.get_stage(0)
stage_0_instance = stage_0.bind(ctx)
print(f"Stage 0 params: {stage_0_instance.params_count_physical / 1e9:.2f}B")
print(f"Stage 0 time: {stage_0_instance.estimate_time(backend) * 1000:.2f}ms")

# 8. 分析PP整体性能
pp_time = pp_model.estimate_pp_time(backend)
print(f"PP bubble ratio: {pp_time['bubble_ratio']:.2%}")
print(f"PP throughput: {pp_time['throughput_tokens_per_sec']:.0f} tokens/s")

# 9. 可视化schedule
schedule = PPSchedule.generate_1f1b_schedule(4, 8)
PPSchedule.visualize_schedule(schedule)
```

#### 6.4 实现细节

##### 6.4.1 文件结构

```
llm_perf/modeling/
├── pp_strategy.py      # PPStrategy, PPSchedule
├── pp_model.py         # PPModel, PPStageModule
└── models.py           # 已有，需要支持bind(pp_strategy)
```

##### 6.4.2 与现有代码集成

**ParallelContext扩展**:
```python
@dataclass
class ParallelContext:
    ...
    pp_strategy: Optional[PPStrategy] = None
```

**LlamaModel.bind()扩展**:
```python
class LlamaModel(ShardedModule):
    def bind(
        self,
        ctx: ParallelContext,
        pp_strategy: Optional[PPStrategy] = None,
    ) -> Union[ModuleInstance, PPModel]:
        if pp_strategy is None:
            return ModuleInstance(self, ctx)
        else:
            return PPModel(self, pp_strategy)
```

##### 6.4.3 测试需求

```python
# tests/test_modeling_pp.py

class TestPPStrategy:
    def test_pp_strategy_creation()
    def test_bubble_ratio_gpipe()
    def test_bubble_ratio_1f1b()
    def test_bubble_ratio_interleaved()
    def test_assign_layers_balanced()
    def test_assign_layers_memory_balanced()
    def test_assign_layers_custom()
    def test_vpp_bubble_ratio()

class TestPPSchedule:
    def test_generate_gpipe_schedule()
    def test_generate_1f1b_schedule()
    def test_generate_interleaved_schedule()
    def test_visualize_schedule()

class TestPPModel:
    def test_pp_model_creation()
    def test_get_stage()
    def test_get_all_stages()
    def test_bind_stage()
    def test_estimate_pp_time()
    def test_pp_time_with_vpp()

class TestPPStageModule:
    def test_pp_stage_creation()
    def test_pp_stage_bind()
    def test_pp_stage_params_count()

class TestPPIntegration:
    def test_llama_with_pp()
    def test_pp_with_tp_sp()
    def test_pp_performance_analysis()
```

#### 6.5 设计文档

完整设计已在本文档 **Section 12: Pipeline Parallelism 独立策略** 中描述。

#### 6.6 待讨论事项

1. **PP与其他并行策略组合**:
   - PP + TP 如何处理？
   - PP + SP 如何处理？
   - PP + EP 如何处理？

2. **Stage间通信带宽**:
   - 如何获取PP stage间的P2P带宽？
   - vpp的通信带宽如何处理？

3. **Activation checkpointing与PP**:
   - PP是否需要考虑activation checkpointing？
   - 如何计算PP下的activation memory？

4. **模型切分与PP**:
   - PP stage划分后，每个stage的模型是否需要独立的切分建模？
   - 还是沿用整体模型的切分约束？

5. **Dynamic scheduling**:
   - 是否需要支持动态调整micro-batch数量？
   - 是否需要支持自适应schedule选择？

### Phase 7: 测试验证

**状态**: 已完成

1. 单并行策略验证 (已完成)
2. 高维并行组合验证 (已完成)
3. PP策略验证 (待Phase 6完成后)
4. 与现有实现对比验证 (已完成)

---

## 15. 实现进度跟踪

| Phase | 内容 | 状态 | 测试数 | 完成日期 |
|-------|------|------|--------|---------|
| Phase 1 | 核心数据结构 | ✅ 完成 | 26 passed | 2026-04-17 |
| Phase 2 | 基础模块 | ✅ 完成 | 22 passed | 2026-04-17 |
| Phase 3 | MoE和组合模块 | ✅ 完成 | (Phase 4 tests) | 2026-04-17 |
| Phase 4 | 完整模型 | ✅ 完成 | 12 passed | 2026-04-17 |
| Phase 5 | ParallelContext和ModuleInstance | ✅ 完成 | (Phase 1 tests) | 2026-04-17 |
| Phase 6 | PP策略 | ✅ 完成 | 33 passed | 2026-04-17 |
| Phase 7 | 测试验证 | ✅ 完成 | 603 total passed | 2026-04-17 |
| Phase 8 | Vision/Video模块 | ✅ 完成 | 29 passed | 2026-04-17 |
| Phase 9 | Wan视频生成模型 | ✅ 完成 | 21 passed | 2026-04-17 |

**总计**: 168 tests for modeling (Phase 1-9), 485 existing tests still passing

---

## 16. 新增模型支持

### 16.1 ShardedMoE

```python
from llm_perf.modeling import ShardedMoE

moe = ShardedMoE(
    hidden_size=4096,
    intermediate_size=2048,
    num_experts=64,
    num_experts_per_token=8,
    shared_expert_intermediate=4096,  # 可选：共享专家
)
```

### 16.2 ShardedMoEBlock

```python
from llm_perf.modeling import ShardedMoEBlock

block = ShardedMoEBlock(
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    intermediate_size=2048,
    num_experts=64,
    num_experts_per_token=8,
)
```

### 16.3 DeepSeekModel

```python
from llm_perf.modeling import DeepSeekModel

model = DeepSeekModel(
    vocab_size=102400,
    hidden_size=5120,
    num_layers=60,
    num_heads=128,
    first_k_dense_layers=1,  # 前1层用Dense FFN
    num_experts=160,
    num_experts_per_token=6,
    shared_expert_intermediate=2048,
)
```

### 16.4 Vision/Video模块

**Conv层** (无切分策略):
```python
from llm_perf.modeling import ShardedConv2d, ShardedConv3d, ShardedGroupNorm

conv2d = ShardedConv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
conv3d = ShardedConv3d(64, 128, kernel_size=(3, 3, 3))  # 视频卷积
norm = ShardedGroupNorm(num_groups=32, num_channels=512)
```

**VAE模型**:
```python
from llm_perf.modeling import ShardedVAE, ShardedVAEEncoder, ShardedVAEDecoder

vae = ShardedVAE(
    in_channels=3,
    out_channels=3,
    latent_channels=4,
    block_out_channels=(128, 256, 512, 512),
    use_3d=True,  # 视频VAE
)

latent = vae.encode(video)  # 编码
reconstructed = vae.decode(latent)  # 解码
```

### 16.5 Wan视频生成模型

**Text Encoder** (umT5-XXL):
```python
from llm_perf.modeling import ShardedWanTextEncoder

encoder = ShardedWanTextEncoder(
    vocab_size=256384,
    hidden_size=4096,
    num_layers=24,
    num_heads=64,
)
text_embed = encoder(input_ids)
```

**DiT Model**:
```python
from llm_perf.modeling import ShardedWanDiT

dit = ShardedWanDiT(
    hidden_size=5120,
    num_layers=40,
    num_heads=40,
    intermediate_size=13824,
    in_channels=16,  # VAE latent channels
    text_dim=4096,
)

output = dit(latent, text_embed, time_embed)
```

**Wan VAE**:
```python
from llm_perf.modeling import ShardedWanVAE

vae = ShardedWanVAE(
    in_channels=3,
    latent_channels=16,  # Wan uses 16 channels
)
```

---

## 17. 新旧接口共存说明

### 17.1 旧接口 (`llm_perf/models/`)

**状态**: Legacy，保留供现有功能使用

**用途**:
- Web API (`web/app.py`)
- Analyzer (`llm_perf/analyzer/`)
- 场景模板 (`llm_perf/scenarios/`)
- 非LLM模型 (ResNet, VAE, WanVideo)

**关键类**:
- `BaseModel`, `ModelConfig`, `LayerConfig`
- `LlamaModel`, `DeepSeekModel`, `MoEModel` (旧版)
- `ShardingInfo`, `ShardedLayerConfig`

### 17.2 新接口 (`llm_perf/modeling/`)

**状态**: 推荐，未来的统一建模方案

**用途**:
- LLM模型性能分析
- 高维并行组合评估
- PP策略规划

**关键类**:
- `ShardedTensor`, `ShardedModule`
- `ShardedEmbedding`, `ShardedAttention`, `ShardedFFN`, `ShardedMoE`
- `LlamaModel`, `DeepSeekModel` (新版)
- `ParallelContext`, `ModuleInstance`
- `PPStrategy`, `PPModel`

### 17.3 迁移路径

**推荐迁移顺序**:
1. 新模型定义使用 `modeling` 接口
2. Analyzer 逐步迁移到 `ModuleInstance`
3. Web API 适配新接口
4. 非LLM模型考虑简化或保留旧接口
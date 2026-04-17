# 统一建模框架设计

## 1. 核心问题

当前方案存在三个关键问题：

### 1.1 切分与不切分各走一套表达
```python
# 当前：两种不同的建模方式
layer.flops  # 未切分的FLOPs
sharded_layer.sharded_flops  # 切分后的FLOPs - 单独一套

# 问题：每种切分策略都需要新的表达
# TP切分 -> build_sharded_layers_for_tp()
# TP+SP切分 -> build_sharded_layers_for_tp_sp()
# TP+EP切分 -> build_sharded_layers_for_tp_ep()
# ... 无穷组合
```

### 1.2 模型建模与开销建模分离
```python
# 模型层只定义shape，不知道通信
LayerConfig(input_shape, output_shape, flops)

# Analyzer单独计算通信开销 - 与模型切分状态不关联
analyzer._estimate_tp_communication_time()
analyzer._estimate_sp_communication_time()
```

### 1.3 不支持高维并行统一表达
```python
# 当前只能处理单一或简单组合
# 无法表达 TP=8, PP=4, EP=2, SP=2, DP=16 的任意组合
```

---

## 2. 统一建模框架

### 2.1 核心概念

#### TensorSpec - 张量规格
一个张量的**唯一逻辑定义**，包含所有维度信息：
```python
@dataclass
class TensorDim:
    """张量的一个维度"""
    name: str  # "batch", "seq", "hidden", "heads", "vocab", "experts"
    logical_size: int  # 逻辑大小（未切分）
    
    # 切分约束：这个维度可以被哪些并行策略切分
    shardable_by: Set[str] = field(default_factory=set)
    # 例如：
    # heads维度 -> shardable_by={"tp"}  (只能被TP切分)
    # seq维度 -> shardable_by={"sp"}  (只能被SP切分)
    # hidden维度 -> shardable_by={"tp", "sp"}  (可以被TP或SP切分)
    # vocab维度 -> shardable_by={"tp"}  (只能被TP切分)
    # experts维度 -> shardable_by={"ep"}  (只能被EP切分)

@dataclass
class TensorSpec:
    """张量规格 - 唯一逻辑定义"""
    dims: List[TensorDim]
    dtype: str = "fp16"
    
    # ===== 核心方法 =====
    
    def logical_shape(self) -> Tuple[int, ...]:
        """逻辑shape（未切分）"""
        return tuple(d.logical_size for d in self.dims)
    
    def physical_shape(self, ctx: ParallelContext) -> Tuple[int, ...]:
        """物理shape（根据并行上下文推导）"""
        shape = []
        for dim in self.dims:
            # 查找切分这个维度的并行策略
            sharded = False
            for ptype in dim.shardable_by:
                degree = ctx.get_degree(ptype)
                if degree > 1 and ctx.is_dim_sharded_by(dim.name, ptype):
                    shape.append(max(1, dim.logical_size // degree))
                    sharded = True
                    break
            if not sharded:
                shape.append(dim.logical_size)
        return tuple(shape)
    
    def get_sharding_info(self, ctx: ParallelContext) -> Dict[str, Any]:
        """获取切分信息"""
        return {
            "logical_shape": self.logical_shape(),
            "physical_shape": self.physical_shape(ctx),
            "sharded_dims": [
                {
                    "dim": dim.name,
                    "logical_size": dim.logical_size,
                    "physical_size": dim.logical_size // ctx.get_degree(ptype),
                    "sharded_by": ptype,
                }
                for dim in self.dims
                for ptype in dim.shardable_by
                if ctx.get_degree(ptype) > 1 and ctx.is_dim_sharded_by(dim.name, ptype)
            ],
        }
```

#### OpSpec - 算子规格
一个算子的**唯一逻辑定义**，包含输入输出和计算语义：
```python
@dataclass
class OpSpec:
    """算子规格 - 唯一逻辑定义"""
    name: str  # "linear", "attention", "softmax", "allreduce"
    op_type: str  # "compute", "communication"
    
    # 输入输出张量规格
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    
    # 计算语义（用于推导FLOPs）
    compute_semantic: str  # "matmul", "elementwise", "reduce", "flash_attn"
    
    # 权重张量（如果有）
    weight: Optional[TensorSpec] = None
    
    # ===== 核心方法 =====
    
    def logical_flops(self) -> int:
        """逻辑FLOPs（未切分）"""
        # 根据计算语义和输入输出推导
        if self.compute_semantic == "matmul":
            # matmul: 2 * M * N * K
            m, n, k = self._get_matmul_dims()
            return 2 * m * n * k
        elif self.compute_semantic == "flash_attn":
            # flash_attn: 4 * batch * seq * heads * head_dim^2
            b, s, h, d = self._get_attn_dims()
            return 4 * b * s * h * d * d
        ...
    
    def physical_flops(self, ctx: ParallelContext) -> int:
        """物理FLOPs（根据并行上下文推导）"""
        # 使用物理shape重新计算
        physical_inputs = [spec.physical_shape(ctx) for spec in self.inputs]
        physical_outputs = [spec.physical_shape(ctx) for spec in self.outputs]
        ...
    
    def infer_comm_ops(self, ctx: ParallelContext) -> List[CommOp]:
        """推导需要的通信操作"""
        # 从输入输出的切分状态变化推导通信
        comm_ops = []
        
        for i, (inp, out) in enumerate(zip(self.inputs, self.outputs)):
            # 检查切分状态变化
            inp_sharded_dims = inp.get_sharded_dims(ctx)
            out_sharded_dims = out.get_sharded_dims(ctx)
            
            # 如果切分状态变化，需要通信
            if inp_sharded_dims != out_sharded_dims:
                comm_op = self._infer_comm_from_sharding_change(inp, out, ctx)
                comm_ops.append(comm_op)
        
        return comm_ops
```

#### ParallelContext - 并行上下文
**核心枢纽**，连接模型与通信域：
```python
@dataclass
class ParallelContext:
    """并行上下文 - 统一模型建模与开销建模"""
    
    # 并行度配置
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    sp_degree: int = 1
    dp_degree: int = 1
    
    # SP类型配置
    sp_type: SPType = SPType.ULYSSES
    ulysses_degree: int = 1
    ring_degree: int = 1
    
    # ===== 通信域映射（来自StrategyConfig） =====
    comm_domains: Dict[str, CommDomain] = field(default_factory=dict)
    
    # ===== 维度切分规则 =====
    # 定义每个维度被哪种并行策略切分
    dim_sharding_rules: Dict[str, str] = field(default_factory=dict)
    # 例如：
    # {"heads": "tp", "kv_heads": "tp", "intermediate": "tp", 
    #  "seq": "sp", "experts": "ep", "vocab": "tp"}
    
    # ===== 核心方法 =====
    
    def get_degree(self, ptype: str) -> int:
        """获取某并行策略的度数"""
        return {
            "tp": self.tp_degree,
            "pp": self.pp_degree,
            "ep": self.ep_degree,
            "sp": self.sp_degree,
            "dp": self.dp_degree,
        }.get(ptype, 1)
    
    def is_dim_sharded_by(self, dim_name: str, ptype: str) -> bool:
        """判断某维度是否被某并行策略切分"""
        return self.dim_sharding_rules.get(dim_name) == ptype
    
    def get_comm_domain(self, ptype: str) -> CommDomain:
        """获取某并行策略的通信域"""
        return self.comm_domains.get(ptype)
    
    def get_total_gpus(self) -> int:
        """总GPU数"""
        return self.tp_degree * self.pp_degree * self.ep_degree * self.sp_degree * self.dp_degree
    
    def build_from_strategy(self, strategy: StrategyConfig, cluster: Cluster) -> "ParallelContext":
        """从StrategyConfig构建"""
        # 1. 提取并行度
        self.tp_degree = strategy.tp_degree
        self.pp_degree = strategy.pp_degree
        ...
        
        # 2. 构建通信域映射
        comm_mapping = strategy.get_communication_domain_mapping(
            devices_per_node=cluster.devices_per_node,
            ...
        )
        
        for ptype, info in comm_mapping.items():
            self.comm_domains[ptype] = CommDomain(
                ptype=ptype,
                degree=info["degree"],
                groups=info["groups"],
                bandwidth_gbps=cluster.get_bandwidth_for_topology_level(
                    info["bandwidth_domain"], info["degree"]
                ),
            )
        
        return self
```

#### OpInstance - 算子实例
**运行时实例**，绑定到具体的并行上下文：
```python
@dataclass
class OpInstance:
    """算子实例 - 在具体并行上下文下的算子"""
    
    # 原始算子规格
    spec: OpSpec
    
    # 并行上下文
    ctx: ParallelContext
    
    # ===== 核心方法 =====
    
    def get_physical_input_shape(self) -> List[Tuple[int, ...]]:
        """获取物理输入shape"""
        return [spec.physical_shape(self.ctx) for spec in self.spec.inputs]
    
    def get_physical_output_shape(self) -> List[Tuple[int, ...]]:
        """获取物理输出shape"""
        return [spec.physical_shape(self.ctx) for spec in self.spec.outputs]
    
    def get_flops(self) -> int:
        """获取物理FLOPs"""
        return self.spec.physical_flops(self.ctx)
    
    def get_comm_ops(self) -> List[CommOp]:
        """获取需要的通信操作"""
        return self.spec.infer_comm_ops(self.ctx)
    
    def estimate_time(self, compute_backend, comm_backend) -> float:
        """估算总时间（计算+通信）"""
        # 计算时间
        compute_time = compute_backend.estimate(
            self.get_physical_input_shape(),
            self.spec.compute_semantic,
            ...
        )
        
        # 通信时间
        comm_time = 0.0
        for comm_op in self.get_comm_ops():
            domain = self.ctx.get_comm_domain(comm_op.ptype)
            comm_time += comm_backend.estimate(
                comm_op.comm_type,
                comm_op.get_data_bytes(),
                domain,
            )
        
        return compute_time + comm_time
```

---

### 2.2 维度切分规则

定义每种算子的维度切分规则，这是框架的核心：

```python
# ===== Attention层切分规则 =====

ATTENTION_DIM_RULES = {
    # Q/K/V projection的输出
    "q_heads": "tp",  # Q heads被TP切分
    "kv_heads": "tp",  # KV heads被TP切分
    "head_dim": None,  # head_dim不切分
    
    # Attention计算
    "seq": "sp",  # seq被SP切分（如果启用）
    
    # O projection的输入
    "attention_heads": "tp",  # 被TP切分
}

# ===== FFN层切分规则 =====

FFN_DIM_RULES = {
    "intermediate": "tp",  # intermediate_size被TP切分
}

# ===== MoE层切分规则 =====

MOE_DIM_RULES = {
    "experts": "ep",  # experts被EP切分
    "expert_intermediate": "tp",  # expert内部的intermediate被TP切分
}

# ===== Embedding层切分规则 =====

EMBEDDING_DIM_RULES = {
    "vocab": "tp",  # vocab被TP切分
}

# ===== LM Head切分规则 =====

LM_HEAD_DIM_RULES = {
    "vocab": "tp",  # vocab被TP切分
}
```

**规则应用逻辑**：
1. 根据算子类型选择对应的规则集
2. 规则集中定义的维度按指定策略切分
3. 未定义的维度不切分
4. 多个并行策略冲突时，按优先级选择

---

### 2.3 通信推导逻辑

从输入输出的切分状态变化自动推导通信：

```python
def _infer_comm_from_sharding_change(
    self,
    input: TensorSpec,
    output: TensorSpec,
    ctx: ParallelContext,
) -> CommOp:
    """从切分状态变化推导通信"""
    
    # 输入切分状态
    inp_sharding = input.get_sharding_info(ctx)
    
    # 输出切分状态
    out_sharding = output.get_sharding_info(ctx)
    
    # 通信推导规则：
    
    # 1. TP列切分 -> TP行切分：需要AllReduce
    #    输入：heads被TP切分
    #    输出：hidden不被切分（聚合）
    #    -> AllReduce(activation_bytes)
    
    # 2. TP行切分 -> TP列切分：不需要通信
    #    输入：hidden不被切分
    #    输出：intermediate被TP切分
    #    -> 无通信
    
    # 3. EP All2All Dispatch：
    #    输入：experts不切分（完整）
    #    输出：experts被EP切分
    #    -> All2All dispatch
    
    # 4. EP All2All Combine：
    #    输入：experts被EP切分
    #    输出：experts不切分（聚合）
    #    -> All2All combine
    
    # 5. SP Ulysses All2All：
    #    输入：seq不被切分
    #    输出：heads被Ulysses切分（seq->heads布局切换）
    #    -> All2All
    
    # 6. SP Ring AllGather：
    #    输入：seq被Ring切分
    #    输出：seq不被切分（完整KV）
    #    -> AllGather
    
    # 7. Megatron-SP：
    #    输入：hidden不被切分
    #    输出：hidden被Megatron切分
    #    -> ReduceScatter (forward)
    #    输入：hidden被Megatron切分
    #    输出：hidden不被切分
    #    -> AllGather (forward)
    
    # 具体实现...
    ...
```

---

### 2.4 完整流程示例

```python
# ===== 1. 定义张量规格 =====

# 输入激活
input_activation = TensorSpec(
    dims=[
        TensorDim("batch", batch_size, shardable_by={}),
        TensorDim("seq", seq_len, shardable_by={"sp"}),
        TensorDim("hidden", hidden_size, shardable_by={"tp", "sp"}),
    ],
    dtype="fp16",
)

# Q projection权重
q_weight = TensorSpec(
    dims=[
        TensorDim("hidden_in", hidden_size, shardable_by={}),
        TensorDim("heads", num_heads, shardable_by={"tp"}),
        TensorDim("head_dim", head_dim, shardable_by={}),
    ],
    dtype="fp16",
)

# Q projection输出
q_output = TensorSpec(
    dims=[
        TensorDim("batch", batch_size, shardable_by={}),
        TensorDim("seq", seq_len, shardable_by={"sp"}),
        TensorDim("heads", num_heads, shardable_by={"tp"}),
        TensorDim("head_dim", head_dim, shardable_by={}),
    ],
    dtype="fp16",
)

# ===== 2. 定义算子规格 =====

q_proj_op = OpSpec(
    name="q_proj",
    op_type="compute",
    inputs=[input_activation],
    outputs=[q_output],
    compute_semantic="matmul",
    weight=q_weight,
)

# ===== 3. 定义并行上下文 =====

ctx = ParallelContext(
    tp_degree=8,
    pp_degree=4,
    sp_degree=2,
    dp_degree=16,
    sp_type=SPType.ULYSSES,
    dim_sharding_rules={
        "heads": "tp",
        "kv_heads": "tp",
        "seq": "sp",
        "intermediate": "tp",
        "vocab": "tp",
    },
)

# 从StrategyConfig和Cluster构建通信域
ctx.build_comm_domains(strategy, cluster)

# ===== 4. 创建算子实例 =====

q_proj_instance = OpInstance(spec=q_proj_op, ctx=ctx)

# ===== 5. 自动推导 =====

# 物理输入shape（自动推导）
# (batch, seq/2, hidden/8)  <- TP切分hidden, SP切分seq
physical_input = q_proj_instance.get_physical_input_shape()

# 物理输出shape（自动推导）
# (batch, seq/2, heads/8, head_dim)  <- TP切分heads, SP切分seq
physical_output = q_proj_instance.get_physical_output_shape()

# FLOPs（自动推导）
# 2 * batch * (seq/2) * (hidden/8) * (heads/8) * head_dim
flops = q_proj_instance.get_flops()

# 通信操作（自动推导）
# 无通信 - TP列切分不产生通信
comm_ops = q_proj_instance.get_comm_ops()

# ===== 6. 估算时间 =====

time = q_proj_instance.estimate_time(compute_backend, comm_backend)
```

---

## 3. 架构对比

### 3.1 当前架构
```
Model (LayerConfig)
    ├─ input_shape (静态)
    ├─ output_shape (静态)
    ├─ flops (静态)
    └─ sharding_info (静态描述)

Analyzer (独立计算)
    ├─ _estimate_compute_time() <- 使用原始shape
    ├─ _estimate_tp_communication_time() <- 独立计算
    ├─ _estimate_sp_communication_time() <- 独立计算
    └─ ...

问题：两层独立，切分状态不一致
```

### 3.2 新架构
```
Model (OpSpec定义)
    ├─ TensorSpec (逻辑定义)
    └─ OpSpec (算子逻辑定义)

ParallelContext (统一枢纽)
    ├─ 并行度配置
    ├─ 通信域映射
    └─ 维度切分规则

OpInstance (运行时实例)
    ├─ spec: OpSpec
    ├─ ctx: ParallelContext
    ├─ get_physical_shape() <- 自动推导
    ├─ get_flops() <- 自动推导
    ├─ get_comm_ops() <- 自动推导
    └─ estimate_time() <- 计算+通信统一估算

优势：一处定义，自动推导所有物理状态和开销
```

---

## 4. 实现路径

### Phase 1: 核心数据结构
1. 实现 `TensorSpec` 和 `TensorDim`
2. 实现 `ParallelContext` 
3. 实现维度切分规则表

### Phase 2: 算子规格
1. 实现 `OpSpec` 基类
2. 实现常见算子规格：`LinearOp`, `AttentionOp`, `FFNOp`, `MoEOp`
3. 实现 `OpInstance`

### Phase 3: 通信推导
1. 实现切分状态变化检测
2. 实现通信类型推导
3. 实现通信量计算

### Phase 4: 模型迁移
1. 将 Model 层迁移到 OpSpec 定义
2. 实现 `build_layers()` 返回 OpSpec 列表
3. Analyzer 使用 OpInstance 估算

### Phase 5: 测试验证
1. 单并行策略验证
2. 高维并行组合验证
3. 与现有实现对比验证

---

## 5. 切分策略全览

| 维度 | 可切分的并行策略 | 切分后大小 | 通信操作 |
|-----|----------------|----------|---------|
| **batch** | DP | batch/dp | AllReduce (梯度) |
| **seq** | SP (Ulysses/Ring/Megatron) | seq/sp | All2All/AllGather/ReduceScatter |
| **hidden** | TP, SP (Megatron) | hidden/tp 或 hidden/sp | AllReduce/ReduceScatter |
| **heads** | TP, SP (Ulysses) | heads/tp 或 heads/sp_ulysses | AllReduce/All2All |
| **kv_heads** | TP | kv_heads/tp | AllReduce |
| **head_dim** | 不切分 | - | - |
| **intermediate** | TP | intermediate/tp | AllReduce |
| **vocab** | TP | vocab/tp | AllGather (推理) |
| **experts** | EP | experts/ep | All2All |
| **expert_intermediate** | TP | expert_intermediate/tp | AllReduce |

---

## 6. 高维并行示例

### TP=8 + EP=2 + SP=4 + PP=4 + DP=16

```python
# 总GPU数 = 8 * 2 * 4 * 4 * 16 = 4096

ctx = ParallelContext(
    tp_degree=8,
    ep_degree=2,
    sp_degree=4,
    pp_degree=4,
    dp_degree=16,
    sp_type=SPType.UNIFIED_2D,
    ulysses_degree=2,
    ring_degree=2,
)

# MoE Attention层
attention_op = AttentionOpSpec(...)

# 创建实例
instance = OpInstance(attention_op, ctx)

# 自动推导：
# - 物理shape: heads被TP切分 -> heads/8
#               seq被SP切分 -> seq/4
#               hidden被TP切分 -> hidden/8
# - FLOPs: 基于物理shape重新计算
# - 通信:
#   - TP AllReduce: activation_bytes (heads聚合)
#   - SP Ulysses All2All: activation_bytes * ulysses_degree (seq布局切换)
#   - SP Ring P2P: kv_bytes * (ring_degree-1) steps

# MoE FFN层
moe_op = MoEOpSpec(...)

instance = OpInstance(moe_op, ctx)

# 自动推导：
# - 物理shape: experts被EP切分 -> experts/2
#               expert_intermediate被TP切分 -> intermediate/8
# - FLOPs: 基于物理shape重新计算
# - 通信:
#   - EP All2All dispatch: token_bytes (expert分发)
#   - EP All2All combine: token_bytes (expert聚合)
#   - TP AllReduce: activation_bytes (intermediate聚合)
```

---

## 7. 关键优势

### 7.1 统一表达
- 一个 `OpSpec` 定义适用于所有切分策略
- 不需要为每种切分组合写新代码

### 7.2 自动推导
- 物理shape从逻辑shape + 并行上下文自动推导
- 通信从输入输出切分状态变化自动推导
- FLOPs从物理shape自动计算

### 7.3 高维并行支持
- 任意组合 TP+PP+EP+SP+DP
- 通信域自动关联到正确的带宽

### 7.4 易扩展
- 新切分策略：添加新的维度切分规则
- 新算子：定义新的 OpSpec
- 新通信模式：扩展通信推导逻辑

---

## 8. 与现有代码的关系

### 8.1 现有架构分层

现有代码有三层 API：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Functional API (functional.py)                │
│  类似 torch.nn.functional - 输入shape，输出KernelResult           │
│  linear(), flash_attention(), rms_norm(), silu() 等              │
│  只定义 FLOPs、memory、arithmetic_intensity，不考虑切分            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Kernel Registry (compute.py, communication.py)│
│  ComputeKernelRegistry: 创建和管理计算kernel                      │
│  CommKernelRegistry: 创建和管理通信kernel                         │
│  包含 device 信息，使用 Roofline 模型估算时间                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Kernel Backend (backend/*.py)                 │
│  TheoryBackend: 理论模型 (Roofline)                               │
│  ProfilingBackend: 实测数据                                       │
│  MicroarchBackend: 微架构模拟                                     │
│  统一的评估接口，可插拔                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 新架构分层

OpSpec/OpInstance 是**新的中间层**，连接逻辑定义和物理评估：

```
┌─────────────────────────────────────────────────────────────────┐
│                      OpSpec (新)                                 │
│  逻辑算子规格 - 扩展 functional API，增加切分约束                  │
│  包含: TensorSpec (输入输出), 切分规则, 计算语义                   │
│  类似: functional API 的 "增强版"                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓ + ParallelContext
┌─────────────────────────────────────────────────────────────────┐
│                      OpInstance (新)                             │
│  运行时实例 - 自动推导物理形态和通信                               │
│  get_physical_shape(): 推导切分后的shape                          │
│  get_comm_ops(): 推导通信操作                                     │
│  get_flops(): 推导切分后的FLOPs                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 调用
┌─────────────────────────────────────────────────────────────────┐
│                      Kernel Backend (保留)                       │
│  estimate_compute_time(): 估算计算时间                            │
│  estimate_comm_time(): 估算通信时间                               │
│  estimate_memory(): 估算内存                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Kernel Registry (保留)                      │
│  ComputeKernel: 计算kernel实现                                    │
│  CommKernel: 通信kernel实现                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 关键区别

| 层级 | 输入 | 输出 | 考虑切分 |
|-----|------|------|---------|
| **Functional API** | shape | KernelResult (FLOPs, memory) | ❌ 不考虑 |
| **OpSpec** | TensorSpec (含切分约束) | 无 (静态定义) | ✅ 定义切分规则 |
| **OpInstance** | OpSpec + ParallelContext | 物理shape, FLOPs, 通信操作 | ✅ 自动推导 |
| **KernelBackend** | 物理shape | 时间 (秒) | ❌ 只评估物理形态 |

### 8.4 代码映射关系

```python
# ===== 现有 functional API =====
# 输入: 逻辑 shape
# 输出: KernelResult (逻辑 FLOPs)
result = functional.linear(
    input=(batch, seq, hidden),  # 逻辑 shape
    weight=(hidden, num_heads * head_dim),
    dtype="fp16"
)
# result.flops = 逻辑 FLOPs (未切分)
# result.bytes_accessed = 逻辑 memory (未切分)

# ===== 新 OpSpec =====
# 输入: TensorSpec (含切分约束)
# 输出: 无 (静态定义)
q_proj_spec = LinearOpSpec(
    name="q_proj",
    input=TensorSpec(dims=[
        TensorDim("batch", batch_size, shardable_by={}),
        TensorDim("seq", seq_len, shardable_by={"sp"}),
        TensorDim("hidden", hidden_size, shardable_by={"tp"}),
    ]),
    output=TensorSpec(dims=[
        TensorDim("batch", batch_size, shardable_by={}),
        TensorDim("seq", seq_len, shardable_by={"sp"}),
        TensorDim("heads", num_heads, shardable_by={"tp"}),
        TensorDim("head_dim", head_dim, shardable_by={}),
    ]),
    compute_semantic="matmul",
)

# ===== 新 OpInstance =====
# 输入: OpSpec + ParallelContext
# 输出: 物理形态和通信 (自动推导)
ctx = ParallelContext(tp_degree=8, sp_degree=4, ...)
instance = OpInstance(spec=q_proj_spec, ctx=ctx)

# 自动推导:
physical_input = instance.get_physical_input_shape()  # (batch, seq/4, hidden/8)
physical_output = instance.get_physical_output_shape()  # (batch, seq/4, heads/8, head_dim)
physical_flops = instance.get_flops()  # 基于 physical shape 计算
comm_ops = instance.get_comm_ops()  # 自动推导通信

# ===== Kernel Backend (保留) =====
# 输入: 物理 shape
# 输出: 时间 (秒)
backend = TheoryBackend(config)
compute_time = backend.estimate_compute_time(
    kernel_name="linear",
    input_shapes=[physical_input],
    output_shape=physical_output,
    dtype="fp16",
    device=device,
)
comm_time = backend.estimate_comm_time(
    comm_type="allreduce",
    data_size_bytes=comm_op.data_bytes,
    num_ranks=ctx.tp_degree,
    bandwidth_gbps=ctx.get_comm_domain("tp").bandwidth_gbps,
)
total_time = compute_time + comm_time
```

### 8.5 Functional API vs OpSpec

**Functional API 的设计**:
```python
# functional.linear 只计算逻辑 FLOPs
def linear(input_shape, weight_shape, dtype):
    # 假设 input_shape = (batch, seq, hidden)
    # 假设 weight_shape = (hidden, num_heads * head_dim)
    flops = 2 * batch * seq * hidden * num_heads * head_dim  # 逻辑 FLOPs
    bytes_accessed = batch * seq * hidden * dtype_size + ...  # 逻辑 memory
    return KernelResult(flops=flops, bytes_accessed=bytes_accessed, ...)
```

**OpSpec 的设计**:
```python
# OpSpec 定义切分规则，OpInstance 推导物理 FLOPs
class LinearOpSpec(OpSpec):
    # 定义输入输出的切分约束
    input = TensorSpec(dims=[...])  # 每个 dim 有 shardable_by
    output = TensorSpec(dims=[...])
    
    # OpInstance.physical_flops() 基于 physical shape 计算
    # 如果 TP=8, 则 physical_flops = functional.linear(physical_shape).flops
```

**关系**:
- `functional API` = `OpSpec` 的**逻辑部分**
- `OpInstance.physical_flops()` 可以调用 `functional.linear(physical_shape).flops`
- OpSpec 扩展了 functional API，增加了切分约束
- OpInstance 负责推导物理 shape，然后调用 functional API 计算物理 FLOPs

### 8.6 保留 vs 重构 vs 新增

**保留**:
- `StrategyConfig` 的通信域划分逻辑 (完整保留)
- `Cluster` 的带宽拓扑 (完整保留)
- `ComputeKernel` / `CommKernel` (完整保留)
- `KernelBackend` (完整保留)
- `functional.py` (完整保留，作为底层计算引擎)

**重构**:
- `LayerConfig` → `OpSpec` (逻辑定义)
- `ShardedLayerConfig` → 删除 (OpInstance 替代)
- `ShardingInfo` → 整合到 TensorSpec (每个 dim 有 shardable_by)
- `Analyzer` → 使用 `OpInstance` 估算 (不再独立计算通信)

**新增**:
- `TensorSpec` / `TensorDim` (张量规格，含切分约束)
- `ParallelContext` (并行上下文，整合 StrategyConfig + 通信域)
- `OpSpec` (算子规格，扩展 functional API)
- `OpInstance` (算子实例，自动推导物理形态和通信)
- 维度切分规则表
- 通信推导逻辑 (从输入输出切分状态变化推导)
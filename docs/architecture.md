# 架构设计（v5.1）

## 整体架构

LLM Performance Evaluator 采用分层架构设计，将应用接口、工作负载配置、统一建模、性能分析、Kernel评估、策略配置、硬件抽象和模型定义解耦。

**设计目标**:
1. **易用性**: 一行代码评估性能，自动搜索最优策略
2. **高可扩展性**: 新模型/新场景通过YAML配置，无需修改代码
3. **正交配置**: Workload与Model独立配置，管线预设不与模型类型绑定
4. **统一建模**: Torch-like接口定义模型，自动推导切分约束和通信开销
5. **分解呈现**: 模块/阶段/子模块的计算/通信/内存分解
6. **核心指标**: MFU/QPS/TTFT/TPOT/TPS

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Application Layer (易用性)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │   Evaluator  │  │  Optimizer   │  │     ColocateAnalyzer     │   │
│  │ (便捷评估)   │  │ (策略搜索)   │  │     (混布评估)           │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     Workload Layer (可配置场景)                       │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  WorkloadConfig (YAML配置)                                  │      │
│  │  - training.yaml (训练：forward+backward+optimizer)        │      │
│  │  - autoregressive-inference.yaml (prefill+decode)          │      │
│  │  - diffusion-pipeline.yaml (encode+denoise+decode)         │      │
│  │  - rl-ppo.yaml / rl-grpo.yaml (RL后训练)                   │      │
│  │  - denoise-training.yaml (扩散训练)                         │      │
│  └────────────────────────────────────────────────────────────┘      │
│  核心概念:                                                          │
│  - ComputePattern: transformer_block/conv_encoder/conv_decoder等    │
│  - Phase: forward/backward/optimizer/prefill/decode/encode等        │
│  - ComponentMapping: generic name -> user component name            │
├─────────────────────────────────────────────────────────────────────┤
│                   Unified Modeling Layer (核心)                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │      Torch-like接口 - 像PyTorch一样定义模型                     │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │  │
│  │  │ShardedTensor│ │ShardedModule│ │    ParallelContext      │  │  │
│  │  │(切分张量)   │ │(切分模块)   │ │   (并行策略上下文)       │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘  │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │  │
│  │  │ModuleInst.  │ │ Basic Layers│ │    Complete Models      │  │  │
│  │  │(物理形态)   │ │Emb/Attn/FFN │ │Llama/DeepSeek/Wan/Custom│  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  核心机制:                                                          │
│  - bind(ctx) → ModuleInstance（物理形态 + 子模块分解）              │
│  - _submodule_instances → 子模块性能分解（emb/attn/ffn等）          │
│  - flops_forward_physical → 物理FLOPs（考虑TP切分）                 │
│  - total_comm_ops → 通信操作分解（AllReduce/AllToAll等）           │
├─────────────────────────────────────────────────────────────────────┤
│                        Analyzer Layer                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  UnifiedAnalyzer (统一分析器)                                 │  │
│  │  - analyze(workload) → UnifiedResult                         │  │
│  │  - PhaseResult.submodules (子模块分解)                       │  │
│  │  - MFU/QPS指标                                               │  │
│  │  - CommunicationBreakdown (通信分解)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ Memory Est.  │ │  Comm Est.   │ │   MFU/QPS    │                │
│  │ - Parameters │ │ - AllReduce  │ │  计算        │                │
│  │ - Activations│ │ - AllToAll   │ │              │                │
│  └──────────────┘ └──────────────┘ └──────────────┘                │
├─────────────────────────────────────────────────────────────────────┤
│                        Strategy Layer                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐  │
│  │ TP/PP/DP/EP  │ │   SP/CP      │ │  Optimizer   │ │ Scheduler  │  │
│  │ (并行策略)   │ │ (序列并行)   │ │ (优化器建模) │ │ (调度配置) │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     Kernel Layer (可插拔 Backend)                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              KernelBackend (可插拔评估方案)                    │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │  │
│  │  │ TheoryBack  │ │ ProfilingBk │ │   MicroarchBackend      │  │  │
│  │  │ (FLOPs理论) │ │ (实测数据)  │ │   (微架构建模)          │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────┐  ┌──────────────────────────────────────┐  │
│  │   Compute Kernels  │  │       Communication Kernels          │  │
│  │  - linear/GEMM     │  │  - allreduce (Ring/Tree)             │  │
│  │  - attention       │  │  - allgather                         │  │
│  │  - conv2d/conv3d   │  │  - alltoall                          │  │
│  └────────────────────┘  └──────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │        Functional API (Torch-like, bind机制调用)               │  │
│  │  linear(), flash_attention(), conv3d(), rms_norm(), silu()    │  │
│  └───────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Hardware Layer                               │
│  ┌────────────────────┐  ┌──────────────────────────────────────┐  │
│  │    Device        │  │    Cluster (Network Topology)        │  │
│  │  - Compute TFLOPS  │  │  - 2-Tier / 3-Tier Clos             │  │
│  │  - Memory BW       │  │  - Fat-Tree / CloudMatrix           │  │
│  │  - Memory Capacity │  │  - Bandwidth Hierarchy              │  │
│  └────────────────────┘  └──────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     Preset Layer (YAML配置)                          │
│  ┌────────────────────┐  ┌──────────────────────────────────────┐  │
│  │   Model Presets    │  │      Workload Presets               │  │
│  │  - llama-7b.yaml   │  │  - configs/workloads/**/*.yaml      │  │
│  │  - wan-t2v-14b.yaml│  │  - 训练/推理/RL/扩散等              │  │
│  │  - preset_type标记 │  │  - 与model解耦                      │  │
│  └────────────────────┘  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. Application Layer

**职责**: 提供便捷的高层接口

| 类 | 职责 | 方法 |
|----|------|------|
| `Evaluator` | 便捷评估 | `evaluate()`, `evaluate_training()`, `evaluate_inference()`, `evaluate_diffusion()` |
| `ColocateAnalyzer` | 混布评估 | `analyze(allocations)` |
| `StrategyOptimizer` | 策略搜索 | `search_best_strategy()` |

**使用示例**:

```python
from llm_perf.app import Evaluator

# 评估训练
result = Evaluator().evaluate_training(
    model="llama-7b",
    hardware="H100-SXM-80GB",
    strategy={"tp": 8},
    batch_size=32,
    seq_len=4096,
)
print(f"MFU: {result.mfu * 100:.1f}%")
print(f"Throughput: {result.throughput['tokens_per_sec']} tok/s")

# 评估扩散推理
result = Evaluator().evaluate_diffusion(
    models={"encoder": "wan-text-encoder", "dit": "wan-dit", "vae": "wan-vae"},
    hardware="H100-SXM-80GB",
    num_frames=81,
    height=720,
    width=1280,
)
print(f"Total time: {result.total_time_sec:.2f}s")
```

---

### 2. Workload Layer

**职责**: 可配置的工作负载预设，与模型类型解耦

**WorkloadConfig结构**:

```yaml
name: training
description: LLM训练场景
workload_type: training

phases:
  - name: forward
    compute_type: forward
    component: main
    compute_pattern: transformer_block
    repeat: 1
    
  - name: backward
    compute_type: backward
    component: main
    compute_pattern: transformer_block
    repeat: 1
    
  - name: optimizer
    compute_type: optimizer
    component: main
    repeat: 1

default_params:
  batch_size: 32
  seq_len: 2048
```

**ComputePattern枚举**:

| Pattern | 描述 | 适用模型 |
|---------|------|---------|
| `transformer_block` | Attention + FFN | LLM, DiT |
| `conv_encoder` | 卷积编码器 | VAE encoder |
| `conv_decoder` | 卷积解码器 | VAE decoder |
| `attention_only` | 纯Attention | DiT (部分) |
| `dense_forward` | 全连接 | MLP |

**工作负载类型**:

| Workload | Phases | 场景 |
|----------|--------|------|
| `training` | forward + backward + optimizer | 训练 |
| `pretraining` | 同training + gradient_accumulation | 预训练 |
| `finetuning` | 同training | 微调 |
| `autoregressive-inference` | prefill + decode | LLM推理 |
| `diffusion-pipeline` | encode + denoise + decode | 扩散推理 |
| `denoise-training` | encode + forward + backward + optimizer | 扩散训练 |
| `rl-ppo` | policy + value + reference | RL-PPO |
| `rl-grpo` | policy + reference | RL-GRPO |
| `speculative-decoding` | target + draft | 推测解码 |

---

### 3. Unified Modeling Layer

**职责**: Torch-like接口定义模型，bind机制获取物理分解

**核心机制**:

```
ShardedModule.forward(input)
    → 记录 op_history (MatmulOp, AttentionOp等)
    
ShardedModule.bind(ctx)
    → ModuleInstance
    
ModuleInstance._submodule_instances
    → {attention_0: SubmoduleInstance, ffn_0: SubmoduleInstance, ...}
    
SubmoduleInstance.flops_forward_physical
    → 物理FLOPs（考虑TP切分）
    
SubmoduleInstance.total_comm_ops
    → [CommOp("allreduce", bytes, ptype), ...]
```

**基础模块**:

| 模块 | 切分方式 | 通信 |
|------|----------|------|
| `ShardedEmbedding` | vocab TP切分 | AllGather |
| `ShardedRMSNorm` | 不切分 | 无 |
| `ShardedAttention` | heads TP切分 | AllReduce/All2All |
| `ShardedFFN` | intermediate TP切分 | AllReduce |
| `ShardedLMHead` | vocab TP切分 | AllGather |
| `ShardedMoE` | experts EP切分 | AllToAll |
| `ShardedMLA` | KV压缩 | AllToAll |
| `ShardedConv3d` | channels切分 | AllReduce |

**类型系统**:

| 类型 | 继承 | 用途 | 注册位置 |
|------|------|------|----------|
| `ShardedTensor` | - | 通用切分张量 | - |
| `ShardedParameter` | ShardedTensor | 模型权重（不可切分逻辑） | `_weights` |
| `ShardedModule` | - | 可绑定模块 | `_submodules` |

**`__setattr__` 自动注册机制**:

```python
def __setattr__(self, name: str, value: Any):
    if isinstance(value, ShardedParameter):
        self._weights[name] = value      # 权重 → _weights
    elif isinstance(value, ShardedModule):
        self._submodules[name] = value   # 子模块 → _submodules
    elif isinstance(value, ShardedTensor):
        pass                              # 普通张量 → 不注册
```

**权重/激活内存统计分离**:

```python
class ShardedModule:
    _weights: Dict[str, ShardedTensor]      # 权重张量
    _activations: Dict[str, ShardedTensor]  # 激活张量
    _intermediate_tensors: Dict[str, ShardedTensor]  # 中间张量
    
    def get_weights(self) -> Dict[str, ShardedTensor]:
        """递归获取所有权重"""
        
    def get_activations(self) -> Dict[str, ShardedTensor]:
        """递归获取所有激活（含中间张量）"""
```

**ShardedFFN 激活类型**:

```python
class FFNActType(str, Enum):
    SWIGLU = "swiglu"  # gated, 3权重
    GELU = "gelu"      # non-gated, 2权重
    RELU = "relu"      # non-gated, 2权重
    SILU = "silu"      # non-gated, 2权重
```

| ffn_act_type | 权重数量 | 权重列表 | 计算路径 |
|--------------|----------|----------|----------|
| `swiglu` | 3 | gate_weight, up_weight, down_weight | `silu(x @ gate) * (x @ up) @ down` |
| `gelu` | 2 | up_weight, down_weight | `gelu(x @ up) @ down` |
| `relu` | 2 | up_weight, down_weight | `relu(x @ up) @ down` |
| `silu` | 2 | up_weight, down_weight | `silu(x @ up) @ down` |

**gated vs non-gated 差异**:

```python
# Gated FFN (SWIGLU)
gate_proj = hidden @ gate_weight    # (batch, seq, intermediate)
gate_silu = silu(gate_proj)         # 门控激活
up_proj = hidden @ up_weight        # (batch, seq, intermediate)
intermediate = gate_silu * up_proj  # 元素级乘法
output = intermediate @ down_weight

# Non-gated FFN (GELU/RELU/SILU)
up_proj = hidden @ up_weight        # (batch, seq, intermediate)
intermediate = act_fn(up_proj)      # 单一激活
output = intermediate @ down_weight
```

**完整模型**:

| 模型 | 结构 | layers |
|------|------|--------|
| `LlamaModel` | Embed + Blocks + Norm + LMHead | ~66 |
| `DeepSeekModel` | MLA + MoE | ~100 |
| `ShardedWanDiT` | Attention + FFN blocks | ~82 |
| `ShardedVAE` | Conv encoder + decoder | ~20 |

---

### 4. Analyzer Layer

**职责**: 统一性能分析，生成分解报告

**UnifiedResult结构**:

```python
@dataclass
class UnifiedResult:
    workload_name: str
    workload_type: WorkloadType
    phases: List[PhaseResult]
    
    total_time_sec: float
    peak_memory_gb: float
    throughput: Dict[str, float]
    
    # 核心指标
    mfu: Optional[float]  # Model FLOPs Utilization (0-1)
    qps: Optional[float]  # Queries Per Second
    
    # 分解
    breakdown: Dict[str, Any]
    detailed_breakdown: Dict[str, Any]
    communication_breakdown: CommunicationBreakdown
```

**PhaseResult结构**:

```python
@dataclass
class PhaseResult:
    name: str
    component: str
    compute_type: ComputeType
    single_time_sec: float
    repeat_count: int
    total_time_sec: float
    flops: int
    memory_gb: float
    
    # 子模块分解（从ModuleInstance._submodule_instances获取）
    submodules: List[SubmoduleResult]
```

**SubmoduleResult结构**:

```python
@dataclass
class SubmoduleResult:
    name: str           # attention_0, ffn_0, embedding等
    submodule_type: str # embedding, attention, ffn, moe, lm_head等
    time_sec: float
    flops: int          # 物理FLOPs
    memory_gb: float    # 物理内存
    communication_bytes: int  # 通信量
```

**MFU计算**:

```
MFU = forward_flops / (peak_tflops × num_devices × total_time)
```

**QPS计算**:

```
QPS = batch_size × dp_degree / total_time
```

---

### 5. Kernel Layer

**职责**: 可插拔的算子性能评估

| Backend | 描述 | 适用 |
|---------|------|------|
| `TheoryBackend` | FLOPs/Roofline | 快速估算 |
| `ProfilingBackend` | 实测数据插值 | 精确建模 |

**Functional API**:

```python
from llm_perf.kernels.functional import linear, flash_attention, conv3d

# KernelResult包含forward和backward信息
result = linear(input_shape=(batch, seq, hidden), 
               weight_shape=(hidden, hidden), 
               dtype="fp16")

print(f"Forward FLOPs: {result.flops}")
print(f"Backward FLOPs: {result.flops_backward}")
print(f"Forward bytes: {result.bytes_accessed}")
print(f"Backward bytes: {result.bytes_accessed_backward}")
print(f"Saved inputs: {result.saved_inputs}")  # backward需要的输入
```

**KernelResult.saved_inputs**: 标记 backward pass 需要保存哪些输入：

| Kernel | saved_inputs | 说明 |
|--------|-------------|------|
| `linear` | `["input"]` | backward: dW = x^T @ dy |
| `bmm` | `["input", "mat2"]` | backward: dA = dC @ B^T, dB = A^T @ dC |
| `scaled_dot_product_attention` | `["query", "key", "value"]` | backward需要Q, K, V |
| `flash_attention` | `[]` | Q/K/V是view，不单独保存 |
| `rms_norm` | `["input"]` | backward需要input |
| `silu/gelu` | `["input"]` | backward需要input |
| `relu` | `[]` | backward使用mask，不保存input |
| `softmax` | `["output"]` | backward需要softmax值 |
| `conv2d/conv3d` | `["input"]` | backward: dW需要input |
| `embedding` | `["input_ids"]` | backward需要indices |

---

### 5.1 Automatic Activation Memory Tracking

**职责**: 自动追踪激活内存，过滤 view tensor

**架构**:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Level 1: ShardedTensor._is_view                                      │
│   - view(), transpose() → _is_view=True (view tensor, no new memory) │
│   - matmul(), silu() → _is_view=False (new tensor allocation)        │
│   - contiguous() → _is_view=False (force non-view copy)              │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Level 2: KernelResult.saved_inputs                                   │
│   - linear: saved_inputs=["input"]                                   │
│   - flash_attention: saved_inputs=[] (Q/K/V are views)               │
│   - Mark which inputs need to be saved for backward pass             │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Level 3: Op.get_saved_tensors()                                      │
│   - MatmulOp: get_saved_tensors() → [self.input]                     │
│   - AttentionOp: get_saved_tensors() → []                            │
│   - RMSNormOp: get_saved_tensors() → [self.input]                    │
│   - ActivationOp: get_saved_tensors() → depends on activation_type   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Level 4: ModuleInstance.activation_memory_physical                   │
│   - Collect from _intermediate_tensors                               │
│   - Filter out _is_view=True tensors                                 │
│   - Calculate physical bytes with get_physical_bytes()               │
│   - Result: only non-view tensors counted                            │
└─────────────────────────────────────────────────────────────────────┘
```

**效果对比**:

| 场景 | 之前（手动追踪） | 之后（自动追踪） |
|------|-----------------|-----------------|
| Flash Attention | ~32GB（错误追踪q, k, v, attn_out, attn_flat） | ~13GB（只追踪q_proj, k_proj, v_proj, output） |
| 训练内存估算 | 可能重复计算view tensor | 自动过滤view，准确估算 |

**使用示例**:

```python
class ShardedAttention(ShardedModule):
    def forward(self, hidden):
        # 这些是真正的张量（需要保存）
        q_proj = self._track_intermediate("q_proj", hidden @ self.q_weight)
        k_proj = self._track_intermediate("k_proj", hidden @ self.k_weight)
        v_proj = self._track_intermediate("v_proj", hidden @ self.v_weight)
        
        # 这些是view tensor（_is_view=True，不保存）
        q = q_proj.view(batch, seq, heads, head_dim).transpose(1, 2)
        k = k_proj.view(...).transpose(1, 2)
        v = v_proj.view(...).transpose(1, 2)
        
        attn_out = flash_attention(q, k, v)  # flash_attention返回view
        
        # 只追踪非view tensor
        output = self._track_intermediate("output", attn_flat @ self.o_weight)
        
        # ModuleInstance.activation_memory_physical 会自动：
        # - 统计 q_proj, k_proj, v_proj, output（非view）
        # - 过滤 q, k, v, attn_out（view）
```

---

### 6. Hardware Layer

**预设设备**:

| 设备 | FP16 TFLOPS | 显存 | 内存带宽 |
|------|-------------|------|----------|
| H100-SXM-80GB | 989 | 80GB | 3.35 TB/s |
| A100-SXM-80GB | 312 | 80GB | 2.04 TB/s |
| MI300X | 1307 | 192GB | 5.3 TB/s |
| Ascend-910B2 | 376 | 64GB | 1.6 TB/s |

**网络拓扑**:

**TopologyLevel 结构**:

```python
@dataclass
class TopologyLevel:
    name: str                    # 层级名称: "node", "rack", "cluster"
    level: int                   # 层级编号: 0=最近设备
    bandwidth_gbps: float       # 该层级带宽
    latency_us: float = 1.0     # 延迟 (μs)
    oversubscription_ratio: float = 1.0  # 超额订阅比
    devices_per_group: int = 1   # 该层设备组大小
```

**拓扑工厂方法**:

| 方法 | 用途 | 层级结构 |
|------|------|----------|
| `create_clos_3tier()` | 3层Clos拓扑 | node → rack → cluster |
| `create_fat_tree()` | 数据中心胖树 | edge → aggregation → core |
| `create_2tier_simple()` | 简单2层拓扑 | node → inter_node |
| `create_cloudmatrix_supernode()` | 华为超节点 | ub_plane → rdma_plane |

**使用示例**:

```python
# 3层Clos拓扑
topology = NetworkTopology.create_clos_3tier(
    node_bw_gbps=900,      # NVLink
    rack_bw_gbps=200,      # Leaf switch
    cluster_bw_gbps=100,   # Spine switch
)

# 华为CloudMatrix 384 NPU超节点
topology = NetworkTopology.create_cloudmatrix_supernode(
    num_npus=384,
    ub_bw_gbps=3136,       # ~392GB/s x 8
)
```

| 拓扑 | 描述 |
|------|------|
| 2-Tier Simple | 机内NVLink + 机间IB |
| 3-Tier Clos | Leaf-Spine-Core |
| Fat-Tree | 数据中心胖树 |
| CloudMatrix | 384 NPU全对等超节点 |

---

### 7. Preset Layer

**职责**: YAML配置文件管理，preset分类标记

**Model Preset (configs/models/*.yaml)**:

```yaml
description: Llama 7B
architecture: llama
preset_type: model  # model/pipeline/component

config:
  hidden_size: 4096
  num_layers: 32
  num_heads: 32
  
param_schema:
  training:
    - name: batch_size
      type: number
      default: 32
    - name: seq_len
      type: number
      default: 2048
  inference:
    - name: batch_size
      type: number
      default: 1
    - name: prompt_len
      type: number
      default: 512
```

**Preset分类**:

| preset_type | 说明 | 推理模式显示 |
|-------------|------|-------------|
| `pipeline` | 完整pipeline | 显示 |
| `model` | 完整单模型 | 显示 |
| `component` | 单组件 | 过滤 |

---

## 数据流

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   User   │───▶│ Evaluator│───▶│ Workload │───▶│ Analyzer │
└──────────┘    └──────────┘    │  Loader  │    └──────────┘
                                  └──────────┘          │
                                                        ▼
┌──────────┐                                    ┌──────────────┐
│  Preset  │                                    │ UnifiedResult│
│  Loader  │                                    │  - phases    │
└──────────┘                                    │  - mfu/qps   │
      │                                         │  - breakdown │
      ▼                                         └──────────────┘
┌──────────────┐                                       │
│ ShardedModule│                                       │
│  .bind(ctx)  │                                       │
└──────────────┘                                       │
      │                                                │
      ▼                                                │
┌──────────────┐                                       │
│ModuleInstance│                                       │
│ - flops_phys │                                       │
│ - submodules │───────────────────────────────────────┘
└──────────────┘
```

---

## 扩展点

### 添加新模型

**方式1**: YAML preset配置

```yaml
# configs/models/my-model.yaml
description: My Custom Model
architecture: my_arch
preset_type: model

config:
  hidden_size: 4096
  num_layers: 32
  
param_schema:
  training:
    - name: batch_size
      type: number
      default: 32
```

**方式2**: 代码定义

```python
from llm_perf.modeling import ShardedModule, ShardedTensor

class MyModel(ShardedModule):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.weight = ShardedTensor(
            shape=(hidden_size, hidden_size),
            shardable={0: "tp"},
        )
    
    def forward(self, x):
        return x @ self.weight
```

---

### 添加新Workload

```yaml
# configs/workloads/autoregressive/my-workload.yaml
name: my-workload
description: My Custom Workload
workload_type: training

phases:
  - name: forward
    compute_type: forward
    component: main
    compute_pattern: transformer_block
    repeat: 1

default_params:
  batch_size: 32
  seq_len: 2048
```

---

## 技术选型

| 技术点 | 选型 | 理由 |
|--------|------|------|
| 性能模型 | Roofline + bind机制 | 统一建模，自动分解 |
| 子模块分解 | ModuleInstance._submodule_instances | 从模型结构自动获取 |
| 工作负载配置 | YAML | 与模型解耦，易扩展 |
| preset分类 | preset_type标记 | 前端过滤无效选项 |
| 指标计算 | MFU/QPS公式化 | 标准化指标 |

---

## 版本历史

### v5.1 (当前)
- **类型系统**: ShardedParameter vs ShardedTensor 区分，__setattr__ 自动注册机制
- **内存统计分离**: get_weights() / get_activations() 独立追踪
- **ShardedFFN**: ffn_act_type 参数支持 SWIGLU/GELU/RELU/SILU
- **FFN权重差异**: gated(3权重) vs non-gated(2权重)
- **TopologyLevel**: 层级结构定义，levels 列表替代 legacy 属性
- **拓扑工厂**: create_clos_3tier/create_fat_tree/create_cloudmatrix_supernode

### v5.0
- **Workload Layer**: YAML配置工作负载，与模型解耦
- **ComputePattern**: 按负载特征分类（transformer_block/conv_encoder等）
- **Analyzer**: UnifiedAnalyzer统一分析器
- **MFU/QPS**: 核心指标
- **子模块分解**: bind机制获取_submodule_instances
- **通信分解**: CommunicationBreakdown
- **混布评估**: ColocateAnalyzer
- **Preset分类**: preset_type标记（model/pipeline/component）
- **前端动态**: param_schema驱动参数渲染
- **自动激活内存追踪**: ShardedTensor._is_view + KernelResult.saved_inputs + Op.get_saved_tensors()

### v4.0
- **Unified Modeling Layer**: Torch-like接口
- **bind机制**: 获取ModuleInstance物理分解
- **Functional API**: kernel定义
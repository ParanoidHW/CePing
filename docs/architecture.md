# 架构设计（v5.0）

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
│  │    Device (GPU)    │  │    Cluster (Network Topology)        │  │
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

### v5.0 (当前)
- **Workload Layer**: YAML配置工作负载，与模型解耦
- **ComputePattern**: 按负载特征分类（transformer_block/conv_encoder等）
- **Analyzer**: UnifiedAnalyzer统一分析器
- **MFU/QPS**: 核心指标
- **子模块分解**: bind机制获取_submodule_instances
- **通信分解**: CommunicationBreakdown
- **混布评估**: ColocateAnalyzer
- **Preset分类**: preset_type标记（model/pipeline/component）
- **前端动态**: param_schema驱动参数渲染

### v4.0
- **Unified Modeling Layer**: Torch-like接口
- **bind机制**: 获取ModuleInstance物理分解
- **Functional API**: kernel定义
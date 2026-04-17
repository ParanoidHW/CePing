# 架构设计

## 整体架构

LLM Performance Evaluator 采用分层架构设计，将应用接口、场景建模、性能分析、Kernel评估、策略配置、硬件抽象和模型定义解耦，实现高内聚低耦合的系统结构。

**设计目标**:
1. **易用性**: 一行代码评估性能，自动搜索最优策略，给定预算求最大Batch
2. **高可扩展性**: 新模型/新场景通过继承基类并注册，无需修改现有代码
3. **合理分层**: Kernel评估支持多种方案（理论/实测/微架构），避免霰弹式修改
4. **框架调度预留**: 为上层训推框架的调度特性建模预留扩展接口

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Application Layer (易用性)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │   Evaluator  │  │  Optimizer   │  │      BatchOptimizer      │   │
│  │ (便捷评估)   │  │ (策略搜索)   │  │  (给定预算求最大Batch)   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     Scenario Layer (高可扩展)                        │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐      │
│  │ LLM Train  │ │ LLM Infer  │ │ PD-Disagg  │ │  RL-Train    │      │
│  │(传统训练)  │ │(传统推理)  │ │(PD分离)    │ │ (RL后训练)   │      │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘      │
│  ┌────────────┐ ┌────────────┐ ┌────────────────────────────────┐   │
│  │ Diffusion  │ │ MultiModal │ │      Custom Scenario           │   │
│  │(扩散生成)  │ │ (多模态)   │ │    (用户自定义场景)            │   │
│  └────────────┘ └────────────┘ └────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                    Scheduler Layer (框架调度预留)                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  SchedulerFeature: Overlap / PipelineBubble / Chunking / ...  │  │
│  │  (计算通信重叠、PP Bubble、序列分块、KV预取等调度特性建模)    │  │
│  └───────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Reporter Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐      │
│  │   Console   │  │    JSON     │  │          HTML           │      │
│  │   Table     │  │   Export    │  │    Visualization        │      │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘      │
├─────────────────────────────────────────────────────────────────────┤
│                        Analyzer Layer                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐  │
│  │Training Ana  │ │Inference Ana │ │ Memory Est.  │ │ Comm Est.  │  │
│  │ - Throughput │ │ - TTFT/TPOT  │ │ - Parameters │ │ - AllRed.  │  │
│  │ - Memory     │ │ - KV Cache   │ │ - Activations│ │ - AllToAll │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘  │
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
├─────────────────────────────────────────────────────────────────────┤
│                        Hardware Layer                               │
│  ┌────────────────────┐  ┌──────────────────────────────────────┐  │
│  │    Device (GPU)    │  │    Cluster (Network Topology)        │  │
│  │  - Compute TFLOPS  │  │  - 2-Tier / 3-Tier Clos             │  │
│  │  - Memory BW       │  │  - Fat-Tree / CloudMatrix           │  │
│  │  - Memory Capacity │  │  - Bandwidth Hierarchy              │  │
│  └────────────────────┘  └──────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Model Layer                                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐      │
│  │ Llama      │ │    MoE     │ │ DeepSeek   │ │    ResNet    │      │
│  │ - Attention│ │ - Experts  │ │ - MLA      │ │ - CNN Blocks │      │
│  │ - FFN      │ │ - Router   │ │ - GQA      │ │ - Conv2d     │      │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘      │
│  ┌────────────┐ ┌────────────┐ ┌────────────────────────────────┐   │
│  │    VAE     │ │ Wan Video  │ │         Custom Model            │   │
│  │ - Conv3d   │ │ - DiT      │ │      (用户自定义模型)           │   │
│  └────────────┘ └────────────┘ └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 模块职责

### 1. Application Layer (易用性API)

**职责**: 提供便捷的高层接口，让用户能快速评估性能、搜索最优策略、优化Batch

**核心类**:

| 类名 | 职责 | 主要方法 |
|------|------|----------|
| `Evaluator` | 便捷评估 | `evaluate_training()`, `evaluate_inference()`, `compare_strategies()` |
| `StrategyOptimizer` | 策略搜索 | `search_best_strategy()`, `compare_strategies()` |
| `BatchOptimizer` | Batch优化 | `find_max_batch()`, `find_max_tps()`, `estimate_memory_for_batch()` |
| `LatencyBudget` | 时延预算 | `ttft_budget_ms`, `tpot_budget_ms`, `total_latency_budget_ms` |

**使用示例**:

```python
from llm_perf import Evaluator, StrategyOptimizer, BatchOptimizer, LatencyBudget

# 一行代码评估训练性能
result = Evaluator.evaluate_training(
    "llama-70b", "h100_8gpu", "tp8",
    batch_size=32, seq_len=4096,
    detailed=True,
)
print(f"Throughput: {result.tokens_per_sec} tokens/s")
print(f"Memory: {result.memory_per_gpu_gb} GB")

# 一行代码评估推理性能
result = Evaluator.evaluate_inference(
    "llama-70b", "h100_8gpu", "tp8",
    batch_size=8, prompt_len=1024, generation_len=128,
)
print(f"TTFT: {result.prefill_time_sec*1000} ms")
print(f"TPOT: {result.decode_time_per_step_sec*1000} ms")

# 自动搜索最优策略
candidates = StrategyOptimizer.search_best_strategy(
    "llama-70b", "h100_32gpu",
    mode="training",
    objective="throughput",
    constraints={"max_memory_gb": 80},
)
for c in candidates:
    print(f"TP={c.tp}, PP={c.pp}, Throughput={c.throughput}")

# 给定预算求最大Batch
result = BatchOptimizer.find_max_batch(
    "llama-70b", "h100_8gpu", "tp8",
    mode="inference",
    latency_budget=LatencyBudget(ttft_budget_ms=50.0, tpot_budget_ms=2.0),
    memory_budget_gb=80,
)
print(f"Max batch: {result.best_batch_size}")
```

### 2. Scenario Layer (高可扩展场景)

**职责**: 封装不同场景的评估逻辑，新场景通过继承并注册即可支持

**核心类**:

| 类名 | 职责 |
|------|------|
| `Scenario` | 场景抽象基类 |
| `ScenarioRegistry` | 场景注册中心 |
| `LLMTrainingScenario` | 传统LLM训练场景 |
| `LLMInferenceScenario` | 传统LLM推理场景 |
| `PDDisaggScenario` | PD分离推理场景（Prefill-Decode分离） |
| `RLTrainingScenario` | RL后训练场景（PPO/DPO等） |
| `DiffusionScenario` | 扩散生成场景（文生图/文生视频） |

**添加新场景**:

```python
from llm_perf.scenarios import Scenario, ScenarioRegistry

class MyCustomScenario(Scenario):
    """自定义场景示例."""
    
    name = "my_custom_scenario"
    
    def get_analyzer(self, model, device, cluster, strategy):
        return MyCustomAnalyzer(model, device, cluster, strategy)
    
    def analyze(self, config):
        # 自定义分析逻辑
        return result

# 注册场景（无需修改现有代码）
ScenarioRegistry.register(MyCustomScenario)

# 使用场景
from llm_perf import Evaluator
result = Evaluator.evaluate_scenario(
    scenario="my_custom_scenario",
    model="my_model",
    hardware="h100_8gpu",
)
```

### 3. Scheduler Layer (框架调度预留)

**职责**: 为上层训推框架的调度特性建模预留扩展接口

**核心类**:

| 类名 | 职责 |
|------|------|
| `SchedulerFeature` | 调度特性抽象基类 |
| `SchedulerModel` | 调度模型组合器 |
| `OverlapFeature` | 计算通信重叠建模 |
| `PipelineBubbleFeature` | Pipeline Bubble 建模 |
| `ChunkingFeature` | 序列分块内存优化 |
| `PrefetchFeature` | KV预取时延优化 |

**使用示例**:

```python
from llm_perf.scheduler import SchedulerModel, OverlapFeature
from llm_perf.strategy.base import StrategyConfig

# 启用计算通信重叠
strategy = StrategyConfig(
    tp_degree=8,
    scheduler_features=["overlap"],
    overlap_enabled=True,
    overlap_efficiency=0.8,
)

# 在分析结果中应用调度特性
scheduler = SchedulerModel.from_strategy(strategy)
result = scheduler.apply_all(analyzer_result)
```

### 4. Kernel Layer (可插拔 Backend)

**职责**: 独立评估算子和通信性能，支持多种评估方案

**Backend 架构**:

| Backend | 描述 | 适用场景 |
|---------|------|----------|
| `TheoryBackend` | FLOPs/Roofline理论评估 | 快速估算、架构分析 |
| `ProfilingBackend` | 实测数据查找和插值 | 精确建模、实际部署 |
| `MicroarchBackend` | 基于硬件微架构建模 | 深度优化、硬件调优 |

**切换 Backend**:

```python
from llm_perf.kernels.backend import KernelBackendRegistry

# 默认使用理论评估
backend = KernelBackendRegistry.get_backend("theory")

# 切换到实测数据评估
KernelBackendRegistry.set_default_backend("profiling")

# 加载实测数据
ProfilingBackend.load_data("profiling_results.json")

# 或使用微架构评估（预留）
KernelBackendRegistry.set_default_backend("microarch")
```

**Kernel 分类**:

| 类型 | 计算单元 | 典型 Kernel | 特点 |
|------|----------|-------------|------|
| Compute (Dense) | CUBE/Tensor Core | `linear`, `bmm`, `conv2d` | 高算术强度 |
| Compute (Sparse) | CUBE/Tensor Core | `flash_attention`, `mla_attention` | 内存优化 |
| Element-wise | VECTOR/CUDA Core | `silu`, `rms_norm`, `softmax` | 低算术强度 |
| Communication | N/A | `allreduce`, `alltoall` | 网络带宽受限 |

### 5. Strategy Layer

**职责**: 管理并行策略、优化器配置、调度特性

**支持的并行方式**:

| 并行类型 | 缩写 | 通信模式 | 适用场景 |
|----------|------|----------|----------|
| Tensor Parallelism | TP | AllReduce | 大模型单节点 |
| Pipeline Parallelism | PP | P2P | 跨节点扩展 |
| Data Parallelism | DP | AllReduce | ZeRO优化 |
| Expert Parallelism | EP | AllToAll | MoE模型 |
| Sequence Parallelism | SP | AllGather | 长序列 |
| Context Parallelism | CP | Ring P2P | 超长序列 |

**优化器建模**:

| 优化器 | 内存倍数 | 特点 |
|--------|----------|------|
| AdamW | 2x params (m+v) | 标准优化器 |
| Lion | 1x params (m only) | 内存节省 |
| Muon | 预留 | 自定义优化器 |

### 6. Analyzer Layer

**职责**: 综合分析性能，生成详细分解报告

**分析维度**:
- **计算时间**: 基于 Roofline 模型，支持 backward 估算
- **通信时间**: 基于通信算法和拓扑带宽
- **内存占用**: 参数、激活、KV Cache、优化器状态
- **详细分解**: 层级分解、Kernel分解、通信分解

**模块拆分**:

| 模块 | 职责 |
|------|------|
| `training.py` | 训练性能分析主逻辑 |
| `inference.py` | 推理性能分析主逻辑 |
| `memory.py` | 内存估算模块 |
| `communication.py` | 通信估算模块 |

### 7. Hardware Layer

**职责**: 抽象硬件能力和网络拓扑

**预设设备**:

| 设备 | FP16 TFLOPS | 显存 | 内存带宽 | 特点 |
|------|-------------|------|----------|------|
| H100-SXM-80GB | 989 | 80GB | 3.35 TB/s | NVIDIA旗舰 |
| A100-SXM-80GB | 312 | 80GB | 2.04 TB/s | Ampere架构 |
| MI300X | 1307 | 192GB | 5.3 TB/s | AMD大显存 |
| Ascend-910B2 | 376 | 64GB | 1.6 TB/s | 华为昇腾 |

**网络拓扑**:

| 拓扑类型 | 描述 | 适用场景 |
|----------|------|----------|
| 2-Tier Simple | 机内NVLink + 机间IB | 单机/小集群 |
| 3-Tier Clos | Leaf-Spine-Core | 中型数据中心 |
| Fat-Tree | 数据中心胖树 | 大规模集群 |
| CloudMatrix | 384 NPU全对等超节点 | 华为超节点 |

### 8. Model Layer

**职责**: 定义模型结构和层配置

**支持的模型架构**:

| 模型 | 特点 | Kernel支持 |
|------|------|-----------|
| Llama | Dense Transformer | standard attention |
| MoE (Mixtral) | 专家混合架构 | expert routing |
| DeepSeek-V2/V3 | MLA + DeepSeekMoE | mla_attention |
| ResNet | CNN视觉模型 | conv2d |
| VAE | 3D卷积编解码 | conv3d |
| Wan Video | DiT视频生成 | flash_attention |

---

## 数据流

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   User   │───▶│Application│───▶│ Scenario │───▶│ Analyzer │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────┐
                    │                                 │                 │
                    ▼                                 ▼                 ▼
              ┌──────────┐                    ┌──────────┐       ┌──────────┐
              │ Strategy │                    │  Kernel  │       │ Hardware │
              └──────────┘                    │(Backend) │       └──────────┘
                    │                         └──────────┘              │
                    │                              │                    │
                    └──────────────────────────────┼────────────────────┘
                                                   ▼
                                           ┌──────────┐
                                           │  Model   │
                                           └──────────┘
                                                   │
                                                   ▼
                                           ┌──────────┐
                                           │ Reporter │───▶ Output
                                           └──────────┘
```

---

## 扩展点

### 添加新模型

```python
from llm_perf.models.base import BaseModel, LayerConfig
from llm_perf.kernels import linear, rms_norm, flash_attention
from llm_perf.kernels.utils import kernel_result_to_layer

class MyModel(BaseModel):
    def build_layers(self) -> List[LayerConfig]:
        layers = []
        
        # 使用 Kernel API 构建层
        q_result = linear(input=(batch, seq, hidden), weight=(q_dim, hidden))
        layers.append(kernel_result_to_layer(name="q_proj", result=q_result))
        
        attn_result = flash_attention(...)
        layers.append(kernel_result_to_layer(name="attention", result=attn_result))
        
        return layers

# 注册模型
from llm_perf.models import ModelRegistry
ModelRegistry.register("my_model", MyModel, MyConfig)
```

### 添加新 Kernel Backend

```python
from llm_perf.kernels.backend import KernelBackend, KernelBackendRegistry

class MyCustomBackend(KernelBackend):
    def estimate_compute_time(self, kernel_name, input_shapes, output_shape, dtype, device):
        # 自定义评估逻辑
        return computed_time
    
    def estimate_comm_time(self, comm_type, data_size_bytes, num_ranks, bandwidth_gbps):
        # 自定义通信评估
        return comm_time

# 注册 Backend
KernelBackendRegistry.register_backend("my_backend", MyCustomBackend)
KernelBackendRegistry.set_default_backend("my_backend")
```

### 添加新调度特性

```python
from llm_perf.scheduler import SchedulerFeature

class MySchedulerFeature(SchedulerFeature):
    name = "my_feature"
    
    def apply_overlap(self, compute_time, comm_time):
        # 自定义重叠逻辑
        return overlapped_time
    
    def apply_memory_optimization(self, base_memory):
        # 自定义内存优化
        return optimized_memory

# 使用特性
strategy = StrategyConfig(
    tp_degree=8,
    scheduler_features=["my_feature"],
)
```

---

## 关键技术选型

| 技术点 | 选型 | 理由 |
|--------|------|------|
| 性能模型 | Roofline + Backend | 统一描述计算和内存瓶颈，支持多种评估方案 |
| 通信模型 | Ring/Tree 算法 | 实际分布式训练常用算法 |
| 架构风格 | 分层 + 插件 + 注册 | 易于扩展，避免霰弹式修改 |
| 易用性 | Application Layer | 一行代码评估，降低使用门槛 |
| 可扩展性 | Scenario + Registry | 新场景无需修改现有代码 |
| Kernel API | Torch-like functional | 开发者友好，易于上手 |

---

## 版本历史

### v3.0 (最新)
- **Application Layer**: Evaluator/StrategyOptimizer/BatchOptimizer 便捷API
- **Scenario Layer**: 场景基类和注册机制，支持LLM/PD-Disagg/RL/Diffusion
- **Kernel Backend**: Theory/Profiling/Microarch 三种可插拔评估方案
- **Scheduler Layer**: Overlap/PipelineBubble/Chunking 调度特性预留
- **LatencyBudget**: 支持 TTFT/TPOT/Total latency 分离预算
- **Analyzer拆分**: memory.py/communication.py 独立模块

### v2.0
- **Kernel API**: 简化 `kernel_result_to_layer`，自动推断 dtype 和 params
- **Attention**: 支持 Flash Attention、GQA、MLA (absorb/non-absorb)
- **Memory Bound**: 基于算术强度和计算单元类型自动判断
- **网络拓扑**: Clos/Fat-Tree/CloudMatrix 支持

### v1.0
- **基础架构**: Model/Hardware/Kernel/Strategy/Analyzer 分层设计
- **模型支持**: Llama/MoE/DeepSeek 基础模型
- **硬件支持**: NVIDIA/AMD/Huawei GPU预设
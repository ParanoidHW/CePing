# 架构改进方案 V3

## 一、当前架构不足分析

### 1. 易用性不足
- **缺乏高层API**：用户需要手动组装 Model、Device、Cluster、Strategy、Analyzer 等多个组件
- **缺乏策略搜索**：没有提供自动搜索最佳并行方案的接口
- **缺乏Batch优化**：没有提供给定预算（时延、内存）确定最大并发数的接口

### 2. 高可扩展性不足
- **新场景扩展困难**：PD分离、RL后训练等场景需要修改多处代码
- **新优化器扩展困难**：AdamW、Lion、Muon等优化器没有统一的建模接口
- **新并行方案扩展困难**：新并行方案（如4D并行）需要修改多个文件

### 3. Kernel评估分层不足
- **单一评估方案**：目前只支持 FLOPs/Roofline 理论评估
- **无 profiling 插值支持**：无法使用实际测试数据校准
- **无微架构评估支持**：无法基于硬件微架构特性建模

### 4. 框架调度预留不足
- **无调度特性建模**：overlap、pipeline bubble、chunking等调度特性没有预留接口
- **无运行时模型**：无法建模实际运行时的动态特性

---

## 二、改进方案

### 新架构分层

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer (新增)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Evaluator  │  │  Optimizer  │  │    BatchOptimizer       │  │
│  │ (便捷评估)  │  │(策略搜索)   │  │  (给定预算求最大batch)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Scenario Layer (新增)                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ LLM Train   │ │ LLM Infer   │ │ PD-Disagg   │ │ RL-Train  │  │
│  │(传统训练)   │ │(传统推理)   │ │(PD分离推理) │ │(RL后训练) │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────────┐   │
│  │ Diffusion   │ │ MultiModal  │ │         Custom            │   │
│  │(扩散生成)   │ │(多模态)     │ │(用户自定义场景)           │   │
│  └─────────────┘ └─────────────┘ └───────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   Scheduler Layer (新增预留)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SchedulerModel: Overlap / Bubble / Chunking / Prefetch   │   │
│  │  (框架调度特性建模预留接口)                               │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        Analyzer Layer                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │TrainingAna  │ │InferenceAna │ │ Memory Est. │ │ Comm Est. │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Strategy Layer                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ TP/PP/DP/EP │ │ SP/CP       │ │ Optimizer   │ │ Scheduler  │  │
│  │ (并行策略)  │ │(序列并行)   │ │(优化器建模) │ │(调度策略) │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Kernel Layer (重构)                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              KernelBackend (可插拔评估方案)                 │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │ │
│  │  │ TheoryBack  │ │ ProfilingBk │ │ MicroarchBackend    │   │ │
│  │  │(FLOPs理论)  │ │(实测数据)   │ │(微架构建模)         │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────┐  ┌──────────────────────────────┐           │
│  │ Compute Kernels│  │   Communication Kernels      │           │
│  └────────────────┘  └──────────────────────────────┘           │
├─────────────────────────────────────────────────────────────────┤
│                       Model Layer                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ Llama       │ │ MoE         │ │ DeepSeek    │ │ Custom    │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       Hardware Layer                            │
│  ┌────────────────┐  ┌──────────────────────────────┐           │
│  │    Device      │  │   Topology (拓扑可插拔)       │           │
│  └────────────────┘  └──────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、新增模块详细设计

### 3.1 Application Layer - 易用性API

#### Evaluator（便捷评估）

```python
# llm_perf/app/evaluator.py

class Evaluator:
    """便捷性能评估接口."""
    
    def evaluate_training(
        model: str | ModelConfig | BaseModel,
        hardware: str | dict,
        strategy: str | StrategyConfig,
        batch_size: int,
        seq_len: int,
        detailed: bool = True,  # 是否返回详细分解
    ) -> TrainingResult:
        """评估训练性能."""
        pass
    
    def evaluate_inference(
        model: str | ModelConfig | BaseModel,
        hardware: str | dict,
        strategy: str | StrategyConfig,
        batch_size: int,
        prompt_len: int,
        generation_len: int,
        detailed: bool = True,
    ) -> InferenceResult:
        """评估推理性能."""
        pass
    
    def evaluate_pipeline(
        pipeline: str | Pipeline,
        hardware: str | dict,
        strategy: str | StrategyConfig,
        **kwargs,
    ) -> PipelineResult:
        """评估 Pipeline 性能."""
        pass
```

#### StrategyOptimizer（策略搜索）

```python
# llm_perf/app/optimizer.py

class StrategyOptimizer:
    """并行策略优化器."""
    
    def search_best_strategy(
        model: str | ModelConfig,
        hardware: str | dict,
        mode: str = "training",  # training | inference
        constraints: Optional[Dict] = None,  # 内存约束、GPU数量约束等
        objective: str = "throughput",  # throughput | latency | memory
        method: str = "grid",  # grid | greedy | genetic
    ) -> List[StrategyCandidate]:
        """搜索最优并行策略."""
        pass
    
    def compare_strategies(
        model: str | ModelConfig,
        hardware: str | dict,
        strategies: List[StrategyConfig],
        mode: str = "training",
    ) -> ComparisonReport:
        """对比多个策略."""
        pass
```

#### BatchOptimizer（Batch优化）

```python
# llm_perf/app/batch_optimizer.py

class BatchOptimizer:
    """给定预算求最大Batch/TPS."""
    
    def find_max_batch(
        model: str | ModelConfig,
        hardware: str | dict,
        strategy: str | StrategyConfig,
        mode: str = "inference",
        latency_budget_ms: Optional[float] = None,  # 时延预算
        memory_budget_gb: Optional[float] = None,  # 内存预算
        target_tps: Optional[float] = None,  # 目标TPS
    ) -> BatchOptimizationResult:
        """给定预算求最大Batch."""
        pass
    
    def find_max_tps(
        model: str | ModelConfig,
        hardware: str | dict,
        mode: str = "inference",
        strategy_constraints: Optional[Dict] = None,
    ) -> TPSOptimizationResult:
        """求最大TPS（自动搜索策略+batch）."""
        pass
```

### 3.2 Scenario Layer - 高可扩展性

#### Scenario 基类

```python
# llm_perf/scenarios/base.py

class ScenarioConfig:
    """场景配置基类."""
    name: str
    description: str
    required_models: List[str]
    supported_parallelisms: List[ParallelType]

class Scenario(ABC):
    """场景抽象基类."""
    
    @abstractmethod
    def build_pipeline(self, models: Dict[str, BaseModel]) -> Pipeline:
        """构建该场景的 Pipeline."""
        pass
    
    @abstractmethod
    def get_analyzer(self) -> Analyzer:
        """获取该场景的分析器."""
        pass
    
    def register(self):
        """注册到 ScenarioRegistry."""
        pass
```

#### 具体场景实现

```python
# llm_perf/scenarios/llm_training.py

class LLMTrainingScenario(Scenario):
    """传统LLM训练场景."""
    
    def build_pipeline(self, models):
        return SimplePipeline(model=models["llm"])
    
    def get_analyzer(self):
        return TrainingAnalyzer

# llm_perf/scenarios/pd_disagg.py

class PDDisaggScenario(Scenario):
    """PD分离推理场景."""
    
    def build_pipeline(self, models):
        # Prefill 和 Decode 分离到不同节点
        return PDDisaggPipeline(
            prefill_model=models["prefill"],
            decode_model=models["decode"],
        )
    
    def get_analyzer(self):
        return PDDisaggAnalyzer

# llm_perf/scenarios/rl_training.py

class RLTrainingScenario(Scenario):
    """RL后训练场景（PPO、DPO等）."""
    
    def build_pipeline(self, models):
        return RLPipeline(
            policy_model=models["policy"],
            reward_model=models["reward"],
            reference_model=models["reference"],
        )
```

### 3.3 Kernel Backend - 多评估方案

```python
# llm_perf/kernels/backend/base.py

class KernelBackend(ABC):
    """Kernel评估后端基类."""
    
    @abstractmethod
    def estimate_compute_time(
        kernel_name: str,
        input_shapes: List[Tuple],
        output_shape: Tuple,
        dtype: str,
        device: Device,
    ) -> float:
        """估算计算时间."""
        pass
    
    @abstractmethod
    def estimate_comm_time(
        comm_type: str,  # allreduce, allgather, alltoall
        data_size_bytes: int,
        num_ranks: int,
        bandwidth_gbps: float,
    ) -> float:
        """估算通信时间."""
        pass

# llm_perf/kernels/backend/theory.py

class TheoryBackend(KernelBackend):
    """理论评估后端（FLOPs/Roofline）."""
    
    def estimate_compute_time(...):
        # 使用 FLOPs 和 Roofline 模型
        flops = calculate_flops(...)
        achievable_flops = min(peak_flops, ai * mem_bw)
        return flops / achievable_flops

# llm_perf/kernels/backend/profiling.py

class ProfilingBackend(KernelBackend):
    """实测数据后端."""
    
    def __init__(self, profiling_data: Dict):
        self.data = profiling_data  # 从实际测试加载
    
    def estimate_compute_time(...):
        # 使用插值或查找表
        return self._lookup_or_interpolate(...)

# llm_perf/kernels/backend/microarch.py

class MicroarchBackend(KernelBackend):
    """微架构评估后端."""
    
    def estimate_compute_time(...):
        # 基于硬件微架构特性建模
        # 如 CUDA Core 数量、Tensor Core 周期、内存层级等
        pass
```

### 3.4 Scheduler Layer - 框架调度预留

```python
# llm_perf/scheduler/base.py

class SchedulerFeature(ABC):
    """调度特性抽象基类."""
    
    @abstractmethod
    def apply_overhead(self, base_time: float) -> float:
        """应用调度开销."""
        pass
    
    @abstractmethod
    def apply_overlap(self, compute_time: float, comm_time: float) -> float:
        """应用overlap优化."""
        pass

class OverlapFeature(SchedulerFeature):
    """计算-通信重叠特性."""
    pass

class PipelineBubbleFeature(SchedulerFeature):
    """Pipeline Bubble 建模."""
    pass

class ChunkingFeature(SchedulerFeature):
    """序列分块特性."""
    pass

class SchedulerModel:
    """调度模型组合."""
    
    def __init__(self, features: List[SchedulerFeature]):
        self.features = features
    
    def apply_all(self, result: AnalysisResult) -> AnalysisResult:
        """应用所有调度特性."""
        pass
```

### 3.5 Optimizer Layer - 优化器建模

```python
# llm_perf/strategy/optimizer.py

class OptimizerConfig:
    """优化器配置."""
    name: str  # adamw, lion, muon, etc.
    memory_multiplier: float  # 优化器状态内存倍数
    # Adam: 2x params (m + v)
    # Lion: 1x params (m only)
    # Muon: ...

class OptimizerRegistry:
    """优化器注册中心."""
    
    def register_optimizer(
        name: str,
        memory_multiplier: float,
        compute_overhead: Optional[float] = None,
    ):
        pass
```

---

## 四、实现计划

### Phase 1: Application Layer（易用性）
1. 创建 `llm_perf/app/` 目录
2. 实现 `Evaluator` 便捷评估接口
3. 实现 `StrategyOptimizer` 策略搜索
4. 实现 `BatchOptimizer` Batch优化

### Phase 2: Scenario Layer（高可扩展性）
1. 创建 `llm_perf/scenarios/` 目录
2. 实现场景基类和注册机制
3. 实现具体场景：LLMTrain, LLMInfer, PD-Disagg, RL-Train

### Phase 3: Kernel Backend（分层）
1. 创建 `llm_perf/kernels/backend/` 目录
2. 实现后端基类
3. 实现 TheoryBackend（保持现有逻辑）
4. 实现 ProfilingBackend 框架

### Phase 4: Scheduler/Optimizer Layer（预留）
1. 创建 `llm_perf/scheduler/` 目录
2. 实现调度特性基类
3. 扩展 OptimizerConfig 和 OptimizerRegistry

---

## 五、易用性示例

### 评估整网性能

```python
from llm_perf import Evaluator

# 一行代码评估训练性能
result = Evaluator.evaluate_training(
    model="llama-70b",
    hardware="h100_8gpu",
    strategy="tp8",
    batch_size=32,
    seq_len=4096,
    detailed=True,
)

# 获取详细分解
print(result.breakdown.compute_time)
print(result.breakdown.comm_time)
print(result.breakdown.memory_gb)
```

### 搜索最优策略

```python
from llm_perf import StrategyOptimizer

# 自动搜索最优策略
candidates = StrategyOptimizer.search_best_strategy(
    model="llama-70b",
    hardware="h100_32gpu",
    mode="training",
    constraints={"max_memory_gb": 80},
    objective="throughput",
)

for candidate in candidates:
    print(f"Strategy: {candidate.strategy}")
    print(f"Throughput: {candidate.throughput}")
```

### 给定预算求最大Batch

```python
from llm_perf import BatchOptimizer

# 给定时延预算求最大Batch
result = BatchOptimizer.find_max_batch(
    model="llama-70b",
    hardware="h100_8gpu",
    strategy="tp8",
    latency_budget_ms=50,  # 要求 TTFT < 50ms
)

print(f"Max batch: {result.max_batch}")
print(f"TPS: {result.tps}")
```

### 自定义场景扩展

```python
from llm_perf.scenarios import Scenario, ScenarioRegistry

class MyCustomScenario(Scenario):
    """自定义场景."""
    
    name = "my_scenario"
    
    def build_pipeline(self, models):
        return MyCustomPipeline(...)
    
    def get_analyzer(self):
        return MyCustomAnalyzer

# 注册场景
ScenarioRegistry.register(MyCustomScenario)

# 使用场景
from llm_perf import Evaluator
result = Evaluator.evaluate_scenario(
    scenario="my_scenario",
    hardware="h100_8gpu",
    ...
)
```

---

## 六、对现有代码的影响

### 需要修改的模块
- `llm_perf/analyzer/`: 添加对 KernelBackend 的支持
- `llm_perf/kernels/`: 重构为使用 Backend
- `llm_perf/strategy/`: 添加 OptimizerConfig
- `docs/architecture.md`: 更新架构文档

### 不需要修改的模块（保持稳定）
- `llm_perf/models/`: 模型层不变
- `llm_perf/hardware/`: 硬件层不变
- `llm_perf/reporter/`: 报告层不变

### 新增模块
- `llm_perf/app/`: Application Layer
- `llm_perf/scenarios/`: Scenario Layer
- `llm_perf/kernels/backend/`: Kernel Backend
- `llm_perf/scheduler/`: Scheduler Layer
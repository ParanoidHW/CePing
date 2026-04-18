# 架构改进设计方案（修订版）

## 分析日期
2026-04-18

## 设计原则

**核心约束**：所有评估过程基于新架构的Sharded模型机制：
- ShardedModule.forward() 记录 op_history
- ModuleInstance.bind(ctx) 获取物理分解
- 从 ModuleInstance._submodule_instances 获取子模块分解

---

## 新架构核心机制理解

### ShardedModule机制

```
ShardedModule（建模）
    ├── forward() → 记录 op_history（MatmulOp, AttentionOp等）
    ├── _submodules → 子模块树（input_norm, attention, ffn等）
    ├── _weights → 权重张量（带shardable标记）
    └── bind(ctx) → ModuleInstance（物理实例）

ModuleInstance（切分后）
    ├── flops_forward_physical → 物理FLOPs（考虑TP/EP切分）
    ├── activation_memory_physical → 物理内存
    ├── total_comm_ops → 通信操作（AllReduce/AllToAll等）
    └── _submodule_instances → 子模块实例分解
        ├── attention_0 → {flops, memory, comm_ops}
        ├── ffn_0 → {flops, memory, comm_ops}
        └── ...
```

**关键发现**：
1. ShardedModule通过`forward()`执行，自动记录op_history
2. `bind(ctx)`后，ModuleInstance自动计算物理形状（考虑TP/PP/EP切分）
3. 子模块分解通过`_submodule_instances`获取，每个子模块有独立的性能数据

---

## Phase 1: MFU指标 + 子模块分解（高优先级）

### 1.1 MFU指标

**实现方案**：

```python
# llm_perf/analyzer/unified.py

def _calculate_mfu(self, total_flops: int, total_time_sec: float) -> float:
    """计算MFU，基于ModuleInstance.flops_forward_physical."""
    
    peak_tflops = self.device.config.compute_tflops_fp16
    num_devices = self.cluster.num_devices
    
    if peak_tflops <= 0 or total_time_sec <= 0:
        return 0.0
    
    theoretical_peak_flops = peak_tflops * 1e12 * num_devices * total_time_sec
    mfu = total_flops / theoretical_peak_flops
    
    return min(mfu, 1.0)
```

---

### 1.2 子模块分解（正确方案）

**核心改动**：使用`bind(ctx)`获取ModuleInstance，从`_submodule_instances`获取分解。

```python
# llm_perf/analyzer/unified.py

from llm_perf.strategy.parallel_context import ParallelContext

def _analyze_phase_with_submodules(
    self,
    component: ShardedModule,
    phase: Phase,
    params: Dict[str, Any],
) -> Tuple[float, float, int, List[SubmoduleResult]]:
    """使用ShardedModule机制分析phase和子模块."""
    
    # Step 1: 创建ParallelContext
    ctx = self._create_parallel_context(params)
    
    # Step 2: 执行forward（记录op_history）
    batch_size = params.get("batch_size", 1)
    seq_len = params.get("seq_len", 512)
    hidden_size = getattr(component, "hidden_size", 4096)
    
    input_tensor = ShardedTensor(shape=(batch_size, seq_len, hidden_size))
    output = component(input_tensor)
    
    # Step 3: bind获取ModuleInstance
    mode = "forward_backward" if phase.compute_type == ComputeType.BACKWARD else "forward"
    module_instance = component.bind(ctx, mode=mode)
    
    # Step 4: 从ModuleInstance获取性能数据
    flops = module_instance.flops_forward_physical
    memory = module_instance.activation_memory_physical / 1e9
    
    # Step 5: 从_submodule_instances获取子模块分解
    submodules = []
    for sub_name, sub_inst in module_instance._submodule_instances.items():
        submodules.append(SubmoduleResult(
            name=sub_name,
            submodule_type=self._infer_submodule_type(sub_name),
            time_sec=self._estimate_submodule_time(sub_inst),
            flops=sub_inst.flops_forward_physical,
            memory_gb=sub_inst.activation_memory_physical / 1e9,
            communication_bytes=sum(op.data_bytes for op in sub_inst.total_comm_ops),
        ))
    
    # Step 6: 时间估算
    backend = TheoryBackend(self.device)
    time_sec = module_instance.estimate_time(backend)
    
    return time_sec, memory, flops, submodules

def _create_parallel_context(self, params: Dict[str, Any]) -> ParallelContext:
    """从StrategyConfig创建ParallelContext."""
    return ParallelContext(
        tp_degree=self.strategy.tp_degree,
        pp_degree=self.strategy.pp_degree,
        ep_degree=self.strategy.ep_degree,
        sp_degree=self.strategy.sp_degree,
        dp_degree=self.strategy.dp_degree,
        dtype=params.get("dtype", "fp16"),
        device=self.device.config,
        activation_checkpointing=self.strategy.activation_checkpointing,
        zero_stage=self.strategy.zero_stage,
    )
```

---

### 1.3 扩展PhaseResult

```python
# llm_perf/analyzer/base.py

@dataclass
class SubmoduleResult:
    """子模块分解结果，从ModuleInstance获取."""
    name: str
    submodule_type: str
    time_sec: float
    flops: int
    memory_gb: float
    communication_bytes: int = 0

@dataclass
class PhaseResult:
    ...
    submodules: List[SubmoduleResult] = field(default_factory=list)
```

---

## Phase 2: QPS指标 + 通信分解

### 2.1 QPS指标

```python
def _calculate_qps(self, batch_size: int, total_time_sec: float) -> float:
    """计算QPS."""
    dp = self.strategy.dp_degree
    return batch_size * dp / total_time_sec if total_time_sec > 0 else 0.0
```

---

### 2.2 通信分解（从ModuleInstance.total_comm_ops获取）

```python
def _extract_communication_breakdown(self, phase_results) -> CommunicationResult:
    """从PhaseResult.submodules的communication_bytes提取."""
    
    comm_ops = {}
    for phase in phase_results:
        for sm in phase.submodules:
            if sm.communication_bytes > 0:
                op_type = self._infer_comm_op_type(sm.submodule_type)
                if op_type not in comm_ops:
                    comm_ops[op_type] = {"total_bytes": 0}
                comm_ops[op_type]["total_bytes"] += sm.communication_bytes
    
    return CommunicationResult(operations=comm_ops)
```

---

## Phase 3: 淩布评估

使用现有UnifiedAnalyzer（基于ShardedModule机制）：

```python
class ColocateAnalyzer:
    def analyze(self, models, allocations) -> Dict[str, UnifiedResult]:
        results = {}
        for model_name, model in models.items():
            analyzer = UnifiedAnalyzer(model, self.device, self.cluster, allocations[model_name])
            results[model_name] = analyzer.analyze(self.workload)
        return results
```

---

## 测试覆盖

```python
class TestSubmoduleBreakdown:
    def test_submodules_from_module_instance(self):
        """验证子模块从ModuleInstance获取."""
        model = LlamaModel(hidden_size=4096, num_layers=32)
        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = model(hidden)
        
        ctx = ParallelContext(tp_degree=8)
        module_inst = model.bind(ctx)
        
        assert len(module_inst._submodule_instances) > 0
        for sub_name, sub_inst in module_inst._submodule_instances.items():
            assert sub_inst.flops_forward_physical > 0
    
    def test_analyzer_uses_bind_mechanism(self):
        """验证UnifiedAnalyzer使用bind机制."""
        result = analyzer.analyze("training")
        forward_phase = result.get_phase("forward")
        assert len(forward_phase.submodules) > 0
```

---

## 新架构依赖总结

| 机制 | 文件 | 用途 |
|------|------|------|
| ShardedModule.forward() | `modeling/module.py` | 记录op_history |
| ModuleInstance.bind() | `modeling/module.py` | 获取物理分解 |
| _submodule_instances | `modeling/module.py` | 子模块性能分解 |
| flops_forward_physical | `modeling/module.py` | 物理FLOPs |
| total_comm_ops | `modeling/module.py` | 通信操作分解 |
| ParallelContext | `strategy/parallel_context.py` | 切分上下文 |
| TheoryBackend | `kernels/backend/theory.py` | 时间估算 |

**不依赖legacy，不直接调用functional.kernel**
# Breakdown数据缺失问题修复设计方案

## 问题根源分析

### 问题1: 新架构缺少 breakdown 和 detailed_breakdown

**旧架构返回格式** (legacy/analyzer/training.py:37-58):
```json
{
  "throughput": {"samples_per_sec": ..., "tokens_per_sec": ...},
  "time": {"time_per_step_sec": ..., "time_to_solution_sec": ...},
  "memory": {"memory_per_gpu_gb": ...},
  "breakdown": {
    "layers": [
      {"name": "embedding", "kernels": [{"name": "linear", "time_sec": ...}]},
      {"name": "attention_0", "kernels": [...]},
      {"name": "ffn_0", "kernels": [...]}
    ]
  },
  "detailed_breakdown": {
    "submodels": [...],
    "memory": {"by_type": {"parameters": ..., "activations": ...}},
    "communication": {"by_parallelism": {"tp": {...}, "dp": {...}}}
  }
}
```

**新架构返回格式** (analyzer/base.py:153-164):
```json
{
  "phases": [{"name": "forward", "total_time_sec": ...}],
  "total_time_sec": ...,
  "peak_memory_gb": ...,
  "throughput": {"tokens_per_sec": ...}
}
```

**根本原因**:
- `UnifiedResult` (analyzer/base.py:129-164) 只包含 phases、total_time、peak_memory、throughput
- 缺少 `breakdown` (layers/kernels breakdown) 和 `detailed_breakdown` (memory/communication breakdown)
- 新架构设计时未考虑前端已有的数据结构依赖

### 问题2: wan pipeline breakdown 数值为0

**前端期望** (app.js:468-475):
- 从 `result.phases` 提取 encode/denoise/decode 各阶段时间
- 计算 `encodePhase.total_time_sec`、`denoisePhase.total_time_sec`、`decodePhase.total_time_sec`

**问题分析**:
- diffusion-pipeline.yaml 配置正确定义了3个阶段: encode/denoise/decode
- `PhaseResult` 包含 `total_time_sec` 字段 (analyzer/base.py:110)
- 前端已正确读取 `phases` 数组

**可能原因**:
1. 模型组件创建失败，导致时间为0
2. component_mapping 配置正确但组件未正确传入
3. compute_pattern 估算逻辑返回0

---

## 修复方案

### 方案概述

采用 **兼容层方案**：在 `UnifiedResult` 中添加 `breakdown` 和 `detailed_breakdown` 字段，通过适配器从 phases 数据生成兼容格式。

### Step1: 扩展 UnifiedResult 数据结构

修改 `analyzer/base.py` 中的 `UnifiedResult`:

```python
@dataclass
class UnifiedResult:
    workload_name: str
    workload_type: WorkloadType
    phases: List[PhaseResult] = field(default_factory=list)
    total_time_sec: float = 0.0
    peak_memory_gb: float = 0.0
    throughput: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 新增兼容字段
    breakdown: Optional[Dict[str, Any]] = None
    detailed_breakdown: Optional[Dict[str, Any]] = None
```

修改 `to_dict()` 方法:
```python
def to_dict(self) -> Dict[str, Any]:
    result = {
        "workload_name": self.workload_name,
        ...
    }
    # 添加兼容字段
    if self.breakdown:
        result["breakdown"] = self.breakdown
    if self.detailed_breakdown:
        result["detailed_breakdown"] = self.detailed_breakdown
    return result
```

### Step2: 在 UnifiedAnalyzer 中生成 breakdown 数据

在 `unified.py` 的 `analyze()` 方法中，分析完成后生成 breakdown:

```python
def analyze(self, workload: Union[str, WorkloadConfig], **kwargs) -> UnifiedResult:
    # ... 现有逻辑 ...
    
    # 生成兼容格式的 breakdown
    breakdown = self._generate_breakdown(phase_results, workload)
    detailed_breakdown = self._generate_detailed_breakdown(phase_results, workload)
    
    return UnifiedResult(
        ...
        breakdown=breakdown,
        detailed_breakdown=detailed_breakdown,
    )
```

#### breakdown 生成逻辑

```python
def _generate_breakdown(self, phases: List[PhaseResult], workload: WorkloadConfig) -> Dict[str, Any]:
    """生成兼容旧前端的 breakdown 格式."""
    total_time = sum(p.total_time_sec for p in phases)
    
    compute_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.FORWARD)
    backward_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.BACKWARD)
    optimizer_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.OPTIMIZER)
    
    # 转换 phases 为 layers 格式
    layers = []
    for phase in phases:
        kernels = []
        if phase.flops:
            kernels.append({
                "name": f"{phase.name}_compute",
                "kernel_type": "compute",
                "time_sec": phase.single_time_sec,
                "flops": phase.flops,
            })
        layers.append({
            "name": phase.component,
            "kernels": kernels,
            "total_time_ms": phase.total_time_sec * 1000,
        })
    
    return {
        "overview": {
            "total_time_sec": total_time,
            "throughput": sum(self.throughput.values()) if self.throughput else 0,
        },
        "time_breakdown": {
            "compute_sec": compute_time,
            "backward_sec": backward_time,
            "optimizer_sec": optimizer_time,
            "communication_sec": 0,  # TODO: 从 metadata 中提取
            "memory_sec": 0,
            "compute_percent": compute_time / total_time * 100 if total_time > 0 else 0,
        },
        "layers": layers,
    }
```

#### detailed_breakdown 生成逻辑

```python
def _generate_detailed_breakdown(self, phases: List[PhaseResult], workload: WorkloadConfig) -> Dict[str, Any]:
    """生成兼容旧前端的 detailed_breakdown 格式."""
    # 按组件分组内存
    by_component = {}
    for phase in phases:
        if phase.component not in by_component:
            by_component[phase.component] = {}
        by_component[phase.component]["activations"] = phase.memory_gb
    
    # 模拟内存类型分解
    total_memory = sum(p.memory_gb for p in phases)
    
    return {
        "submodels": [
            {
                "model_name": phase.component,
                "model_type": phase.compute_type.value,
                "compute_time_sec": phase.total_time_sec,
                "memory": {"by_type": {"activations": phase.memory_gb}},
            }
            for phase in phases
        ],
        "memory": {
            "by_type": {"activations": total_memory},
            "by_submodel": by_component,
            "by_block_type": {},  # TODO: 需要更细粒度的分析
        },
        "communication": {
            "by_parallelism": {},  # TODO: 需要通信分析
        },
    }
```

### Step3: 前端适配

前端 `displayResults` 函数已有两种处理路径:
1. `diffusion-video` pipeline: 读取 `result.phases`
2. `training` 模式: 读取 `result.breakdown` 和 `result.detailed_breakdown`

新增兼容处理:

```javascript
function displayResults(result) {
    // 优先使用 phases 格式
    if (result.phases && result.phases.length > 0) {
        // 检查是否有 meaningful phases 数据
        const hasPhaseTime = result.phases.some(p => p.total_time_sec > 0);
        if (hasPhaseTime) {
            // 使用 phases 渲染
            if (state.currentPipeline === 'diffusion-video') {
                renderPipelineResult(result);
            } else if (result.workload_type === 'training') {
                renderTrainingFromPhases(result);
            }
            return;
        }
    }
    
    // 回退使用 breakdown 格式
    if (result.breakdown) {
        renderTrainingFromBreakdown(result);
    }
}
```

---

## wan pipeline 数值为0的专项修复

### 根本原因排查

检查 `UnifiedAnalyzer._analyze_phases()` 中组件获取逻辑:

```python
# unified.py:106-108
for phase in workload.phases:
    user_component_name = workload.resolve_component(phase.component)
    component = self._get_component(user_component_name)
```

检查 `_get_component()` 是否正确返回组件:
- 对于 pipeline，`self.model` 是 `Dict[str, ShardedModule]`
- 需要 `component_name` 在字典中存在

### 可能问题

web/app.py 中创建 models:
```python
models = {
    "encoder": create_model_from_config({"type": "wan-text-encoder"}),
    "backbone": create_model_from_config({"type": "wan-dit"}),
    "decoder": create_model_from_config({"type": "wan-vae"}),
}
```

component_mapping 配置:
```yaml
component_mapping:
  encoder: encoder
  backbone: backbone
  decoder: decoder
```

**排查点**:
1. `wan-text-encoder`, `wan-dit`, `wan-vae` 模型是否正确创建
2. `_estimate_attention_only()`, `_estimate_transformer_block()`, `_estimate_conv_decoder()` 是否返回有效值

### 修复措施

在 `_estimate_phase()` 中添加调试日志，验证返回值不为0。

---

## 实施步骤

1. **修改 analyzer/base.py**
   - 扩展 `UnifiedResult` 添加 `breakdown` 和 `detailed_breakdown` 字段

2. **修改 analyzer/unified.py**
   - 添加 `_generate_breakdown()` 方法
   - 添加 `_generate_detailed_breakdown()` 方法
   - 在 `analyze()` 中调用生成 breakdown

3. **验证 wan pipeline**
   - 启动 web 服务测试 diffusion-video pipeline
   - 确认 phases 数据不为0

4. **前端测试**
   - 测试 llama-7b training 结果展示
   - 测试 diffusion-video pipeline 结果展示

---

## 验证数据对比

### 期望输出格式

**Training 结果**:
```json
{
  "workload_name": "training",
  "workload_type": "training",
  "phases": [{"name": "forward", "total_time_sec": 0.5}, ...],
  "total_time_sec": 1.5,
  "peak_memory_gb": 40.0,
  "throughput": {"tokens_per_sec": 100000},
  "breakdown": {
    "time_breakdown": {"compute_sec": 1.2, "compute_percent": 80},
    "layers": [...]
  },
  "detailed_breakdown": {
    "memory": {"by_type": {"activations": 20}},
    ...
  }
}
```

**Diffusion-video 结果**:
```json
{
  "phases": [
    {"name": "encode", "total_time_sec": 0.1, "component": "encoder"},
    {"name": "denoise", "total_time_sec": 10.5, "component": "backbone"},
    {"name": "decode", "total_time_sec": 0.3, "component": "decoder"}
  ],
  "total_time_sec": 10.9,
  ...
}
```
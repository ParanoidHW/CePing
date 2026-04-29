# Web2 前后端解耦评估服务架构设计方案

## 1. 设计目标

### 1.1 问题分析

当前 web 服务前后端耦合严重，主要问题：

| 问题 | 当前状态 | 影响 |
|------|----------|------|
| Workload 场景硬编码 | 前端 HTML 硬编码 5 种场景配置表单 | 新增场景需修改前端代码 |
| 参数映射硬编码 | 后端 `SCENARIO_PARAM_MAP` 硬编码参数映射 | 新增参数需修改后端代码 |
| 模型切换逻辑硬编码 | 前端 `switchWorkloadScenario()` 硬编码切换逻辑 | 新增场景需修改前端逻辑 |
| 并行策略 UI 硬编码 | HTML 硬编码不同场景的并行策略表单 | 新增策略组合需修改前端 |
| 分解渲染部分解耦 | 已实现 Object.keys() 自动发现 | 仅部分解耦，场景切换仍硬编码 |

### 1.2 设计目标

实现完全的前后端解耦：

- **前端零硬编码**：所有表单、渲染逻辑由后端 schema 驱动
- **后端 schema-driven**：所有配置项、验证逻辑由 schema 定义
- **Workload 驱动**：Workload 作为配置核心，驱动模型选择和参数配置
- **动态可扩展**：新增 workload/模型/场景无需修改前端代码
- **llm_perf 核心复用**：Web 服务和 CLI 工具共享核心能力，避免 schema 重复

---

## 2. 架构分层设计（修正版）

### 2.1 核心原则

**llm_perf 作为核心层，Web 服务和 CLI 工具都是调用层**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Application Layer (Web + CLI)                             │
│  ┌────────────────────────────┐  ┌───────────────────────────────────────┐ │
│  │  Web Layer                 │  │  CLI Layer                            │ │
│  │  ├── web2/ (Frontend)      │  │  ├── bin/eval-cli                    │ │
│  │  ├── web2_api/ (HTTP API)  │  │  ├── scripts/configs/                │ │
│  │  只做 HTTP 适配             │  │  统一 CLI + 配置文件                  │ │
│  └────────────────────────────┘  └───────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Core Layer (llm_perf)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ llm_perf/ (核心层，所有能力定义在此)                                       ││
│  │ ├── workload/        Workload Schema + Engine + Breakdown               ││
│  │ │   ├── schema.py    Workload 类型定义、参数 schema、策略 schema          ││
│  │ │   ├── engine.py    评估引擎，调用 Analyzer                              ││
│  │ │   └── breakdown.py 分解计算，Stage/Phase/Submodule 分解                 ││
│  │ ├── modeling/        ModelRegistry + ShardedModule                       ││
│  │ ├── analyzer/        UnifiedAnalyzer + Handler                           ││
│  │ ├── strategy/        ParallelContext + Strategy                          ││
│  │ ├── hardware/        Device + Cluster + Topology                         ││
│  │ ├── kernels/         Compute + Communication + Backend                   ││
│  │ └── validation/      Memory + Strategy + Sequence Validator              ││
│  └─────────────────────────────────────────────────────────────────────────┐│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 层次职责（修正版）

| 层次 | 目录 | 职责 | 关键原则 |
|------|------|------|----------|
| **Frontend Layer** | `web2/` | 动态表单渲染、动态结果渲染、前端验证 | 不定义 schema，从 API 获取 |
| **API Layer** | `web2_api/` | HTTP 适配、请求解析、响应构建 | **只做数据转换，不定义 schema** |
| **Core Layer** | `llm_perf/` | Schema 定义、评估引擎、建模、分析 | **所有核心能力定义在此** |
| **CLI Layer** | `bin/`, `scripts/configs/` | 统一 CLI 工具、配置文件示例 | 通过参数支持所有 workload |

### 2.3 数据流向（修正版）

```
┌─────────────┐    1. GET /api/schema/workload     ┌─────────────┐
│   Frontend  │ ───────────────────────────────→  │  API Layer  │
│  (web2/)    │                                    │ (web2_api/) │
│             │ ←──────────────────────────────    │             │
│  动态渲染   │    WorkloadSchema                  │  HTTP适配   │
│             │                                    │             │
│             │                                    │  从 llm_perf│
│             │                                    │  导出 schema│
│             │    2. POST /api/evaluate           │             │
│             │ ───────────────────────────────→  │             │
│             │                                    │             │
│             │ ←──────────────────────────────    │             │
│             │    EvaluationResult                │             │
└─────────────┘                                    └─────────────┘
                                                          │
                                                          │ 调用
                                                          ↓
                                                   ┌─────────────┐
                                                   │ llm_perf    │
                                                   │ (核心层)    │
                                                   │             │
                                                   │ workload/   │
                                                   │ ├── schema  │
                                                   │ ├── engine  │
                                                   │ ├── breakdown│
                                                   │             │
                                                   │ analyzer/   │
                                                   │ modeling/   │
                                                   └─────────────┘

┌─────────────┐                                    ┌─────────────┐
│   CLI 工具  │ ───────────────────────────────→  │ llm_perf    │
│ (eval-cli)  │    直接调用 llm_perf              │ (核心层)    │
│             │                                    │             │
│ 配置文件    │ ←──────────────────────────────    │ workload/   │
│ (YAML/JSON) │    EvaluationResult                │ ├── schema  │
└─────────────┘                                    │ ├── engine  │
                                                   │ ├── breakdown│
                                                   │             │
                                                   │ analyzer/   │
                                                   │ modeling/   │
                                                   └─────────────┘
```

---

## 3. llm_perf 扩展设计（新增）

### 3.1 配置体系架构

**核心原则：复用现有配置，避免重复定义**

项目已有完整的配置体系：

```
configs/
├── models/                    # 模型配置（20+ 模型）
│   ├── llama-7b.yaml
│   ├── deepseek-v3.yaml
│   ├── qwen3-5.yaml
│   ├── wan-dit.yaml
│   └── ...
├── workloads/                 # Workload 配置（应用场景 + compute_mode 属性划分）
│   ├── training/              # compute_mode: base
│   ├── rl_training/           # compute_mode: autoregressive  
│   ├── inference/             # compute_mode: autoregressive
│   ├── pd_disagg/             # compute_mode: autoregressive
│   ├── multimodal/            # compute_mode: autoregressive
│   ├── diffusion/             # compute_mode: iterative
│   ├── conv/                  # compute_mode: conv
│   └── custom/                # 用户自定义
└── hardware/                  # 硬件配置
```

**llm_perf/workload/ 职责**：加载和解析现有 configs，而非重新定义

```
llm_perf/workload/
├── __init__.py
├── loader.py              # 加载 configs/workloads/*.yaml 和 configs/models/*.yaml
├── validator.py           # 校验配置合法性
├── schema.py              # Workload 类型定义（从 YAML 解析后的数据结构）
├── engine.py              # 评估引擎，调用 UnifiedAnalyzer
├── breakdown.py           # 分解计算模块
└── tests/
```

### 3.2 Workload Loader 设计

**核心职责**：加载 `configs/workloads/` 和 `configs/models/` 目录下的 YAML 配置

```python
# llm_perf/workload/loader.py

from pathlib import Path
from typing import Dict, List, Optional
import yaml

class WorkloadLoader:
    """Workload 配置加载器"""
    
    WORKLOADS_DIR = Path("configs/workloads")
    MODELS_DIR = Path("configs/models")
    
    def __init__(self):
        self._workload_cache: Dict[str, Dict] = {}
        self._model_cache: Dict[str, Dict] = {}
    
    def list_workloads(self) -> List[str]:
        """列出所有 workload 名称
        
        Returns:
            如 ["training/training", "inference/autoregressive", "diffusion/pipeline", ...]
        """
        workloads = []
        for category_dir in self.WORKLOADS_DIR.iterdir():
            if category_dir.is_dir() and category_dir.name != "custom":
                for yaml_file in category_dir.glob("*.yaml"):
                    workloads.append(f"{category_dir.name}/{yaml_file.stem}")
        return sorted(workloads)
    
    def list_workload_categories(self) -> Dict[str, List[str]]:
        """按类别列出 workload
        
        Returns:
            {
                "training": ["training", "denoise"],
                "rl_training": ["rl_ppo", "rl_grpo"],
                "inference": ["inference", "autoregressive", "speculative_decoding"],
                "diffusion": ["denoise", "pipeline"],
                "conv": ["encoder", "decoder", "resnet"],
            }
        """
        categories = {}
        for category_dir in self.WORKLOADS_DIR.iterdir():
            if category_dir.is_dir() and category_dir.name != "custom":
                workloads = [f.stem for f in category_dir.glob("*.yaml")]
                categories[category_dir.name] = sorted(workloads)
        return categories
    
    def load_workload(self, workload_name: str) -> Dict:
        """加载 workload 配置
        
        Args:
workload_name: 如 "inference/autoregressive" 或 "training"
        
        Returns:
            workload 配置字典
        """
        if workload_name in self._workload_cache:
            return self._workload_cache[workload_name]
        
        # 支持简写：training -> training/training
        if "/" not in workload_name:
            workload_path = self._resolve_workload_path(workload_name)
        else:
            workload_path = self.WORKLOADS_DIR / f"{workload_name}.yaml"
        
        if not workload_path.exists():
            raise FileNotFoundError(f"Workload not found: {workload_name}")
        
        with open(workload_path) as f:
            config = yaml.safe_load(f)
        
        self._workload_cache[workload_name] = config
        return config
    
    def _resolve_workload_path(self, name: str) -> Path:
        """解析 workload 路径（支持简写）"""
        # 先检查 training/
        training_path = self.WORKLOADS_DIR / "training" / f"{name}.yaml"
        if training_path.exists():
            return training_path
        
        # 遍历所有目录查找
        for category_dir in self.WORKLOADS_DIR.iterdir():
            if category_dir.is_dir():
                candidate = category_dir / f"{name}.yaml"
                if candidate.exists():
                    return candidate
        
        raise FileNotFoundError(f"Workload not found: {name}")
    
    def list_models(self) -> List[str]:
        """列出所有模型名称
        
        Returns:
            如 ["llama-7b", "llama-13b", "llama-70b", "deepseek-v3", ...]
        """
        models = []
        for yaml_file in self.MODELS_DIR.glob("*.yaml"):
            models.append(yaml_file.stem)
        return sorted(models)
    
    def load_model(self, model_name: str) -> Dict:
        """加载模型配置
        
        Args:
            model_name: 如 "llama-7b", "deepseek-v3"
        
        Returns:
            模型配置字典
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        model_path = self.MODELS_DIR / f"{model_name}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_name}")
        
        with open(model_path) as f:
            config = yaml.safe_load(f)
        
        self._model_cache[model_name] = config
        return config


# 单例
_loader = None

def get_loader() -> WorkloadLoader:
    global _loader
    if _loader is None:
        _loader = WorkloadLoader()
    return _loader
```

### 3.3 Workload Validator 设计

```python
# llm_perf/workload/validator.py

from typing import Dict, List, Tuple

class WorkloadValidator:
    """Workload 配置校验器"""
    
    REQUIRED_WORKLOAD_FIELDS = ["name", "description", "workload_type", "phases"]
    REQUIRED_MODEL_FIELDS = ["description", "preset_type", "architecture", "config"]
    
    VALID_WORKLOAD_TYPES = ["training", "inference", "diffusion", "mixed"]
    VALID_COMPUTE_TYPES = ["forward", "backward", "optimizer"]
    VALID_COMPUTE_PATTERNS = [
        "transformer_block", "conv_encoder", "conv_decoder", 
        "attention_only", "dense_forward"
    ]
    
    def validate_workload(self, config: Dict) -> Tuple[bool, List[str]]:
        """校验 workload 配置
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        # 检查必需字段
        for field in self.REQUIRED_WORKLOAD_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # 校验 workload_type
        if "workload_type" in config:
            if config["workload_type"] not in self.VALID_WORKLOAD_TYPES:
                errors.append(f"Invalid workload_type: {config['workload_type']}")
        
        # 校验 phases
        if "phases" in config:
            for i, phase in enumerate(config["phases"]):
                phase_errors = self._validate_phase(phase, i)
                errors.extend(phase_errors)
        
        return len(errors) == 0, errors
    
    def _validate_phase(self, phase: Dict, index: int) -> List[str]:
        errors = []
        
        if "name" not in phase:
            errors.append(f"Phase {index}: missing 'name'")
        
        if "compute_type" not in phase:
            errors.append(f"Phase {index}: missing 'compute_type'")
        elif phase["compute_type"] not in self.VALID_COMPUTE_TYPES:
            errors.append(f"Phase {index}: invalid compute_type '{phase['compute_type']}'")
        
        if "component" not in phase:
            errors.append(f"Phase {index}: missing 'component'")
        
        if "compute_pattern" in phase and phase["compute_pattern"] not in self.VALID_COMPUTE_PATTERNS:
            errors.append(f"Phase {index}: invalid compute_pattern '{phase['compute_pattern']}'")
        
        return errors
    
    def validate_model(self, config: Dict) -> Tuple[bool, List[str]]:
        """校验模型配置
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        for field in self.REQUIRED_MODEL_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        if "config" in config:
            config_errors = self._validate_model_config(config["config"])
            errors.extend(config_errors)
        
        return len(errors) == 0, errors
    
    def _validate_model_config(self, config: Dict) -> List[str]:
        errors = []
        
        required = ["hidden_size", "num_layers", "num_heads", "vocab_size"]
        for field in required:
            if field not in config:
                errors.append(f"Model config missing: {field}")
        
        return errors
```

### 3.4 Workload Schema 定义（从 YAML 解析）

```python
# llm_perf/workload/schema.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class WorkloadType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    DIFFUSION = "diffusion"
    MIXED = "mixed"

class ComputeType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"

@dataclass
class PhaseDefinition:
    """Phase 定义（从 YAML 解析）"""
    name: str
    compute_type: ComputeType
    component: str
    compute_pattern: Optional[str] = None
    repeat: int = 1
    extra_params: Dict = field(default_factory=dict)

@dataclass
class WorkloadConfig:
    """Workload 配置（从 YAML 解析后）"""
    name: str
    description: str
    workload_type: WorkloadType
    phases: List[PhaseDefinition]
    component_mapping: Dict[str, str] = field(default_factory=dict)
    default_params: Dict = field(default_factory=dict)
    optimizer_factor: Optional[float] = None
    throughput_metric: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_config: Dict) -> "WorkloadConfig":
        """从 YAML 配置创建"""
        phases = []
        for p in yaml_config.get("phases", []):
            phases.append(PhaseDefinition(
                name=p["name"],
                compute_type=ComputeType(p["compute_type"]),
                component=p["component"],
                compute_pattern=p.get("compute_pattern"),
                repeat=p.get("repeat", 1),
                extra_params=p.get("extra_params", {})
            ))
        
        return cls(
            name=yaml_config["name"],
            description=yaml_config["description"],
            workload_type=WorkloadType(yaml_config["workload_type"]),
            phases=phases,
            component_mapping=yaml_config.get("component_mapping", {}),
            default_params=yaml_config.get("default_params", {}),
            optimizer_factor=yaml_config.get("optimizer_factor"),
            throughput_metric=yaml_config.get("throughput_metric")
        )

@dataclass
class ModelConfig:
    """模型配置（从 YAML 解析后）"""
    name: str
    description: str
    architecture: str
    sparse_type: str
    attention_features: List[str]
    supported_workloads: List[str]
    config: Dict
    param_schema: Dict
    
    @classmethod
    def from_yaml(cls, name: str, yaml_config: Dict) -> "ModelConfig":
        """从 YAML 配置创建"""
        return cls(
            name=name,
            description=yaml_config["description"],
            architecture=yaml_config["architecture"],
            sparse_type=yaml_config.get("sparse_type", "dense"),
            attention_features=yaml_config.get("attention_features", []),
            supported_workloads=yaml_config.get("supported_workloads", []),
            config=yaml_config["config"],
            param_schema=yaml_config.get("param_schema", {})
        )
```

### 3.4 评估引擎

```python
# llm_perf/workload/engine.py

from llm_perf.analyzer.unified import UnifiedAnalyzer
from llm_perf.workload.registry import WorkloadSchemaRegistry

class EvaluationEngine:
    """评估引擎（统一入口）"""
    
    def __init__(self):
        self.registry = WorkloadSchemaRegistry()
        self.analyzer = UnifiedAnalyzer()
    
    def evaluate(self, request: Dict) -> Dict:
        """执行评估
        
        Args:
            request: EvaluationRequest 格式
        
        Returns:
            EvaluationResult 格式
        """
        workload_type = request["workload"]["type"]
        schema = self.registry.get(workload_type)
        
        # 1. 验证请求
        validation_result = self._validate_request(request, schema)
        if validation_result["errors"]:
            return {"success": False, "validation": validation_result}
        
        # 2. 构建 WorkloadConfig
        workload_config = self._build_workload_config(request, schema)
        
        # 3. 调用 UnifiedAnalyzer
        unified_result = self.analyzer.analyze(workload_config)
        
        # 4. 构建分解结果
        breakdown = self._build_breakdown(unified_result, schema)
        
        # 5. 构建响应
        return {
            "success": True,
            "result": {
                "workload_type": workload_type,
                "total_time_sec": unified_result.total_time_sec,
                "peak_memory_gb": unified_result.peak_memory_gb,
                "throughput": unified_result.throughput,
                "metrics": unified_result.metrics,
                "stages": unified_result.stages,
                "breakdown": breakdown
            },
            "validation": validation_result
        }
```

### 3.5 分解计算模块

```python
# llm_perf/workload/breakdown.py

class BreakdownCalculator:
    """分解计算模块"""
    
    def calculate(self, unified_result, schema: WorkloadTypeDefinition) -> Dict:
        """计算分解数据
        
        根据 schema.result_breakdown.levels 计算对应层级分解
        """
        levels = schema.result_breakdown.get("levels", [])
        
        breakdown = {}
        
        if "stage" in levels:
            breakdown["by_stage"] = self._aggregate_by_stage(unified_result)
        
        if "submodule" in levels:
            breakdown["by_submodule_type"] = self._aggregate_by_submodule(unified_result)
        
        if "memory" in levels:
            breakdown["memory"] = self._aggregate_memory(unified_result)
        
        if "communication" in levels:
            breakdown["communication"] = self._aggregate_communication(unified_result)
        
        return breakdown
```

---

## 4. API Layer 设计（修正版）

### 4.1 API 调用现有配置

**核心原则**: API 层加载 `configs/models/` 和 `configs/workloads/`，不定义新配置

```python
# web2_api/routes/workloads.py

from llm_perf.workload.loader import get_loader
from llm_perf.workload.validator import WorkloadValidator

loader = get_loader()
validator = WorkloadValidator()

def list_workloads():
    """GET /api/workloads
    
    Returns:
        {
            "categories": {
                "training": ["training", "denoise"],
                "rl_training": ["rl_ppo", "rl_grpo"],
                "inference": ["inference", "autoregressive", "speculative_decoding"],
                "diffusion": ["denoise", "pipeline"],
                "conv": ["encoder", "decoder", "resnet"],
                "multimodal": ["multimodal_inference"]
            },
            "total": 15
        }
    """
    categories = loader.list_workload_categories()
    total = sum(len(v) for v in categories.values())
    return {"categories": categories, "total": total}

def get_workload(workload_name: str):
    """GET /api/workload/{workload_name}
    
    Args:
        workload_name: 如 "inference/autoregressive" 或 "training"
    
    Returns:
        {
            "name": "autoregressive-inference",
            "description": "自回归生成（prefill + decode）",
            "workload_type": "inference",
            "compute_mode": "autoregressive",
            "phases": [...],
            "default_params": {...},
            "throughput_metric": "tokens_per_sec"
        }
    """
    config = loader.load_workload(workload_name)
    is_valid, errors = validator.validate_workload(config)
    
    if not is_valid:
        return {"error": "Invalid workload", "details": errors}
    
    return config

def list_models():
    """GET /api/models
    
    Returns:
        {
            "models": ["llama-7b", "llama-13b", "llama-70b", "deepseek-v3", "qwen3-5", ...],
            "total": 20
        }
    """
    models = loader.list_models()
    return {"models": models, "total": len(models)}

def get_model(model_name: str):
    """GET /api/model/{model_name}
    
    Args:
        model_name: 如 "llama-7b", "deepseek-v3"
    
    Returns:
        {
            "description": "LLaMA 7B",
            "architecture": "llama",
            "config": {...},
            "param_schema": {...}
        }
    """
    config = loader.load_model(model_name)
    is_valid, errors = validator.validate_model(config)
    
    if not is_valid:
        return {"error": "Invalid model", "details": errors}
    
    return config
```

### 4.2 Schema API（基于现有配置生成前端表单 schema）

```python
# web2_api/routes/schema.py

from llm_perf.workload.loader import get_loader
from llm_perf.workload.schema import WorkloadConfig, ModelConfig

loader = get_loader()

def get_workload_form_schema(workload_name: str):
    """GET /api/schema/workload/{workload_name}
    
    基于 configs/workloads/*.yaml 和 configs/models/*.yaml 生成前端表单 schema
    """
    workload_config = loader.load_workload(workload_name)
    workload = WorkloadConfig.from_yaml(workload_config)
    
    # 获取支持的模型列表
    models = loader.list_models()
    
    # 构建参数 schema（从 workload 的 default_params 提取）
    params_schema = _build_params_schema(workload)
    
    # 构建阶段 schema（从 phases 提取）
    stages_schema = _build_stages_schema(workload)
    
    return {
        "workload_type": workload.name,
        "description": workload.description,
        "category": workload.workload_type.value,
        "stages": stages_schema,
        "parameters": params_schema,
        "models": models,
        "throughput_metric": workload.throughput_metric
    }

def get_model_form_schema(model_name: str, workload_type: str):
    """GET /api/schema/model/{model_name}?workload={workload_type}
    
    基于 configs/models/*.yaml 和 workload_type 生成模型参数表单 schema
    """
    model_config = loader.load_model(model_name)
    model = ModelConfig.from_yaml(model_name, model_config)
    
    # 获取对应 workload 的 param_schema
    param_schema = model.param_schema.get(workload_type, [])
    
    return {
        "model_name": model.name,
        "description": model.description,
        "architecture": model.architecture,
        "config": model.config,
        "param_schema": param_schema,
        "supported_workloads": model.supported_workloads
    }

def _build_params_schema(workload: WorkloadConfig) -> Dict:
    """从 workload.default_params 构建参数 schema"""
    schema = {}
    for param_name, default_value in workload.default_params.items():
        param_type = "number" if isinstance(default_value, (int, float)) else "string"
        schema[param_name] = {
            "type": param_type,
            "default": default_value,
            "label": _format_param_label(param_name)
        }
    return schema

def _build_stages_schema(workload: WorkloadConfig) -> List[Dict]:
    """从 workload.phases 构建阶段 schema"""
    stages = []
    for phase in workload.phases:
        stages.append({
            "name": phase.name,
            "compute_type": phase.compute_type.value,
            "component": phase.component,
            "repeat": phase.repeat
        })
    return stages

def _format_param_label(param_name: str) -> str:
    """参数名转换为显示标签"""
    labels = {
        "batch_size": "Batch Size",
        "seq_len": "Sequence Length",
        "prompt_len": "Prompt Length",
        "generation_len": "Generation Length",
        "num_steps": "Diffusion Steps",
        "num_frames": "Number of Frames",
        "height": "Height",
        "width": "Width",
    }
    return labels.get(param_name, param_name.replace("_", " ").title())
```

### 4.3 API 目录结构

```
web2_api/
├── __init__.py
├── app.py                             # Flask 应用
├── routes/
│   ├── __init__.py
│   ├── schema.py                      # Schema API（从 llm_perf 导出）
│   ├── evaluate.py                    # Evaluate API（调用 llm_perf）
│   ├── resources.py                   # Resources API（从 llm_perf 导出）
│   └── export.py                      # Export API
├── utils/
│   ├── request_parser.py              # 请求解析（HTTP -> Dict）
│   ├── result_builder.py              # 结果构建（Dict -> HTTP Response）
│   └── error_handler.py               # 错误处理
└── tests/
```

**注意**: 删除 `web2_api/services/schema_service.py`，schema 直接从 llm_perf 导出

---

## 5. CLI 工具设计（修正版）

### 5.1 统一 CLI 工具

**核心原则**: 一个 CLI 工具支持所有 workload 类型，加载 `configs/models/` 和 `configs/workloads/`

```
bin/
├── eval-cli                           # 统一评估 CLI（支持所有 workload）
└── eval-batch                         # 批量评估工具
```

**注意**: 不再需要 `scripts/configs/` 目录，直接使用 `configs/models/` 和 `configs/workloads/`

### 5.2 eval-cli 命令行参数设计

```bash
# 基本用法
eval-cli --workload <WORKLOAD> --model <MODEL> [OPTIONS]

# Workload 参数支持多种格式：
#   - 完整路径: inference/autoregressive
#   - 简写: training (自动解析为 training/training)
#   - YAML 文件: configs/workloads/custom/my-workload.yaml

# 完整参数列表
eval-cli \
  --workload <WORKLOAD>              # workload 名称或路径（必需）
  --model <MODEL>                    # 模型名称或配置文件（必需）
  --config <FILE>                    # 完整评估配置文件（可选）
  --hardware <SPEC>                   # 硬件规格（可选）
  --strategy <SPEC>                   # 并行策略（可选）
  --output <FORMAT>                   # 输出格式（json/yaml/table，默认 json）
  --output-file <FILE>                # 输出文件路径（可选）
  --breakdown-level <LEVEL>           # 分解层级（stage/phase/submodule/memory）
  --verbose                           # 详细输出
  --interactive                       # 交互式模式
```

### 5.3 参数详细说明

#### 5.3.1 --workload 参数

支持多种格式：

```bash
# 格式1: 完整路径（推荐）
eval-cli --workload inference/autoregressive

# 格式2: 简写（自动解析）
eval-cli --workload training          # 解析为 training/training
eval-cli --workload inference          # 解析为 inference/inference

# 格式3: 自定义 YAML 文件
eval-cli --workload configs/workloads/custom/my-workload.yaml
```

支持的 workload 列表（来自 configs/workloads/）：

| Workload | 说明 | 配置文件 | compute_mode |
|----------|------|----------|--------------|
| `training/training` | 通用训练 | `configs/workloads/training/training.yaml` | base |
| `training/denoise` | 去噪训练 | `configs/workloads/training/denoise.yaml` | base |
| `rl_training/rl_ppo` | PPO 训练 | `configs/workloads/rl_training/rl_ppo.yaml` | autoregressive |
| `rl_training/rl_grpo` | GRPO 训练 | `configs/workloads/rl_training/rl_grpo.yaml` | autoregressive |
| `inference/inference` | 通用推理 | `configs/workloads/inference/inference.yaml` | autoregressive |
| `inference/autoregressive` | 自回归生成 | `configs/workloads/inference/autoregressive.yaml` | autoregressive |
| `inference/speculative_decoding` | 投机解码 | `configs/workloads/inference/speculative_decoding.yaml` | autoregressive |
| `diffusion/denoise` | 多步去噪推理 | `configs/workloads/diffusion/denoise.yaml` | iterative |
| `diffusion/pipeline` | 扩散 Pipeline | `configs/workloads/diffusion/pipeline.yaml` | iterative |
| `conv/encoder` | 卷积编码 | `configs/workloads/conv/encoder.yaml` | conv |
| `conv/decoder` | 卷积解码 | `configs/workloads/conv/decoder.yaml` | conv |
| `conv/resnet` | ResNet 分类 | `configs/workloads/conv/resnet.yaml` | conv |

#### 5.3.2 --model 参数

支持多种格式：

```bash
# 格式1: 使用 preset 名称（加载 configs/models/{name}.yaml）
eval-cli --model llama-7b
eval-cli --model deepseek-v3
eval-cli --model qwen3-5

# 格式2: 使用配置文件
eval-cli --model configs/models/llama-7b.yaml
eval-cli --model configs/models/custom/my-model.yaml
```

支持的模型列表（来自 configs/models/）：

| Model | Architecture | Sparse Type | Config File |
|-------|-------------|-------------|-------------|
| `llama-7b` | llama | dense | `configs/models/llama-7b.yaml` |
| `llama-13b` | llama | dense | `configs/models/llama-13b.yaml` |
| `llama-70b` | llama | dense | `configs/models/llama-70b.yaml` |
| `deepseek-v3` | deepseek | deepseek_moe | `configs/models/deepseek-v3.yaml` |
| `qwen3-5` | qwen | dense | `configs/models/qwen3-5.yaml` |
| `mixtral-8x7b` | mixtral | moe | `configs/models/mixtral-8x7b.yaml` |
| `wan-dit` | dit | dense | `configs/models/wan-dit.yaml` |
| `resnet50` | resnet | dense | `configs/models/resnet50.yaml` |

#### 5.3.3 --config 参数

完整评估配置文件格式：

```yaml
# 评估配置示例
workload:
  name: inference/autoregressive    # 或使用 path: configs/workloads/inference/autoregressive.yaml

model:
  name: llama-7b                    # 或使用 path: configs/models/llama-7b.yaml

hardware:
  device: H100-SXM-80GB
  num_devices: 8

strategy:
  tp_degree: 8
  pp_degree: 1

params:
  batch_size: 8
  prompt_len: 1024
  generation_len: 128

output:
  format: json
  breakdown_level: [stage, submodule]
```

#### 5.3.4 --hardware 参数

```bash
# 简写格式
eval-cli --hardware H100-SXM-80GB:8

# 完整格式
eval-cli --hardware device=H100-SXM-80GB,num_devices=8
```

#### 5.3.5 --strategy 参数

```bash
# 简写格式
eval-cli --strategy tp=8,pp=1

# 完整格式
eval-cli --strategy tp_degree=8,pp_degree=1,dp_degree=1
```

#### 5.3.6 --output 参数

| 格式 | 说明 | 示例 |
|------|------|------|
| `json` | JSON 格式（默认） | `--output json` |
| `yaml` | YAML 格式 | `--output yaml` |
| `table` | 表格格式 | `--output table` |

### 5.4 交互式模式

```bash
# 进入交互式模式
eval-cli
eval-cli --interactive
```

交互式配置流程：

```
$ eval-cli
╔══════════════════════════════════════════════════════════╗
║           LLM Performance Evaluator CLI                  ║
╚══════════════════════════════════════════════════════════╝

Step 1: Select Workload Category
  1. base              (基础负载)
  2. autoregressive    (自回归生成)
  3. iterative         (迭代去噪)
  4. conv              (卷积类)
  5. multimodal        (多模态推理)
  
Enter choice [1-5]: 2

Step 2: Select Workload
  1. inference              (自回归生成)
  2. rl-ppo                 (PPO 训练)
  3. rl-grpo                (GRPO 训练)
  4. speculative-decoding   (投机解码)
  
Enter choice [1-4]: 1

Step 3: Select Model
  1. llama-7b
  2. llama-13b
  3. llama-70b
  4. deepseek-v3
  5. qwen3-5
  ...
  20. custom model...
  
Enter choice [1-20]: 1

Step 4: Configure Hardware
Device [H100-SXM-80GB]: 
Number of devices [8]: 16

Step 5: Configure Workload Parameters
Batch size [1]: 8
Prompt length [512]: 1024
Generation length [128]: 256

Step 6: Configure Parallel Strategy
TP degree [1]: 8
PP degree [1]: 2
DP degree [1]: 1

Step 7: Select Output Format
  1. json
  2. yaml
  3. table
  
Enter choice [1-3]: 3

────────────────────────────────────────
Configuration Summary:
  Workload: inference/autoregressive
  Model: llama-7b
  Hardware: 16x H100-SXM-80GB
  Strategy: TP=8, PP=2, DP=1
  
  Parameters:
    batch_size: 8
    prompt_len: 1024
    generation_len: 256
────────────────────────────────────────

Run evaluation? [Y/n]: Y

Evaluating...
[████████████████████████████████] 100%

═══════════════════════════════════════
              Results Summary
═══════════════════════════════════════
Throughput:    12,500 tokens/sec
MFU:           45.2%
Peak Memory:   65.3 GB/device
Total Time:    2.34 sec/batch

Save results? [Y/n]: Y
Output file [results/inference_llama7b_20260429.json]: 
Configuration saved to: configs/saved/inference_llama7b_20260429.yaml
```

### 5.5 使用示例

#### 5.5.1 基本用法

```bash
# Training workload
eval-cli \
  --workload training/training \
  --model llama-7b \
  --hardware H100-SXM-80GB:8 \
  --strategy tp=8,pp=1 \
  --output table

# Inference workload (自回归生成)
eval-cli \
  --workload inference/autoregressive \
  --model deepseek-v3 \
  --hardware H100-SXM-80GB:8 \
  --strategy tp=8 \
  --output json

# Diffusion pipeline workload
eval-cli \
  --workload diffusion/pipeline \
  --model wan-dit \
  --hardware H100-SXM-80GB:4 \
  --strategy tp=4 \
  --output yaml

# Conv encoder workload
eval-cli \
  --workload conv/encoder \
  --model resnet50 \
  --hardware H100-SXM-80GB:1 \
  --output table
```

#### 5.5.2 使用配置文件

```bash
# 从配置文件加载
eval-cli --config configs/eval/llama7b_training.yaml

# 配置文件 + 覆盖参数
eval-cli \
  --config configs/eval/llama7b_training.yaml \
  --hardware H100-SXM-80GB:16 \
  --strategy tp=16,pp=1
```

#### 5.5.3 使用自定义 workload

```bash
# 使用自定义 workload YAML
eval-cli \
  --workload configs/workloads/custom/my-workload.yaml \
  --model llama-7b \
  --hardware H100-SXM-80GB:8

# 使用自定义模型 YAML
eval-cli \
  --workload training/training \
  --model configs/models/custom/my-model.yaml \
  --hardware H100-SXM-80GB:8
```

### 5.6 CLI 实现框架

```python
# bin/eval-cli

#!/usr/bin/env python3
import argparse
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from llm_perf.workload.loader import get_loader
from llm_perf.workload.validator import WorkloadValidator
from llm_perf.workload.engine import EvaluationEngine
from llm_perf.reporter import JsonReporter, YamlReporter, TableReporter


def main():
    args = parse_args()
    
    if args.interactive or len(sys.argv) == 1:
        run_interactive_mode()
    else:
        run_cli_mode(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Performance Evaluator CLI"
    )
    
    parser.add_argument(
        "--workload", "-w",
        help="Workload name or path (e.g., inference/autoregressive)"
    )
    
    parser.add_argument(
        "--model", "-m",
        help="Model name or config path (e.g., llama-7b)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Evaluation config file path"
    )
    
    parser.add_argument(
        "--hardware",
        help="Hardware spec (e.g., H100-SXM-80GB:8)"
    )
    
    parser.add_argument(
        "--strategy",
        help="Parallel strategy (e.g., tp=8,pp=1)"
    )
    
    parser.add_argument(
        "--output", "-o",
        choices=["json", "yaml", "table"],
        default="json",
        help="Output format"
    )
    
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file path"
    )
    
    parser.add_argument(
        "--breakdown-level",
        choices=["stage", "phase", "submodule", "memory"],
        default=["stage"],
        nargs="+",
        help="Breakdown level"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    
    parser.add_argument(
        "--list-workloads",
        action="store_true",
        help="List all available workloads"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models"
    )
    
    return parser.parse_args()


def run_cli_mode(args):
    loader = get_loader()
    
    # 列出可用选项
    if args.list_workloads:
        categories = loader.list_workload_categories()
        print("Available Workloads:")
        for cat, workloads in categories.items():
            print(f"\n  [{cat}]")
            for w in workloads:
                print(f"    - {w}")
        return
    
    if args.list_models:
        models = loader.list_models()
        print("Available Models:")
        for m in models:
            print(f"  - {m}")
        return
    
    # 构建配置
    config = build_config(args, loader)
    
    if args.verbose:
        print_config_summary(config)
    
    # 执行评估
    engine = EvaluationEngine()
    result = engine.evaluate(config)
    
    # 输出结果
    reporter = get_reporter(args.output)
    output = reporter.format(result, breakdown_level=args.breakdown_level)
    
    if args.output_file:
        args.output_file.write_text(output)
        print(f"Results saved to: {args.output_file}")
    else:
        print(output)


def build_config(args, loader) -> Dict[str, Any]:
    config = {}
    
    # 从配置文件加载
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    
    # 加载 workload
    if args.workload:
        workload_config = loader.load_workload(args.workload)
        config["workload"] = workload_config
    
    # 加载模型
    if args.model:
        model_config = loader.load_model(args.model)
        config["model"] = model_config
    
    # 覆盖参数
    if args.hardware:
        config["hardware"] = parse_hardware(args.hardware)
    
    if args.strategy:
        config["strategy"] = parse_strategy(args.strategy)
    
    return config


def run_interactive_mode():
    from llm_perf.cli.interactive import InteractiveSession
    
    loader = get_loader()
    session = InteractiveSession(loader)
    config = session.run()
    
    if config:
        engine = EvaluationEngine()
        result = engine.evaluate(config)
        
        output_format = session.get_output_format()
        reporter = get_reporter(output_format)
        output = reporter.format(result)
        
        print(output)
        
        if session.should_save():
            save_path = session.get_save_path()
            save_path.write_text(output)
            print(f"Results saved to: {save_path}")


def get_reporter(format: str):
    reporters = {
        "json": JsonReporter,
        "yaml": YamlReporter,
        "table": TableReporter
    }
    return reporters[format]()


if __name__ == "__main__":
    main()
```
bin/
├── eval-cli                           # 统一评估 CLI（支持所有 workload）
└── eval-batch                         # 批量评估工具

scripts/
├── configs/                           # 配置文件示例（JSON/YAML）
│   ├── training/
│   │   ├── llama_7b_h100.yaml
│   │   └── deepseek_v3_h100.yaml
│   ├── inference/
│   │   ├── llama_7b_inference.yaml
│   │   └── pd_disagg.yaml
│   ├── diffusion/
│   │   └── sdxl_t2i.yaml
│   ├── rl_training/
│   │   └── ppo_llama.yaml
│   └── multimodal/
│       └── llava_inference.yaml
└── README.md                          # 配置文件说明
```

### 5.2 eval-cli 命令行参数设计

```bash
# 基本用法
eval-cli --workload-type <TYPE> --model <MODEL> [OPTIONS]

# 完整参数列表
eval-cli \
  --workload-type <TYPE>           # workload 类型（必需）
  --model <MODEL>                  # 模型名称或 preset（必需）
  --config <FILE>                  # 配置文件路径（JSON/YAML，可选）
  --hardware <SPEC>                 # 硬件规格（可选）
  --strategy <SPEC>                 # 并行策略（可选）
  --output <FORMAT>                 # 输出格式（json/yaml/table，默认 json）
  --output-file <FILE>              # 输出文件路径（可选）
  --breakdown-level <LEVEL>         # 分解层级（stage/phase/submodule/memory，默认 stage）
  --verbose                         # 详细输出
  --interactive                     # 交互式模式
```

### 5.3 参数详细说明

#### 5.3.1 --workload-type 参数

支持的所有 workload 类型：

| Workload 类型 | 说明 | 示例 |
|---------------|------|------|
| `training` | 预训练 | `--workload-type training` |
| `rl-training` | RL 后训练 | `--workload-type rl-training` |
| `inference` | LLM 推理（PD 混布） | `--workload-type inference` |
| `pd-disagg` | LLM 推理（PD 分离） | `--workload-type pd-disagg` |
| `multimodal-inference` | 多模态推理 | `--workload-type multimodal-inference` |
| `diffusion-pipeline` | 扩散生成 | `--workload-type diffusion-pipeline` |

#### 5.3.2 --model 参数

支持多种模型指定方式：

```bash
# 使用 preset 名称
--model llama-7b
--model deepseek-v3

# 使用配置文件
--model @configs/models/llama_7b.yaml

# 直接指定参数
--model name=llama-7b,hidden_size=4096,num_layers=32
```

#### 5.3.3 --config 参数

从配置文件加载完整评估配置：

```bash
# JSON 配置
--config configs/training/llama_7b_h100.json

# YAML 配置
--config configs/training/llama_7b_h100.yaml
```

配置文件格式：

```yaml
# configs/training/llama_7b_h100.yaml
workload:
  type: training
  parameters:
    batch_size: 32
    seq_len: 4096
    micro_batch_size: 4

stages:
  - stage_name: main
    model:
      preset: llama-7b

hardware:
  device: H100-SXM-80GB
  num_devices: 8

strategy:
  tp_degree: 8
  pp_degree: 1
```

#### 5.3.4 --hardware 参数

硬件规格指定：

```bash
# 简写格式
--hardware H100-SXM-80GB:8

# 完整格式
--hardware device=H100-SXM-80GB,num_devices=8

# 使用配置文件
--hardware @configs/hardware/h100_cluster.yaml
```

#### 5.3.5 --strategy 参数

并行策略指定：

```bash
# 简写格式
--strategy tp=8,pp=1

# 完整格式
--strategy tp_degree=8,pp_degree=1,dp_degree=1

# Multi-stage workload（不同 stage 不同策略）
--strategy main:tp=8,pp=1 encoder:tp=4,pp=2
```

#### 5.3.6 --output 参数

输出格式支持：

| 格式 | 说明 | 示例 |
|------|------|------|
| `json` | JSON 格式（默认） | `--output json` |
| `yaml` | YAML 格式 | `--output yaml` |
| `table` | 表格格式 | `--output table` |

#### 5.3.7 --breakdown-level 参数

分解层级输出：

```bash
# Stage 级别（默认）
--breakdown-level stage

# Phase 级别（适用于 rl-training）
--breakdown-level phase

# Submodule 级别（详细）
--breakdown-level submodule

# Memory 分解
--breakdown-level memory

# 多级别组合
--breakdown-level stage,submodule,memory
```

### 5.4 交互式模式

无参数或使用 `--interactive` 标志进入交互式配置：

```bash
# 进入交互式模式
eval-cli
eval-cli --interactive
```

交互式配置流程：

```
$ eval-cli
╔══════════════════════════════════════════════════════════╗
║           LLM Performance Evaluator CLI                  ║
╚══════════════════════════════════════════════════════════╝

Step 1: Select Workload Type
  1. training           (预训练)
  2. rl-training        (RL 后训练)
  3. inference          (LLM 推理 - PD 混布)
  4. pd-disagg          (LLM 推理 - PD 分离)
  5. multimodal-inference (多模态推理)
  6. diffusion-pipeline (扩散生成)
  
Enter choice [1-6]: 1

Step 2: Select Model
  1. llama-7b
  2. llama-13b
  3. llama-70b
  4. deepseek-v3
  5. custom model...
  
Enter choice [1-5]: 1

Step 3: Configure Hardware
Device [H100-SXM-80GB]: 
Number of devices [8]: 16

Step 4: Configure Workload Parameters
Batch size [32]: 64
Sequence length [2048]: 4096
Micro batch size [1]: 4

Step 5: Configure Parallel Strategy
TP degree [1]: 8
PP degree [1]: 2
DP degree [1]: 1

Step 6: Select Output Format
  1. json
  2. yaml
  3. table
  
Enter choice [1-3]: 3

────────────────────────────────────────
Configuration Summary:
  Workload: training
  Model: llama-7b
  Hardware: 16x H100-SXM-80GB
  Strategy: TP=8, PP=2, DP=1
  
  Parameters:
    batch_size: 64
    seq_len: 4096
    micro_batch_size: 4
────────────────────────────────────────

Run evaluation? [Y/n]: Y

Evaluating...
[████████████████████████████████] 100%

═══════════════════════════════════════
              Results Summary
═══════════════════════════════════════
Throughput:    12,500 tokens/sec
MFU:           45.2%
Peak Memory:   65.3 GB/device
Total Time:    2.34 sec/batch

Memory Breakdown:
  ─────────────────────────────────
  Component           Memory (GB)
  ─────────────────────────────────
  Parameters              12.5
  Activations             28.3
  Optimizer States        24.5
  ─────────────────────────────────
  Total                   65.3
  ─────────────────────────────────

Save results? [Y/n]: Y
Output file [results/training_llama7b_20260429.json]: 
Configuration saved to: configs/training/llama_7b_h100_saved.yaml
```

### 5.5 使用示例

#### 5.5.1 基本用法

```bash
# Training workload
eval-cli \
  --workload-type training \
  --model llama-7b \
  --hardware H100-SXM-80GB:8 \
  --strategy tp=8,pp=1 \
  --output table

# Inference workload
eval-cli \
  --workload-type inference \
  --model deepseek-v3 \
  --hardware H100-SXM-80GB:8 \
  --strategy tp=8 \
  --output json

# Diffusion workload
eval-cli \
  --workload-type diffusion-pipeline \
  --model sdxl \
  --hardware H100-SXM-80GB:4 \
  --strategy tp=4 \
  --output yaml
```

#### 5.5.2 使用配置文件

```bash
# 从配置文件加载
eval-cli --config configs/training/llama_7b_h100.yaml

# 配置文件 + 覆盖参数
eval-cli \
  --config configs/training/llama_7b_h100.yaml \
  --hardware H100-SXM-80GB:16 \
  --strategy tp=16,pp=1
```

#### 5.5.3 Multi-Stage Workload

```bash
# Diffusion Pipeline（multi-stage）
eval-cli \
  --workload-type diffusion-pipeline \
  --model sdxl \
  --hardware H100-SXM-80GB:4 \
  --strategy "encoder:tp=2 main:tp=4 decoder:tp=2" \
  --output table

# RL Training（multi-phase）
eval-cli \
  --workload-type rl-training \
  --model llama-7b \
  --hardware H100-SXM-80GB:8 \
  --strategy tp=8 \
  --output json \
  --breakdown-level phase
```

#### 5.5.4 批量评估

```bash
# 使用 eval-batch 批量评估
eval-batch \
  --config-list configs/batch/scaling_study.yaml \
  --output-dir results/scaling_study \
  --parallel 4

# 批量配置文件
cat configs/batch/scaling_study.yaml
```

```yaml
# configs/batch/scaling_study.yaml
workload_type: training
model: llama-7b
base_config:
  hardware:
    device: H100-SXM-80GB
  strategy:
    pp_degree: 1

sweep:
  - num_devices: [8, 16, 32, 64]
  - tp_degree: [1, 2, 4, 8]
  - batch_size: [32, 64, 128]
```

### 5.6 配置文件示例

#### 5.6.1 Training 配置示例

```yaml
# scripts/configs/training/llama_7b_h100.yaml
workload:
  type: training
  parameters:
    batch_size: 32
    seq_len: 4096
    micro_batch_size: 4

stages:
  - stage_name: main
    model:
      preset: llama-7b

hardware:
  device: H100-SXM-80GB
  num_devices: 8

strategy:
  tp_degree: 8
  pp_degree: 1
  dp_degree: 1

output:
  format: json
  breakdown_level: [stage, submodule, memory]
```

#### 5.6.2 PD-Disagg 配置示例

```yaml
# scripts/configs/inference/pd_disagg.yaml
workload:
  type: pd-disagg
  parameters:
    input_tokens: 128
    output_tokens: 512

stages:
  - stage_name: prefill
    model:
      preset: deepseek-v3
    strategy:
      tp_degree: 8
      pp_degree: 1

  - stage_name: decode
    model:
      preset: deepseek-v3
    strategy:
      tp_degree: 4
      pp_degree: 1

hardware:
  device: H100-SXM-80GB
  num_devices: 12

output:
  format: table
  breakdown_level: [stage]
```

#### 5.6.3 Diffusion Pipeline 配置示例

```yaml
# scripts/configs/diffusion/sdxl_t2i.yaml
workload:
  type: diffusion-pipeline
  parameters:
    diffusion_steps: 50
    image_size: 1024
    batch_size: 1

stages:
  - stage_name: encoder
    model:
      preset: sdxl-encoder
    strategy:
      tp_degree: 2

  - stage_name: diffusion
    model:
      preset: sdxl-unet
    strategy:
      tp_degree: 4

  - stage_name: decoder
    model:
      preset: sdxl-decoder
    strategy:
      tp_degree: 2

hardware:
  device: H100-SXM-80GB
  num_devices: 8

output:
  format: yaml
```

### 5.7 输出格式

#### 5.7.1 JSON 输出

```bash
eval-cli --config configs/training/llama_7b_h100.yaml --output json
```

```json
{
  "success": true,
  "result": {
    "workload_type": "training",
    "model": "llama-7b",
    "hardware": {
      "device": "H100-SXM-80GB",
      "num_devices": 8
    },
    "strategy": {
      "tp_degree": 8,
      "pp_degree": 1,
      "dp_degree": 1
    },
    "metrics": {
      "throughput_tokens_per_sec": 12500,
      "mfu": 0.452,
      "peak_memory_gb": 65.3,
      "total_time_sec": 2.34
    },
    "breakdown": {
      "stage": {
        "main": {
          "compute_time_sec": 1.85,
          "communication_time_sec": 0.42,
          "memory_gb": 65.3
        }
      },
      "submodule": {
        "attention": { "time_sec": 0.92, "memory_gb": 28.5 },
        "mlp": { "time_sec": 0.78, "memory_gb": 32.1 },
        "embedding": { "time_sec": 0.15, "memory_gb": 4.7 }
      },
      "memory": {
        "parameters": 12.5,
        "activations": 28.3,
        "optimizer_states": 24.5
      }
    }
  }
}
```

#### 5.7.2 YAML 输出

```bash
eval-cli --config configs/training/llama_7b_h100.yaml --output yaml
```

```yaml
success: true
result:
  workload_type: training
  model: llama-7b
  hardware:
    device: H100-SXM-80GB
    num_devices: 8
  strategy:
    tp_degree: 8
    pp_degree: 1
    dp_degree: 1
  metrics:
    throughput_tokens_per_sec: 12500
    mfu: 0.452
    peak_memory_gb: 65.3
    total_time_sec: 2.34
  breakdown:
    stage:
      main:
        compute_time_sec: 1.85
        communication_time_sec: 0.42
        memory_gb: 65.3
    submodule:
      attention:
        time_sec: 0.92
        memory_gb: 28.5
      mlp:
        time_sec: 0.78
        memory_gb: 32.1
      embedding:
        time_sec: 0.15
        memory_gb: 4.7
    memory:
      parameters: 12.5
      activations: 28.3
      optimizer_states: 24.5
```

#### 5.7.3 Table 输出

```bash
eval-cli --config configs/training/llama_7b_h100.yaml --output table
```

```
═══════════════════════════════════════════════════════════════
                    Training Evaluation Results
═══════════════════════════════════════════════════════════════

Configuration:
  Workload:     training
  Model:        llama-7b
  Hardware:     8x H100-SXM-80GB
  Strategy:     TP=8, PP=1, DP=1

───────────────────────────────────────────────────────────────
Metrics:
───────────────────────────────────────────────────────────────
  Throughput:         12,500 tokens/sec
  MFU:                45.2%
  Peak Memory:        65.3 GB/device
  Total Time:         2.34 sec/batch

───────────────────────────────────────────────────────────────
Stage Breakdown:
───────────────────────────────────────────────────────────────
  Stage        Compute(s)  Comm(s)  Memory(GB)  %Time
  ──────────────────────────────────────────────────────
  main              1.85     0.42       65.3    100.0%

───────────────────────────────────────────────────────────────
Submodule Breakdown:
───────────────────────────────────────────────────────────────
  Submodule      Time(s)  Memory(GB)  %Time
  ────────────────────────────────────────────────
  attention        0.92       28.5    39.3%
  mlp              0.78       32.1    33.3%
  embedding        0.15        4.7     6.4%

───────────────────────────────────────────────────────────────
Memory Breakdown:
───────────────────────────────────────────────────────────────
  Component             Memory (GB)    %
  ──────────────────────────────────────────
  Parameters                 12.5     19.1%
  Activations                28.3     43.4%
  Optimizer States           24.5     37.5%
  ──────────────────────────────────────────
  Total                      65.3    100.0%

═══════════════════════════════════════════════════════════════
```

### 5.8 CLI 实现框架

```python
# bin/eval-cli

#!/usr/bin/env python3
import argparse
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from llm_perf.workload.engine import EvaluationEngine
from llm_perf.workload.registry import WorkloadSchemaRegistry
from llm_perf.reporter import JsonReporter, YamlReporter, TableReporter
from llm_perf.config import ConfigLoader


def main():
    args = parse_args()
    
    if args.interactive or len(sys.argv) == 1:
        run_interactive_mode()
    else:
        run_cli_mode(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Performance Evaluator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--workload-type", "-w",
        choices=["training", "rl-training", "inference", "pd-disagg", 
                 "multimodal-inference", "diffusion-pipeline"],
        help="Workload type"
    )
    
    parser.add_argument(
        "--model", "-m",
        help="Model name or preset"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file path (JSON/YAML)"
    )
    
    parser.add_argument(
        "--hardware",
        help="Hardware specification (e.g., H100-SXM-80GB:8)"
    )
    
    parser.add_argument(
        "--strategy",
        help="Parallel strategy (e.g., tp=8,pp=1)"
    )
    
    parser.add_argument(
        "--output", "-o",
        choices=["json", "yaml", "table"],
        default="json",
        help="Output format"
    )
    
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file path"
    )
    
    parser.add_argument(
        "--breakdown-level",
        choices=["stage", "phase", "submodule", "memory"],
        default=["stage"],
        nargs="+",
        help="Breakdown level"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    
    return parser.parse_args()


def run_cli_mode(args):
    config = build_config(args)
    
    if args.verbose:
        print_config_summary(config)
    
    engine = EvaluationEngine()
    result = engine.evaluate(config)
    
    reporter = get_reporter(args.output)
    output = reporter.format(result, breakdown_level=args.breakdown_level)
    
    if args.output_file:
        args.output_file.write_text(output)
        print(f"Results saved to: {args.output_file}")
    else:
        print(output)


def run_interactive_mode():
    from llm_perf.cli.interactive import InteractiveSession
    
    session = InteractiveSession()
    config = session.run()
    
    if config:
        engine = EvaluationEngine()
        result = engine.evaluate(config)
        
        output_format = session.get_output_format()
        reporter = get_reporter(output_format)
        output = reporter.format(result)
        
        print(output)
        
        if session.should_save():
            save_path = session.get_save_path()
            save_path.write_text(output)
            print(f"Results saved to: {save_path}")


def build_config(args) -> Dict[str, Any]:
    config = {}
    
    if args.config:
        loader = ConfigLoader()
        config = loader.load(args.config)
    
    if args.workload_type:
        config.setdefault("workload", {})["type"] = args.workload_type
    
    if args.model:
        config.setdefault("stages", [{"stage_name": "main"}])
        config["stages"][0]["model"] = {"preset": args.model}
    
    if args.hardware:
        config["hardware"] = parse_hardware(args.hardware)
    
    if args.strategy:
        config["strategy"] = parse_strategy(args.strategy)
    
    return config


def get_reporter(format: str):
    reporters = {
        "json": JsonReporter,
        "yaml": YamlReporter,
        "table": TableReporter
    }
    return reporters[format]()


if __name__ == "__main__":
    main()
```

---

## 6. 数据契约设计

### 6.1 Schema 请求格式

**GET /api/schema/workload?type={workload_type}**

Response: （与原设计一致，略）

### 6.2 Multi-Stage Workload Schema

**GET /api/schema/workload?type=diffusion-pipeline**

Response: （与原设计一致，略）

### 6.3 评估请求格式

**POST /api/evaluate**

Request: （与原设计一致，略）

### 6.4 评估结果格式

Response: （与原设计一致，略）

### 6.5 错误响应格式

（与原设计一致，略）

---

## 7. Workload 模型设计（移到 llm_perf）

### 7.1 Workload 类型定义

```python
# llm_perf/workload/schema.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class WorkloadCategory(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    GENERATION = "generation"
    MIXED = "mixed"

@dataclass
class WorkloadTypeDefinition:
    name: str
    category: WorkloadCategory
    display_name: str
    description: str
    
    stages_type: str
    stage_definitions: List[Dict]
    
    parameters_schema: Dict[str, Dict]
    parameters_layout: Dict
    
    strategy_schema: Dict[str, Dict]
    strategy_mode: str
    
    validation_rules: List[Dict]
    
    result_metrics: Dict
    result_breakdown: Dict
```

### 7.2 Workload 类型列表

| Workload 类型 | Category | Stage 类型 | 策略模式 | 主要参数 |
|---------------|----------|------------|----------|----------|
| `training` | TRAINING | single | single | batch_size, seq_len, micro_batch_size |
| `inference` | INFERENCE | single | single | batch_size, prompt_len, generation_len |
| `pd-disagg` | INFERENCE | separated | separated | input_tokens, output_tokens |
| `rl-training` | MIXED | phases | separated | seq_len, num_rollouts, ppo_epochs |
| `diffusion-pipeline` | GENERATION | pipeline | shared | diffusion_steps, generation_mode |
| `multimodal-inference` | INFERENCE | pipeline | shared | prompt_len, image_size |

### 7.3 Stage 模型定义

```python
# llm_perf/workload/schema.py

@dataclass
class StageDefinition:
    stage_name: str
    display_name: str
    description: str
    
    model_filter: Dict[str, List]
    optional: bool
    default_model: Optional[str]
    independent_strategy: bool
    parameter_overrides: Dict
```

---

## 8. 并行策略模型设计（移到 llm_perf）

### 8.1 策略 Schema 定义

```python
# llm_perf/workload/strategy.py

@dataclass
class StrategyParameterDefinition:
    name: str
    type: str                              # integer/select/boolean/computed
    label: str
    default: Any
    
    min: Optional[int]
    max: Optional[int]
    options: Optional[List]
    formula: Optional[str]
    
    constraints: List[Dict]
    visible_when: Optional[Dict]
    group: Optional[str]
```

### 8.2 标准策略参数

（与原设计 5.1 节一致，略）

### 8.3 策略模式

（与原设计 5.3 节一致，略）

---

## 9. 硬件拓扑模型设计（移到 llm_perf）

（与原设计第 6 章一致，略）

---

## 10. 评估结果模型设计（移到 llm_perf）

（与原设计第 7 章一致，略）

---

## 11. 目录结构设计（修正版）

### 11.1 前端目录 (web2/)

（与原设计 11.1 一致，略）

### 11.2 API 目录 (web2_api/) - 修正版

```
web2_api/
├── __init__.py
├── app.py                             # Flask 应用
├── routes/
│   ├── __init__.py
│   ├── workloads.py                   # Workload API（加载 configs/workloads/）
│   ├── models.py                      # Model API（加载 configs/models/）
│   ├── schema.py                      # Schema API（基于配置生成表单 schema）
│   ├── evaluate.py                    # Evaluate API（调用 llm_perf）
│   └── export.py                      # Export API
├── utils/
│   ├── __init__.py
│   ├── request_parser.py              # 请求解析
│   ├── result_builder.py              # 结果构建
│   └ error_handler.py                 # 错误处理
└── tests/
```

**核心变更**：
- 删除 `web2_api/services/`（直接调用 llm_perf/workload/loader）
- 新增 `routes/workloads.py`（加载 configs/workloads/）
- 新增 `routes/models.py`（加载 configs/models/）

### 11.3 llm_perf 目录 - 修正版

```
llm_perf/
├── __init__.py
├── workload/                          # 【新增】Workload 模块
│   ├── __init__.py
│   ├── loader.py                      # 加载 configs/workloads/*.yaml 和 configs/models/*.yaml
│   ├── validator.py                   # 校验配置合法性
│   ├── schema.py                      # Workload/Model 数据结构（从 YAML 解析）
│   ├── engine.py                      # 评估引擎
│   ├── breakdown.py                   # 分解计算
│   └── tests/
│   ├── modeling/                          # 【已有】模型建模
│   ├── registry.py                    # ModelRegistry
│   ├── module.py                      # ShardedModule
│   └── ...
├── analyzer/                          # 【已有】分析器
│   ├── unified.py                     # UnifiedAnalyzer
│   ├── breakdown.py                   # Breakdown
│   └── handlers/                      # Handler
│   └ ...
├── strategy/                          # 【已有】策略
│   ├── parallel_context.py            # ParallelContext
│   └ ...
├── hardware/                          # 【已有】硬件
│   ├── device.py                      # Device
│   ├── cluster.py                     # Cluster
│   └ topology.py                      # Topology
│   └ ...
├── kernels/                           # 【已有】Kernel
│   ├── functional.py                  # Kernel API
│   ├── compute.py                     # Compute kernels
│   └ communication.py                 # Communication kernels
│   └ ...
├── validation/                        # 【已有】验证
│   ├── memory_validator.py
│   ├── strategy_validator.py
│   └ ...
├── reporter/                          # 【已有】报告
│   ├── json_reporter.py
│   ├── xlsx_reporter.py
│   └ ...
├── cli/                               # 【已有】CLI
│   ├── main.py                        # CLI 入口
│   └ ...
└── utils/
    └ ...
```

**核心变更**：
- 删除 `llm_perf/workload/presets/`（加载 configs/workloads/ 而非重新定义）
- 新增 `loader.py`（加载 configs/ 下的 YAML 配置）
- 新增 `validator.py`（校验配置合法性）

### 11.4 CLI 目录 - 修正版

```
bin/
├── eval-cli                           # 统一评估 CLI（支持所有 workload）
└── eval-batch                         # 批量评估工具
```

**删除**：不再需要 `scripts/configs/` 目录，直接使用 `configs/models/` 和 `configs/workloads/`

### 11.5 配置目录结构（复用现有）

```
configs/
├── workloads/                         # Workload 配置（应用场景 + compute_mode）
│   ├── training/                      # compute_mode: base
│   │   ├── training.yaml              # 通用训练
│   │   └ denoise.yaml                 # 去噪训练
│   ├── rl_training/                   # compute_mode: autoregressive
│   │   ├── rl_ppo.yaml                # PPO 训练
│   │   └ rl_grpo.yaml                 # GRPO 训练
│   ├── inference/                     # compute_mode: autoregressive
│   │   ├── inference.yaml             # 通用推理
│   │   ├── autoregressive.yaml        # 自回归生成
│   │   └ speculative_decoding.yaml    # 投机解码
│   ├── pd_disagg/                     # compute_mode: autoregressive
│   ├── multimodal/                    # compute_mode: autoregressive
│   │   └ multimodal_inference.yaml    # 多模态推理
│   ├── diffusion/                     # compute_mode: iterative
│   │   ├── denoise.yaml               # 多步去噪推理
│   │   └ pipeline.yaml                # 扩散 Pipeline
│   ├── conv/                          # compute_mode: conv
│   │   ├── encoder.yaml               # 卷积编码
│   │   ├── decoder.yaml               # 卷积解码
│   │   └ resnet.yaml                  # ResNet 分类
│   └ custom/                          # 用户自定义
│   └ README.md                        # 配置说明文档
│
├── models/                            # 模型配置（已有）
│   ├── llama-7b.yaml
│   ├── llama-13b.yaml
│   ├── llama-70b.yaml
│   ├── deepseek-v3.yaml
│   ├── qwen3-5.yaml
│   ├── qwen3-5-moe.yaml
│   ├── mixtral-8x7b.yaml
│   ├── wan-dit.yaml
│   ├── wan-t2v-14b.yaml
│   ├── resnet50.yaml
│   └ ...                              # 其他模型
│
├── hardware/                          # 硬件配置（已有）
│   ├── h100.yaml
│   ├── a100.yaml
│   └ ...
│
└── eval/                              # 【可选】评估配置示例
    ├── llama7b_training.yaml          # 评估配置示例
    ├── deepseekv3_inference.yaml
    └ ...
```

### 11.6 整体项目结构 - 修正版

```
CePing/
├── web/                               # 旧版 web (保留兼容)
├── web2/                              # 新版前端 (React)
├── web2_api/                          # 新版 API（HTTP 适配层）
├── llm_perf/                          # 【核心层】
│   ├── workload/                      # 【新增】Workload 模块
│   │   ├── loader.py                  # 加载 configs/*.yaml
│   │   ├── validator.py               # 校验配置
│   │   ├── schema.py                  # 数据结构
│   │   ├── engine.py                  # 评估引擎
│   │   └ breakdown.py                 # 分解计算
│   ├── modeling/                      # 模型建模
│   ├── analyzer/                      # 分析器
│   ├── strategy/                      # 策略
│   ├── hardware/                      # 硬件
│   ├── kernels/                       # Kernel
│   ├── validation/                    # 验证
│   ├── reporter/                      # 报告
│   └ cli/                             # CLI
├── bin/                               # 【新增】CLI 工具
│   ├── eval-cli                       # 统一评估 CLI
│   └ eval-batch                       # 批量评估
├── configs/                           # 【复用】YAML 配置
│   ├── workloads/                     # Workload 配置
│   ├── models/                        # Model 配置
│   ├── hardware/                      # Hardware 配置
│   └ eval/                            # 评估配置示例（可选）
├── docs/
│   ├── web2_design.md                 # 本文档
│   ├── architecture.md
│   └ ...
├── tests/
├── .agents/
│   └ skills/
└── README.md
```

**核心变更**：
- 删除 `web2_schema/`（Schema 定义在 llm_perf/workload/schema.py）
- 删除 `scripts/configs/`（复用 configs/models/ 和 configs/workloads/）
- 删除 `llm_perf/workload/presets/`（加载 configs/workloads/ 而非重新定义）

---

## 12. 实现步骤（修正版）

### Phase 1: llm_perf/workload 模块（预计 3 天）

1. **Day 1**: Workload Loader + Validator
   - 实现 `llm_perf/workload/loader.py`（加载 configs/*.yaml）
   - 实现 `llm_perf/workload/validator.py`（校验配置）
   - 实现 `llm_perf/workload/schema.py`（数据结构）

2. **Day 2**: 评估引擎
   - 实现 `llm_perf/workload/engine.py`（调用 UnifiedAnalyzer）
   - 实现 `llm_perf/workload/breakdown.py`（分解计算）
   - 编写 loader/validator 测试用例

3. **Day 3**: 补充缺失配置
   - 新增 `configs/workloads/multimodal/multimodal-inference.yaml`
   - 新增 `configs/models/` 中的多模态模型配置（如 llava、qwen-vl）
   - 更新 `configs/workloads/README.md`

### Phase 2: API Layer（预计 2 天）

1. **Day 4**: Workload/Model API
   - 实现 `/api/workloads`（列出 configs/workloads/）
   - 实现 `/api/workload/{name}`（加载 configs/workloads/{name}.yaml）
   - 实现 `/api/models`（列出 configs/models/）
   - 实现 `/api/model/{name}`（加载 configs/models/{name}.yaml）

2. **Day 5**: Schema API + Evaluate API
   - 实现 `/api/schema/workload/{name}`（基于配置生成表单 schema）
   - 实现 `/api/schema/model/{name}`（模型参数表单 schema）
   - 实现 `/api/evaluate`（调用 llm_perf/workload/engine）

### Phase 3: Frontend Layer（预计 5 天）

（与原设计一致，略）

### Phase 4: CLI 工具（预计 3 天）

1. **Day 11**: eval-cli 核心
   - 实现 `bin/eval-cli` 命令行参数解析
   - 实现 `--workload` 参数（加载 configs/workloads/*.yaml）
   - 实现 `--model` 参数（加载 configs/models/*.yaml）
   - 实现 `--list-workloads` 和 `--list-models` 命令
   - 实现 JSON/YAML/Table 输出格式

2. **Day 12**: eval-cli 交互式模式
   - 实现交互式配置流程
   - 实现 workload 类别选择、模型选择
   - 实现配置保存功能
   - 实现批量评估工具 `bin/eval-batch`

3. **Day 13**: 配置示例和文档
   - 创建 `configs/eval/` 目录（可选评估配置示例）
   - 编写 CLI 使用文档
   - 更新 README.md

### Phase 5: 验证和文档（预计 2 天）

1. **Day 14**: 验证
   - llm_perf/workload/loader 测试
   - API 测试（workloads、models、schema）
   - CLI 测试（所有 workload 类型）

2. **Day 15**: 文档
   - 更新 configs/workloads/README.md
   - CLI 使用指南
   - API 文档

---

## 13. 兼容性考虑

（与原设计一致，略）

---

## 14. 检查清单

### 设计阶段

- [x] 确认复用 configs/models/ 和 configs/workloads/
- [ ] llm_perf/workload/loader.py 设计完成
- [ ] llm_perf/workload/validator.py 设计完成
- [ ] llm_perf/workload/schema.py 设计完成
- [ ] 评估引擎设计完成
- [ ] API Layer 设计完成（加载 configs/）
- [ ] 统一 CLI 工具设计完成
- [ ] 多模态 workload 配置设计完成
- [ ] 文档评审通过

### 实现阶段

- [ ] llm_perf/workload/loader.py 实现完成
- [ ] llm_perf/workload/validator.py 实现完成
- [ ] llm_perf/workload/engine.py 实现完成
- [ ] API Layer 实现完成
- [ ] Frontend Layer 实现完成
- [ ] 统一 CLI 工具实现完成
- [ ] configs/workloads/multimodal/ 配置新增完成
- [ ] 测试覆盖完成

### 验证阶段

- [ ] loader 加载 configs/*.yaml 测试通过
- [ ] validator 校验配置测试通过
- [ ] API 测试通过（workloads、models、schema）
- [ ] CLI 测试通过（所有 workload 类型）
- [ ] 端到端测试通过
- [ ] 存量功能兼容验证

---

## 15. 修正摘要

### 修正要点

1. **复用现有配置体系（核心变更）**
   - ❌ 删除 `scripts/configs/` 目录定义（避免重复）
   - ❌ 删除 `llm_perf/workload/presets/`（加载 configs/workloads/ 而非重新定义）
   - ✅ 复用 `configs/models/`（20+ 模型配置）
   - ✅ 复用 `configs/workloads/`（training/rl_training/inference/diffusion/conv）
   - 新增 `llm_perf/workload/loader.py` 加载现有配置

2. **llm_perf/workload 职责调整**
   - **loader.py**: 加载 configs/workloads/*.yaml 和 configs/models/*.yaml
   - **validator.py**: 校验配置合法性
   - **schema.py**: Workload/Model 数据结构（从 YAML 解析）
   - **engine.py**: 评估引擎（调用 UnifiedAnalyzer）
   - **breakdown.py**: 分解计算

3. **Web API 调用现有配置**
   - `/api/workloads` → 返回 configs/workloads/ 目录列表
   - `/api/workload/{name}` → 加载 configs/workloads/{name}.yaml
   - `/api/models` → 返回 configs/models/ 目录列表
   - `/api/model/{name}` → 加载 configs/models/{name}.yaml
   - `/api/schema/workload/{name}` → 基于配置生成前端表单 schema

4. **CLI 工具调用现有配置**
    - `bin/eval-cli --workload inference/autoregressive` → 加载 configs/workloads/inference/autoregressive.yaml
    - `bin/eval-cli --model llama-7b` → 加载 configs/models/llama-7b.yaml
    - 支持 `--list-workloads` 和 `--list-models` 查看可用配置

5. **Workload 划分方式**
    - 采用"应用场景 + compute_mode 属性"双重划分
    - 目录：training、rl_training、inference、pd_disagg、multimodal、diffusion、conv、custom
    - compute_mode：base、autoregressive、iterative、conv

### 架构对比

| 方面 | 原设计 | 修正版 |
|------|--------|--------|
| 配置定义 | scripts/configs/ + llm_perf/workload/presets/ | **复用 configs/models/ + configs/workloads/** |
| Schema 来源 | llm_perf/workload/ 定义 | **从 configs/*.yaml 加载解析** |
| loader.py | 无 | **新增，加载 configs/*.yaml** |
| validator.py | 无 | **新增，校验配置合法性** |
| Web API | 从 llm_perf 导出 schema | **加载 configs/*.yaml 并生成 schema** |
| CLI --workload | --workload-type 参数 | **--workload 参数，支持路径格式** |
| CLI --model | preset 名称 | **加载 configs/models/*.yaml** |

### 删除的重复定义

| 原定义位置 | 状态 |
|------------|------|
| `scripts/configs/` | ❌ 删除（复用 configs/） |
| `llm_perf/workload/presets/` | ❌ 删除（加载 configs/workloads/） |
| `web2_schema/` | ❌ 删除（Schema 在 llm_perf/workload/schema.py） |

### 新增内容

| 新增位置 | 说明 |
|----------|------|
| `llm_perf/workload/loader.py` | 加载 configs/*.yaml |
| `llm_perf/workload/validator.py` | 校验配置合法性 |
| `web2_api/routes/workloads.py` | Workload API（加载 configs/workloads/） |
| `web2_api/routes/models.py` | Model API（加载 configs/models/） |
| `configs/workloads/multimodal/` | **待新增**多模态推理 workload |

### 待新增的 workload 配置

```yaml
# configs/workloads/multimodal/multimodal-inference.yaml（待创建）
name: multimodal-inference
description: 多模态理解推理（vision encoder + LLM backbone）
workload_type: inference

component_mapping:
  encoder: vision_encoder
  backbone: llm_backbone

phases:
  - name: encode
    compute_type: forward
    component: encoder
    compute_pattern: conv_encoder
    repeat: 1
    
  - name: prefill
    compute_type: forward
    component: backbone
    compute_pattern: transformer_block
    repeat: 1
    extra_params:
      seq_len: prompt_len
    
  - name: decode
    compute_type: forward
    component: backbone
    compute_pattern: transformer_block
    repeat: generation_len
    extra_params:
      seq_len: 1

default_params:
  batch_size: 1
  prompt_len: 512
  generation_len: 128
  image_size: 336

throughput_metric: tokens_per_sec
```

---

## 附录 A: Workload Schema 示例

（与原设计一致，略）

---

## 附录 B: 前端类型定义

（与原设计一致，略）

---

*文档版本: 2.2*
*创建日期: 2026-04-29*
*修正日期: 2026-04-29*
*修正原因: 整合现有 configs/models/ 和 configs/workloads/，避免重复定义配置*
*作者: CePing Architecture Team*
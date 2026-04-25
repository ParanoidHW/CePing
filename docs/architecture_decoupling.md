# 架构解耦设计文档

本文档记录 CePing 项目的前后端解耦设计，确保新增类型/模型无需修改代码即可自动显示。

---

## 目录

1. [解耦设计原则](#1-解耦设计原则)
2. [子模块分解解耦](#2-子模块分解解耦)
3. [前端渲染解耦](#3-前端渲染解耦)
4. [新增模型天然兼容](#4-新增模型天然兼容)
5. [所有分解类型解耦状态](#5-所有分解类型解耦状态)
6. [结构签名缓存机制](#6-结构签名缓存机制)

---

## 1. 解耦设计原则

### 1.1 后端数据结构原则

所有分解数据使用字典结构，自动包含所有类别/类型：

```python
{
    "by_submodule_type": {
        "embedding": {...},
        "transformer_block": {...},
        "lm_head": {...},
        "new_type": {...}  # 新增类型自动出现
    },
    "time_breakdown": {
        "compute_sec": 0.1,
        "backward_sec": 0.02,
        "new_category_sec": 0.05  # 新增类别自动出现
    }
}
```

**优势**：
- 新增类别自动出现在数据中
- 无需硬编码任何列表
- 数据结构驱动，而非代码驱动

### 1.2 前端渲染原则

前端使用 `Object.keys(data)` 自动获取字段：

```javascript
const bySubmoduleType = detailed.by_submodule_type || {};
const allTypes = Object.keys(bySubmoduleType);  // 自动发现所有类型

const timeBreakdown = result.breakdown.time_breakdown || {};
const allSecKeys = Object.keys(tb).filter(k => k.endsWith('_sec'));  // 自动发现所有时间类别
```

**优势**：
- 不硬编码任何类别列表
- 新增类型自动显示
- 前端代码完全通用

### 1.3 模型定义原则

ShardedModule 基类定义 `_submodule_name` 属性：

```python
class ShardedModule:
    _submodule_name: str = ""  # 子模块类型名称（显式声明）
```

子模块通过 `_submodules` 注册：

```python
def __setattr__(self, name: str, value: Any):
    if isinstance(value, ShardedModule):
        self._submodules[name] = value  # 子模块自动注册
```

分析器自动遍历所有子模块：

```python
def _analyze_phase_with_submodules(self, ...):
    module_inst = component.bind(ctx)
    for sub_name, sub_inst in module_inst._submodule_instances.items():
        # 自动分析所有子模块
```

---

## 2. 子模块分解解耦

### 2.1 旧架构（耦合）

之前使用 `MODULE_CLASS_TO_TYPE` 注册表硬编码 16 个类名映射：

```python
# 旧架构（已移除）
MODULE_CLASS_TO_TYPE = {
    "ShardedEmbedding": "embedding",
    "ShardedAttention": "attention",
    "ShardedLinearAttention": "linear_attention",
    "ShardedMLA": "mla",
    "ShardedFFN": "ffn",
    "ShardedMoE": "moe",
    "ShardedLMHead": "lm_head",
    "ShardedRMSNorm": "rms_norm",
    "ShardedLayerNorm": "layer_norm",
    "ShardedTransformerBlock": "transformer_block",
    "ShardedWanDiTBlock": "wan_dit",
    "ShardedWanVAE": "vae",
    "ShardedWanTextEncoder": "text_encoder",
    "ShardedQwen3_5DenseBlock": "qwen3_5_dense",
    "ShardedQwen3_5MoEBlock": "qwen3_5_moe",
    "ShardedDeepSeekBlock": "deepseek",
}
```

**问题**：
- 新增模型需手动注册
- 前端依赖固定类型列表
- 维护成本高

### 2.2 新架构（解耦）

使用 `_submodule_name` 属性 + 智能推断：

```python
class ShardedModule:
    _submodule_name: str = ""  # 显式声明

def infer_submodule_name_from_class(cls_name: str) -> str:
    """从类名推断子模块名称（fallback方案）."""
    # 移除 "Sharded" 前缀
    # 移除 "Block" 后缀
    # CamelCase to snake_case
    # 特殊缩写处理: MLA, FFN, MoE, ViT, DiT, VAE
```

**优先级**：显式声明 > 类名推断

```python
def _infer_submodule_type(self, sub_name: str, sub_inst: Any = None) -> str:
    """识别子模块类型."""
    if sub_inst and hasattr(sub_inst, "module"):
        module = sub_inst.module
        
        # 优先：显式声明
        if hasattr(module, "_submodule_name") and module._submodule_name:
            return module._submodule_name
        
        # fallback：类名推断
        module_class = type(module).__name__
        inferred_name = infer_submodule_name_from_class(module_class)
        if inferred_name:
            return inferred_name
    
    # fallback：名称推断
    sub_lower = sub_name.lower()
    if "attention" in sub_lower:
        return "attention"
    ...
```

### 2.3 使用示例

**显式声明**：

```python
class ShardedNewAttention(ShardedModule):
    _submodule_name = "new_attention"  # 显式声明
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        ...
```

**自动推断**：

```python
class ShardedNewFFN(ShardedModule):
    # 无 _submodule_name，使用类名推断
    # ShardedNewFFN → new_ffn
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        ...
```

**无需任何注册表修改！**

---

## 3. 前端渲染解耦

### 3.1 旧架构（硬编码）

```javascript
// 旧架构（已移除）
const categories = ['Compute', 'Backward', 'Optimizer'];
const types = ['embedding', 'transformer_block', 'lm_head'];
```

**问题**：
- 新增类型需修改前端代码
- 维护两份列表（后端 + 前端）

### 3.2 新架构（自动读取）

**内存分解**：

```javascript
const memByType = detailed.memory?.by_type || {};
const { total, ...breakdownItems } = memByType;

// 自动发现所有类型
const allTypes = Object.keys(breakdownItems).filter(k => !k.startsWith('total'));

// 优先顺序排列（非硬编码）
const orderedPriority = ['weight', 'gradient', 'optimizer', 'activation'];
const orderedTypes = [
    ...orderedPriority.filter(t => breakdownItems[t] !== undefined),
    ...allTypes.filter(t => !orderedPriority.includes(t))
];
```

**子模块分解**：

```javascript
const bySubmoduleType = detailed.by_submodule_type || {};

// 自动遍历所有类型
for (const [submoduleType, data] of Object.entries(bySubmoduleType)) {
    // 渲染 submoduleType 行
    ...
    
    // 自动遍历嵌套类型
    if (data.nested_breakdown) {
        for (const [nestedType, nestedData] of Object.entries(data.nested_breakdown)) {
            // 渲染 nestedType 行
        }
    }
}
```

**时间分解**：

```javascript
const tb = breakdown.time_breakdown || {};
const allSecKeys = Object.keys(tb).filter(k => k.endsWith('_sec'));

// 自动发现所有时间类别
// compute_sec, backward_sec, optimizer_sec, new_category_sec
```

**通信分解**：

```javascript
const commByPara = detailed.communication?.by_parallelism || {};
const commByOp = detailed.communication?.by_operation || {};

// 自动遍历所有并行类型
for (const [paraType, data] of Object.entries(commByPara)) {
    // tp, pp, dp, ep, new_parallelism
}

// 自动遍历所有原语类型
for (const [opType, data] of Object.entries(commByOp)) {
    // all_reduce, all_gather, reduce_scatter, all_to_all, new_op
}
```

### 3.3 效果

**新增类型自动显示**：

- 新增时间类别（如 `new_category_sec`）：自动出现在性能分解表格
- 新增子模块类型（如 `new_attention`）：自动出现在子模块分解表格
- 新增通信原语（如 `all_to_all`）：自动出现在通信分解表格
- 新增嵌套类型（如 `full_attention`）：自动出现在嵌套分解表格

**无需修改前端代码！**

---

## 4. 新增模型天然兼容

### 4.1 步骤 1：定义模型类

```python
from llm_perf.modeling import ShardedModule

class ShardedNewAttention(ShardedModule):
    _submodule_name = "new_attention"  # 显式声明
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.q_weight = ShardedTensor(...)
        self.k_weight = ShardedTensor(...)
        self.v_weight = ShardedTensor(...)
        
    def forward(self, hidden):
        q = hidden @ self.q_weight
        k = hidden @ self.k_weight
        v = hidden @ self.v_weight
        return flash_attention(q, k, v)

class ShardedNewFFN(ShardedModule):
    # 自动推断：ShardedNewFFN → new_ffn
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_weight = ShardedTensor(...)
        self.up_weight = ShardedTensor(...)
        self.down_weight = ShardedTensor(...)

class ShardedNewModelBlock(ShardedModule):
    _submodule_name = "transformer_block"  # 继承通用类型
    
    def __init__(self, ...):
        self.attention = ShardedNewAttention(...)  # 自动识别
        self.ffn = ShardedNewFFN(...)              # 自动识别
        
    def forward(self, hidden):
        norm1 = rms_norm(hidden)
        attn_out = self.attention(norm1)
        ...
```

### 4.2 步骤 2：前端自动显示

**无需修改 unified.py**：

- `_infer_submodule_type` 自动识别 `new_attention`, `new_ffn`
- 数据结构自动包含新类型

**无需修改 app.js**：

- `Object.keys(bySubmoduleType)` 自动发现新类型
- 嵌套分解自动显示 `new_attention`, `new_ffn`

**自动显示效果**：

```
子模块分解表格：
| transformer_block | xxx ms | xx% | xxx ms | xx% | xxx GB |
| - new_attention   | xxx ms | xx% | xxx ms | xx% | xxx GB |
| - new_ffn         | xxx ms | xx% | xxx ms | xx% | xxx GB |
```

---

## 5. 所有分解类型解耦状态

| 分解类型 | 解耦状态 | 实现方式 | 位置 |
|---------|---------|---------|------|
| 耗时分解 | ✅ 已解耦 | `Object.keys(time_breakdown)` | app.js:1175 |
| 推理分解 | ✅ 已解耦 | `Object.keys(inference_breakdown)` | app.js:1216 |
| 子模块分解 | ✅ 已解耦 | `Object.keys(bySubmoduleType)` | app.js:1004 |
| 嵌套分解 | ✅ 已解耦 | `Object.keys(nested_breakdown)` | app.js:1031 |
| 内存类型分解 | ✅ 已解耦 | `Object.keys(memory_by_type)` | app.js:983 |
| 通信域分解 | ✅ 已解耦 | `Object.keys(communication_by_domain)` | app.js:1076+ |
| 通信原语分解 | ✅ 已解耦 | `Object.keys(communication_by_op)` | app.js:1076+ |
| Phase分解 | ✅ 已解耦 | `Object.keys(phase_breakdown)` | app.js:1076+ |

### 5.1 新增类别自动显示机制

**新增时间类别**：

```python
# 后端添加新类别
time_breakdown["new_category_sec"] = 0.05
time_breakdown["new_category_percent"] = 10.0
```

```javascript
// 前端自动显示
const allSecKeys = Object.keys(tb).filter(k => k.endsWith('_sec'));
// → ['compute_sec', 'backward_sec', ..., 'new_category_sec']
```

**新增子模块类型**：

```python
# 后端添加新类型
by_submodule_type["new_type"] = {
    "compute": {"time_sec": 0.05},
    "memory": {"activations_gb": 1.2}
}
```

```javascript
// 前端自动显示
const allTypes = Object.keys(bySubmoduleType);
// → ['embedding', 'transformer_block', ..., 'new_type']
```

**新增通信域**：

```python
# 后端添加新通信域
communication_by_domain["new_domain"] = {
    "total_bytes": 1e9,
    "total_time_sec": 0.01
}
```

```javascript
// 前端自动显示
for (const [domain, data] of Object.entries(commByDomain)) {
    // → tp, pp, dp, ..., new_domain
}
```

---

## 6. 结构签名缓存机制

### 6.1 问题背景

Qwen3.5 使用混合注意力（linear + full），所有 Block 被视为相同结构，只分析第一个。

**原因**：

```python
# 旧签名（不含 layer_type）
sig = f"{class_name}:{hidden_size},{num_heads},..."
# 所有 Block 签名相同 → 只分析一次
```

**结果**：

- 前端只显示 `linear_attention` 分解
- `full_attention` 缺失

### 6.2 解决方案

`_compute_structure_signature` 添加 `layer_type` 属性：

```python
def _compute_structure_signature(self, sub_inst: ModuleInstance) -> str:
    """计算子模块结构签名."""
    module = sub_inst.module
    class_name = type(module).__name__
    
    config_attrs = [
        "hidden_size",
        "num_heads",
        "head_dim",
        "intermediate_size",
        "num_experts",
        "num_experts_per_token",
        "vocab_size",
        "embedding_dim",
        "dtype",
        "num_key_value_heads",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "qk_nope_head_dim",
        "v_head_dim",
        "layer_type",  # 新增！区分 linear vs full
    ]
    
    config_values = []
    for attr in config_attrs:
        if hasattr(module, attr):
            val = getattr(module, attr)
            config_values.append(f"{attr}={val}")
    
    return f"{class_name}:{','.join(config_values)}"
```

### 6.3 效果

**不同签名**：

```python
# linear_attention Block
sig0 = "ShardedQwen3_5DenseBlock:hidden_size=2048,...,layer_type=linear_attention"

# full_attention Block
sig3 = "ShardedQwen3_5DenseBlock:hidden_size=2048,...,layer_type=full_attention"

# 签名不同 → 分别缓存和分析
```

**前端显示完整分解**：

```
transformer_block 嵌套分解：
| - linear_attention | xxx ms | xx% |
| - attention (full) | xxx ms | xx% |
| - moe              | xxx ms | xx% |
```

### 6.4 测试验证

```python
def test_qwen35_signature_different_for_different_attention_types(self, setup):
    """Test different attention types have different signatures."""
    analyzer = setup["analyzer"]
    
    block0_sig = analyzer._compute_structure_signature(
        module_inst._submodule_instances["layers.0"]
    )
    block3_sig = analyzer._compute_structure_signature(
        module_inst._submodule_instances["layers.3"]
    )
    
    assert "layer_type=linear_attention" in block0_sig
    assert "layer_type=full_attention" in block3_sig
    assert block0_sig != block3_sig
```

---

## 版本历史

### v1.0 (当前)
- **移除 MODULE_CLASS_TO_TYPE 注册表**
- **新增 _submodule_name 属性**
- **infer_submodule_name_from_class 智能推断**
- **前端 Object.keys 自动发现**
- **结构签名添加 layer_type**
- **Qwen3.5 full_attention 正确显示**
- **所有分解类型完全解耦**
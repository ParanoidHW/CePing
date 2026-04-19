# Phase 4 设计方案：Preset YAML配置管理

## 1. YAML文件格式定义

```yaml
# 预设名称（文件名即preset名称）
# configs/models/llama-7b.yaml

# 元信息
description: "LLaMA 7B"
architecture: llama
sparse_type: dense
attention_features:
  - gqa

# 模型配置参数
config:
  vocab_size: 32000
  hidden_size: 4096
  num_layers: 32
  num_heads: 32
  num_kv_heads: 32
  intermediate_size: 11008
  max_seq_len: 4096
  dtype: fp16

# 参数schema（可选，用于Web UI）
param_schema:
  training:
    - name: batch_size
      label: Batch Size
      type: number
      default: 32
    - name: seq_len
      label: Sequence Length
      type: number
      default: 4096
  inference:
    - name: batch_size
      label: Batch Size
      type: number
      default: 8
    - name: prompt_len
      label: Prompt Length
      type: number
      default: 1024
    - name: generation_len
      label: Generation Length
      type: number
      default: 128
```

## 2. configs/models/目录结构

```
configs/models/
├── llama-7b.yaml
├── llama-13b.yaml
├── llama-70b.yaml
├── mixtral-8x7b.yaml
├── deepseek-v3.yaml
├── resnet50.yaml
├── video-vae.yaml
├── wan-t2v-14b.yaml
└── wan-dit.yaml
```

## 3. registry.py改动方案

### 3.1 新增函数

```python
def _load_presets_from_yaml() -> dict:
    """Load preset configurations from YAML files.
    
    Scans configs/models/*.yaml and returns a dict of presets.
    Falls back to hardcoded presets if YAML files not found.
    """
    import yaml
    from pathlib import Path
    
    presets = {}
    config_dir = Path(__file__).parent.parent.parent / "configs" / "models"
    
    if not config_dir.exists():
        return _get_hardcoded_presets()
    
    for yaml_file in config_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            preset_data = yaml.safe_load(f)
        
        preset_name = yaml_file.stem
        
        # Flatten config into top-level for backward compatibility
        preset = {
            "description": preset_data.get("description", ""),
            "architecture": preset_data.get("architecture", preset_name),
            "sparse_type": preset_data.get("sparse_type", "dense"),
            "attention_features": preset_data.get("attention_features", []),
        }
        
        # Merge config params
        if "config" in preset_data:
            preset.update(preset_data["config"])
        
        # Add param_schema if present
        if "param_schema" in preset_data:
            preset["param_schema"] = preset_data["param_schema"]
        
        presets[preset_name] = preset
    
    return presets
```

### 3.2 修改 get_model_presets()

```python
# 模块级缓存
_PRESETS_CACHE: Optional[dict] = None

def get_model_presets() -> dict:
    """Get preset configurations."""
    global _PRESETS_CACHE
    if _PRESETS_CACHE is None:
        _PRESETS_CACHE = _load_presets_from_yaml()
    return _PRESETS_CACHE
```

### 3.3 删除硬编码preset定义

将 `get_model_presets()` 中的硬编码presets移到 `_get_hardcoded_presets()` 作为fallback。

## 4. 多版本preset支持方案

### 4.1 文件命名约定

```
configs/models/
├── llama-7b.yaml          # 默认版本
├── llama-7b-v2.yaml       # v2版本
├── llama-7b-chat.yaml     # chat版本
└── llama-7b-instruct.yaml # instruct版本
```

### 4.2 版本发现机制

```python
def list_preset_versions(base_name: str) -> list:
    """List all versions of a preset.
    
    Example: list_preset_versions("llama-7b") 
    -> ["llama-7b", "llama-7b-v2", "llama-7b-chat"]
    """
    presets = get_model_presets()
    return [k for k in presets.keys() if k.startswith(base_name)]
```

### 4.3 继承机制（可选，Phase 5）

```yaml
# configs/models/llama-7b-chat.yaml
base: llama-7b
description: "LLaMA 7B Chat"
config:
  # 只覆盖差异部分
  max_seq_len: 8192
```

## 5. 实施步骤

1. 创建 `configs/models/` 目录
2. 将现有presets迁移到YAML文件
3. 修改 `registry.py`:
   - 添加 `_load_presets_from_yaml()`
   - 添加 `_get_hardcoded_presets()` 作为fallback
   - 修改 `get_model_presets()` 使用YAML加载
4. 添加单元测试
5. 运行验证测试

## 6. 向后兼容性

- 如果 `configs/models/` 不存在，使用硬编码presets
- YAML格式与现有dict格式兼容
- `create_model_from_config()` 无需修改
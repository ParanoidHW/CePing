---
name: new-model
description: 新增模型时行为
---

## 新模型评估支持

- 优先从[Huggingface](https://huggingface.co/models)上检索和获取**官方模型**的结构文件；
- 严格根据结构配置文件的超参进行评估，不得随意修改其中任何一个模型配置参数；如有不明确的结构参数，需进行备注；
- 如果Hugginface上有多个模型配置文件，并且无法区分主要backbone，同时支持评估，并在DEVELOP_LOG里进行提示，提示我进行区分；

## 模型架构开发规范

### 模型配置类
所有模型配置必须继承 `ModelConfig`，并实现 `__post_init__` 验证：

```python
@dataclass
class ModelConfig(ModelConfig):
    """模型配置类文档字符串.
    
    必须列出所有字段及其含义。
    """
    # 模型架构参数
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    
    # 派生参数（在 __post_init__ 中计算）
    intermediate_size: int = 0  # 0 表示自动计算
    
    def __post_init__(self):
        """验证和计算派生参数."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.intermediate_size == 0:
            # 默认比例（如 Llama 使用 8/3 ≈ 2.67x）
            self.intermediate_size = int(self.hidden_size * 8 / 3)
```

### 2.2 模型类结构
```python
class ModelName(BaseModel):
    """模型类文档字符串.
    
    描述模型架构特点、参考论文/实现链接。
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """构建模型层配置.
        
        必须使用 kernel API，禁止手动计算 activation_bytes。
        """
        pass
    
    def _build_transformer_block(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """构建单个 transformer 层.
        
        私有方法前缀 `_`，返回 LayerConfig 列表。
        """
        pass
```

### 2.3 架构修正规范
当修正模型架构时（如 Wan2.1 的 AdaLN）：
1. **参考官方实现**：基于 HuggingFace 或论文官方代码
2. **记录修正原因**：在代码注释中说明为什么修正
3. **更新测试**：修正后必须更新对应测试用例
4. **验证数值**：确保修正后的参数量和 FLOPs 与官方一致

```python
# Wan2.1 AdaLN 修正示例（参考 model.py line 276）
# 修正前：错误的参数数量
# 修正后：6 个调制参数（self-attn 的 shift/scale/gate + FFN 的 shift/scale/gate）
# Cross-attention 无调制（model.py line 310 确认）
num_modulation = 6 * cfg.hidden_size
```

### 2.4 特别注意事项

1. FFN/MoE intermediate size和其激活函数有关，如果是带gated的激活函数，intermediate size需要乘以2，否则不用乘；
2. 确定FFN激活函数是否带gate，不能只看huggingface官方config.json，也得看实际代码实现；如果没有调用带gated的融合算子，或者没有在down前进行act * gate，就可以认为不是带gate的；
2. Norm类的评估数据合并到其后续的模块中，不单独呈现
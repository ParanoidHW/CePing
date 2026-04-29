---
name: architecture
description: 项目架构设计准则
---

## 设计原则

1. 新特性需先进行分析合理性，并从外部公开实践中获取数据、信息，参考相关方案；
2. 设计需审视所有注册过的现有模型、kernel、硬件等如何支持待开发新特性，而非只基于目标模型/kernel/硬件进行定制化设计；
3. 原则2的例外情况，需提示我进行决策。

---

## 架构设计规范

### 1. 模块职责清晰

每个模块应有明确的单一职责：

```python
# 好的设计：职责清晰
class ShardedTensor:
    """张量数据结构，管理shape和shardable约束"""
    pass

class ShardedModule:
    """模块基类，管理子模块和权重注册"""
    pass

class ModuleInstance:
    """物理实例，管理并行切分后的物理指标"""
    pass

# 不好的设计：职责混杂
class TensorModule:
    """同时管理张量和模块，职责不清"""
    pass
```

### 2. 层次分明

遵循清晰的层次结构，避免跨层调用：

```
层次结构：
├── 基础层：Tensor, Op
├── 模块层：ShardedModule, Layers
├── 模型层：LlamaModel, DeepSeekModel
├── 实例层：ModuleInstance, WeightInstance
├── 分析层：UnifiedAnalyzer, TheoryBackend
```

### 3. 避免循环依赖

模块间依赖应为单向：

```python
# 正确：单向依赖
module.py -> tensor.py
instance.py -> module.py, tensor.py

# 错误：循环依赖
module.py -> instance.py
instance.py -> module.py  # 循环！
```

### 4. 合并或抽取功能相似的模块

例如大模型普遍都是transformer架构，attn/ffn/moe等模块可以复用，不需要为每个模型单独定义一个子模块。

多模理解也是类似，很多结构是类似的，优先考虑统一复用，在已有基础上新增必要的新配置入参。

---

## 数据结构设计规范

### 1. 语义清晰

数据结构命名和属性应清晰表达语义：

```python
# 好的设计：语义清晰
@dataclass
class KVCacheConfig:
    max_seq_len: int      # 最大序列长度
    num_layers: int       # 层数
    num_kv_heads: int     # KV头数（GQA）
    head_dim: int         # 头维度
    cache_dtype: str      # 缓存数据类型

# 不好的设计：语义模糊
@dataclass  
class CacheConfig:
    s1: int  # 不清楚代表什么
    s2: int
    n: int
```

### 2. 可扩展性

设计时应考虑未来扩展：

```python
# 好的设计：可扩展
class CommPatternDeriver:
    def derive_comm_ops(self, op: Op) -> List[CommOp]:
        """统一接口，易于添加新Op类型"""
        if isinstance(op, MatmulOp):
            return self._derive_matmul_comm(op)
        elif isinstance(op, AttentionOp):
            return self._derive_attention_comm(op)
        # 未来可以轻松添加新的Op类型

# 不好的设计：不可扩展
def infer_comm(op):
    """硬编码，难以扩展"""
    if op.type == "matmul":
        return ...
    # 添加新类型需要修改多处
```

### 3. 类型安全

使用类型注解确保类型安全：

```python
# 好的设计：类型安全
def bind(
    self,
    ctx: ParallelContext,
    pp_strategy: Optional[PPStrategy] = None,
) -> Union[ModuleInstance, PPModel]:
    """明确的类型注解"""
    pass

# 不好的设计：类型不安全
def bind(self, ctx, pp_strategy=None):
    """缺少类型注解"""
    pass
```

### 4. 架构设计保持解耦

设计方案时需要做好解耦，不同模块间减少信息的重复读写和存取，保持最小集的数据生产者，所有消费者统一从这些生产者获取数据，消费者段尽可能减少显式的注册表信息：

```python
# 好的设计：模块解耦
class Producer:
    def data_producer(self, inputs):
        # some codes to produce data
        self.intermediate_data = ....

    @property
    def data(self):
        return self.intermediate_data


class Consumer:
    def data_consumer(self, inputs, producer):
        # data process
        abstract = self.data_process(producer.data)

        return abstract


# 坏的设计：消费者自己生产和组装数据
class Consumer:
    def data_producer(self, inputs):
        self.intermediate_data = ...

    def data_consumer(self, inputs):
        # data process
        abstract = self.data_producer(data)

        return abstract

```

---

## 代码组织规范

### 1. 基类优先

公共逻辑应提取到基类：

```python
# 好的设计：公共逻辑在基类
class ShardedModule:
    def bind(self, ctx, pp_strategy=None):
        """所有子类共享"""
        pass

class LlamaModel(ShardedModule):
    # 不需要重写bind()
    pass

# 不好的设计：重复代码
class LlamaModel:
    def bind(self, ctx, pp_strategy=None):
        """重复实现"""
        pass

class DeepSeekModel:
    def bind(self, ctx, pp_strategy=None):
        """重复实现"""
        pass
```

### 2. 工具函数集中

工具函数应集中管理：

```python
# 好的设计：集中管理
# utils/constants.py
DTYPE_SIZES = {"fp16": 2, "fp32": 4, "bf16": 2}

# utils/helpers.py
def get_physical_shape(tensor, parallel_degrees):
    pass

# 不好的设计：分散定义
# module.py
DTYPE_SIZES = {...}  # 重复定义

# layers.py
DTYPE_SIZES = {...}  # 重复定义
```

### 3. 组合优于继承

优先使用组合而非深层继承：

```python
# 好的设计：组合
class LlamaModel:
    def __init__(self):
        self.embedding = ShardedEmbedding(...)
        self.layers = [ShardedTransformerBlock(...)]
        self.lm_head = ShardedLMHead(...)

# 不好的设计：深层继承
class BaseModel:hu
    pass
class LanguageModel(BaseModel):
    pass
class TransformerModel(LanguageModel):
    pass
class LlamaModel(TransformerModel):
    pass  # 继承链过长
```

---

## 接口设计规范

### 1. 一致性

相似操作的接口应保持一致：

```python
# 好的设计：接口一致
class ShardedModule:
    def bind(self, ctx) -> ModuleInstance:
        pass

class ShardedEmbedding:
    def bind(self, ctx) -> ModuleInstance:
        pass  # 相同接口

# 不好的设计：接口不一致
class ShardedModule:
    def bind(self, ctx) -> ModuleInstance:
        pass

class ShardedEmbedding:
    def create_instance(self, ctx) -> Instance:
        pass  # 不同接口
```

### 2. 向后兼容

接口变更应保持向后兼容：

```python
# 好的设计：向后兼容
def bind(
    self,
    ctx: ParallelContext,
    pp_strategy: Optional[PPStrategy] = None,  # 新增参数，默认值保持兼容
    pp_stage: Optional[int] = None,            # 保持原有参数
    mode: str = "forward_backward",            # 保持原有参数
) -> Union[ModuleInstance, PPModel]:
    pass

# 不好的设计：破坏兼容
def bind(self, ctx, strategy):  # 参数名改变
    pass
```

---

## 测试设计规范

### 1. 测试先行

新增特性前先设计测试：

```python
# 开发流程
1. 设计测试用例
2. 实现特性
3. 运行测试验证

# 测试用例示例
def test_bind_with_pp_strategy(self):
    """Test bind returns PPModel when pp_strategy provided"""
    model = LlamaModel(...)
    ctx = ParallelContext(tp_degree=8)
    pp_strategy = PPStrategy(num_stages=2)
    
    pp_model = model.bind(ctx, pp_strategy=pp_strategy)
    
    assert pp_model is not None
    assert hasattr(pp_model, "stages")
```

### 2. 边界覆盖

测试应覆盖边界情况：

```python
def test_bind_edge_cases(self):
    """边界情况测试"""
    # TP=1（无切分）
    ctx_tp1 = ParallelContext(tp_degree=1)
    
    # TP=8（正常切分）
    ctx_tp8 = ParallelContext(tp_degree=8)
    
    # pp_strategy=None（默认情况）
    instance = model.bind(ctx, pp_strategy=None)
    
    # pp_strategy提供（特殊情况）
    pp_model = model.bind(ctx, pp_strategy=strategy)
```

### 3. 集成测试

除了单元测试，还需集成测试：

```python
def test_unified_analyzer_integration(self):
    """集成测试：完整流程"""
    model = LlamaModel(...)
    device = Device.from_preset("H100-SXM-80GB")
    cluster = Cluster.create_homogeneous(...)
    strategy = StrategyConfig(tp_degree=8)
    
    analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
    result = analyzer.analyze("training", batch_size=32)
    
    # 验证完整流程
    assert result.total_time_sec > 0
    assert result.peak_memory_gb > 0
    assert len(result.phases) > 0
```

---

## 问题分析规范

### 1. TODO优先级

发现问题时按优先级排序：

```
P0: 最高优先级，立即修复（如：数据丢失、核心功能错误）
P1: 高优先级，尽快修复（如：估算不准确、逻辑错误）
P2: 中优先级，计划修复（如：代码重复、设计不完整）
P3: 低优先级，可选修复（如：性能优化、功能扩展）
```

### 2. 根因分析

不急于修改，先分析根因：

```
问题分析流程：
1. 观察现象（测试失败、数据错误）
2. 定位代码（使用grep、read）
3. 分析根因（设计缺陷、实现错误）
4. 设计方案（系统性修复）
5. 小步实现（每个TODO单独提交）
```

### 3. 设计方案输出

发现问题后，给出完整设计方案：

```
设计方案格式：
1. 问题分析：详细描述问题和根因
2. 设计方案：给出修复方案，包括代码示例
3. 实现步骤：分步骤说明如何实现
4. 测试方案：说明如何验证修复
5. 影响评估：说明改动的影响范围，如果几个需求之间有耦合，需分析严重程度、依赖程度和优先级，按优先级逐个处理
```

---

## 文档规范

### 1. 代码注释

复杂逻辑必须有注释说明：

```python
def _infer_view_shardable(self, new_shape):
    """推导reshape后的shardable约束.
    
    算法：
    1. 计算原shape和新shape的累积乘积
    2. 对于每个原shardable维度，找到在新shape中对应的维度
    3. 映射规则：原维度累积范围完全包含在新维度累积范围内
    
    参考：PyTorch view语义
    """
    pass
```

### 2. 设计文档

重要设计需有文档说明：

```
文档内容：
- 设计动机（为什么这样设计）
- 设计方案（具体实现）
- 使用示例（如何使用）
- 注意事项（边界情况）
```

---

## 检查清单

### 设计阶段
- [ ] 分析需求合理性
- [ ] 参考外部实践
- [ ] 审视现有组件兼容性
- [ ] 设计测试用例
- [ ] 输出设计方案

### 实现阶段
- [ ] 模块职责清晰
- [ ] 层次分明
- [ ] 避免循环依赖
- [ ] 公共逻辑提取基类
- [ ] 类型安全

### 测试阶段
- [ ] 单元测试覆盖
- [ ] 边界情况测试
- [ ] 集成测试
- [ ] 存量测试通过

---

## 测试执行规范

### 分布式测试

默认使用 pytest-xdist 进行分布式测试，自动检测 CPU 核数：

```bash
# 自动检测 CPU 核数（推荐）
python -m pytest tests/

# 固定 worker 数量
python -m pytest tests/ -n 4

# 禁用分布式（单进程调试）
python -m pytest tests/ -n 0
```

配置已在 `pyproject.toml` 中设置：
- `-n auto`：自动使用所有可用 CPU 核心
- `-v`：详细输出
- `--tb=short`：简洁的错误追溯

### 提交阶段
- [ ] 小步提交（每个特性一个commit）
- [ ] commit message清晰
- [ ] 推送到远程

---

## Modeling 分层结构规范

### 目录结构

modeling 必须分层组织：

```
llm_perf/modeling/
├── base/                    # 基础通用层（可复用、无模型特定逻辑）
│   ├── layers.py            # Transformer基础层
│   ├── vision.py            # 视觉层
│   ├── dit_layers.py        # DiT基础层
│   ├── dit_blocks.py        # DiT块
│   ├── vae_3d.py            # 3D VAE
│
├── models/                  # 模型层（具体模型定义）
│   ├── llama.py
│   ├── deepseek.py
│   ├── hunyuan_video.py
│   ├── wan_video.py
│   ...
│
├── module.py                # 基类（不动）
├── tensor.py                # 张量定义（不动）
```

### 强制要求

| 要求 | 说明 |
|------|------|
| 基础层放 base/ | 可被多个模型使用的层必须放 base/ |
| 模型层放 models/ | 具体模型定义放 models/ |
| 禁止单独模型目录 | ❌ 不允许 `hunyuan_video/layers.py` 这样的单独目录 |
| 禁止重复定义 | ❌ ModulateDiT、PatchEmbed3D 应是通用层 |
| 禁止模型特定逻辑 | ❌ base/ 目录禁止包含任何模型特定的命名或逻辑 |

### 检查清单

设计阶段必须检查：
- [ ] 新增层是否可被多个模型使用（通用性）
- [ ] 是否已有类似的基础层（复用）
- [ ] 是否放入正确的分层位置（base vs models）
- [ ] 是否避免单独模型目录

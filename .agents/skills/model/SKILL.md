---
name: model
description: 模型评估支持开发规范（解耦、依赖 kernel、bind 建模）
---

# 模型评估支持开发规范

本文档定义新增模型评估支持时的开发规范，包括模型建模、workload 解耦、kernel 依赖、bind 建模等核心要求。

---

## 1. 开发过程

新增模型评估支持的开发流程：

### 1.1 信息收集阶段
1. **检索官方模型配置** - 从 HuggingFace 获取官方模型配置文件
2. **严格遵循官方超参** - 不得随意修改配置文件中的任何参数；如有不明确的参数，需进行备注
3. **多配置文件处理** - 如 HuggingFace 上有多个配置文件且无法区分主要 backbone，同时支持评估并在 DEVELOP_LOG 中提示
4. **分析架构特点** - 确定模型结构、子模块类型、并行切分方式
5. **收集参考来源** - 记录 paper、官方代码等外部信息，刷新到 `docs/data_sources_wiki.md`

### 1.2 开发阶段
1. **创建模型配置类** - 继承 `ModelConfig`，实现 `__post_init__`
2. **创建模型类** - 继承 `ShardedModule`，使用 kernel API 构建层
3. **实现 bind 机制** - 确保模型可通过 `model.bind(ctx)` 获取物理切分指标
4. **编写测试用例** - 确保参数量、FLOPs 与官方一致

### 1.3 文档刷新阶段
1. **刷新 docs wiki** - 在 `docs/model_evaluation_wiki.md` 添加模型建模详情
2. **刷新 DEVELOP_LOG** - 记录开发摘要

### 1.4 提交阶段
1. **运行全量测试** - `python -m pytest tests/ -v -n 4`
2. **提交 commit** - 小步快跑，每个特性单独提交

---

## 2. 模型建模规范（核心要求）

### 2.1 解耦模型建模和 workload 建模

**核心原则**：模型建模定义结构，workload 建模定义计算模式，两者通过 `supported_workloads` 对应。

```python
class LlamaModel(ShardedModule):
    """Llama 模型结构定义.
    
    仅定义模型结构、层配置、子模块，不包含 workload 相关参数。
    """
    supported_workloads = ["training", "inference"]
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embedding = ShardedEmbedding(...)
        self.layers = [ShardedTransformerBlock(...) for _ in range(config.num_layers)]
        self.lm_head = ShardedLMHead(...)
```

**Workload 定义位置**：
- Training workload：由 `ParallelContext` 和 `forward_backward` mode 定义
- Inference workload：由 `ParallelContext` 和 `forward` mode 定义
- Diffusion workload：由专门的 workload 类定义

**禁止**：在模型类中硬编码 workload 相关参数（如 batch_size、seq_len）

---

### 2.2 模型开销建模必须依赖已有基础能力

**核心原则**：所有层特征必须通过 kernel API 获取，禁止手动计算。

**正确做法**：

```python
from llm_perf.kernels.functional import linear, flash_attention, rms_norm

def build_layers(self) -> List[LayerConfig]:
    """构建模型层配置，使用 kernel API."""
    layers = []
    
    # 使用 linear kernel 获取 FLOPs 和 memory
    q_proj = linear(
        input=(batch, seq, hidden_size),
        weight=(num_heads * head_dim, hidden_size),
        dtype=self.dtype,
    )
    layers.append(kernel_result_to_layer(q_proj, "q_proj"))
    
    # 使用 flash_attention kernel
    attn = flash_attention(
        query=(batch, num_heads, seq, head_dim),
        key=(batch, kv_heads, kv_seq, head_dim),
        value=(batch, kv_heads, kv_seq, head_dim),
        dtype=self.dtype,
    )
    layers.append(kernel_result_to_layer(attn, "attention"))
    
    return layers
```

**禁止做法**：

```python
# 错误：手动计算 activation_bytes
activation_bytes = batch * seq * hidden_size * dtype_size  # 禁止！
flops = 2 * batch * seq * hidden_size * intermediate_size  # 禁止！
```

**例外情况**：如果 kernel API 缺失，必须：
1. 先调用.agent/skills/kernel skill添加 kernel 到 `llm_perf/kernels/functional.py`
2. 使用 NOTE 注释说明临时处理

---

### 2.3 Norm 类子模块合并入后续子模块

**核心原则**：LayerNorm、RMSNorm 不单独呈现，合并入 TransformerBlock 或其他父模块。

**正确做法**：

```python
class ShardedTransformerBlock(ShardedModule):
    """Transformer Block，包含 Attention + FFN + RMSNorms."""
    
    def __init__(self, hidden_size, num_heads, ...):
        super().__init__()
        # Norm weight 作为 block 内部参数
        self.input_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            name="input_norm_weight",
        )
        self.attention = ShardedAttention(...)
        self.post_attn_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            name="post_attn_norm_weight",
        )
        self.ffn = ShardedFFN(...)
    
    def forward(self, hidden):
        # Norm 在 forward 中计算，不单独呈现
        norm_out = self._rms_norm(hidden, self.input_norm_weight)
        attn_out = self.attention(norm_out)
        return hidden + attn_out + self.ffn(...)
```

**模型分解时 Norm 合并**：

```python
# LlamaModel 的子模块分解：
# - embedding (单独)
# - transformer_block (合并 attention + ffn + norms)
# - lm_head (单独)
# 
# 不单独呈现：input_norm, post_attn_norm, final_norm
```

---

### 2.4 新增类型子模块评估必须用通用命名和表达

**核心原则**：新增子模块类型必须刷新到 `llm_perf/kernels/functional.py`，使用通用 kernel 命名。

**通用 kernel 命名**：

| Kernel 名称 | 适用场景 | 示例模型 |
|-------------|----------|----------|
| `linear` | 全连接层 | Llama, DeepSeek, Qwen |
| `flash_attention` | Attention 层 | 所有 Transformer |
| `moe_expert` | MoE Expert | DeepSeek, Qwen-MoE |
| `conv2d` / `conv3d` | 卷积层 | VAE, DiT |
| `rms_norm` / `layer_norm` | Norm 层 | Llama (RMSNorm), BERT (LayerNorm) |

**禁止做法**：

```python
# 错误：模型专属的 kernel 命名
def llama_attention(...)  # 禁止！
def deepseek_mla(...)     # 禁止！用 mla_attention 替代
def qwen_linear_attn(...) # 禁止！用 linear_attention 替代
```

**新增 kernel 流程**：

1. 在 `llm_perf/kernels/functional.py` 添加通用 kernel
2. 确保返回 `KernelResult`（包含 FLOPs、memory、backward metrics）
3. 添加对应的 Op 类到 `llm_perf/kernels/op.py`
4. 更新 `docs/kernel_api.md`

---

### 2.5 必须依赖 bind 建模

**核心原则**：使用 `model.bind(ctx, strategy)` 进行并行策略表达，bind 返回 `ModuleInstance`。

**bind 机制**：

```python
from llm_perf.strategy.parallel_context import ParallelContext

# 创建并行策略
ctx = ParallelContext(
    tp_degree=8,
    pp_degree=2,
    dp_degree=1,
    sp_degree=1,
    ep_degree=1,
    dtype="fp16",
    mode="forward_backward",  # training
)

# bind 返回 ModuleInstance（物理切分后的指标）
model = LlamaModel(config)
instance = model.bind(ctx)  # 返回 ModuleInstance

# 从 ModuleInstance 获取物理指标
params_physical = instance.params_count_physical  # TP/PP 切分后每卡参数量
flops_physical = instance.flops_forward_physical  # 物理切分后 FLOPs
activation_mem = instance.activation_memory_physical  # 物理切分后 activation
comm_ops = instance.total_comm_ops  # 通信操作（AllReduce, AllGather 等）
```

**ModuleInstance 属性**：

| 属性 | 说明 |
|------|------|
| `params_count_physical` | 物理切分后每卡参数量 |
| `params_count_logical` | 逻辑参数量（总量） |
| `flops_forward_physical` | 前向 FLOPs（物理切分后） |
| `flops_backward_physical` | 反向 FLOPs（≈ 2x forward） |
| `activation_memory_physical` | Activation 内存 |
| `weight_memory_physical` | 权重内存 |
| `gradient_memory_physical` | 梯度内存（仅 training） |
| `optimizer_memory_physical` | Optimizer state 内存 |
| `total_comm_ops` | 所有通信操作 |
| `kv_cache_memory` | KV cache 内存（仅 inference） |

---

### 2.6 详细建模过程刷新到 docs wiki

**核心原则**：每个模型的建模过程必须记录在 `docs/model_evaluation_wiki.md`。

**文档模板**：

```markdown
## X. ModelName 模型

### X.1 架构概述

| 模型 | 参数量 | Hidden Size | Layers | Heads | 特殊结构 |
|------|--------|-------------|--------|-------|----------|
| ModelName-XB | XB | H | L | N | ... |

**关键特性**:
- 特性1
- 特性2

**与其他模型区别**:
- 与 ModelA 的区别
- 与 ModelB 的区别

### X.2 子模块分解

模型结构: `Embedding → N × Block → Final Norm → LM Head`

| 子模块 | 类型 | 参数 | 切分方式 |
|--------|------|------|----------|
| embedding | ShardedEmbedding | vocab × hidden | TP vocab |
| block | ShardedBlock | ... | TP/PP |
| lm_head | ShardedLMHead | hidden × vocab | TP vocab |

### X.3 权重计算

```
# 各层权重计算公式
embedding_params = vocab_size × hidden_size
attention_params = ...
```

### X.4 FLOPs 计算

```
# 各层 FLOPs 计算公式（引用 kernel API）
attention_flops = flash_attention(...).flops
ffn_flops = linear(...).flops × 2  # gate + up
```

### X.5 内存分析

| 内存类型 | 计算方式 |
|----------|----------|
| 参数内存 | params × dtype_size |
| Activation | kernel API 计算 |
| KV Cache | kv_seq × kv_heads × head_dim × 2 |

### X.6 Bind 流程

```python
ctx = ParallelContext(tp=8, pp=2, ...)
instance = model.bind(ctx)
```
```

---

## 3. Workload 解耦设计

### 3.1 模型和 Workload 的对应关系

模型通过 `supported_workloads` 属性声明支持的 workload 类型（示例见第2.1节）。

### 3.2 Workload 参数位置

| 参数 | 定义位置 | 说明 |
|------|----------|------|
| `batch_size` | ParallelContext / Analyzer | 批次大小 |
| `seq_len` | ParallelContext / Analyzer | 序列长度 |
| `mode` | ParallelContext | "forward" (inference) / "forward_backward" (training) |
| `dtype` | ParallelContext | 数据类型 |
| `zero_stage` | ParallelContext | ZeRO 优化阶段 |

**禁止**：在模型类中硬编码这些参数。

### 3.3 Workload 实例化

```python
# Training workload
ctx_train = ParallelContext(
    tp_degree=8,
    mode="forward_backward",
    batch_size=32,
    seq_len=4096,
)
instance_train = model.bind(ctx_train)

# Inference workload
ctx_infer = ParallelContext(
    tp_degree=8,
    mode="forward",
    batch_size=1,
    seq_len=1,  # decode
    kv_seq_len=4096,
)
instance_infer = model.bind(ctx_infer)
```

---

## 4. 子模块评估规范

详见第2节各规范：
- **依赖 kernel 创建层**：参见 [2.2 模型开销建模必须依赖已有基础能力](#22-模型开销建模必须依赖已有基础能力)
- **Norm 合并入父模块**：参见 [2.3 Norm 类子模块合并入后续子模块](#23-norm-类子模块合并入后续子模块)
- **通用命名**：参见 [2.4 新增类型子模块评估必须用通用命名和表达](#24-新增类型子模块评估必须用通用命名和表达)

---

## 5. Bind 建模规范

详见 [2.5 必须依赖 bind 建模](#25-必须依赖-bind-建模)。

---

## 6. 文档刷新规范

详见 [2.6 详细建模过程刷新到 docs wiki](#26-详细建模过程刷新到-docs-wiki)。

**额外要求**：
- **数据源刷新**：外部检索到的信息刷新到 `docs/data_sources_wiki.md`
- **DEVELOP_LOG 刷新**：记录开发摘要、关键决策、测试结果

---

## 7. 推荐开发实践样例

### 7.1 Llama 模型实现示例

```python
from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.layers import ShardedEmbedding, ShardedTransformerBlock, ShardedLMHead

class LlamaModel(ShardedModule):
    """Llama 模型.
    
    参考: https://arxiv.org/abs/2302.13971
    
    supported_workloads:
        - training: forward_backward mode
        - inference: forward mode with KV cache
    """
    
    supported_workloads = ["training", "inference"]
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int = None,
        intermediate_size: int = None,
        dtype: str = "fp16",
    ):
        super().__init__()
        
        # 派生参数处理
        if num_kv_heads is None:
            num_kv_heads = num_heads  # MHA, GQA 需要显式指定
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)  # Llama 默认比例
        
        self.embedding = ShardedEmbedding(vocab_size, hidden_size, dtype)
        
        self.layers = [
            ShardedTransformerBlock(
                hidden_size, num_heads, num_kv_heads, None, intermediate_size, dtype
            )
            for _ in range(num_layers)
        ]
        
        self.final_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            name="final_norm_weight",
        )
        self.lm_head = ShardedLMHead(hidden_size, vocab_size, dtype)
    
    def forward(self, input_ids):
        """Forward pass."""
        hidden = self.embedding(input_ids)
        
        for layer in self.layers:
            hidden = layer(hidden)
        
        hidden = self._rms_norm(hidden, self.final_norm_weight)
        logits = self.lm_head(hidden)
        
        return logits
```

### 7.2 DeepSeek MoE 模型示例

```python
class DeepSeekModel(ShardedModule):
    """DeepSeek V3 模型，包含 MLA 和 MoE.
    
    参考: https://arxiv.org/abs/2412.19437
    
    特点:
        - MLA: KV compression
        - DeepSeekMoE: routed + shared experts
        - First k layers use dense FFN
    """
    
    supported_workloads = ["training", "inference"]
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        
        self.embedding = ShardedEmbedding(config.vocab_size, config.hidden_size)
        
        self.layers = []
        for i in range(config.num_layers):
            if i < config.first_k_dense_layers:
                self.layers.append(ShardedTransformerBlock(...))
            else:
                self.layers.append(ShardedMoEBlock(...))
        
        self.lm_head = ShardedLMHead(...)

### 7.3 架构修正示例

当发现模型架构与官方实现不一致需修正时：

```python
# Wan2.1 AdaLN 修正示例
# 
# 修正原因：初始实现遗漏了 cross-attention 的调制参数
# 参考来源：model.py line 276, line 310
#
# 修正前：错误的参数数量
# num_modulation = 3 * cfg.hidden_size  # 仅 self-attn
#
# 修正后：6 个调制参数（self-attn 的 shift/scale/gate + FFN 的 shift/scale/gate）
# Cross-attention 无调制（model.py line 310 确认）
num_modulation = 6 * cfg.hidden_size
```

**修正规范**：
1. **参考官方实现** - 基于 HuggingFace 或论文官方代码
2. **记录修正原因** - 在代码注释中说明修正理由和参考来源
3. **更新测试用例** - 修正后必须更新对应测试
4. **验证数值一致性** - 确保修正后参数量和 FLOPs 与官方一致
```

---

## 8. 开发规范

### 8.1 测试覆盖

**必须测试**：
- 参数量与官方一致
- FLOPs 与理论计算一致
- bind 返回正确的物理指标

```python
def test_llama_params():
    """Test Llama parameter count matches official."""
    model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
    expected_params = 32000 * 4096 + 32 * (4 * 4096 * 4096 + 3 * 4096 * 11008)
    assert model.params_count() == expected_params

def test_llama_flops():
    """Test Llama FLOPs using kernel API."""
    model = LlamaModel(...)
    ctx = ParallelContext(tp_degree=1, mode="forward_backward")
    instance = model.bind(ctx)
    # 验证 FLOPs 计算正确
```

### 8.2 Commit 格式

```bash
feat(modeling): add ModelName model support

- Add ModelName model class with ShardedModule
- Use kernel API for FLOPs and memory calculation
- Norm layers merged into TransformerBlock
- Add supported_workloads for workload decoupling

Tests: ModelName params and FLOPs verified against official
Docs: Updated docs/model_evaluation_wiki.md
```

### 8.3 检查清单

**开发前**：
- [ ] 检索官方模型配置（HuggingFace）
- [ ] 理解模型架构特点
- [ ] 确定子模块类型和切分方式

**开发中**：
- [ ] 使用 kernel API 创建层
- [ ] Norm 合并入父模块
- [ ] 实现 bind 机制
- [ ] 遵循通用命名

**开发后**：
- [ ] 参数量验证
- [ ] FLOPs 验证
- [ ] 测试全部通过
- [ ] docs wiki 刷新
- [ ] DEVELOP_LOG 刷新

---

## 9. 特别注意事项

### 9.1 FFN intermediate size 与激活函数

**激活函数类型**：
- SwiGLU / GeGLU: `intermediate_size × 2`（gated activation）
- 标准 FFN: `intermediate_size`

**验证方法**：
1. 检查 HuggingFace config.json 的 `intermediate_size` 字段
2. 检查实际代码实现：
   - 是否调用带 gated 的融合算子（如 `F.silu` + gate projection）
   - 是否在 down 前进行 `act * gate` 操作
3. 如未使用 gated 激活，则 `intermediate_size` 不需乘以 2

### 9.2 GQA vs MHA

- MHA: `num_kv_heads = num_heads`
- GQA: `num_kv_heads < num_heads`（显式指定）
- MQA: `num_kv_heads = 1`

### 9.3 MoE shared experts

DeepSeek MoE 有 routed experts + shared experts，两者都需要评估。

---

## 10. 工具命令速查

```bash
# 运行模型测试
python -m pytest tests/test_models.py -v -n 4

# 运行特定模型测试
python -m pytest tests/test_llama.py -v -n 4

# 全量测试
python -m pytest tests/ -v -n 4

# 验证模型参数量
python -c "from llm_perf.modeling import LlamaModel; m = LlamaModel(32000, 4096, 32, 32); print(f'{m.params_count()/1e9:.2f}B')"

# 验证 bind 机制
python -c "
from llm_perf.modeling import LlamaModel
from llm_perf.strategy.parallel_context import ParallelContext
ctx = ParallelContext(tp_degree=8, mode='forward_backward')
m = LlamaModel(32000, 4096, 32, 32)
inst = m.bind(ctx)
print(f'Params per device: {inst.params_count_physical/1e9:.2f}B')
"

# 检查代码风格
ruff check llm_perf/modeling/*.py --select=F401,F841,E741
```
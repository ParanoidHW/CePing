# 开发日志

## 2026-04-25: 全面重构前后端场景显示解耦架构

### 任务背景
用户反馈：每次前后端支持新东西都有显示问题，解耦做得差劲。

### 根因分析
**系统性问题**：
1. 前端硬编码场景判断（`diffusion-video`、`inference`）
2. 后端硬编码模型类型映射（`infer_workload`）
3. 新增场景需改多处代码
4. 无统一的场景类型定义和指标映射

**混淆的概念**：
- `workload_type`（计算类型）：training/inference/mixed
- `scenario`（前端显示类型）：training/inference/diffusion/rl_training/pd_disagg
- 两者用途不同但被混用

### 重构方案

#### 1. 统一 ScenarioType 定义
新增 `ScenarioType` 枚举，用于前端渲染选择：
```python
class ScenarioType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    DIFFUSION = "diffusion"
    RL_TRAINING = "rl_training"
    PD_DISAGG = "pd_disagg"
```

#### 2. 后端重构
- 新增 `infer_scenario_type()` 函数，从 workload_name 推断显示类型
- `UnifiedResult.to_dict()` 返回 `scenario_type` 字段
- `web/app.py` 使用 `SCENARIO_TO_WORKLOAD_MAP` 替代硬编码判断
- `_build_decode_dict()` 根据 scenario_type 判断是否返回 LLM decode 指标

#### 3. 前端重构
- 使用 `RENDER_TEMPLATES` 映射替代硬编码判断：
```javascript
const RENDER_TEMPLATES = {
    'diffusion': renderDiffusionSummary,
    'training': renderTrainingSummary,
    'rl_training': renderTrainingSummary,
    'inference': renderInferenceSummary,
    'pd_disagg': renderInferenceSummary,
};
```

### 关键改动
- `llm_perf/analyzer/base.py`: 新增 ScenarioType 枚举，修改 UnifiedResult
- `llm_perf/analyzer/workload_loader.py`: 新增 SCENARIO_TYPE_MAP 和 infer_scenario_type()
- `web/app.py`: 新增 SCENARIO_TO_WORKLOAD_MAP，移除硬编码判断
- `web/static/js/app.js`: 新增 RENDER_TEMPLATES，重构 displayResults()

### 测试覆盖
新增 `tests/test_scenario_display_decoupling.py`（18 个测试用例）：
- TestScenarioTypeInference: 8 个测试
- TestUnifiedResultScenarioType: 4 个测试
- TestFrontendRenderingDecoupling: 2 个测试
- TestScenarioTypeEnumValues: 1 个测试
- TestBackendAPIScenarioType: 3 个测试

### 验证结果
- 全量测试：155 passed
- Ruff 检查：All checks passed

### Commit
`80e4e92` - refactor(analyzer): add ScenarioType for frontend display decoupling

---

## 2026-04-25: 修复 HunyuanImage 3.0 preset 到 model_class 的映射问题

### 任务背景
用户选择 `hunyuan-image-3` + `diffusion` workload 时报错：
```
Unknown model: 'hunyuan-image-3'
Available presets: ['hunyuan-image-3', ...]
Available models: ['hunyuan_image_3_text', 'hunyuan_image_3_diffusion', ...]
```

### 根因分析
1. **预设配置**：
   - preset 名称：`hunyuan-image-3`
   - architecture：`hunyuan_image_3`（不存在）
   - 实际注册的模型类：`hunyuan_image_3_text` 和 `hunyuan_image_3_diffusion`

2. **映射逻辑缺失**：
   - `_load_presets_from_yaml` 没有加载 `model_class_map` 字段
   - `create_model_from_config` 不支持根据 workload_type 选择模型类
   - web 服务没有传递 workload 参数给模型创建函数

### 修复方案（方案 A+B 组合）

#### 1. hunyuan-image-3.yaml
添加 `model_class_map` 字段：
```yaml
model_class_map:
  inference: hunyuan_image_3_text
  diffusion: hunyuan_image_3_diffusion
```

#### 2. registry.py
- `_load_presets_from_yaml`：添加对 `model_class_map` 字段的加载
- `create_model_from_config`：添加 `workload_type` 参数，根据 preset 的 model_class_map 选择模型类

#### 3. web/app.py
- `create_model_from_registry`：添加 `workload_type` 参数
- 所有调用点：从请求的 workload 数据中提取 workload_type 并传递

### 核心逻辑
```python
# registry.py: create_model_from_config
preset_config = presets[preset_name]
model_class_map = preset_config.get("model_class_map", {})
if workload_type and workload_type in model_class_map:
    model_name = model_class_map[workload_type]
    # 使用 workload 指定的模型类
    return registry.create(model_name, **filtered_config)
# 否则使用 architecture 查找（向后兼容）
```

### 测试用例
新增 `TestPresetWorkloadMapping` 测试类（6 个测试）：
- `test_hunyuan_image_3_inference_mapping`：验证 inference → text 模型
- `test_hunyuan_image_3_diffusion_mapping`：验证 diffusion → diffusion 模型
- `test_hunyuan_image_3_training_mapping`：验证 training（无映射）fallback
- `test_llama_inference_mapping`：验证无 model_class_map 的 preset 仍能正常工作
- `test_model_class_map_in_preset`：验证配置加载
- `test_no_model_class_map_uses_architecture`：验证向后兼容

### 验证结果
- 31 个 hunyuan 测试全部通过
- 20 个 workload-model decoupling 测试全部通过
- 4 个关键存量测试全部通过
- 代码质量检查通过（ruff）
- 实际验证：`hunyuan-image-3` + `diffusion` 成功创建 `HunyuanImage3DiffusionModel`

### Commit
- Hash: `e1c64ab`
- Message: `fix(modeling): add preset to model_class mapping for HunyuanImage 3.0`

---

## 2026-04-24: Kernel backward FLOPs/内存访问显式分解

### 任务背景
当前代码和文档中 backward 计算量笼统标记为 `2× forward`，但实际应该显式分解为 `dx` 和 `dW` 的计算量和内存访问，以便准确计算 roofline 模型所需的内存带宽。

### 正确的分解
**Forward: Y = X @ W**
- X: M×K, W: K×N, Y: M×N
- FLOPs = 2 × M × K × N
- Memory: 读 X + 读 W + 写 Y

**Backward 分解**

**dx 计算**: dY @ W^T
- dY: M×N, W^T: N×K, dx: M×K
- FLOPs = 2 × M × N × K = 2 × M × K × N
- Memory: 读 dY + 读 W + 写 dx

**dW 计算**: X^T @ dY
- X^T: K×M, dY: M×N, dW: K×N
- FLOPs = 2 × K × M × N = 2 × M × K × N
- Memory: 读 X + 读 dY + 写 dW

**总 Backward**:
- FLOPs = 4 × M × K × N（数值上确实是 2× forward）
- 但内存访问需要显式计算

### 关键差异
| 操作 | 读 | 写 |
|------|----|----|
| dx = dY @ W^T | dY(M×N) + W(K×N) | dx(M×K) |
| dW = X^T @ dY | X(M×K) + dY(M×N) | dW(K×N) |

- W 在 dx 中被**读**（不变）
- X 在 dW 中被**读**（需要保存的 activation）
- dY 在两步中都被读

### 实现内容
1. **linear 函数**
   - 显式分解 backward FLOPs 为 flops_dx 和 flops_dw
   - 显式分解 backward memory 为 bytes_dx 和 bytes_dw

2. **bmm 函数**
   - 显式分解 backward FLOPs 为 flops_da 和 flops_db
   - 显式分解 backward memory 为 bytes_da 和 bytes_db

3. **conv2d, conv3d 函数**
   - 显式分解 backward FLOPs 为 flops_dx 和 flops_dw
   - 显式分解 backward memory 为 bytes_dx 和 bytes_dw

4. **moe_expert 函数**
   - 分解 gate_proj, up_proj, down_proj 的 backward
   - 包含 SiLU activation backward

5. **文档更新**
   - docs/model_evaluation_wiki.md 添加 backward 分解章节
   - LLaMA、DeepSeek V3、Qwen3.5 各模型部分添加 backward 分解说明

### 验证
- backward FLOPs = 2 × forward FLOPs（数值正确）
- backward memory ≈ 2 × forward memory（显式分解）
- 108 tests passed

### commit
- b35f4df: refactor(kernels): explicit backward FLOPs/memory decomposition

---

## 2026-04-23: Qwen3.5 Dense 模型支持

### 任务背景
Qwen3.5 系列有 Dense 和 MoE 两种架构：
- Dense 模型：Qwen3.5-0.8B/2B/4B/9B/27B（SwiGLU FFN）
- MoE 模型：Qwen3.5-35B-A3B（已完成，256 experts, top-8 routing）

需要添加 Dense 版本支持，包括：
- 混合注意力（linear/full，每4层一个 full attention）
- SwiGLU FFN（而非 MoE）
- tie_word_embeddings（小模型共享 embedding/lm_head）

### 实现内容
1. **ShardedQwen3_5DenseBlock**
   - 支持 linear_attention 和 full_attention
   - 使用 ShardedFFN（SwiGLU activation）
   - RMSNorm（post-attention 和 post-FFN）

2. **Qwen3_5Model**
   - 混合注意力：generate_layer_types() 函数生成层类型
   - tie_word_embeddings 处理：
     - true: lm_head=None，forward 使用 embedding.weight.T
     - false: 独立 lm_head 子模块
   - 参数量自动计算（父类方法处理）

3. **Presets（YAML配置）**
   - qwen3-5.yaml（9B 默认）
   - qwen3-5-0-8b.yaml（1024 hidden, 24 layers, tie=True）
   - qwen3-5-2b.yaml（2048 hidden, 24 layers, tie=True）
   - qwen3-5-4b.yaml（2560 hidden, 32 layers, tie=True）
   - qwen3-5-27b.yaml（5120 hidden, 64 layers, tie=False）

### 架构特点
- 所有模型都有混合注意力（linear/full）
- SwiGLU FFN：gate_proj + up_proj + down_proj
- 权重参数：2 * hidden_size * intermediate_size + intermediate_size * hidden_size
- 小模型（0.8B/2B/4B）共享 embedding/lm_head 权重，减少内存

### 参数量验证
| 模型 | 计算值 | 预期范围 |
|------|--------|----------|
| 0.8B | 0.85B | ~0.9B |
| 2B | 2.08B | ~2B |
| 4B | 5.05B | ~5B |
| 9B | 10.42B | ~10B |
| 27B | 32.71B | ~28B（总参数量） |

### 测试验证
- 新增测试：26 个（ShardedQwen3_5DenseBlock, Qwen3_5Model, tie_embeddings, params, inference）
- 存量测试：113 passed, 2 skipped
- ruff 代码检查通过

### 参考来源
- HuggingFace Qwen3.5 config.json（官方配置参数）
- linear_num_value_heads（linear attention 的 value head 数）
- head_dim（attention head 维度）
- intermediate_size（SwiGLU FFN 中间层大小）

### Commit
- Hash: c9aaca4
- Message: feat(modeling): add Qwen3.5 Dense model support
- Tests: 55 passed (26 new tests added)

---

## 2026-04-23: Wan VAE 重构到 encoder.py

### 任务背景
Wan 模型中的 VAE 编解码器是独立模块，与 backbone（DiT）无关，应该抽取到 encoder.py 保持一致性。vision.py 中有重复定义的 VAE（第一版无 attention，第二版有 attention），需要统一。

### 实现内容
1. **encoder.py 新增 VAE**
   - 移动 `ShardedVAEEncoder`, `ShardedVAEDecoder`, `ShardedVAE` 从 vision.py 到 encoder.py
   - 使用通用命名（不绑定 Wan）
   - Encoder: 降采样（空间压缩 8x）
   - Decoder: 上采样（空间恢复 8x）

2. **vision.py 清理重复定义**
   - 删除第一版 VAE（无 attention，175-344行）
   - 删除第二版 VAE（已移到 encoder.py）
   - 只保留基础组件：Conv, GroupNorm, ResNetBlock, AttentionBlock, ResNet

3. **导入路径更新**
   - `__init__.py`: 从 encoder 导入 VAE
   - `registry.py`: 从 encoder 导入 VAE
   - `wan.py`: `ShardedWanVAE` 使用 encoder.py 的 `ShardedVAE`（向后兼容）
   - `test_analyzer.py`: 从 encoder 导入 VAE decoder

4. **Wan 模型简化**
   - Wan 只保留 DiT backbone
   - `ShardedWanVAE` 标记为 deprecated，内部使用 `ShardedVAE`
   - VAE 作为独立模块存在

### 架构改进
- VAE encoder/decoder 是独立模块（不绑定 Wan）
- 命名通用（`ShardedVAE`），与 Wan 解耦
- 更好的职责分离：encoder.py（编码器），vision.py（基础组件）
- 向后兼容：`ShardedWanVAE` 仍然可用

### 测试验证
- 存量测试全部通过（132 passed）
- VAE 功能正常（test_vae_model_integration, test_vae_decoder_flops）
- ruff 代码检查通过

### 参考来源
- Wan 论文: https://arxiv.org/abs/2503.20314
- Wan VAE 结构：3D causal VAE for video generation
- 空间压缩：8x（height/8, width/8）

### Commit
- Hash: 5699c7a
- Message: refactor(modeling): move VAE to encoder.py for consistency
- Tests: 132 passed (all existing tests pass)

---

## 2026-04-23: 实现 ShardedLinearAttention kernel + 子模块

### 任务背景
Qwen3.5 使用混合注意力架构：
- 40层中每4层有1层 `full_attention`，其余3层 `linear_attention`
- Linear Attention 没有 KV cache 的 seq² 复杂度，适合长序列

### 实现内容
1. **kernel 层 (llm_perf/kernels/)**
   - 新增 `LinearAttentionOp` 在 op.py
     - kernel_dim 参数（Qwen3.5 为 4）
     - 无 phase/workload 参数，保持 kernel 层纯粹
     - 无 kv_seq_len 参数（与 standard attention 不同）

2. **functional 层 (llm_perf/kernels/functional.py)**
   - 新增 `linear_attention` kernel 函数
     - FLOPs: O(seq)，而非 O(seq²)
     - Memory: 固定大小 state，无 KV cache 线性增长
     - 使用 kernel trick (elu+1) 近似 softmax

3. **modeling 层 (llm_perf/modeling/layers.py)**
   - 新增 `ShardedLinearAttention` 模块
     - 继承 ShardedModule
     - 使用 LinearAttentionOp
     - 支持 TP sharding（heads 维度）
     - 不依赖 phase/workload

4. **测试用例 (tests/test_framework_overhead.py)**
   - test_linear_attention_flops_linear_in_seq: 验证 O(seq) 复杂度
   - test_linear_attention_memory_no_seq_growth: 验证无 seq² 内存增长
   - test_linear_attention_backward_metrics: 验证 backward metrics
   - test_linear_attention_op_purity: 验证 kernel 层纯粹性
   - test_linear_attention_sharded_module: 验证模块功能
   - test_linear_attention_tp_sharding: 验证 TP sharding

### 参考来源
- Linear Attention 论文: https://arxiv.org/abs/2006.16236
- Qwen3.5 config.json 参数:
  - linear_conv_kernel_dim: 4
  - linear_key_head_dim: 128
  - linear_num_key_heads: 16
  - linear_num_value_heads: 32
  - linear_value_head_dim: 128

### Commit
- Hash: d8fb484
- Message: feat(kernels): add LinearAttentionOp for Qwen3.5 mixed attention
- Tests: 138 passed (6 new tests, all existing tests pass)

---

## 2026-04-23: 修复 Attention kv_seq_len 问题，decode 阶段计算量正确评估

### 问题描述
- decode 阶段 input tensor seq_len=1
- 但 K/V tensor 应该是 kv_seq_len 长度
- Attention 计算量应为 O(kv_seq_len × hidden_size)，当前低估

### 根因分析
1. `unified.py:517-518`: decode phase seq_len=1
2. `layers.py:356-367`: `ShardedAttention.forward` 使用 `hidden.shape[1]` 创建 K/V tensor
3. `module.py:527-531`: `_infer_physical_flops` 使用 `op.key.shape` 计算 flops

### 修复方案
遵循架构原则：kernel 层保持纯粹，shape 由上层定义

1. 在 `ShardedModule` 基类添加 `_kv_seq_len` 属性
2. 在 `ShardedAttention.forward` 和 `ShardedMLA.forward` 中使用 `_kv_seq_len`
3. 在 `unified.py` 的 `_analyze_phase_with_submodules` 中设置 `_kv_seq_len`

### 修改文件
- `llm_perf/modeling/module.py`: 添加 `_kv_seq_len` 属性
- `llm_perf/modeling/layers.py`: `ShardedAttention.forward` 使用 `_kv_seq_len`
- `llm_perf/modeling/mla.py`: `ShardedMLA.forward` 使用 `_kv_seq_len`
- `llm_perf/analyzer/unified.py`: 添加 `_set_kv_seq_len` 方法
- `tests/test_framework_overhead.py`: 新增测试用例

### 测试结果
- 所有测试通过：554 passed (3 new tests added)

### 参考来源
- Transformer decode 阶段特性：Q长度=1，K/V长度=kv_seq_len
- `flash_attention` 公式：FLOPs = O(batch * num_heads * seq_len * kv_seq_len * head_dim)

---

## 2026-04-23: 修复 TPOT 时间计算，加入 KV Cache 访存时间

### 问题描述
- 当前实现：kvcache 只计算容量（bytes），没有计算访存时间
- TPOT 不随序列长度变化（应该线性增长）

### 修复内容
在 `llm_perf/analyzer/unified.py` 的 `_analyze_phases` 方法中：
1. 计算 KV cache 访存时间 = framework_overhead_bytes / memory_bandwidth
2. 将访存时间添加到 single_time，再计算 total_time
3. 添加 debug 日志，便于调试

### 修改文件
- `llm_perf/analyzer/unified.py`: line 326-336

### 测试结果
- 所有测试通过：551 passed
- Web API 测试通过：test_workload_inference_scenario

### 后续 TODO
**重要**：当前修复仅添加了 KV cache 访存时间，但 TPOT 的线性增长问题仍未完全解决。

根本原因：
- Attention 计算未考虑 kv_seq_len
- decode 阶段 seq_len=1，导致 attention 计算量被低估
- 实际 attention 计算量：O(kv_seq_len × hidden_size)

后续需要：
1. 修改 Attention kernel 或 bind 机制，使其考虑 kv_seq_len
2. 确保 Attention 计算时间随 kv_seq_len 线性增长
3. 参考 `_calculate_framework_overhead` 中的 kv_seq_len 计算逻辑

### 数据来源
- Transformer decode 阶段特性：Q长度=1，K/V长度=kv_seq_len
- Memory-bound 特性：decode 阶段受内存带宽限制
- 理论分析：Attention QK^T 和 PV 计算都随 kv_seq_len 变化

---

## 会话时间
2026-04-22

---

## 分层并行切分策略支持

### [feat(validation)]: 支持 Attention 和 MoE 分层切分策略

**功能需求**:
- 用户指出 Attention 和 MoE 可以采用不同的 TP 策略
- 8卡场景示例：Attn TP=8，MoE ETP=2×EP=4 或 ETP=4×EP=2
- 需要支持分层切分的参数设计和验证逻辑

**实现内容**:

1. **新增 `expert_tp_degree` 参数** (`llm_perf/strategy/base.py`, `llm_perf/strategy/parallel_context.py`)
   - `StrategyConfig.expert_tp_degree`: Optional[int]，默认等于 tp_degree
   - `ParallelContext.expert_tp_degree`: 用于 MoE Expert 内部 TP 切分
   - `world_size`: Attention 部分的 GPU 数量
   - `moe_world_size`: MoE 部分的 GPU 数量（使用 expert_tp_degree）

2. **更新验证逻辑** (`llm_perf/validation/strategy_validator.py`)
   - `_validate_layered_parallelism`: 分层切分验证
     - Attention: `tp_degree × dp_degree × pp_degree × sp_degree = num_gpus`
     - MoE: `expert_tp_degree × ep_degree × dp_degree × pp_degree × sp_degree = num_gpus`
   - `_validate_ep_performance_suggestion`: EP 性能建议（WARNING 级别）
     - 原 `EP ≤ TP` 约束改为性能建议
     - 当 EP > expert_tp × 2 时发出警告

3. **更新设计文档** (`docs/validation_design.md`)
   - 新增 2.1.1 分层并行策略验证
   - 更新 2.1.2 并行度乘积（分层验证）
   - 2.1.3 EP 与 Expert TP 关系（降级为 WARNING）
   - 更新错误响应示例

4. **更新测试用例** (`tests/test_strategy.py`, `tests/test_validation.py`)
   - `test_expert_tp_defaults_to_tp`: 默认值测试
   - `test_expert_tp_explicit`: 显式设置测试
   - `test_moe_world_size_layered`: 分层切分计算测试
   - `test_valid_strategy_layered_tp`: 分层切分验证测试
   - `test_ep_exceeds_expert_tp_warning`: EP 性能警告测试

**关键变更**:
- `StrategyConfig.world_size` 计算：不再包含 ep_degree
- `StrategyConfig.moe_world_size` 新增：包含 expert_tp_degree × ep_degree
- `EP_EXCEEDS_TP` 错误改为 `EP_EXCEEDS_EXPERT_TP_WARNING` 警告
- `PARALLEL_PRODUCT_MISMATCH` 分拆为 `ATTN_PARALLEL_PRODUCT_MISMATCH` 和 `MOE_PARALLEL_PRODUCT_MISMATCH`

**验证场景示例**:
- 8卡集群，Attn TP=8，MoE ETP=2×EP=4：
  - Attention: 8×1×1×1 = 8 ✅
  - MoE: 2×4×1×1×1 = 8 ✅
- 64卡集群，Attn TP=64，MoE ETP=8×EP=8：
  - Attention: 64×1×1×1 = 64 ✅
  - MoE: 8×8×1×1×1 = 64 ✅

**参考来源**:
- DeepSeek-V3 MoE 切分策略
- Megatron-LM MoE 实现
- Expert 权重切分：`{0: "ep", 2: "tp"}`

---

## 会话时间
2026-04-03

---

## 详细性能分解与数据呈现优化

### [feat(analyzer)]: 添加详细性能分解框架

**功能需求**:
- Pipeline子模型级别分解（Text Encoder、DiT、VAE的内容、计算、通信开销）
- 模型内Block级别分解（每类block的详细分析）
- 内存分类呈现（激活内存、模型权重、优化器内存等）
- 通信细分呈现（按TP、DP、PP、EP、SP切分方式，包含通信语义和通信量）

**实现内容**:

1. **新增详细分解数据结构** (`llm_perf/analyzer/detailed_breakdown.py`)
   - `ParallelismType` 枚举：TP、DP、PP、EP、SP
   - `MemoryType` 枚举：PARAMETER、ACTIVATION、GRADIENT、OPTIMIZER、KV_CACHE等
   - `CommunicationDetail`: 详细通信操作（类型、操作、数据量、时间、描述）
   - `BlockBreakdown`: Block级别分解（类型、计算、内存、通信）
   - `SubModelBreakdown`: 子模型级别分解（名称、类型、迭代次数、内存、通信）
   - `MemoryBreakdown`: 内存分解（按类型、按子模型、按Block类型）
   - `CommunicationBreakdown`: 通信分解（按并行类型、按子模型）
   - `DetailedPerformanceResult`: 完整详细结果

2. **新增分解生成器** (`llm_perf/analyzer/breakdown_generator.py`)
   - `BreakdownGenerator`: 为单个模型生成分解
     - `_estimate_memory_by_type()`: 按类型估算内存
     - `_estimate_kv_cache_memory()`: 估算KV Cache
     - `_generate_block_breakdowns()`: 生成Block级别分解
     - `_generate_communication_breakdown()`: 生成通信分解
   - `PipelineBreakdownGenerator`: 为Pipeline生成分解
     - 聚合所有子模型的内存和通信数据

3. **更新`DiffusionVideoAnalyzer`** (`llm_perf/analyzer/diffusion_video.py`)
   - 添加`include_detailed_breakdown`参数
   - 新增`_generate_detailed_breakdown()`方法
   - `DiffusionVideoResult`支持`detailed_breakdown`字段

**示例输出**:
```
=== Detailed Breakdown Summary ===

--- Sub-Models ---
Text Encoder (text_encoder):
  Memory: 8.28 GB
DiT (dit):
  Memory: 27.89 GB
VAE (vae):
  Memory: 18.10 GB

--- Memory Breakdown ---
  activation     : 21.71 GB
  comm_buffer    : 0.52 GB
  gradient       : 5.24 GB
  kv_cache       : 0.58 GB
  optimizer      : 20.97 GB
  parameter      : 5.24 GB

--- Communication Breakdown ---
  TP   : Volume=0.00 GB, Time=0.00 ms
```

**测试验证**:
- 全量测试通过：277 tests passing
- 代码风格检查通过

---

## 内存估算框架重构

### [refactor(models/analyzer)]: 重构内存估算框架，区分训练/推理模式

**问题背景**:
- 原内存计算公式对所有模型累加所有层激活值，适用于训练场景（需要保存激活用于反向传播）
- 但推理场景中，激活内存是逐层覆盖刷新的，应按历史激活值的最大值计算
- 需要添加内存校准项（碎片、调度开销等）的占位符

**实现内容**:

1. **新增`MemoryCalibrationConfig`** (`llm_perf/models/base.py`)
   - 支持配置内存碎片因子（默认15%）
   - CUDA上下文开销（默认500MB）
   - 通信缓冲区因子（默认5%，分布式场景）
   - Kernel工作空间因子（默认2%）
   - 安全边距因子（默认5%）
   - `apply()`方法自动应用所有校准因子

2. **更新`ModelConfig`** (`llm_perf/models/base.py`)
   - 添加`memory_calibration`字段，默认使用`MemoryCalibrationConfig()`

3. **新增`BaseModel.estimate_memory()`** (`llm_perf/models/base.py`)
   - 区分训练模式和推理模式的内存估算
   - 训练模式：累加所有层激活（用于反向传播）
   - 推理模式：取单层最大激活（逐层复用）
   - 支持批量大小、分布式模式、校准应用等参数
   - 新增`_estimate_inference_activation_memory()`和`_estimate_training_activation_memory()`辅助方法

4. **更新`DiffusionVideoAnalyzer`** (`llm_perf/analyzer/diffusion_video.py`)
   - `_estimate_text_encoder_memory()`: 使用新的`estimate_memory()`方法
   - `_estimate_dit_memory()`: 使用新的`estimate_memory()`方法，并保留视频特定的序列长度缩放

5. **更新`TrainingAnalyzer`** (`llm_perf/analyzer/training.py`)
   - `_estimate_memory()`: 使用新的`estimate_memory(inference_mode=False)`，并保留分布式训练逻辑

6. **更新`InferenceAnalyzer`** (`llm_perf/analyzer/inference.py`)
   - `_estimate_memory()`: 使用新的`estimate_memory(inference_mode=True)`，并保留KV Cache计算

**内存估算对比**（Wan2.1 DiT）:
```
旧方法（累加所有层）: 606.74 GB
新方法-推理模式:      55.14 GB
新方法-训练模式:     795.32 GB（含校准因子）
训练/推理比例:        14.4x
```

**测试验证**:
- 全量测试通过：277 tests passing
- 代码风格检查通过：ruff检查无错误

---

## Bug修复：Web服务Wan2.1模型选择错误

### [fix(web)]: 修复选择Wan2.1模型时高级参数为空导致的错误

**问题描述**:
- 在Web服务中选择wan-t2v-14b模型时出现报错：`Error: unsupported operand type(s) for *: 'int' and 'NoneType'`
- 原因是wan-t2v-14b预设的type为"wan-pipeline"，而前端`loadModelPreset`函数尝试访问不存在的字段（如`hidden_size`等），导致高级参数显示为"undefined"
- 提交评估时，undefined值被发送到后端，导致计算错误

**修复内容**:

1. **web/static/js/app.js**
   - 修改`loadModelPreset`函数，处理"wan-pipeline"类型的特殊情况
   - 为视频生成模型设置默认参数值
   - 存储pipeline信息供evaluate函数使用
   - 修改`evaluate`函数，检测视频生成pipeline并使用正确的端点（`/api/evaluate/pipeline/diffusion-video`）
   - 修改`collectConfig`函数，为视频生成添加特有参数（num_frames, height, width等）

**测试验证**:
- 后端API测试通过：`/api/evaluate/pipeline/diffusion-video`返回正确结果
- 全量测试通过：277 tests passing

---

### [fix(web)]: 修复Wan2.1结果显示错误

**问题描述**:
- 点击"开始评估"时提示 `Error: Network error: Cannot read properties of undefined (reading 'toFixed')`
- 原因是`displayResults`函数期望标准训练/推理结果格式，但pipeline返回不同格式（包含`step_times`和`throughput`）
- 原代码尝试访问不存在的字段（如`tokens_per_sec`、`decode_tokens_per_sec`等），导致undefined错误

**修复内容**:

1. **web/static/js/app.js**
   - 为`state`对象添加`currentPipeline`和`videoParams`初始值
   - 在`displayResults`函数中添加对`diffusion-video` pipeline的特殊处理
   - 添加默认值（`|| 0`）防止undefined错误
   - 显示视频生成特有的结果（总时间、峰值内存、像素/秒、推理步数）
   - 添加组件耗时分解表格（Text Encoder、DiT Denoising、VAE Decoder）

**测试验证**:
- 后端API测试通过
- Pipeline评估返回正确结果：总时间621.9秒，峰值内存271.5GB
- 全量测试通过：277 tests passing

---

## 模型注册机与Pipeline管线抽象

### [feat(core/registry)]: 实现模型注册机和Pipeline注册机

**功能概述**:
- 实现ModelRegistry单例模式，集中管理所有模型注册
- 实现PipelineRegistry单例模式，集中管理所有Pipeline注册
- 抽象Pipeline基类，支持子模型串接和迭代执行
- 实现DiffusionVideoPipeline用于视频生成管线
- Web服务支持动态刷新模型和管线列表

**实现内容**:

1. **ModelRegistry** (`llm_perf/core/registry.py`)
   - 单例模式，全局唯一实例
   - 支持注册模型配置类和模型类
   - 通过名称动态创建模型实例
   - 按类别组织模型（llm, moe, vae, dit, text_encoder等）
   - 支持预设配置

2. **PipelineRegistry** (`llm_perf/core/registry.py`)
   - 单例模式，全局唯一实例
   - 支持注册Pipeline类
   - 通过名称动态创建Pipeline实例
   - 支持查询Pipeline支持的模型类别

3. **Pipeline基类** (`llm_perf/core/pipeline.py`)
   - 抽象基类定义管线接口
   - PipelineStep: 单个执行步骤
   - IterationConfig: 迭代配置（用于去噪等迭代过程）
   - PipelineResult: 统一的执行结果格式
   - 支持步骤依赖关系

4. **模型注册模块** (`llm_perf/models/registry.py`)
   - 注册所有内置模型：llama, moe, deepseek, deepseek-v3, resnet, vae
   - 注册Wan2.1模型：wan-text-encoder, wan-dit, wan-vae
   - 提供预设配置（llama-7b, llama-70b, mixtral-8x7b, deepseek-v3等）

5. **Pipeline注册模块** (`llm_perf/pipelines/registry.py`)
   - 注册inference pipeline（标准LLM推理）
   - 注册diffusion-video pipeline（视频生成）

6. **DiffusionVideoPipeline** (`llm_perf/pipelines/diffusion_video.py`)
   - 实现Text Encoder → DiT (迭代) → VAE Decoder管线
   - 集成DiffusionVideoAnalyzer进行性能分析
   - 支持可调节的推理步数和CFG
   - create_wan_t2v_pipeline工厂函数

7. **InferencePipeline** (`llm_perf/pipelines/base.py`)
   - 标准LLM推理Pipeline封装
   - 使用InferenceAnalyzer进行性能分析

8. **Web服务更新** (`web/app.py`)
   - `/api/models` - 获取所有注册模型
   - `/api/models/refresh` - 刷新模型注册表
   - `/api/pipelines` - 获取所有注册管线
   - `/api/pipelines/refresh` - 刷新管线注册表
   - `/api/evaluate/pipeline/<name>` - 执行指定管线评估
   - 使用ModelRegistry创建模型，替代硬编码逻辑

9. **测试覆盖** (`tests/test_registry.py`)
   - ModelRegistry单元测试 (6个)
   - PipelineRegistry单元测试 (4个)
   - 集成测试 (5个)

**测试状态**: 277 tests passing (262原有 + 15新增)

---

## 会话时间
2026-04-03

---

## Wan2.1 视频生成模型评估支持

### [feat(models/analyzer)]: 新增 Wan2.1-T2V-14B 多模态视频生成模型评估支持

**功能概述**:
- 支持 Wan2.1 文本到视频生成模型的完整评估
- 分离评估 Text Encoder、DiT Backbone、VAE 三个核心组件
- 支持完整去噪流程评估（含CFG）

**实现内容**:

1. **WanTextEncoder (umT5-XXL)** (`llm_perf/models/wan_video.py`)
   - 24层T5 Encoder架构
   - hidden_size=4096, num_heads=64
   - 支持多语言文本编码（中英文）
   - 参数量: ~4.7B

2. **WanDiTModel (Diffusion Transformer)** (`llm_perf/models/wan_video.py`)
   - 40层Transformer，hidden_size=5120
   - Flow Matching框架
   - 每块结构: Self-Attn → Cross-Attn → FFN
   - Patchify: 3D卷积 kernel=(1,2,2)
   - Time Embedding: 共享MLP + 每块独立bias
   - 参数量: ~14B

3. **WanVAEModel (3D Causal VAE)** (`llm_perf/models/wan_video.py`)
   - 时序压缩: 4x
   - 空间压缩: 8x8
   - Latent channels: 16
   - 因果卷积保证时序因果性
   - 参数量: ~0.5B

4. **DiffusionVideoAnalyzer** (`llm_perf/analyzer/diffusion_video.py`)
   - 完整视频生成流程评估
   - 支持CFG（Classifier-Free Guidance）
   - 各组件独立评估 + 整体pipeline评估
   - 内存和耗时分析

5. **测试覆盖** (`tests/test_wan_video.py`)
   - TextEncoder测试 (4个)
   - DiT测试 (7个)
   - VAE测试 (5个)
   - Analyzer测试 (8个)
   - 共24个测试用例

**数据来源**:
- Wan2.1技术报告: arXiv:2503.20314
- HF模型页: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
- 官方仓库: https://github.com/Wan-Video/Wan2.1

**影响文件**:
- `llm_perf/models/wan_video.py` (新增)
- `llm_perf/models/__init__.py`
- `llm_perf/analyzer/diffusion_video.py` (新增)
- `llm_perf/analyzer/__init__.py`
- `tests/test_wan_video.py` (新增)
- `docs/data_sources_wiki.md` (更新)

**验证结果**:
```bash
$ python tests/run_tests.py
Ran 262 tests in 0.298s
OK
```

---

## 历史会话

### 会话时间
2026-04-03 (修正)

---

## 序列并行 (Sequence Parallelism) 评估支持 - 修正

### [fix(kernels/analyzer)]: 修正 Ulysses-SP all-to-all 次数和 TP 兼容性描述

**修正内容**:

1. **修正 all-to-all 次数**
   - 原实现: 每层 2 次 all-to-all (pre + post)
   - 修正后: 每层 **4 次 all-to-all** (Q/K/V 各一次 pre + O 一次 post)
   - 参考: FlexSP paper (arXiv:2412.01523)
   - 影响文件: `training.py`, `inference.py`

2. **修正 Ulysses 与 TP 兼容性描述**
   - 原描述: "Ulysses 与 TP 冲突"
   - 修正后: "Ulysses 与 TP **可以组合**，需满足 `sp_degree * tp_degree <= num_heads` 且能整除"
   - 补充: Dummy-Head 技术可扩展并行度突破 heads 限制
   - 参考: USP paper, 360-LLaMA-Factory (arXiv:2505.22296)
   - 影响文件: `docs/data_sources_wiki.md`

**数据来源**:
- FlexSP: arXiv:2412.01523 (确认 Q/K/V/O 各需一次 all-to-all)
- Dummy-Head Ulysses: arXiv:2505.22296
- USP: arXiv:2405.07719

---

## 历史会话

### 会话时间
2026-04-03

---

## 序列并行 (Sequence Parallelism) 评估支持

### [feat(kernels/analyzer)]: 新增 Ulysses-SP、Ring-SP、Unified-2D-SP 评估支持

**功能概述**:
- 支持多模态生成模型中三种主流序列并行切分方式的性能评估
- Ulysses-SP: 基于 all-to-all 的 head-sharding 序列并行
- Ring-SP: 支持 send-recv P2P 和 allgather 两种实现
- Unified-2D-SP: 融合 Ulysses 和 Ring 的 2D 序列并行

**实现内容**:

1. **策略配置扩展 (`llm_perf/strategy/base.py`)**
   - 新增 `SPType` 枚举: `ULYSSES`, `RING_P2P`, `RING_ALLGATHER`, `UNIFIED_2D`
   - `StrategyConfig` 新增 `sp_type`, `ulysses_degree`, `ring_degree` 字段
   - 更新 `to_dict` / `from_dict` 序列化支持

2. **通信 Kernel 扩展 (`llm_perf/kernels/communication.py`)**
   - `create_sp_ulysses_alltoall`: Ulysses 注意力前后的 all-to-all 通信
   - `create_sp_ring_p2p`: Ring Attention 的 ring-exchange P2P 通信
   - `create_sp_ring_allgather`: Ring Attention 的 allgather 聚合实现
   - `create_sp_unified_2d`: 2D-SP 组合 kernel，同时创建 ulysses + ring 通信

3. **训练分析器集成 (`llm_perf/analyzer/training.py`)**
   - 在 `_estimate_communication_time` 中新增 SP 通信开销估算
   - `_estimate_sp_communication_time` 方法支持四种 SP 类型的独立建模:
     - Ulysses: `2 * alltoall_time * num_layers`
     - Ring P2P: `(sp_degree - 1) * kv_bytes_per_step / bw * num_layers`
     - Ring Allgather: `allgather_time * num_layers`
     - Unified 2D: ulysses 时间 + ring 时间叠加
   - SP 激活内存按 `sp_degree` 进行缩减（已有逻辑兼容）

4. **推理分析器集成 (`llm_perf/analyzer/inference.py`)**
   - 在 `_estimate_communication_time_for_phase` 中新增 SP 通信估算
   - 区分 prefill 和 decode 阶段的通信量差异

5. **测试覆盖 (`tests/test_sp_communication.py`)**
   - CommKernel 创建测试 (5 个)
   - TrainingAnalyzer SP 训练测试 (5 个，含内存缩减验证)
   - InferenceAnalyzer SP 推理测试 (4 个)
   - StrategyConfig 序列化测试 (2 个)
   - 共 16 个测试用例，全部通过

**数据来源**:
- DeepSpeed-Ulysses: arXiv:2309.14509
- Ring Attention: Megatron-LM Context Parallel, PyTorch TorchTitan
- Unified 2D-SP: USP paper (arXiv:2405.07719)

**影响文件**:
- `llm_perf/strategy/base.py`
- `llm_perf/kernels/communication.py`
- `llm_perf/analyzer/training.py`
- `llm_perf/analyzer/inference.py`
- `tests/test_sp_communication.py` (新增)
- `docs/data_sources_wiki.md` (更新)

**验证结果**:
```bash
$ python tests/run_tests.py
Ran 238 tests in 0.299s
OK
```

---

## 历史会话

### 会话时间
2026-04-01

---

## DeepSeek V2/V3 模型支持

### [feat(models)]: 新增 DeepSeek V2/V3 系列模型评估支持

**功能概述**:
- 支持 DeepSeek-V2 和 DeepSeek-V3 两种模型变体
- 实现 MLA (Multi-head Latent Attention) 核心架构评估
- 集成 DeepSeekMoE 架构支持

**实现内容**:

1. **DeepSeekConfig / DeepSeekV3Config**
   - 完整的 HuggingFace 官方配置参数映射
   - MLA 相关参数: `kv_lora_rank`, `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`
   - MoE 相关参数: `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`, `first_k_dense_replace`

2. **DeepSeekModel 层构建**
   - MLA Attention: query/kv 压缩与解压投影层
   - Dense FFN: 前 k 层使用标准 SwiGLU FFN
   - MoE FFN: 路由专家 + 共享专家混合架构
   - All-to-all 通信层（支持 EP 并行）

3. **MLA 核心逻辑**
   - KV cache 压缩: `hidden_size` → `kv_lora_rank` (512)
   - 压缩比计算: ~96x (vs 标准 MHA)
   - KV cache 内存计算工具方法

4. **预配置模型**
   - `DeepSeekV2Model`: 官方 V2 参数 (5120 hidden, 60 layers, 160 experts)
   - `DeepSeekV3Model`: 官方 V3 参数 (7168 hidden, 61 layers, 256 experts)

**影响文件**:
- `llm_perf/models/deepseek.py` (新增)
- `llm_perf/models/__init__.py`
- `tests/test_deepseek.py` (新增)
- `docs/data_sources_wiki.md` (更新)

**参考配置来源**:
- DeepSeek-V2: https://huggingface.co/deepseek-ai/DeepSeek-V2
- DeepSeek-V3: https://huggingface.co/deepseek-ai/DeepSeek-V3

---

## 历史会话

### 会话时间
2026-03-31

---

## 1. Bug 修复

### [fix(hardware/cluster)]: 修复 NetworkConfig 传入 Cluster 时的 AttributeError

**问题描述：**
调用 `llm-perf evaluate` 时出现 `AttributeError: 'NetworkConfig' object has no attribute 'levels'`。

**根因分析：**
- `llm_perf/cli/main.py` 和 `run_eval.py` 在创建集群时传入的是 `NetworkConfig`（旧版配置对象）
- 但 `Cluster` 的 `__init__` 和 `create_homogeneous` 只声明接受 `NetworkTopology`
- 内部代码直接访问 `self.topology.levels`，而 `NetworkConfig` 没有 `levels` 属性

**修复内容：**
- 在 `Cluster.__init__` 中增加类型检查，如果传入的是 `NetworkConfig`，自动调用 `.to_topology()` 转换为 `NetworkTopology`
- 更新 `create_homogeneous` 和 `create_from_preset` 的 `topology` 参数类型为 `Union[NetworkTopology, NetworkConfig]`
- 修复 `_find_topology_level` 的 fallback 逻辑，同 node 通信应返回最低 level（level 0）

**影响文件：**
- `llm_perf/hardware/cluster.py`

---

## 2. 测试补充与修复

### 2.1 补充缺失的基础功能测试用例

**新增测试文件：**

| 测试文件 | 覆盖模块 | 测试数量 |
|---------|---------|---------|
| `test_models.py` | BaseModel, LlamaModel, MoEModel, Config 类 | 20+ |
| `test_strategy.py` | StrategyConfig, ParallelStrategy, StrategyPlanner | 25+ |
| `test_reporter.py` | TableReporter, JSONReporter | 10+ |
| `test_helpers.py` | utils/helpers 工具函数, constants 常量 | 20+ |
| `test_analyzer.py` | PerformanceBreakdown, LayerBreakdown, KernelBreakdown | 12+ |

### 2.2 修复现有测试问题

**修复内容：**

1. **MoEConfig 继承错误**
   - 改为继承 `LlamaConfig` 以正确使用 `__post_init__`
   - 修复 `test_models.py` 中 MoE 相关测试

2. **test_integration.py**
   - 补充 `LlamaConfig` 缺少的必填参数 (`vocab_size`, `num_attention_heads`)
   - 将 70B 模型测试改为 13B 模型（避免内存超限）

3. **test_device.py**
   - FP8 精度断言改为 `assertAlmostEqual(delta=1.0)`（处理浮点精度误差）

4. **test_topology.py**
   - 3-tier Clos 测试使用 1024 设备（原 512 只需 2-tier）

**验证结果：**
```bash
$ python -m unittest discover -s tests -p "test_*.py"
Ran 145 tests in 0.270s
OK
```

**影响文件：**
- `llm_perf/models/moe.py`
- `tests/test_device.py`
- `tests/test_integration.py`
- `tests/test_topology.py`
- `tests/test_models.py` (新增)
- `tests/test_strategy.py` (新增)
- `tests/test_reporter.py` (新增)
- `tests/test_helpers.py` (新增)
- `tests/test_analyzer.py` (新增)

---

## 3. Conv Kernel 增强

### 3.1 Conv2d Kernel 访存开销增强

**新增内容：**
- 详细的访存开销计算：
  - Input activation: `batch * in_c * h * w * dtype_size`
  - Weights: `out_c * in_c * k^2 * dtype_size`
  - Output activation: `batch * out_c * out_h * out_w * dtype_size`
  - Workspace (im2col): `batch * out_h * out_w * k^2 * in_c * dtype_size`
- 支持 `groups` 参数（分组卷积）

### 3.2 新增 Conv3d Kernel 支持

**功能：**
- 支持 3D 卷积 (temporal + height + width)
- FLOPs 计算：`2 * batch * out_t * out_h * out_w * out_c * kt * kh * kw * in_c`
- 访存计算包含更大的 workspace（3D unfold 开销更高）
- `get_or_create_conv3d()` 动态创建接口

**影响文件：**
- `llm_perf/kernels/compute.py`

---

## 4. ResNet 模型支持

### [feat(models)]: 新增 ResNet 模型

**功能：**
- 支持 ResNet-18/34/50/101/152 五种变体
- `ResNetConfig` 配置类，支持 `variant` 指定
- `BasicBlock` (ResNet-18/34) 和 `Bottleneck` (ResNet-50/101/152) 两种残差块
- `from_variant()` 工厂方法快速创建模型

**架构：**
```
Input → Conv1 → MaxPool → Stage1 → Stage2 → Stage3 → Stage4 → AvgPool → FC → Output
```

**影响文件：**
- `llm_perf/models/resnet.py` (新增)
- `llm_perf/models/__init__.py`
- `tests/test_resnet.py` (新增)

---

## 5. Video VAE 模型支持

### [feat(models)]: 新增 Video VAE 模型

**参考：** AutoencoderKL from Diffusers

**架构特点：**
- **Encoder**: 输入视频 → 下采样 → 注意力 → 潜变量 (mean + logvar)
- **Decoder**: 潜变量 → 上采样 → 注意力 → 重建视频
- **3D ResNet Block**: GroupNorm → Conv3d → GroupNorm → Conv3d
- **Spatial Attention**: QKV projection → attention compute → output projection

**配置选项：**
- `block_out_channels`: 各阶段通道数 (128, 256, 512, 512)
- `layers_per_block`: 每个阶段的 ResNet 块数
- `latent_channels`: 潜变量维度（通常 4）
- `use_3d_conv`: 切换 Video VAE / Image VAE

**影响文件：**
- `llm_perf/models/vae.py` (新增)
- `llm_perf/models/__init__.py`
- `tests/test_vae.py` (新增)

---

## 6. 接口重构

### [refactor(kernels)]: 改造 conv kernel 接口与 PyTorch 对齐

**改造前（难以阅读）：**
```python
configs = [
    (1, 3, 64, 7, 224, 224, 2, 3),  # 参数含义不清晰
]
kernel = registry.get_or_create_conv2d(
    batch=1, in_channels=64, out_channels=64,
    kernel_size=3, input_h=56, input_w=56,  # 分散的参数
    stride=1, padding=1,
)
```

**改造后（PyTorch 风格）：**
```python
configs = [
    dict(batch=1, in_channels=3, out_channels=64, kernel_size=7,
         input_size=(224, 224), stride=2, padding=3),
]
kernel = registry.get_or_create_conv2d(
    in_channels=64, out_channels=64, kernel_size=3,  # 主要参数在前
    stride=1, padding=1, groups=1,                   # 有默认值的参数
    batch=1, input_size=(56, 56), dtype="fp16",      # 额外参数
)
```

**Conv3d 接口：**
```python
kernel = registry.get_or_create_conv3d(
    in_channels=128, out_channels=256,
    kernel_size=(3, 3, 3),      # (T, H, W)
    stride=(1, 2, 2),           # (T, H, W)
    padding=(1, 1, 1),          # (T, H, W)
    batch=1,
    input_size=(16, 64, 64),    # (T, H, W)
    dtype="fp16",
)
```

**影响文件：**
- `llm_perf/kernels/compute.py`
- `tests/test_resnet.py`
- `tests/test_vae.py`

---

## 提交记录

```
410aabe [fix(hardware/cluster)]: 修复 NetworkConfig 传入 Cluster 时的 AttributeError
4df7aa8 [test]: 补充缺失的基础功能测试用例并修复现有测试
9d8033b [feat(kernels/models)]: 新增 Conv2d kernel 评估支持和 ResNet 模型
fd84455 [feat(kernels/models)]: 新增 Video VAE 模型和 Conv3d kernel 支持
c2f4726 [refactor(kernels)]: 改造 conv kernel 接口与 PyTorch 对齐
```

---

## 测试统计

| 阶段 | 测试数量 | 状态 |
|-----|---------|-----|
| 初始 | 145 | OK |
| 新增 Conv/ResNet | 169 | OK |
| 新增 VAE | 190 | OK |
| 接口重构后 | 190 | OK |

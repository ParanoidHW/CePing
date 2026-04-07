# 开发日志

## 会话时间
2026-04-03

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

# Reviewer 检视清单

本文档记录代码 reviewer 在检视 Kernel API 迁移和架构修正相关代码时应该检查的事项。

---

## 1. Kernel API 迁移检查

### 1.1 导入检查
- [ ] 模型文件是否正确导入 `kernel_result_to_layer`
  ```python
  from ..kernels.utils import kernel_result_to_layer
  ```
- [ ] 是否删除本地定义的 `_kernel_to_layer` 函数（避免重复定义）

### 1.2 Activation 计算检查
**重点检查项**：所有 activation_bytes 必须通过 kernel API 获取

- [ ] 检查是否还有手动计算 `activation_bytes` 的情况
  ```bash
  grep -n "activation_bytes=" llm_perf/models/*.py | grep -v "kernel_result_to_layer" | grep -v "# NOTE"
  ```
- [ ] 对于保留的手动计算，是否添加了 NOTE 注释说明原因
  ```python
  # NOTE: Manual calculation for communication layer (alltoall)
  activation_bytes=cfg.hidden_size * dtype_size * cfg.num_experts_per_tok,
  ```

### 1.3 LayerConfig 创建检查
- [ ] 使用 kernel 的层是否使用 `kernel_result_to_layer`
  ```python
  # ✅ 正确
  layers.append(kernel_result_to_layer(
      name=f"{prefix}_q_proj",
      result=q_result,
      params=cfg.hidden_size * q_heads * head_dim,
      dtype_size=dtype_size
  ))
  
  # ❌ 错误（手动计算 activation_bytes）
  layers.append(LayerConfig(
      name=f"{prefix}_q_proj",
      input_shape=...,
      output_shape=...,
      params_count=...,
      flops=q_result.flops,
      activation_bytes=seq_len * out_dim * dtype_size,  # 禁止
  ))
  ```

### 1.4 例外情况检查
允许手动计算的情况（必须添加 NOTE 注释）：
- [ ] **通信层**：alltoall_dispatch, alltoall_combine 等
- [ ] **归一化层**：GroupNorm, InstanceNorm 等（无 kernel 实现）
- [ ] **池化层**：AvgPool, MaxPool 等
- [ ] **特殊计算**：attention_compute 等简化计算

---

## 2. 架构修正检查

### 2.1 Wan2.1 架构修正检查
- [ ] **AdaLN 参数数量**：是否为 6 个参数
  - self-attn: shift, scale, gate (3 params)
  - FFN: shift, scale, gate (3 params)
  - 共 `6 * hidden_size`
- [ ] **Q/K RMSNorm**：是否在 self-attention 中对 Q 和 K 应用 RMSNorm
- [ ] **Cross-attn 无调制**：cross-attention 是否确实没有 modulation
- [ ] **参考来源**：注释中是否注明参考自 Wan2.1 model.py 的哪一行

### 2.2 FLOPs 计算检查
- [ ] **线性操作**：是否使用 `flops = 2 * M * N * K`
  ```python
  # ✅ 正确（乘加 = 2 FLOPs）
  flops = 2 * batch_size * in_features * out_features
  
  # ❌ 错误
  flops = batch_size * in_features * out_features  # 少了乘2
  ```
- [ ] **卷积操作**：是否包含所有维度
  ```python
  flops = 2 * N * C_out * H_out * W_out * C_in * K_h * K_w
  ```

### 2.3 参数量检查
- [ ] 模型总参数量是否与官方一致
  - DeepSeek V3: ~671B (saved in checkpoint)
  - Wan DiT 14B: ~14B
  - Llama 7B: ~7B
- [ ] 每层参数量是否正确计算（包含 bias 如果有）

---

## 3. MoE 模型特殊检查

### 3.1 专家层缩放检查
- [ ] routed expert 的 flops 是否根据 `num_experts_per_tok` 缩放
  ```python
  flops=int(up_result.flops * cfg.num_experts_per_tok)
  ```
- [ ] routed expert 的 activation_bytes 是否正确缩放
  ```python
  activation_bytes=int(ffn_intermediate * dtype_size * cfg.num_experts_per_tok)
  ```
- [ ] shared expert 是否没有缩放（始终激活）
- [ ] `is_moe=True` 是否正确设置

### 3.2 通信层检查
- [ ] alltoall_dispatch 层是否正确建模
- [ ] alltoall_combine 层是否正确建模
- [ ] activation_bytes 是否反映通信数据量

---

## 4. 代码质量检查

### 4.1 Ruff 检查
必须无警告：
```bash
ruff check llm_perf/models/*.py --select=F401,F841,E741
```

- [ ] **F401**: 无未使用的导入
- [ ] **F841**: 无未使用的变量
- [ ] **E741**: 无歧义变量名（如 `l`, `O`, `I`）

### 4.2 导入检查
- [ ] 标准库 → 第三方库 → 本地模块 的顺序
- [ ] 相对导入使用 `..module` 而非 `llm_perf.xxx`

### 4.3 命名规范
- [ ] 类名：PascalCase (`LlamaModel`)
- [ ] 函数名：snake_case (`build_layers`)
- [ ] 常量：UPPER_CASE (`DTYPE_SIZES`)
- [ ] 私有方法：`_` 前缀 (`_build_transformer_layer`)

### 4.4 注释检查
- [ ] 类和方法是否有 docstring
- [ ] 复杂逻辑是否有行内注释
- [ ] 架构修正是否注明参考来源
- [ ] 例外情况是否有 NOTE 注释

---

## 5. 测试检查

### 5.1 测试通过率
```bash
python -m pytest tests/test_models.py tests/test_deepseek.py \
    tests/test_resnet.py tests/test_vae.py tests/test_wan_video.py -v
```
- [ ] 所有测试通过（无 Error 或 Failure）

### 5.2 测试更新检查
- [ ] 层数变更时测试期望是否更新
  ```python
  # 测试 Llama 层数
  expected = 1 + 2 * 12 + 2  # 27 layers
  self.assertEqual(len(model.layers), expected)
  ```
- [ ] 参数量测试是否通过
- [ ] FLOPs 测试是否通过

### 5.3 新功能测试
- [ ] 新增 kernel 是否有对应测试
- [ ] 新增模型是否有对应测试

---

## 6. Web 服务检查

### 6.1 详细分解展示
- [ ] Training 模型是否显示 memory/communication breakdown
- [ ] Inference 模型是否显示 memory/communication breakdown
- [ ] Pipeline 模型是否显示 memory/communication breakdown

### 6.2 数据格式检查
- [ ] memory_breakdown 包含：parameters_gb, activations_gb, communication_gb
- [ ] communication_breakdown 包含：all_reduce_gb, all_gather_gb, reduce_scatter_gb
- [ ] 数值格式化：GB 保留 2 位小数

---

## 7. 文档检查

### 7.1 代码注释
- [ ] 架构修正是否说明参考来源（论文/官方代码链接）
- [ ] 复杂计算是否有注释说明
- [ ] 例外情况是否有 NOTE 注释

### 7.2 文档更新
- [ ] `docs/kernel_api.md` 是否更新（新增 kernel）
- [ ] `docs/kernel_migration_guide.md` 是否更新（重构指南）
- [ ] `examples/` 是否有示例代码

---

## 8. 快速检查命令

```bash
# 1. 代码质量检查
ruff check llm_perf/models/*.py --select=F401,F841,E741

# 2. 查找手动计算 activation
grep -n "activation_bytes=" llm_perf/models/*.py | \
    grep -v "kernel_result_to_layer" | grep -v "# NOTE"

# 3. 运行测试
python -m pytest tests/test_models.py tests/test_deepseek.py \
    tests/test_resnet.py tests/test_vae.py tests/test_wan_video.py -v --tb=short

# 4. 验证模型
python -c "
from llm_perf.models.llama import LlamaModel, LlamaConfig
from llm_perf.models.deepseek import DeepSeekV3Model
print('Llama:', len(LlamaModel(LlamaConfig()).layers), 'layers')
print('DeepSeek V3:', len(DeepSeekV3Model().layers), 'layers')
"

# 5. 检查 NOTE 注释
grep -n "# NOTE" llm_perf/models/*.py
```

---

## 9. Review 流程

### Step 1: 静态检查
1. 运行 ruff 检查
2. 查找手动计算 activation
3. 检查 NOTE 注释

### Step 2: 功能检查
1. 运行单元测试
2. 验证模型结构
3. 检查参数量和 FLOPs

### Step 3: 架构检查
1. 对照官方实现验证架构
2. 检查修正是否完整
3. 验证 FLOPs 计算公式

### Step 4: 文档检查
1. 检查代码注释
2. 检查是否更新文档

### Step 5: 最终确认
- [ ] 检视完成，功能精度确认 ok
- [ ] 或：列出问题清单要求修复

---

## 10. 常见问题清单

### 问题 1：手动计算 activation_bytes 未处理
**症状**：grep 找到未使用 kernel_result_to_layer 的 activation_bytes
**修复**：替换为 kernel_result_to_layer，或添加 NOTE 注释

### 问题 2：MoE 专家层未缩放
**症状**：expert_up/gate/down 的 flops/activation 未乘 num_experts_per_tok
**修复**：添加 int() 缩放计算

### 问题 3：架构修正遗漏
**症状**：Wan2.1 的 AdaLN 参数数量错误，或 cross-attn 有 modulation
**修复**：对照官方 model.py 修正

### 问题 4：FLOPs 计算错误
**症状**：线性操作未乘 2（乘加 = 2 FLOPs）
**修复**：`flops = 2 * M * N * K`

### 问题 5：测试未更新
**症状**：层数变更但测试期望未更新
**修复**：更新测试中的 expected 值

---

## 附录：本次会话 Review 要点

### 主要变更
1. **Kernel API 迁移**：6 个模型（Llama, MoE, DeepSeek, ResNet, VAE, Wan）
2. **架构修正**：Wan2.1 AdaLN（6 params）、Q/K RMSNorm、cross-attn 无 modulation
3. **FLOPs 修正**：所有线性操作 `flops = 2 * M * N * K`
4. **Web 服务**：详细分解展示（memory/communication breakdown）

### Review 重点
- 所有 activation 必须通过 kernel API
- Wan2.1 架构修正是否完整
- FLOPs 计算是否乘 2
- 测试是否全部通过
- ruff 检查是否无警告

---

*本文档记录于 2026-04-03，基于 Kernel API 迁移 Review 经验整理。*

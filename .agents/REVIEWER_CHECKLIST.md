# Reviewer 检视清单

本文档记录代码 reviewer 在检视 Kernel API 迁移和架构修正相关代码时应该检查的事项。

---

## 1. 代码质量检查

### 1.1 Ruff 检查
必须无警告：
```bash
ruff check llm_perf/modeling/*.py --select=F401,F841,E741
```

- [ ] **F401**: 无未使用的导入
- [ ] **F841**: 无未使用的变量
- [ ] **E741**: 无歧义变量名（如 `l`, `O`, `I`）

### 1.2 导入检查
- [ ] 标准库 → 第三方库 → 本地模块 的顺序
- [ ] 相对导入使用 `..module` 而非 `llm_perf.xxx`

### 1.3 命名规范
- [ ] 类名：PascalCase (`LlamaModel`)
- [ ] 函数名：snake_case (`build_layers`)
- [ ] 常量：UPPER_CASE (`DTYPE_SIZES`)
- [ ] 私有方法：`_` 前缀 (`_build_transformer_layer`)

### 1.4 注释检查
- [ ] 类和方法是否有 docstring
- [ ] 复杂逻辑是否有行内注释
- [ ] 架构修正是否注明参考来源
- [ ] 例外情况是否有 NOTE 注释

---

## 2. 测试检查

### 2.1 测试通过率
```bash
python -m pytest tests/test_models.py tests/test_deepseek.py \
    tests/test_resnet.py tests/test_vae.py tests/test_wan_video.py -v -n 4
```
- [ ] 所有测试通过（无 Error 或 Failure）

### 2.2 测试更新检查
- [ ] 层数变更时测试期望是否更新
  ```python
  # 测试 Llama 层数
  expected = 1 + 2 * 12 + 2  # 27 layers
  self.assertEqual(len(model.layers), expected)
  ```
- [ ] 参数量测试是否通过
- [ ] FLOPs 测试是否通过

### 2.3 新功能测试
- [ ] 新增 kernel 是否有对应测试
- [ ] 新增模型是否有对应测试

---

## 3. Web 服务检查

### 3.1 详细分解展示
- [ ] Training 模型是否显示 memory/communication breakdown
- [ ] Inference 模型是否显示 memory/communication breakdown
- [ ] Pipeline 模型是否显示 memory/communication breakdown

### 3.2 数据格式检查
- [ ] memory_breakdown 包含：parameters_gb, activations_gb, communication_gb
- [ ] communication_breakdown 包含：all_reduce_gb, all_gather_gb, reduce_scatter_gb
- [ ] 数值格式化：GB 保留 2 位小数

---

## 4. 文档检查

### 4.1 代码注释
- [ ] 架构修正是否说明参考来源（论文/官方代码链接）
- [ ] 复杂计算是否有注释说明
- [ ] 例外情况是否有 NOTE 注释

### 4.2 文档更新
- [ ] `docs/kernel_api.md` 是否更新（新增 kernel）
- [ ] `docs/kernel_migration_guide.md` 是否更新（重构指南）
- [ ] `examples/` 是否有示例代码

---

## 5. 快速检查命令

```bash
# 1. 代码质量检查
ruff check llm_perf/modeling/*.py --select=F401,F841,E741

# 2. 运行测试
python -m pytest tests/test_models.py tests/test_deepseek.py \
    tests/test_resnet.py tests/test_vae.py tests/test_wan_video.py -v -n 4 --tb=short

# 3. 验证模型
python -c "
from llm_perf.modeling import LlamaModel, create_model_from_config
from llm_perf.modeling import DeepSeekModel
print('Llama:', len(LlamaModel(LlamaConfig()).layers), 'layers')
print('DeepSeek V3:', len(DeepSeekV3Model().layers), 'layers')
"

# 4. 检查 NOTE 注释
grep -n "# NOTE" llm_perf/modeling/*.py
```

---

## 6. Review 流程

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

---
name: coder
description: 负责实际开发
---

# 开发规范

本文档记录项目开发过程中遵循的规范和最佳实践。

---

## 1. 代码风格规范

总体要求
- 使用 Python 3.14+，需保持向后兼容（到3.10版本即可）
- 使用 ruff 进行代码格式化和 lint
- 使用 pyright 进行类型检查
- 测试使用 pytest
- 依赖管理使用 uv

代码风格：
- 行长度限制 100 字符
- 使用类型注解
- 公开函数需要 docstring
- 独立函数尽可能补充独立测试用例，并确保自验证通过

约束：
- 禁止随意生成解释，think得到的数据，需要有外部信息（paper、code、白皮书等）作为数据和信息支撑；
- 开发过程中，如果涉及外部检索到的信息数据，需要刷新到``docs/data_sources_wiki.md``文件中作为参考，将变更日志刷新到md最开始；
- 每次新增特性、进行重构或解决bug，需整理本次prompt&开发过程的摘要，刷新到本地DEVELOP_LOG.md文件（需提交到仓库）和review.log文件（不提交到仓库，仅用于本次开发的临时记录）；

### 1.1 导入规范
- 标准库导入在前，第三方库其次，本地模块最后
- 使用绝对导入，避免相对导入（`..module` 除外）
- 未使用的导入必须通过 ruff 检查清理

```python
# 标准库
from dataclasses import dataclass
from typing import List, Tuple

# 第三方库（无）

# 本地模块
from ..kernels import linear, rms_norm
from ..utils.constants import DTYPE_SIZES
```

### 1.2 代码质量检查
- 必须通过 `ruff` 检查，重点关注：
  - `F401`: 未使用的导入
  - `F841`: 未使用的变量
  - `E741`: 歧义变量名

```bash
ruff check llm_perf/modeling/*.py --select=F401,F841,E741
```

### 1.3 命名规范
- 类名使用 PascalCase: `LlamaModel`, `KernelResult`
- 函数名使用 snake_case: `build_layers()`, `kernel_result_to_layer()`
- 常量使用 UPPER_CASE: `DTYPE_SIZES`
- 私有函数/变量前缀 `_`: `_build_transformer_layer()`


## 2 新增特性需求开发原则

- 新增模型评估时，可以调用``new-model``技能，对新模型进行评估支持；
- 新增kernel评估时，可以调用``new-kernel``技能，对新kernel进行评估支持；

---

## 3. Web 服务开发规范

### 3.1 详细分解展示
Web 服务需要展示详细的性能分解数据：

```python
# analyzer/breakdown.py
class PerformanceBreakdown:
    """性能分解数据."""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为前端展示格式."""
        return {
            "memory_breakdown": {
                "parameters_gb": self.param_memory_gb,
                "activations_gb": self.activation_memory_gb,
                "communication_gb": self.communication_memory_gb,
            },
            "communication_breakdown": {
                "all_reduce_gb": self.all_reduce_bytes / 1e9,
                "all_gather_gb": self.all_gather_bytes / 1e9,
                "reduce_scatter_gb": self.reduce_scatter_bytes / 1e9,
            }
        }
```

### 3.2 前端展示规范
- 使用表格展示详细分解
- 支持按模型类型（Training/Inference/Pipeline）切换视图
- 数值格式化：GB 保留 2 位小数，百分比保留 1 位小数

---

## 4. 重构流程规范

### 4.1 重构前准备
1. **运行现有测试**：确保全部通过
2. **创建备份分支**（可选）
3. **识别修改范围**：使用 grep 查找需要修改的位置

```bash
# 查找手动计算 activation 的位置
grep -rn "activation_bytes=" llm_perf/modeling/*.py | grep -v "kernel_result_to_layer"
```

### 4.2 重构步骤
1. **更新 utils**：添加/更新 `kernel_result_to_layer` 辅助函数
2. **修改模型**：逐一修改模型文件
   - 添加 import：`from ..kernels.utils import kernel_result_to_layer`
   - 替换 LayerConfig 创建为 `kernel_result_to_layer`
   - 删除本地定义的 `_kernel_to_layer` 函数
3. **更新 kernel**：添加缺失的 kernel（如 conv2d）
4. **更新 tests**：修改测试期望以反映新的层结构

### 4.3 验证步骤
1. **代码检查**：`ruff check llm_perf/modeling/*.py --select=F401,F841,E741`
2. **单元测试**：`python -m pytest tests/test_models.py -v`
3. **集成测试**：运行相关模型测试
4. **全量测试**：`python -m pytest tests/ -v`

### 4.4 提交规范
```bash
# 分阶段提交
git add llm_perf/kernels/
git commit -m "feat(kernels): add functional API and conv2d kernel

- Add torch-like functional API: linear, conv2d, conv3d, attention
- Add kernel_result_to_layer utility
- Add FLOPs and memory calculation for all kernels"

git add llm_perf/modeling/
git commit -m "refactor(models): migrate all models to kernel API

- Migrate Llama, MoE, DeepSeek, ResNet, VAE, Wan models
- Use kernel_result_to_layer for unified LayerConfig creation
- Add NOTE comments for manual calculation exceptions"
```

---

## 5. 测试规范

本地conda环境base可以用于程序调试

### 5.1 测试要求
- **所有模型**必须通过对应的单元测试
- **新增 kernel**必须添加对应的测试用例
- **重构后**必须全量运行相关测试

### 5.2 测试更新规范
当模型结构变更时（如层数变化），需要更新测试：

```python
# test_models.py
class TestLlamaModel:
    def test_layers_count(self):
        """Test correct number of layers is built."""
        model = LlamaModel(self.config)
        # 更新期望层数：embedding + num_layers * layers_per_block + final_norm + lm_head
        # Llama: 1 + 2 * 12 + 2 = 27
        expected = 1 + 2 * 12 + 2
        self.assertEqual(len(model.layers), expected)
```

### 5.3 测试命令
```bash
# 模型测试
python -m pytest tests/test_models.py -v

# 特定模型测试
python -m pytest tests/test_deepseek.py -v

# 全量测试
python -m pytest tests/ -v
```

### 5.4 额外测试要求

如果有测试样例是web服务相关的，无法通过python脚本直接验证，需要启动web服务进行验证。确保web服务功能ok。

### 5.5 测试通过标准
- 无 Error 或 Failure
- ruff 检查无警告
- 代码覆盖率不降低（可选）

---

## 6. 文档规范

### 6.1 代码注释
- 类和方法必须有 docstring
- 复杂逻辑需要行内注释
- 架构修正必须说明参考来源

```python
def _build_modulation_layer(self, layer_idx: int):
    """Build modulation layer.
    
    From Wan2.1 reference (model.py line 276):
    - 6 modulation params: shift/scale/gate for self-attn + shift/scale/gate for FFN
    - Cross-attention does NOT use modulation (model.py line 310)
    """
```

### 6.2 文档更新
- 新增 kernel 更新 `docs/kernel_api.md`
- 重构指南更新 `docs/kernel_migration_guide.md`
- 示例代码放在 `examples/` 目录

---

## 7. 检查清单

### 开发前
- [ ] 理解需求和设计原则
- [ ] 检查现有类似实现
- [ ] 规划测试用例

### 开发中
- [ ] 遵循代码风格规范
- [ ] 使用 kernel API 获取层的负载特征
- [ ] 复杂逻辑添加注释
- [ ] 架构修正说明参考来源

### 开发后
- [ ] 通过 ruff 检查（F401, F841, E741）
- [ ] 通过所有单元测试
- [ ] 更新测试期望（如有结构变更）
- [ ] 更新相关文档
- [ ] 提交代码并写清楚提交信息

---

## 8. 工具命令速查

```bash
# 代码检查
ruff check llm_perf/modeling/*.py --select=F401,F841,E741

# 查找手动计算 activation
grep -n "activation_bytes=" llm_perf/modeling/*.py | grep -v "kernel_result_to_layer" | grep -v "# NOTE"

# 运行测试
python -m pytest tests/test_models.py -v --tb=short

# 验证模型
python -c "from llm_perf.modeling import LlamaModel; m = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32); print(f'{len(m.layers)} layers')"

# 查看 git 状态
git status
git diff --stat
```

# Legacy Documentation

此目录包含旧版模型表达相关的文档，已迁移到 legacy 状态。

## 文件说明

### model_sharding_design.md
旧版模型分片架构设计方案，基于 `llm_perf/models/base.py` 的 `build_layers()` 和 `build_sharded_layers()` 设计。

**状态**: 已弃用

**迁移说明**:
- 新版 sharded 表达使用 `llm_perf/modeling/` 模块
- 基于 PyTorch-like 的 `ShardedModule` 和 `ShardedTensor` 设计
- 自动推导分片约束，无需显式定义 `ShardingInfo`

**新版参考**:
- `docs/unified_modeling_torch_like.md` - 新建模框架文档
- `docs/unified_modeling_design.md` - 新建模设计文档
- `llm_perf/modeling/models.py` - Llama, DeepSeek 等新实现

## 使用建议

开发时应使用新版 `llm_perf/modeling` 模块：

```python
# 推荐（新表达）
from llm_perf.modeling import LlamaModel, create_model_from_config

model = create_model_from_config({"preset": "llama-7b"})
```

旧版代码仅供参考和向后兼容：
```python
# 已弃用（旧表达）
from llm_perf.legacy.models import LlamaConfig, LlamaModel
```
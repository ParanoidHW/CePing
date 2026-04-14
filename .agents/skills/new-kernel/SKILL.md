---
name: new-kernel
description: 新增kernel评估时的行为
---

## 新kernel评估支持

评估新kernel时
- kernel的注册方式需尽量对齐torch原生对应接口，入参数量和形式保持和torch接口一致。
- 参考的torch版本为2.10

kernel计算的参考材料：
- [kernel_modeling](../../../docs/kernel_modeling.md)


---

## Kernel API 开发规范

### 强制使用 Kernel API
**所有模型的 `activation_bytes`/`params`/`FLOPs`/内存开销等特征，必须通过 kernel API 获取**，禁止kernel外手动计算。通信开销也应该通过kernel表达来独立处理通信量和不同拓扑下的开销。

#### ✅ 正确做法
```python
from ..kernels import linear, rms_norm
from ..kernels.utils import kernel_result_to_layer

# 调用 kernel
result = linear(
    input=(seq_len, hidden_size),
    weight=(out_dim, hidden_size),
    bias=None,
    dtype=cfg.dtype
)

# 使用工具函数创建 LayerConfig
layers.append(kernel_result_to_layer(
    name="layer_name",
    result=result,
))
```

#### ❌ 错误做法
```python
# kernel外独立计算（禁止）
layers.append(LayerConfig(
    name="layer_name",
    input_shape=(1, seq_len, hidden_size),
    output_shape=(1, seq_len, out_dim),
    params_count=hidden_size * out_dim,
    flops=result.flops,
    activation_bytes=seq_len * out_dim * dtype_size,  # 禁止手动计算
))
```

### 计算类Kernel 函数签名规范
```python
def kernel_name(
    input: Tuple[int, ...],
    weight: Optional[Tuple[int, ...]] = None,
    bias: Optional[Tuple[int, ...]] = None,
    dtype: str = "fp16"
) -> KernelResult:
    """Kernel 文档字符串.
    
    Args:
        input: 输入形状
        weight: 权重形状（如果有）
        bias: 偏置形状（如果有）
        dtype: 数据类型
    
    Returns:
        KernelResult 包含 output, flops, bytes_accessed
    """
    pass
```
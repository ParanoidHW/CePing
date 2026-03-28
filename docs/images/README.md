# 图片资源目录

此目录用于存放文档中引用的图片资源。

## 支持的图片类型

- PNG - 架构图、流程图导出
- SVG - 矢量图，可缩放
- GIF - 动画演示

## 生成工具推荐

1. **Mermaid** - 直接在 Markdown 中使用，无需外部图片
2. **Draw.io** - 绘制架构图，导出 PNG/SVG
3. **Python + Matplotlib** - 性能曲线图

## 示例：性能曲线生成

```python
import matplotlib.pyplot as plt
import numpy as np

# 序列长度 vs 吞吐量
seq_lens = [1024, 2048, 4096, 8192]
throughput = [45.2, 38.5, 28.3, 18.7]

plt.figure(figsize=(10, 6))
plt.plot(seq_lens, throughput, 'o-', linewidth=2, markersize=8)
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Throughput (K tokens/s)', fontsize=12)
plt.title('Llama-7B Inference Performance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('scaling_curve.png', dpi=150, bbox_inches='tight')
```

## 架构图示例

可以使用 Draw.io 创建如下架构图：

```
┌─────────────────────────────────────────────┐
│              User Interface                  │
│         (CLI / API / Web)                   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            Analyzer Layer                    │
│    (Training / Inference Analyzer)          │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│   Kernel     │    │   Communication  │
│   Registry   │    │     Registry     │
└──────────────┘    └──────────────────┘
```

保存为 `architecture.png` 并在文档中引用：

```markdown
![架构图](images/architecture.png)
```

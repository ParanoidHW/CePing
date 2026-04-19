# LLM Performance Evaluator 文档

本文档介绍 LLM Performance Evaluator 的架构设计、理论基础和自定义扩展方法。

## 文档目录

### 整体架构

| 文档 | 内容 |
|------|------|
| [architecture.md](./architecture.md) | 系统整体架构 v5.0（分层设计、核心模块、数据流、扩展点） |

### 理论基础

| 文档 | 内容 |
|------|------|
| [roofline_model.md](./roofline_model.md) | Roofline 性能模型（运算强度、脊点、瓶颈判断） |
| [kernel_modeling.md](./kernel_modeling.md) | Kernel 建模方法（GEMM、FlashAttention、通信Kernel） |
| [communication.md](./communication.md) | 通信建模与并行策略（TP/PP/DP/EP/SP） |
| [topology.md](./topology.md) | 网络拓扑模型（Clos、Fat-Tree、CloudMatrix） |
| [data_sources_wiki.md](./data_sources_wiki.md) | 数据来源汇总（硬件参数、公式来源、模型架构） |

### 自定义扩展

| 文档 | 内容 |
|------|------|
| [kernel_api.md](./kernel_api.md) | Kernel Functional API 使用指南 |
| [examples.md](./examples.md) | 典型使用场景示例 |

## 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Application Layer                                │
│  Evaluator · Optimizer · ColocateAnalyzer                           │
├─────────────────────────────────────────────────────────────────────┤
│                     Workload Layer (YAML配置)                        │
│  training.yaml · autoregressive-inference.yaml · diffusion-pipeline │
├─────────────────────────────────────────────────────────────────────┤
│                   Unified Modeling Layer                             │
│  ShardedTensor · ShardedModule · ParallelContext · ModuleInstance   │
├─────────────────────────────────────────────────────────────────────┤
│                     Analyzer Layer                                   │
│  UnifiedAnalyzer · PhaseResult · MFU/QPS                            │
├─────────────────────────────────────────────────────────────────────┤
│                     Kernel Layer                                     │
│  TheoryBackend · ProfilingBackend · Functional API                   │
├─────────────────────────────────────────────────────────────────────┤
│                     Hardware Layer                                   │
│  Device · Cluster · NetworkTopology                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 性能评估指标

| 指标 | 训练 | 推理 |
|------|------|------|
| **吞吐量** | tokens/sec | TTFT / TPOT / TPS |
| **利用率** | MFU (Model FLOPs Utilization) | QPS (Queries Per Second) |
| **内存** | 参数 + 梯度 + 优化器 + 激活 | 参数 + KV Cache |
| **通信** | AllReduce / AllToAll | P2P / AllGather |

## 快速开始

```python
from llm_perf import create_model_from_config, Device, Cluster, StrategyConfig
from llm_perf.analyzer import UnifiedAnalyzer, get_workload

# 1. 创建模型
model = create_model_from_config({"preset": "llama-7b"})

# 2. 配置硬件
device = Device.from_preset("H100-SXM-80GB")
cluster = Cluster(device, num_devices=8)

# 3. 设置策略
strategy = StrategyConfig(tp_degree=8)

# 4. 分析性能
analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze("training", batch_size=32, seq_len=4096)

# 5. 查看结果
print(f"MFU: {result.mfu * 100:.1f}%")
print(f"吞吐量: {result.throughput['tokens_per_sec']:.0f} tokens/s")
print(f"显存: {result.peak_memory_gb:.2f} GB")
```

## 扩展指南

### 添加新模型

1. **YAML preset方式**：创建 `configs/models/my-model.yaml`
2. **代码方式**：继承 `ShardedModule`，定义 `forward()` 方法

详见 [architecture.md](./architecture.md) 扩展点章节。

### 添加新Workload

创建 `configs/workloads/custom/my-workload.yaml`：

```yaml
name: my-workload
workload_type: training

phases:
  - name: forward
    compute_type: forward
    compute_pattern: transformer_block
    repeat: 1
```

### 添加新Kernel

继承 `KernelBackend`，注册到 `KernelBackendRegistry`。

详见 [kernel_api.md](./kernel_api.md)。
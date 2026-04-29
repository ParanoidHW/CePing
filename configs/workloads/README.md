# Workload Configurations

本目录包含负载特征配置文件，用于描述计算流程，而非模型类型。

## Workload 划分方式

采用**应用场景 + compute_mode 属性**的双重划分方式：

- **目录划分**：按应用场景划分（training、inference、rl_training 等）
- **compute_mode 属性**：描述底层计算特征（base、autoregressive、iterative、conv）

## 目录结构

```
configs/workloads/
├── training/             # compute_mode: base
│   ├── training.yaml     # 通用训练流程
│   └── denoise.yaml      # 去噪训练
│
├── rl_training/          # compute_mode: autoregressive
│   ├── rl_ppo.yaml       # PPO训练
│   └── rl_grpo.yaml      # GRPO训练
│
├── inference/            # compute_mode: autoregressive
│   ├── inference.yaml    # 通用单次推理
│   ├── autoregressive.yaml  # 自回归生成（prefill + decode）
│   └── speculative_decoding.yaml  # 投机解码
│
├── pd_disagg/            # compute_mode: autoregressive
│   # Prefill-Decode分离推理（待添加）
│
├── multimodal/           # compute_mode: autoregressive
│   # 多模态推理（待添加）
│
├── diffusion/            # compute_mode: iterative
│   ├── denoise.yaml      # 多步去噪推理
│   └── pipeline.yaml     # 完整Pipeline（encoder + backbone + decoder）
│
├── conv/                 # compute_mode: conv
│   ├── encoder.yaml      # 卷积编码
│   ├── decoder.yaml      # 卷积解码
│   └── resnet.yaml       # ResNet分类
│
└── custom/               # 用户自定义（可在此添加）
```

## compute_mode 属性说明

| compute_mode | 描述 | 典型应用 |
|--------------|------|----------|
| `base` | 基础计算（单次 forward/backward） | 通用训练、单步推理 |
| `autoregressive` | 自回归生成（prefill + decode） | LLM推理、RL训练 |
| `iterative` | 迭代去噪（多步迭代） | 扩散模型推理 |
| `conv` | 卷积计算（2D/3D 卷积） | VAE、ResNet |

## YAML格式说明

```yaml
name: <负载特征名>           # 如 training, autoregressive-inference
description: <描述>
workload_type: <training/inference/mixed>
compute_mode: <base/autoregressive/iterative/conv>  # 新增属性

component_mapping:         # 可选，用于多组件Pipeline
  <通用标识>: <用户组件名>
  # 示例：
  # encoder: text_encoder
  # backbone: dit
  # decoder: vae

phases:
  - name: <phase名>
    compute_type: <forward/backward/optimizer>
    component: <通用标识>        # main/backbone/encoder/decoder
    compute_pattern: <计算模式>   # 可选，见下文
    repeat: <int或动态参数名>
    seq_len_factor: <float或表达式>

default_params:            # 默认参数值
  batch_size: 32
  seq_len: 2048

optimizer_factor: 1.5      # 可选，optimizer开销因子
throughput_metric: <吞吐量指标>
```

## 计算模式（ComputePattern）

| Pattern | 描述 | 适用场景 |
|---------|------|----------|
| `transformer_block` | Attention + FFN | LLM、DiT等Transformer结构 |
| `conv_encoder` | 卷积编码 | VAE Encoder、ResNet |
| `conv_decoder` | 卷积解码 | VAE Decoder |
| `attention_only` | 纯注意力 | Text Encoder |
| `dense_forward` | 全连接 | MLP |

## component_mapping用法

`component_mapping` 用于将通用标识映射到用户实际组件名：

```yaml
# 配置文件中使用通用标识
component_mapping:
  encoder: text_encoder    # encoder → text_encoder
  backbone: dit            # backbone → dit
  decoder: vae             # decoder → vae

# Phase中使用通用标识
phases:
  - name: encode
    component: encoder      # 会映射到 text_encoder
```

调用时：
```python
models = {"text_encoder": encoder, "dit": dit, "vae": vae}
result = analyzer.analyze("diffusion-pipeline", models=models)
```

## 自定义workload

1. 在 `configs/workloads/custom/` 创建YAML文件
2. 使用 `load_workload("custom/my-workload.yaml")` 加载
3. 或直接传入路径：`analyzer.analyze("configs/workloads/custom/my-workload.yaml", ...)`

## 负载特征命名规范

命名应反映计算特点，而非模型类型：

- ✓ `autoregressive-inference` - 描述自回归生成特点
- ✓ `iterative-denoise` - 描述迭代去噪特点
- ✓ `training` - 通用训练流程
- ✗ `llm-training` - 包含模型类型（错误）
- ✗ `diffusion-inference` - 包含模型类型（错误）
# 模型评估过程 Wiki

本文档详细描述 CePing 项目支持的各个模型的评估过程，包括架构概述、子模块分解、权重计算、FLOPs 计算、内存分析和切分策略。

---

## 目录

1. [LLaMA 模型](#1-llama-模型)
2. [DeepSeek V3 模型](#2-deepseek-v3-模型)
3. [Wan DiT 模型](#3-wan-dit-模型)
4. [Qwen3.5 模型](#4-qwen35-模型)

---

## 1. LLaMA 模型

### 1.1 架构概述

LLaMA 是 Meta AI 开发的开源大语言模型系列，采用标准 Transformer 架构。

| 模型 | 参数量 | Hidden Size | Layers | Heads | KV Heads | Intermediate Size |
|------|--------|-------------|--------|-------|----------|-------------------|
| LLaMA-7B | 7B | 4096 | 32 | 32 | 32 | 11008 |
| LLaMA-13B | 13B | 5120 | 40 | 40 | 40 | 13824 |
| LLaMA-70B | 70B | 8192 | 80 | 64 | 8 (GQA) | 28672 |

**关键特性**:
- **GQA (Grouped Query Attention)**: LLaMA-70B 使用 8 KV heads 而非 64，减少 KV cache 内存
- **RMSNorm**: 使用 RMS Normalization而非 LayerNorm
- **SwiGLU FFN**: Gated FFN 激活函数
- **RoPE**: Rotary Position Embedding

**与其他模型区别**:
- 标准 Transformer 架构，无特殊压缩或稀疏化
- 相比 DeepSeek，无 MLA 和 MoE
- 相比 Qwen3.5，无 Linear Attention

### 1.2 子模块分解

LLaMA 模型结构: `Embedding → N × TransformerBlock → Final RMSNorm → LM Head`

每个 TransformerBlock 包含:

| 子模块 | 类型 | 参数 | 切分方式 |
|--------|------|------|----------|
| input_norm | RMSNorm | `hidden_size` | 无切分 |
| attention | Standard Attention | Q/K/V/O weights | TP (heads维度) |
| post_attn_norm | RMSNorm | `hidden_size` | 无切分 |
| ffn | SwiGLU FFN | gate/up/down weights | TP (intermediate维度) |

**Attention 详细参数**:
```
q_weight: (hidden_size, num_heads × head_dim)
k_weight: (hidden_size, num_kv_heads × head_dim)
v_weight: (hidden_size, num_kv_heads × head_dim)
o_weight: (num_heads × head_dim, hidden_size)
```

**FFN 详细参数 (SwiGLU)**:
```
gate_weight: (hidden_size, intermediate_size)
up_weight: (hidden_size, intermediate_size)
down_weight: (intermediate_size, hidden_size)
```

### 1.3 权重计算

#### Embedding 权重
```
params_embedding = vocab_size × hidden_size
```

#### TransformerBlock 权重
```
# Attention
params_q = hidden_size × num_heads × head_dim
params_k = hidden_size × num_kv_heads × head_dim
params_v = hidden_size × num_kv_heads × head_dim
params_o = num_heads × head_dim × hidden_size
params_attn = params_q + params_k + params_v + params_o

# FFN (SwiGLU)
params_ffn = 3 × hidden_size × intermediate_size

# Norms
params_norms = 2 × hidden_size

# 每层总参数
params_layer = params_attn + params_ffn + params_norms
```

#### 总参数量
```
params_total = params_embedding + num_layers × params_layer + hidden_size + hidden_size × vocab_size

# LLaMA-7B 示例
vocab_size = 32000
hidden_size = 4096
num_layers = 32
num_heads = 32
num_kv_heads = 32
head_dim = 128
intermediate_size = 11008

params_embedding = 32000 × 4096 = 131.07M
params_attn = 4096 × 32 × 128 × 4 = 67.11M
params_ffn = 3 × 4096 × 11008 = 135.27M
params_layer = 67.11M + 135.27M + 8.19M = 210.57M
params_total = 131.07M + 32 × 210.57M + 4.09M + 131.07M = ~6.74B
```

### 1.4 FLOPs 计算

#### Prefill 阶段 (forward)

```
# Attention FLOPs
flops_qkv_proj = 3 × batch × seq_len × hidden_size × num_heads × head_dim
flops_attention = 4 × batch × num_heads × seq_len × seq_len × head_dim  # QK^T + @V
flops_o_proj = batch × seq_len × num_heads × head_dim × hidden_size

# FFN FLOPs (SwiGLU)
flops_gate_proj = batch × seq_len × hidden_size × intermediate_size
flops_up_proj = batch × seq_len × hidden_size × intermediate_size
flops_down_proj = batch × seq_len × intermediate_size × hidden_size
flops_activation = batch × seq_len × intermediate_size × 6  # silu + elementwise mul

# 每层总 FLOPs
flops_layer = flops_attn + flops_ffn
flops_total = num_layers × flops_layer + embedding_flops + lm_head_flops
```

#### Decode 阶段 (生成一个 token)

Decode 阶段 seq_len = 1, kv_seq_len = prompt_len + generated_len

```
# Attention (生成 1 个 token)
flops_qkv = 3 × batch × 1 × hidden_size × num_heads × head_dim
flops_attention = 4 × batch × num_heads × 1 × kv_seq_len × head_dim
flops_o = batch × 1 × num_heads × head_dim × hidden_size

# FFN (不变)
flops_ffn = 3 × batch × 1 × hidden_size × intermediate_size
```

**关键差异**: Decode 阶段 attention 计算量与 KV cache 长度成正比，而非 O(seq²)

### 1.5 内存分析

#### 权重内存
```
memory_weights = params_total × dtype_size

# LLaMA-7B fp16
memory_weights = 6.74B × 2 bytes = ~13.48 GB
```

#### KV Cache 内存
```
# Standard Attention
memory_kv_per_layer = 2 × batch × seq_len × num_kv_heads × head_dim × dtype_size
memory_kv_total = num_layers × memory_kv_per_layer

# LLaMA-7B, batch=1, seq=4096
memory_kv = 32 × 2 × 1 × 4096 × 32 × 128 × 2 = ~8.6 GB
```

#### Activation 内存 (训练)

Activation 内存取决于 backward pass 需要保存的中间结果:

| 子模块 | 需保存的张量 | 说明 |
|--------|-------------|------|
| Attention | q_proj, k_proj, v_proj, output | q, k, v 是 view，不保存 |
| FFN | gate_proj, up_proj, intermediate, output | silu 输出需保存 |
| RMSNorm | input | backward 需要输入计算梯度 |

```
memory_activation = batch × seq × (hidden_size × 4 + intermediate_size × 4) × num_layers
```

### 1.6 切分策略

#### TP (Tensor Parallelism)

```python
ctx = ParallelContext(tp_degree=8)

# Attention TP切分
q_weight.shardable = {1: "tp"}  # heads维度切分
k_weight.shardable = {1: "tp"}
v_weight.shardable = {1: "tp"}
o_weight.shardable = {0: "tp"}  # row切分，需要AllReduce

# FFN TP切分
gate_weight.shardable = {1: "tp"}
up_weight.shardable = {1: "tp"}
down_weight.shardable = {0: "tp"}  # row切分，需要AllReduce
```

**切分后参数量 (TP=8)**:
```
params_per_gpu = params_total / 8
memory_per_gpu = memory_weights / 8 + memory_kv / 8 + memory_activation
```

#### PP (Pipeline Parallelism)

将 layers 分配到不同 stage:
```
layers_per_stage = num_layers / pp_degree
```

#### DP (Data Parallelism)

复制整个模型到多个 GPU，每个 GPU 处理不同 batch:
```
effective_batch_size = batch_size × dp_degree
```

### 1.7 评估示例

#### 配置
```python
model = create_model_from_config({"preset": "llama-7b"})
device = Device.from_preset("H100-SXM-80GB")
cluster = Cluster.create_homogeneous(device.config, num_devices=8)
strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)
```

#### 训练评估
```python
analyzer = TrainingAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze(batch_size=32, seq_len=4096)

# 输出
print(f"吞吐量: {result.tokens_per_sec:.1f} tokens/s")
print(f"MFU: {result.mfu:.2%}")
print(f"显存/GPU: {result.memory_per_gpu_gb:.2f} GB")
```

#### 推理评估
```python
analyzer = InferenceAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze(batch_size=8, prompt_len=1024, generation_len=128)

# 输出
print(f"TTFT: {result.prefill_time_sec * 1000:.1f} ms")
print(f"TPOT: {result.decode_time_per_step_sec * 1000:.1f} ms")
print(f"TPS: {result.decode_tokens_per_sec:.1f} tokens/s")
```

---

## 2. DeepSeek V3 模型

### 2.1 架构概述

DeepSeek V3 是 DeepSeek AI 开发的 MoE 大语言模型，采用 MLA 和 DeepSeekMoE 架构。

| 参数 | 值 |
|------|-----|
| 总参数量 | 671B |
| 活跃参数量 | 37B (per token) |
| Hidden Size | 7168 |
| Layers | 61 |
| Heads | 128 |
| KV Heads | 128 |
| Experts | 256 |
| Active Experts | 8 (top-8 routing) |
| KV LoRA Rank | 512 |

**关键特性**:
- **MLA (Multi-head Latent Attention)**: KV cache 压缩，减少内存占用
- **DeepSeekMoE**: 256 routed experts + shared expert，每个 token 激活 top-8
- **First-k Dense**: 第1层使用 dense FFN，其余层使用 MoE

**与其他模型区别**:
- 相比 LLaMA: MLA vs 标准 Attention，MoE vs Dense FFN
- 相比 Qwen3.5: 无 Linear Attention，MoE 结构相同但参数更大

### 2.2 子模块分解

DeepSeek V3 结构: `Embedding → DenseLayer + 60 × MoELayer → Final RMSNorm → LM Head`

#### Dense Layer (第 1 层)
```
input_norm → Attention (MLA) → post_attn_norm → FFN (SwiGLU)
```

#### MoE Layer (第 2-61 层)
```
input_norm → Attention (MLA) → post_attn_norm → MoE (Routed + Shared)
```

#### MLA 详细结构

| 权重 | Shape | 说明 |
|------|-------|------|
| kv_down_weight | `(hidden_size, kv_lora_rank)` | KV 压缩投影 |
| k_up_weight | `(kv_lora_rank, num_kv_heads × qk_nope_head_dim)` | K 解压缩 |
| v_up_weight | `(kv_lora_rank, num_kv_heads × v_head_dim)` | V 解压缩 |
| q_down_weight | `(hidden_size, q_lora_rank)` | Q 压缩 (可选) |
| q_up_weight | `(q_lora_rank, num_heads × qk_nope_head_dim)` | Q 解压缩 |
| q_rope_weight | `(hidden_size, num_heads × qk_rope_head_dim)` | Q RoPE 部分 |
| k_rope_weight | `(hidden_size, num_kv_heads × qk_rope_head_dim)` | K RoPE 部分 |
| o_weight | `(num_heads × v_head_dim, hidden_size)` | Output 投影 |

#### MoE 详细结构

| 权重 | Shape | 说明 |
|------|-------|------|
| router_weight | `(hidden_size, num_experts)` | Router 投影 |
| expert_gate_weight | `(num_experts, hidden_size, intermediate_size)` | Expert gate |
| expert_up_weight | `(num_experts, hidden_size, intermediate_size)` | Expert up |
| expert_down_weight | `(num_experts, intermediate_size, hidden_size)` | Expert down |
| shared_gate_weight | `(hidden_size, shared_intermediate)` | Shared expert gate |
| shared_up_weight | `(hidden_size, shared_intermediate)` | Shared expert up |
| shared_down_weight | `(shared_intermediate, hidden_size)` | Shared expert down |

### 2.3 权重计算

#### MLA 权重
```
# KV 压缩路径
params_kv_down = hidden_size × kv_lora_rank
params_k_up = kv_lora_rank × num_kv_heads × qk_nope_head_dim
params_v_up = kv_lora_rank × num_kv_heads × v_head_dim

# Q 路径
params_q_down = hidden_size × q_lora_rank
params_q_up = q_lora_rank × num_heads × qk_nope_head_dim

# RoPE 部分
params_q_rope = hidden_size × num_heads × qk_rope_head_dim
params_k_rope = hidden_size × num_kv_heads × qk_rope_head_dim

# Output
params_o = num_heads × v_head_dim × hidden_size

params_mla_total = 上述所有参数之和
```

#### MoE 权重
```
# Routed experts
params_router = hidden_size × num_experts
params_expert = num_experts × (hidden_size × intermediate_size × 2 + intermediate_size × hidden_size)
              = num_experts × 3 × hidden_size × intermediate_size

# Shared expert
params_shared = 3 × hidden_size × shared_intermediate_size

params_moe_total = params_router + params_expert + params_shared
```

#### DeepSeek V3 参数量
```
hidden_size = 7168
num_layers = 61
num_heads = 128
kv_lora_rank = 512
intermediate_size = 2048 (expert)
shared_intermediate_size = 2048
num_experts = 256

# MLA params per layer
params_mla ≈ 7168 × 512 × 3 + 512 × 128 × 128 × 2 + 7168 × 128 × 64 × 2 + 128 × 128 × 7168
           ≈ 11M + 16.8M + 117M + 118M ≈ ~263M

# MoE params per layer
params_moe = 7168 × 256 + 256 × 3 × 7168 × 2048 + 3 × 7168 × 2048
           = 1.8M + 11B + 44M ≈ ~11B

# 总参数量
params_total = embedding + 61 × (params_mla + params_moe) + lm_head
             ≈ 129280 × 7168 + 61 × (263M + 11B) + 7168 × 129280
             ≈ ~671B
```

### 2.4 FLOPs 计算

#### MLA Attention FLOPs

MLA 的计算流程:
1. KV 压缩: `hidden → latent_kv` (kv_lora_rank)
2. KV 解压缩: `latent_kv → k, v`
3. Q 路径: `hidden → latent_q → q`
4. Attention 计算

```
# KV 路径
flops_kv_down = batch × seq × hidden_size × kv_lora_rank
flops_k_up = batch × seq × kv_lora_rank × num_kv_heads × qk_nope_head_dim
flops_v_up = batch × seq × kv_lora_rank × num_kv_heads × v_head_dim

# Q 路径
flops_q_down = batch × seq × hidden_size × q_lora_rank
flops_q_up = batch × seq × q_lora_rank × num_heads × qk_nope_head_dim

# RoPE
flops_q_rope = batch × seq × hidden_size × num_heads × qk_rope_head_dim
flops_k_rope = batch × seq × hidden_size × num_kv_heads × qk_rope_head_dim

# Attention
flops_attn = 4 × batch × num_heads × seq × seq × head_dim

# Output
flops_o = batch × seq × num_heads × v_head_dim × hidden_size
```

#### MoE FLOPs

MoE 的计算流程:
1. Router: 计算每个 token 的 expert 路由
2. Expert 计算: top-8 experts + shared expert

```
# Router
flops_router = batch × seq × hidden_size × num_experts

# Routed experts (top-8)
flops_per_expert = batch × seq × hidden_size × intermediate_size × 2 + intermediate_size × hidden_size
flops_experts = num_experts_per_token × flops_per_expert

# Shared expert
flops_shared = batch × seq × hidden_size × shared_intermediate × 2 + shared_intermediate × hidden_size

# MoE total
flops_moe = flops_router + flops_experts + flops_shared
```

**注意**: MoE FLOPs 只计算 active experts (top-8)，而非全部 256 experts

### 2.5 内存分析

#### MLA KV Cache (关键优势)

MLA 通过 KV 压缩大幅减少 KV cache 内存:

```
# Standard Attention KV cache
memory_kv_standard = 2 × batch × seq × num_kv_heads × head_dim × dtype_size

# MLA KV cache (compressed)
memory_kv_mla = batch × seq × kv_lora_rank × dtype_size

# 压缩比
compression_ratio = (num_kv_heads × head_dim) / kv_lora_rank
                 = (128 × 192) / 512
                 = 48x
```

DeepSeek V3 的 KV cache 内存减少 ~48x，这对于长序列推理非常关键。

#### 权重内存
```
memory_weights = 671B × 2 bytes = ~1342 GB (fp16)

# TP=8 切分后
memory_per_gpu = 1342 / 8 = ~168 GB (超出 80GB H100)
```

需要 EP (Expert Parallelism) + TP 组合切分。

### 2.6 切分策略

#### EP (Expert Parallelism)

MoE 的 experts 切分到不同 GPU:
```
experts_per_gpu = num_experts / ep_degree

# EP=8 切分
experts_per_gpu = 256 / 8 = 32 experts per GPU
```

EP 切分需要 AllToAll 通信:
```
# AllToAll 通信量
comm_bytes = batch × seq × hidden_size × dtype_size × 2 (send + receive)
```

#### TP + EP 组合

```python
ctx = ParallelContext(tp_degree=4, ep_degree=8)

# MLA weights TP切分
k_up_weight.shardable = {1: "tp"}
v_up_weight.shardable = {1: "tp"}
q_up_weight.shardable = {1: "tp"}
o_weight.shardable = {0: "tp"}

# MoE weights EP + TP切分
expert_gate_weight.shardable = {0: "ep", 2: "tp"}
expert_up_weight.shardable = {0: "ep", 2: "tp"}
expert_down_weight.shardable = {0: "ep", 1: "tp"}
```

### 2.7 评估示例

#### 配置
```python
model = create_model_from_config({"preset": "deepseek-v3"})
device = Device.from_preset("H100-SXM-80GB")
cluster = Cluster.create_homogeneous(device.config, num_devices=64)
strategy = StrategyConfig(tp_degree=8, ep_degree=8, num_experts=256)
```

#### 训练评估
```python
analyzer = TrainingAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze(batch_size=16, seq_len=8192)

print(f"吞吐量: {result.tokens_per_sec:.1f} tokens/s")
print(f"MoE 活跃参数: {result.active_params_per_token / 1e9:.1f}B")
print(f"EP 通信占比: {result.ep_communication_ratio:.1f}%")
```

#### MLA KV Cache 评估
```python
# 对比 Standard vs MLA KV cache
seq_len = 163840  # DeepSeek V3 支持 160K context

kv_standard = 2 * batch * seq_len * 128 * 192 * 2
kv_mla = batch * seq_len * 512 * 2

print(f"Standard KV: {kv_standard / 1024**3:.2f} GB")
print(f"MLA KV: {kv_mla / 1024**3:.2f} GB")
print(f"压缩比: {kv_standard / kv_mla:.1f}x")
```

---

## 3. Wan DiT 模型

### 3.1 架构概述

Wan (Wan2.1) 是视频生成模型，采用 DiT (Diffusion Transformer) 架构。

| 参数 | 值 |
|------|-----|
| Hidden Size | 5120 |
| Layers | 40 |
| Heads | 40 |
| Intermediate Size | 13824 |
| In/Out Channels | 16 (latent) |
| Patch Size | (1, 2, 2) |

**关键特性**:
- **Patchify**: 3D Conv 将视频 latent 转换为 patches
- **Self-Attention**: Spatial-temporal attention
- **Cross-Attention**: Text conditioning (来自 T5 encoder)
- **GELU FFN**: 使用 GELU 激活而非 SwiGLU
- **Modulation**: Time-dependent modulation 参数

**与其他模型区别**:
- 相比 LLaMA: DiT 架构，无 causal mask，使用 LayerNorm
- 相比 DeepSeek: 无 MoE，使用 standard attention
- 相比 Qwen3.5: Diffusion 模型，而非 LLM

### 3.2 子模块分解

Wan DiT 结构: `Patchify → TimeEmbed → 40 × DiTBlock → FinalNorm → Unpatchify`

每个 DiTBlock 包含:

| 子模块 | 类型 | 说明 |
|--------|------|------|
| norm1 | LayerNorm | 无 affine weight |
| self_attn_qkv | Q/K/V weights | Self-attention |
| self_attn_o | O weight | Self-attention output |
| norm2 | LayerNorm | Cross-attention 前的 norm |
| cross_attn_q | Q weight | Cross-attention query |
| cross_attn_kv | KV weight | Cross-attention key-value (from text) |
| cross_attn_o | O weight | Cross-attention output |
| norm3 | LayerNorm | FFN 前的 norm |
| ffn | GELU FFN | 2 weights (up, down) |

**特殊点**:
- Self-attention 使用 Q/K norm (RMSNorm)
- Cross-attention 的 KV 来自 text embedding (4096 → 2×5120)
- Modulation 参数: 6 × hidden_size (shift/scale/gate for self-attn + FFN)

### 3.3 权重计算

#### Patchify 权重
```
params_patchify = in_channels × hidden_size × pt × ph × pw
               = 16 × 5120 × 1 × 2 × 2
               = 327,680
```

#### DiTBlock 权重
```
# Self-attention
params_self_q = hidden_size × hidden_size
params_self_k = hidden_size × hidden_size
params_self_v = hidden_size × hidden_size
params_self_o = hidden_size × hidden_size
params_self_norms = 2 × hidden_size  # q_norm, k_norm

# Cross-attention
params_cross_q = hidden_size × hidden_size
params_cross_kv = text_dim × 2 × hidden_size  # text_dim=4096
params_cross_o = hidden_size × hidden_size

# FFN (GELU, non-gated)
params_up = hidden_size × intermediate_size
params_down = intermediate_size × hidden_size

# Per block total
params_block = 4 × 5120² + 2 × 5120 + 5120² + 4096 × 10240 + 5120² + 5120 × 13824 + 13824 × 5120
            ≈ ~105M + 10K + 26M + 52M + 71M + 71M
            ≈ ~225M

# 40 blocks total
params_blocks = 40 × 225M ≈ ~9B
```

#### Time Embedding 权重
```
params_time_in = freq_dim × hidden_size
params_time_out = hidden_size × hidden_size
params_time_proj = hidden_size × 6 × hidden_size
```

### 3.4 FLOPs 计算

视频生成是 diffusion 过程，每一步包含:
1. Patchify: 3D Conv
2. Time Embedding: MLP
3. DiT Blocks: Self-attention + Cross-attention + FFN
4. Unpatchify: Linear projection

#### Patchify FLOPs
```
# 3D Conv
batch = 1
num_frames = 81  (latent frames)
height = 90  (latent height, 720/8)
width = 160 (latent width, 1280/8)
seq_len = num_frames × height × width = 81 × 90 × 160 = 1,166,400

flops_patchify = batch × seq_len × in_channels × hidden_size × pt × ph × pw
              = 1 × 1.17M × 16 × 5120 × 4
              ≈ 240 GFLOPs
```

#### DiT Block FLOPs

```
# Self-attention
flops_self_qkv = 3 × batch × seq_len × hidden_size²
flops_self_attn = 4 × batch × heads × seq_len² × head_dim
flops_self_o = batch × seq_len × hidden_size²

# Cross-attention (text_len = 512)
text_len = 512
flops_cross_q = batch × seq_len × hidden_size²
flops_cross_kv = batch × text_len × text_dim × 2 × hidden_size
flops_cross_attn = 4 × batch × heads × seq_len × text_len × head_dim
flops_cross_o = batch × seq_len × hidden_size²

# FFN (GELU)
flops_ffn = batch × seq_len × hidden_size × intermediate_size × 2 + batch × seq_len × intermediate_size × hidden_size

# Per block total
flops_block ≈ 3 × seq × 5120² + 4 × 40 × seq² × 128 + seq × 5120² + ...
```

**注意**: Self-attention 是 O(seq²)，seq = 1.17M 时计算量极大。这是视频生成慢的主要原因。

#### Diffusion 步数

视频生成通常需要 50 步 diffusion:
```
total_flops = num_steps × flops_per_step
            = 50 × (flops_patchify + 40 × flops_block + flops_unpatchify)
```

### 3.5 内存分析

#### Activation 内存

DiT 的 activation 内存主要来自:
- Self-attention: seq² × heads × dtype_size (attention scores)
- Cross-attention: seq × text_len × heads × dtype_size
- FFN intermediate: seq × intermediate_size × dtype_size

```
# Self-attention activation (最关键)
memory_self_attn = batch × heads × seq_len² × dtype_size
               = 1 × 40 × (1.17M)² × 2
               ≈ 110 GB (远超 80GB H100)

# 需要 FlashAttention 优化
```

FlashAttention 将 attention scores 保持在 SRAM，避免 HBM 爆炸。

#### 权重内存
```
memory_weights = params_total × dtype_size
              = 14B × 2 bytes
              ≈ 28 GB
```

### 3.6 切分策略

#### TP (Tensor Parallelism)

```python
ctx = ParallelContext(tp_degree=8)

# Self-attention TP切分
self_attn_qkv.q_weight.shardable = {1: "tp"}
self_attn_qkv.k_weight.shardable = {1: "tp"}
self_attn_qkv.v_weight.shardable = {1: "tp"}
self_attn_qkv.o_weight.shardable = {0: "tp"}

# Cross-attention TP切分
cross_attn_q_weight.shardable = {1: "tp"}
cross_attn_kv_weight.shardable = {1: "tp"}
cross_attn_o_weight.shardable = {0: "tp"}

# FFN TP切分
ffn.up_weight.shardable = {1: "tp"}
ffn.down_weight.shardable = {0: "tp"}
```

#### Sequence Parallelism (SP)

对于长序列 (seq > 1M)，需要 SP 切分:
```
seq_per_gpu = seq_len / sp_degree
```

SP 切分可以:
- 减少 self-attention 的内存压力
- 减少 communication overhead

### 3.7 评估示例

#### 配置
```python
model = create_model_from_config({"preset": "wan-dit"})
device = Device.from_preset("H100-SXM-80GB")
cluster = Cluster.create_homogeneous(device.config, num_devices=8)
strategy = StrategyConfig(tp_degree=8)
```

#### Diffusion 推理评估
```python
analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze_diffusion(
    num_frames=81,
    height=720,
    width=1280,
    num_steps=50,
    use_cfg=True,  # Classifier-Free Guidance (2x compute)
)

print(f"总时间: {result.total_time_sec:.2f} s")
print(f"每步时间: {result.step_time_sec:.2f} s")
print(f"峰值显存: {result.peak_memory_gb:.2f} GB")
```

---

## 4. Qwen3.5 模型

### 4.1 架构概述

Qwen3.5 是阿里巴巴开发的大语言模型系列，包含 Dense 和 MoE 两种架构，均采用 Hybrid Attention。

#### Dense 模型规格

| 模型 | Hidden | Layers | Heads | KV Heads | Intermediate | tie_embed | 参数量 |
|------|--------|--------|-------|----------|--------------|-----------|--------|
| 0.8B | 1024 | 24 | 8 | 2 | 3584 | ✅ | 0.9B |
| 2B | 2048 | 24 | 8 | 2 | 6144 | ✅ | 2B |
| 4B | 2560 | 32 | 16 | 4 | 9216 | ✅ | 5B |
| 9B | 4096 | 32 | 16 | 4 | 12288 | ❌ | 10B |
| 27B | 5120 | 64 | 24 | 4 | 17408 | ❌ | 28B |

#### MoE 模型规格

| 模型 | Hidden | Layers | Heads | Experts | Top_K | MoE Intermediate | 总参数 | 活跃参数 |
|------|--------|--------|-------|---------|-------|------------------|--------|----------|
| 35B-A3B | 2048 | 40 | 16 | 256 | 8 | 512 | 35B | 3B |
| 122B-A10B | 3072 | 48 | 32 | 256 | 8 | 1024 | 122B | 10B |
| 397B-A17B | 4096 | 60 | 32 | 512 | 10 | 1024 | 397B | 17B |

**关键特性**:
- **Hybrid Attention**: 3 linear + 1 full per 4-layer cycle
- **Linear Attention**: O(seq) 复杂度，无 KV cache，kernel_dim=4
- **Dense FFN**: SwiGLU (gate_proj + up_proj + down_proj)
- **MoE FFN**: 256/512 routed experts + 1 shared expert
- **tie_word_embeddings**: Small models share embedding/lm_head weights
- **Vision Encoder**: 支持多模态，不同规格配置不同

**与其他模型区别**:
- 相比 LLaMA: Hybrid Attention vs Standard，Dense/MoE vs Dense
- 相比 DeepSeek: Linear Attention vs MLA，MoE 结构相似
- 相比 Wan: LLM vs DiT，有 causal mask

#### Dense vs MoE 架构对比

| 特性 | Dense 模型 | MoE 模型 |
|------|-----------|----------|
| FFN 类型 | SwiGLU FFN | MoE (routed + shared experts) |
| 参数效率 | 固定激活 | 稀疏激活，活跃参数 << 总参数 |
| 推理计算 | 全参数 | 只计算 top-k experts |
| 内存占用 | 固定 | 权重更大，但激活内存小 |
| 训练成本 | 高 | 低（稀疏激活） |

### 4.2 子模块分解

#### Dense 模型结构

```
Embedding → N × ShardedQwen3_5DenseBlock → Final RMSNorm → LM Head

ShardedQwen3_5DenseBlock:
  input_norm → (Linear/Full) Attention → post_attn_norm → SwiGLU FFN
```

#### MoE 模型结构

```
Embedding → N × ShardedQwen3_5MoEBlock → Final RMSNorm → LM Head

ShardedQwen3_5MoEBlock:
  input_norm → (Linear/Full) Attention → post_attn_norm → MoE FFN
```

#### Hybrid Attention 详细结构

每 4 层为一个 cycle：3 层 linear_attention + 1 层 full_attention

```
# layer_types 生成逻辑
pattern = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
num_cycles = num_layers // 4
layer_types = pattern * num_cycles + pattern[:remainder]
```

**Linear Attention**:
- kernel_dim = 4 (feature map 维度)
- 状态大小固定 = `kernel_dim × head_dim` per head
- 无 KV cache，O(seq) 复杂度

**Full Attention**:
- 标准 Attention，使用 GQA (num_kv_heads < num_heads)
- KV cache = `2 × batch × seq × num_kv_heads × head_dim × dtype_size`

### 4.3 权重计算

#### Dense 模型权重计算

**Embedding 权重**:
```
params_embedding = vocab_size × hidden_size
```

**SwiGLU FFN 权重**:
```
FFN_weight = (hidden_size × intermediate_size × 2) + (intermediate_size × hidden_size)
           = 3 × hidden_size × intermediate_size

# Qwen3.5-9B 示例
hidden_size = 4096
intermediate_size = 12288
params_ffn = 3 × 4096 × 12288 = 151M
```

**Hybrid Attention 权重** (linear + full 混合):
```
# Linear Attention (每层的 3/4)
params_linear_q = hidden_size × linear_num_heads × linear_key_head_dim
params_linear_k = hidden_size × linear_num_kv_heads × linear_key_head_dim
params_linear_v = hidden_size × linear_num_kv_heads × linear_value_head_dim
params_linear_o = linear_num_heads × linear_value_head_dim × hidden_size

# Full Attention (每层的 1/4)
params_full_q = hidden_size × num_heads × head_dim
params_full_k = hidden_size × num_kv_heads × head_dim
params_full_v = hidden_size × num_kv_heads × head_dim
params_full_o = num_heads × head_dim × hidden_size
```

**tie_word_embeddings 影响**:
```
# tie=True: embedding 和 lm_head 共享权重
params_lm_head = 0 (共享)
params_total -= vocab_size × hidden_size

# tie=False: 独立权重
params_lm_head = vocab_size × hidden_size
```

**Dense 模型总参数量公式**:
```
params_total = params_embedding + params_layers + params_norms + params_lm_head

# Qwen3.5-9B 示例 (tie=False)
vocab_size = 248320
hidden_size = 4096
num_layers = 32
num_heads = 16
num_kv_heads = 4
head_dim = 256
intermediate_size = 12288

params_embedding = 248320 × 4096 = 1.02B
params_ffn = 3 × 4096 × 12288 = 151M
params_attn = (3/4 × linear_attn + 1/4 × full_attn) × num_layers
params_layers ≈ 32 × (linear_attn + ffn)
params_lm_head = 1.02B
params_total ≈ 10B
```

#### MoE 模型权重计算

**MoE FFN 权重**:
```
params_router = hidden_size × num_experts
params_experts = num_experts × 3 × hidden_size × moe_intermediate_size
params_shared = 3 × hidden_size × shared_intermediate

params_moe = params_router + params_experts + params_shared
```

**不同规格对比**:

| 参数 | 35B-A3B | 122B-A10B | 397B-A17B |
|------|---------|-----------|-----------|
| moe_intermediate_size | 512 | 1024 | 1024 |
| num_experts | 256 | 256 | 512 |
| top_k | 8 | 8 | 10 |
| params_router | 0.52M | 0.78M | 2.1M |
| params_experts | 0.8B | 2.4B | 12.6B |
| params_shared | 3.15M | 9.2M | 12.3M |

**MoE 参数量公式**:
```
Total = Embedding + Layers × (Attention + MoE_experts + Shared_expert) + LM_head

MoE_experts = num_experts × moe_intermediate × hidden × 2 + moe_intermediate × hidden
            = num_experts × 3 × hidden × moe_intermediate

Active = top_k × (MoE_experts / num_experts) + Shared_expert
       = top_k × 3 × hidden × moe_intermediate + 3 × hidden × shared_intermediate
```

**35B-A3B 示例**:
```
hidden_size = 2048
num_layers = 40
num_experts = 256
moe_intermediate = 512
shared_intermediate = 512
top_k = 8

params_embedding = 248320 × 2048 = 0.51B
params_router = 2048 × 256 = 0.52M
params_experts = 256 × 3 × 2048 × 512 = 0.8B
params_shared = 3 × 2048 × 512 = 3.15M
params_moe = 0.52M + 0.8B + 3.15M ≈ 0.81B

params_layers = 40 × (27M + 0.81B) ≈ 33.6B
params_lm_head = 0.51B
params_total ≈ 34.6B

# Active params per token
active_experts = 8 × 3 × 2048 × 512 = 25M
active_shared = 3 × 2048 × 512 = 3.15M
active_total ≈ 28M + attention ≈ 3B
```

### 4.4 Vision Encoder 配置

Qwen3.5 支持多模态，不同规格的 Vision Encoder 配置不同：

| 模型 | Vision Depth | Vision Hidden | out_hidden_size |
|------|--------------|---------------|-----------------|
| 0.8B | 12 | 768 | 1024 |
| 2B | 24 | 1024 | 2048 |
| 4B | 24 | 1024 | 2560 |
| 9B | 27 | 1152 | 4096 |
| 27B | 27 | 1152 | 5120 |
| 35B-A3B | 27 | 1152 | 2048 |
| 122B-A10B | 27 | 1152 | 3072 |
| 397B-A17B | 27 | 1152 | 4096 |

**Vision Encoder 权重**:
```
params_vision = vision_depth × (attention_params + ffn_params)
attention_params = 4 × vision_hidden²
ffn_params = vision_hidden × vision_intermediate × 3
```

### 4.5 FLOPs 计算

#### Linear Attention FLOPs

```
# Projections
flops_qkv = 3 × batch × seq × hidden × linear_heads × linear_head_dim

# Kernel feature map
flops_kernel = batch × seq × linear_heads × kernel_dim × linear_head_dim × 2

# Output projection
flops_o = batch × seq × linear_heads × linear_head_dim × hidden

# Total: O(seq) 无 seq² 项
```

**关键优势**: Decode 阶段计算量不随 KV cache 长度增长。

#### Full Attention FLOPs

```
flops_attn = 4 × batch × heads × seq × seq × head_dim
```

#### Dense SwiGLU FFN FLOPs

```
flops_ffn = batch × seq × (hidden × intermediate × 2 + intermediate × hidden)
          = 3 × batch × seq × hidden × intermediate
```

#### MoE FFN FLOPs

```
flops_router = batch × seq × hidden × num_experts
flops_active_experts = top_k × 3 × batch × seq × hidden × moe_intermediate
flops_shared = 3 × batch × seq × hidden × shared_intermediate

flops_moe = flops_router + flops_active_experts + flops_shared
```

**注意**: MoE FLOPs 只计算 top-k active experts，而非全部 experts。

### 4.6 内存分析

#### Linear Attention 内存 (关键优势)

Linear Attention 无 KV cache，状态大小固定：
```
# Standard Attention KV cache (随 seq 增长)
memory_kv_standard = 2 × batch × seq × kv_heads × head_dim × dtype × layers

# Linear Attention state (固定大小)
memory_linear_state = heads × kernel_dim × head_dim × dtype
                   = 16 × 4 × 128 × 2 = 16 KB per head
                   ≈ 0.5 MB total (固定)
```

**对比**:
- seq=4096: Standard KV ≈ 1.3 GB, Linear state = 0.5 MB
- seq=100K: Standard KV ≈ 32 GB, Linear state = 0.5 MB

#### Full Attention KV cache

只有 1/4 层使用 full attention：
```
memory_kv_full = (num_layers / 4) × 2 × batch × seq × kv_heads × head_dim × dtype
```

#### Dense 模型权重内存

```
# Qwen3.5-9B (tie=False)
memory_weights = 10B × 2 bytes = 20 GB

# Qwen3.5-2B (tie=True)
memory_weights = 2B × 2 bytes = 4 GB
```

#### MoE 模型权重内存

```
# 35B-A3B
memory_weights = 35B × 2 bytes = 70 GB
memory_per_gpu (TP=8) = 70 / 8 ≈ 9 GB (可行)

# 397B-A17B
memory_weights = 397B × 2 bytes = 794 GB
memory_per_gpu (TP=8, EP=64) = 794 / 512 ≈ 1.5 GB per expert shard
```

### 4.7 切分策略

#### Dense 模型切分

```python
ctx = ParallelContext(tp_degree=8)

# Attention TP切分
q_weight.shardable = {1: "tp"}
k_weight.shardable = {1: "tp"}
v_weight.shardable = {1: "tp"}
o_weight.shardable = {0: "tp"}

# SwiGLU FFN TP切分
gate_weight.shardable = {1: "tp"}
up_weight.shardable = {1: "tp"}
down_weight.shardable = {0: "tp"}
```

**TP 建议**:
- 0.8B/2B: TP=1 (单卡可行)
- 4B: TP=2
- 9B: TP=2-4
- 27B: TP=4-8

#### MoE 模型切分 (TP + EP)

```python
ctx = ParallelContext(tp_degree=8, ep_degree=8)

# Attention TP切分
q_weight.shardable = {1: "tp"}
k_weight.shardable = {1: "tp"}
v_weight.shardable = {1: "tp"}
o_weight.shardable = {0: "tp"}

# MoE EP + TP切分
expert_gate_weight.shardable = {0: "ep", 2: "tp"}
expert_up_weight.shardable = {0: "ep", 2: "tp"}
expert_down_weight.shardable = {0: "ep", 1: "tp"}
shared_expert.shardable = {1: "tp"}  # shared expert 用 TP
```

**切分建议**:

| 模型 | TP | EP | 总 GPU |
|------|----|----|--------|
| 35B-A3B | 4-8 | 2-4 | 8-32 |
| 122B-A10B | 8 | 8-16 | 64-128 |
| 397B-A17B | 8 | 32-64 | 256-512 |

### 4.8 评估示例

#### Dense 模型配置

```python
# Qwen3.5-9B
model = Qwen3_5Model(
    vocab_size=248320,
    hidden_size=4096,
    num_layers=32,
    num_heads=16,
    num_kv_heads=4,
    head_dim=256,
    intermediate_size=12288,
    tie_word_embeddings=False,
)

device = Device.from_preset("H100-SXM-80GB")
cluster = Cluster.create_homogeneous(device.config, num_devices=4)
strategy = StrategyConfig(tp_degree=4)
```

#### MoE 模型配置

```python
# Qwen3.5-35B-A3B
model = Qwen3_5MoEModel(
    vocab_size=248320,
    hidden_size=2048,
    num_layers=40,
    num_heads=16,
    num_kv_heads=2,
    head_dim=256,
    linear_num_heads=16,
    linear_num_kv_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    intermediate_size=512,
    num_experts=256,
    num_experts_per_token=8,
    shared_expert_intermediate=512,
)

device = Device.from_preset("H100-SXM-80GB")
cluster = Cluster.create_homogeneous(device.config, num_devices=16)
strategy = StrategyConfig(tp_degree=8, ep_degree=2)
```

#### Qwen3.5-122B-A10B 配置

```python
model = Qwen3_5MoEModel(
    vocab_size=248320,
    hidden_size=3072,
    num_layers=48,
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    intermediate_size=1024,
    num_experts=256,
    num_experts_per_token=8,
    shared_expert_intermediate=1024,
)

strategy = StrategyConfig(tp_degree=8, ep_degree=16)
```

#### Qwen3.5-397B-A17B 配置

```python
model = Qwen3_5MoEModel(
    vocab_size=248320,
    hidden_size=4096,
    num_layers=60,
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    intermediate_size=1024,
    num_experts=512,
    num_experts_per_token=10,  # top-10 routing
    shared_expert_intermediate=1024,
)

strategy = StrategyConfig(tp_degree=8, ep_degree=64)
```

#### 训练评估

```python
analyzer = TrainingAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze(batch_size=32, seq_len=4096)

print(f"吞吐量: {result.tokens_per_sec:.1f} tokens/s")
print(f"MFU: {result.mfu:.2%}")
print(f"Linear Attention layers: {num_layers * 3 // 4}")
print(f"Full Attention layers: {num_layers // 4}")
```

#### 长序列推理评估

```python
analyzer = InferenceAnalyzer(model, device, cluster, strategy)

# 对比 Linear vs Full Attention KV cache
prompt_len = 100000  # 100K context

result = analyzer.analyze(batch_size=1, prompt_len=prompt_len, generation_len=100)

print(f"TTFT: {result.prefill_time_sec * 1000:.1f} ms")
print(f"Linear Attention 内存: 固定 ~0.5 MB")
print(f"Full Attention KV cache: {(num_layers // 4) * 2 * 1 * prompt_len * kv_heads * head_dim * 2 / 1024**2:.2f} MB")
```

#### MTP 推测解码评估

```python
model_with_mtp = Qwen3_5MoEModel(
    vocab_size=248320,
    hidden_size=2048,
    num_layers=40,
    num_heads=16,
    num_experts=256,
    mtp_num_layers=1,
    mtp_share_embeddings=True,
)

# MTP 可以并行生成多个 token，提高吞吐
result = analyzer.analyze_with_mtp(batch_size=8, prompt_len=1024, generation_len=128)
print(f"MTP throughput gain: {result.throughput_gain:.1f}%")
```

---

## 5. 模型对比总结

### 5.1 Attention 类型对比

| 模型 | Attention 类型 | KV Cache | 内存复杂度 | Decode 计算复杂度 |
|------|---------------|----------|-----------|------------------|
| LLaMA | Standard (GQA) | 标准 KV | O(seq) | O(seq × head_dim) |
| DeepSeek V3 | MLA | 压缩 KV (512) | O(seq) | O(seq × kv_lora_rank) |
| Wan DiT | Standard | 无 KV cache | O(seq²) scores | N/A (diffusion) |
| Qwen3.5 | Hybrid (Linear + Full) | Linear 固定 + Full O(seq) | Fixed + O(seq) | O(1) + O(seq) |

### 5.2 FFN 类型对比

| 模型 | FFN 类型 | 参数量 | 激活参数 | 激活函数 |
|------|----------|--------|----------|----------|
| LLaMA | Dense SwiGLU | 固定 | 固定 | SiLU + gate |
| DeepSeek V3 | MoE (256+1) | 671B total | 37B active | SiLU + gate |
| Wan DiT | Dense GELU | 固定 | 固定 | GELU |
| Qwen3.5-Dense | SwiGLU | 0.9B-28B | 固定 | SiLU + gate |
| Qwen3.5-MoE | MoE (256/512+1) | 35B-397B total | 3B-17B active | SiLU + gate |

### 5.3 切分策略对比

| 模型 | TP | EP | PP | SP/CP | 特殊切分 |
|------|----|----|----|-------|----------|
| LLaMA | 支持 | N/A | 支持 | 支持 | GQA heads |
| DeepSeek V3 | 支持 | 支持 | 支持 | 支持 | MLA KV compressed, experts EP |
| Wan DiT | 支持 | N/A | 支持 | 必需 (长序列) | Patchify TP |
| Qwen3.5 | 支持 | 支持 | 支持 | 支持 | Linear state 固定, experts EP |

### 5.4 内存对比 (batch=1, seq=4096, fp16)

| 模型 | 权重内存 | KV Cache 内存 | Activation 内存 (训练) |
|------|----------|---------------|----------------------|
| LLaMA-7B | 13.5 GB | 8.6 GB | ~4 GB |
| DeepSeek V3 | 1342 GB (需要多卡) | 0.5 GB (MLA compressed) | ~10 GB |
| Wan DiT-14B | 28 GB | N/A | ~110 GB (需要 FlashAttn) |
| Qwen3.5-9B | 20 GB | 0.5 MB + 0.13 GB | ~2 GB |
| Qwen3.5-27B | 56 GB | 0.5 MB + 0.3 GB | ~5 GB |
| Qwen3.5-35B-A3B | 70 GB | 0.5 MB + 0.2 GB | ~2 GB |
| Qwen3.5-397B-A17B | 794 GB (需要多卡) | 0.5 MB + 0.3 GB | ~3 GB |

---

## 6. 评估流程总结

### 6.1 通用评估流程

```python
# 1. 创建模型
model = create_model_from_config({"preset": "model-name"})

# 2. 配置硬件
device = Device.from_preset("H100-SXM-80GB")
cluster = Cluster.create_homogeneous(device.config, num_devices=8)

# 3. 设置策略
strategy = StrategyConfig(tp_degree=8, ep_degree=2)

# 4. 创建分析器
analyzer = TrainingAnalyzer(model, device, cluster, strategy)

# 5. 执行评估
result = analyzer.analyze(batch_size=32, seq_len=4096)

# 6. 输出结果
print(f"吞吐量: {result.tokens_per_sec:.1f} tokens/s")
print(f"MFU: {result.mfu:.2%}")
print(f"显存: {result.memory_per_gpu_gb:.2f} GB")
```

### 6.2 关键评估指标

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| MFU | Model FLOPs Utilization | 训练 |
| tokens_per_sec | Token 吞吐量 | 训练/推理 |
| TTFT | Time To First Token | 推理 (prefill) |
| TPOT | Time Per Output Token | 推理 (decode) |
| TPS | Tokens Per Second | 推理 (decode) |
| peak_memory_gb | 峰值显存占用 | 训练/推理 |

### 6.3 分解分析

```python
# 子模块分解
breakdown = result.breakdown

# 计算分解
print(f"Attention FLOPs: {breakdown['attention_flops'] / 1e12:.2f}T")
print(f"FFN FLOPs: {breakdown['ffn_flops'] / 1e12:.2f}T")

# 内存分解
print(f"权重内存: {breakdown['param_memory_gb']:.2f} GB")
print(f"激活内存: {breakdown['activation_memory_gb']:.2f} GB")
print(f"KV cache: {breakdown['kv_cache_memory_gb']:.2f} GB")

# 通信分解
print(f"AllReduce: {breakdown['allreduce_bytes'] / 1e9:.2f} GB")
print(f"AllToAll: {breakdown['alltoall_bytes'] / 1e9:.2f} GB")
```

---

## 参考资料

- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [DeepSeek V3 Paper](https://arxiv.org/abs/2412.19437)
- [Wan Paper](https://arxiv.org/abs/2503.20314)
- [Qwen3.5 Technical Report](https://qwenlm.github.io/blog/qwen2.5/)
- [Linear Attention](https://arxiv.org/abs/2006.16236)
- [CePing Architecture](architecture.md)
- [Kernel API](kernel_api.md)
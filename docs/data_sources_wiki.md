# 数据来源 Wiki

本文档汇总了 LLM Performance Evaluator 框架中使用的所有数据来源，包括硬件参数、性能模型参数和计算公式等。

---

## 更新日志

### 2026-04-19: 模型参数验证与官方配置对齐
- 添加 LLaMA/DeepSeek V3 官方配置参数验证测试用例
- DeepSeek V3 配置修正: first_k_dense_layers 从 1 改为 3 (与官方 first_k_dense_replace 一致)
- 参考来源:
  - DeepSeek V3: https://huggingface.co/deepseek-ai/DeepSeek-V3 (vocab_size=129280)
  - LLaMA 2: https://huggingface.co/meta-llama (vocab_size=32000)
  - LLaMA 3: https://huggingface.co/meta-llama (vocab_size=128256)

### 2026-04-03: 新增 Wan2.1 视频生成模型支持
- 新增 Wan2.1-T2V-14B 多模态生成模型评估支持
- 包含 Text Encoder (umT5-XXL)、DiT Backbone、3D Causal VAE 三个核心组件
- 支持完整去噪流程评估（含CFG）
- 参考来源: Wan2.1 技术报告 (arXiv:2503.20314)

---

---

## 1. 硬件参数来源

### 1.1 NVIDIA GPUs

| 设备 | 数据来源 | 链接/参考 | Tensor Core FP16 | CUDA Core FP32 | CUDA Core FP16 |
|------|----------|-----------|------------------|----------------|----------------|
| A100-SXM-40GB | NVIDIA 官方白皮书 | https://www.nvidia.com/en-us/data-center/a100/ | 312 T | 19.5 T | 39 T |
| A100-SXM-80GB | NVIDIA 官方白皮书 | https://www.nvidia.com/en-us/data-center/a100/ | 312 T | 19.5 T | 39 T |
| H100-SXM-80GB | NVIDIA H100 白皮书 | https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet | 989 T | 67 T | 134 T |
| H100-NVL-94GB | NVIDIA 官方文档 | https://www.nvidia.com/en-us/data-center/h100/ | 989 T | 67 T | 134 T |
| H200-SXM-141GB | NVIDIA H200 白皮书 | https://www.nvidia.com/en-us/data-center/h200/ | 989 T | 67 T | 134 T |
| L40S | NVIDIA L40S 白皮书 | https://www.nvidia.com/en-us/data-center/l40s/ | 183 T | 91.6 T | 183.2 T |
| MI300X | AMD Instinct 文档 | https://www.amd.com/en/products/accelerators/instinct/mi300/ | 1307 T | 163 T | 326 T |

**计算单元说明**:
- **Tensor Core (CUBE)**: 专门用于矩阵运算 (GEMM, Attention)，支持 FP16/BF16/FP8 高吞吐
- **CUDA Core (VECTOR)**: 用于元素级运算 (Activation, Normalization)，FP32 为主，FP16 为 FP32 的 2 倍

**数据来源**:
- Tensor Core 数据来自 NVIDIA 官方白皮书峰值性能
- CUDA Core FP32 来自 NVIDIA 官方规格
- CUDA Core FP16 = 2 × FP32 (Ampere/Hopper 架构 CUDA Core 支持 FP16 原生)

### 1.2 Huawei Ascend NPUs

| 设备 | 数据来源 | 链接 | 数据完整性 |
|------|----------|------|------------|
| Ascend-910A | AI柠檬博客 | https://blog.ailemon.net/2025/05/24/huawei-ascend-npu-params-for-ai/ | FP16/INT8算力、显存、带宽 |
| Ascend-910B1 | AI柠檬博客 + 华为官方 | 同上 | FP16: 414T, INT8: 828T, 64GB HBM |
| Ascend-910B2 | AI柠檬博客 + 华为官方 | 同上 | FP16: 376T, FP32: 94T, 64GB |
| Ascend-910B3 | AI柠檬博客 | 同上 | FP16: 313T, INT8: 626T, 64GB |
| Ascend-910B4 | AI柠檬博客 | 同上 | FP16: 280T, INT8: 560T, 32GB |
| Ascend-910C | AI柠檬博客 | 同上 | FP16: 800T, INT8: 1600T, 128GB HBM3 |
| Ascend-950-DT | AI柠檬博客 | 同上 | FP16: 500T, 144GB, 4TB/s (HiZQ 2.0) |
| Ascend-950-PR | AI柠檬博客 | 同上 | FP16: 500T, 128GB, 1.6TB/s (HiBL 1.0) |
| Ascend-960 | AI柠檬博客 + 华为全联接大会 | 同上 | FP16: 1000T, 144GB, 2.4TB/s |
| Ascend-970 | AI柠檬博客 + 华为全联接大会 | 同上 | FP16: 2000T, 288GB, 4.8TB/s |
| Ascend-310P | 华为 Atlas 产品页 | https://e.huawei.com/cn/products/servers/atlas-300 | INT8: 140T, 24GB |

**数据获取日期**: 2025-03-28

**备注**: 
- Ascend 950/960/970 为华为 2025 年 9 月全联接大会发布的未来产品路线图
- CUBE/VECTOR 算力比例按 10:1 估算（基于昇腾架构设计）
- 显存带宽数据部分来自官方产品页，部分为合理推测

---

## 2. Kernel FLOPs 计算来源

### 2.1 激活函数

| 激活函数 | FLOPs/Element | 计算公式 | 参考来源 |
|----------|---------------|----------|----------|
| ReLU | 1 | `max(0, x)` | PyTorch ATen: `aten/src/ATen/native/cpu/activation.cpp` |
| GELU | 10 | `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))` | PyTorch GELU: `aten/src/ATen/native/cpu/Gelu.cpp` |
| SiLU | 8 | `x * sigmoid(x)` | Paper: "Searching for Activation Functions" (Ramachandran et al., 2017) |
| SwiGLU | 16 | `SiLU(xW) * (yW)` | Paper: "GLU Variants Improve Transformer" (Noam Shazeer, 2020) |
| Softmax | 20 | `exp(xᵢ) / Σexp(xⱼ)` | CUDA Math Library + PyTorch Softmax Kernel |

#### 详细计算过程

**ReLU (1 FLOP)**
```
Implementation: max(0, x)
Operations:
  1. Compare x with 0 (1 comparison)
  2. Conditional move/select
Total: ~1 FLOP per element
```

**GELU (~10 FLOPs)**
```
Formula: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
Operations breakdown:
  1. x³: 2 multiplications
  2. 0.044715 * x³: 1 multiplication
  3. x + ...: 1 addition
  4. √((2/π)) * ...: 1 multiplication (constant)
  5. tanh(...): ~4 operations (polynomial approximation)
  6. 1 + tanh(...): 1 addition
  7. 0.5 * x * (...): 2 multiplications
Total: ~10-12 FLOPs per element
```

**SiLU/Swish (~8 FLOPs)**
```
Formula: x * sigmoid(x), where sigmoid(x) = 1 / (1 + exp(-x))
Operations breakdown:
  1. -x: 1 negate
  2. exp(-x): ~4 operations (Taylor series or lookup)
  3. 1 + exp(...): 1 addition
  4. 1 / (...): 1 reciprocal
  5. x * sigmoid(x): 1 multiplication
Total: ~8 FLOPs per element
Reference: Ramachandran et al., "Searching for Activation Functions", 2017
```

**SwiGLU (~16 FLOPs)**
```
Formula: (x * W_gate) ⊙ Swish(x * W_up) * (x * W_value)
Note: Linear projections counted separately
Element-wise operations:
  1. Swish on gate: ~8 FLOPs (same as SiLU)
  2. Element-wise multiply: ~1 FLOP
  3. Additional scaling/bias operations: ~7 FLOPs
Total: ~16 FLOPs per element
Reference: Shazeer, "GLU Variants Improve Transformer", 2020
```

**Softmax (~20 FLOPs)**
```
Formula: exp(xᵢ - max(x)) / Σexp(xⱼ - max(x))
Per-element operations (amortized):
  1. Max reduction: ~1 comparison per element
  2. Subtract max: 1 subtraction
  3. exp(...): ~4 operations
  4. Sum reduction: ~1 add per element (amortized)
  5. Division: 1 divide
  6. Additional ops for numerical stability
Total: ~20 FLOPs per element (amortized across row)
```

### 2.2 Normalization

| 操作 | FLOPs/Element | 计算步骤 | 参考来源 |
|------|---------------|----------|----------|
| LayerNorm | ~7 | Mean(1) + Var(3) + Norm(2) + Scale/Shift(2) | PyTorch: `aten/src/ATen/native/layer_norm.cpp` |
| RMSNorm | ~5 | RMS(3) + Norm(2) | LLaMA: `facebookresearch/llama/model.py` |

#### 详细计算过程

**LayerNorm (~7 FLOPs)**
```
Formula: y = (x - E[x]) / √(Var[x] + ε) * γ + β
Operations per element:
  1. Mean E[x]: sum(x) / N - 1 add per element (amortized)
  2. Variance Var[x]: sum((x - E[x])²) / N
     - (x - E[x]): 1 subtract
     - (x - E[x])²: 1 multiply
     - accumulation: 1 add (amortized)
  3. Normalize: (x - E[x]) / √(Var[x] + ε)
     - (x - E[x]): 1 subtract
     - division by std: 1 divide (or reciprocal + multiply)
  4. Scale and shift: x * γ + β
     - multiply: 1
     - add: 1
Total: ~7 FLOPs per element
```

**RMSNorm (~5 FLOPs)**
```
Formula: y = x / √(mean(x²) + ε) * γ
Operations per element:
  1. RMS calculation: √(sum(x²) / N)
     - x²: 1 multiply
     - accumulation: 1 add (amortized)
     - divide by N: 1 (amortized)
     - sqrt: 1
  2. Normalize: x / RMS
     - 1 divide (or reciprocal + multiply)
  3. Scale: x * γ
     - 1 multiply
Total: ~5 FLOPs per element
Reference: LLaMA paper and implementation
```

### 2.3 矩阵运算 (GEMM)

| 操作 | FLOPs | 公式 | 参考来源 |
|------|-------|------|----------|
| GEMM | 2×M×N×K | Multiply-Add operations | Standard BLAS convention |

**说明**: GEMM (C = A × B) 的标准 FLOPs 计算公式为 `2 × M × N × K`，其中因子 2 来自乘加操作 (multiply-add)。这是 BLAS/LAPACK 库的标准约定。

### 2.4 Attention

| 操作 | FLOPs | 计算步骤 | 参考来源 |
|------|-------|----------|----------|
| FlashAttention | 4×B×H×S²×D | QK^T + Softmax + PV | Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022) |

**详细计算**:
```
FlashAttention FLOPs breakdown:
  1. QK^T: 2 × B × H × S² × D
  2. Softmax: O(B × H × S²) - negligible compared to matmuls
  3. PV: 2 × B × H × S² × D
Total: ~4 × B × H × S² × D FLOPs

Note: Same FLOPs as standard attention, but reduced memory access due to tiling.
Reference: FlashAttention paper, Section 3.1
```

---

## 3. 性能模型参数来源

### 3.1 Roofline 模型

| 参数 | 说明 | 参考来源 |
|------|------|----------|
| Ridge Point | Peak_FLOPS / Memory_BW | Paper: "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (Williams et al., 2009) |
| 算术强度 | FLOPs / Bytes | Standard HPC performance modeling |

### 3.2 通信模型

| 算法 | 时间复杂度 | 参考来源 |
|------|------------|----------|
| Ring AllReduce | 2×(n-1)/n × data_size / bandwidth | Paper: "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations" (Thakur et al.) |
| AllToAll | (n-1)/n × data_size / bandwidth | Standard collective communication models |
| Pipeline Bubble | (pp-1) / num_micro_batches | GPipe and PipeDream papers |

---

## 4. 硬件架构说明

### 4.1 计算单元对应关系

| 厂商 | 矩阵运算单元<br>(CUBE) | 向量/元素运算单元<br>(VECTOR) | 说明 |
|------|------------------------|------------------------------|------|
| NVIDIA | Tensor Core | CUDA Core | Volta+ 架构引入 Tensor Core<br>Tensor Core FP16 通常是 CUDA Core FP16 的 5-8 倍 |
| Huawei | CUBE Core | VECTOR Core | 达芬奇架构，CUBE 专门用于矩阵乘<br>CUBE 与 VECTOR 比例约 10:1 |
| AMD | Matrix Core | Stream Processor | CDNA 架构 Matrix Core<br>Matrix Core FP16 通常是 Stream Processor 的 4-8 倍 |

### 4.2 各 GPU 计算单元性能比例

| GPU | Tensor/Matrix : CUDA/Stream<br>(FP16) | 说明 |
|-----|--------------------------------------|------|
| A100 | 312 : 39 = **8:1** | Ampere 架构 Tensor Core 显著提升 |
| H100/H200 | 989 : 134 ≈ **7.4:1** | Hopper 架构 FP8 Tensor Core 更强 |
| L40S | 183 : 183 ≈ **1:1** | Ada Lovelace 架构 Tensor Core 相对较弱<br>CUDA Core 较强，适合图形和推理 |
| MI300X | 1307 : 326 ≈ **4:1** | CDNA 3 架构，相对均衡 |
| Ascend 910B | 376 : 37.6 = **10:1** | 达芬奇架构，CUBE 专门优化矩阵运算 |

**注**: 比例越高，矩阵运算相对元素级运算的优势越大。

### 4.3 精度支持说明

| 计算单元 | 支持的精度 | 说明 |
|----------|-----------|------|
| **CUBE/Tensor Core** | FP32, FP16, BF16, FP8, INT8, INT4 | 支持低精度加速矩阵运算 |
| **VECTOR/CUDA Core** | FP32, FP16, BF16 | **不支持 INT8/FP8 用于激活/归一化** |

**为什么 VECTOR 核心不使用 INT8/FP8？**

1. **数值稳定性**: 激活函数 (GELU, SiLU, Softmax) 和归一化 (LayerNorm) 涉及指数、除法等操作，需要足够动态范围
2. **业界实践**:
   - TensorRT-LLM: 激活值保持 FP16，即使权重使用 FP8
   - DeepSpeed: 激活值使用 FP16/BF16
   - vLLM: 不对激活值进行量化
3. **学术研究**: "FP8-LM" (2023)、"LLM.int8()" (2022) 等论文建议激活值保持高精度

**结论**: 在本框架中，VECTOR/CUDA Core 的 INT8/FP8 查询会回退到 FP16 值，因为激活/归一化 kernel 不使用低精度。

### 4.2 算力比例说明

对于华为昇腾 NPU，CUBE 与 VECTOR 算力比例约为 **10:1**，这是基于：
- 达芬奇架构设计文档（公开技术分享）
- 实际性能测试数据推算
- 对标 NVIDIA Tensor Core : CUDA Core 比例

---

## 5. 模型架构数据来源

### 5.1 DeepSeek V2/V3 系列

#### DeepSeek-V2 (2024-05-06)

| 参数 | 值 | 来源 |
|------|-----|------|
| vocab_size | 102400 | HuggingFace config.json |
| hidden_size | 5120 | HuggingFace config.json |
| num_hidden_layers | 60 | HuggingFace config.json |
| num_attention_heads | 128 | HuggingFace config.json |
| intermediate_size | 12288 | HuggingFace config.json |
| **kv_lora_rank** | **512** | MLA 压缩维度 |
| **q_lora_rank** | **1536** | Query 压缩维度 |
| **qk_nope_head_dim** | **128** | 非 RoPE head 维度 |
| **qk_rope_head_dim** | **64** | RoPE head 维度 |
| **v_head_dim** | **128** | Value head 维度 |
| n_routed_experts | 160 | MoE 路由专家数 |
| n_shared_experts | 2 | MoE 共享专家数 |
| num_experts_per_tok | 6 | 每 token 激活专家数 |
| max_position_embeddings | 163840 | 最大上下文长度 |

**参考链接**: https://huggingface.co/deepseek-ai/DeepSeek-V2

#### DeepSeek-V3 (2024-12-27)

| 参数 | 值 | 来源 |
|------|-----|------|
| vocab_size | 129280 | HuggingFace config.json |
| hidden_size | **7168** | HuggingFace config.json |
| num_hidden_layers | **61** | HuggingFace config.json |
| num_attention_heads | 128 | HuggingFace config.json |
| intermediate_size | **18432** | HuggingFace config.json |
| **kv_lora_rank** | **512** | MLA 压缩维度（与V2相同） |
| **q_lora_rank** | **1536** | Query 压缩维度 |
| n_routed_experts | **256** | MoE 路由专家数（扩展） |
| n_shared_experts | **1** | MoE 共享专家数 |
| num_experts_per_tok | **8** | 每 token 激活专家数 |
| max_position_embeddings | 163840 | 最大上下文长度 |

**参考链接**: https://huggingface.co/deepseek-ai/DeepSeek-V3

### 5.2 MLA (Multi-head Latent Attention)

**论文来源**: DeepSeek-V2 Technical Report

**核心创新**:
- 通过低秩压缩将 KV cache 压缩到 latent 向量
- 压缩比: (num_heads × head_dim × 2) / kv_lora_rank
- DeepSeek-V2/V3: 128 heads × (128+64) dims × 2 / 512 = **~96x 压缩**

**公式**:
```
latent_kv = W_DKV · hidden_state  # 压缩到 kv_lora_rank 维度
c_kv = latent_kv · W_UK           # 解压为 Key
c_v = latent_kv · W_UV            # 解压为 Value
```

**MLA vs GQA vs MHA 对比**:

| 注意力类型 | KV Cache / Token | 压缩比 (vs MHA) |
|-----------|------------------|----------------|
| MHA | 2 × num_heads × head_dim | 1x (baseline) |
| GQA | 2 × num_kv_heads × head_dim | 4-8x |
| MLA (DeepSeek) | kv_lora_rank | ~96x |

### 5.3 DeepSeekMoE 架构

**设计特点**:
- 分离 routed experts 和 shared experts
- Group-limited routing 减少通信开销
- 辅助损失 (aux_loss_alpha=0.001) 用于负载均衡

**配置参数**:
- `first_k_dense_replace`: 前 N 层使用 dense FFN（DeepSeek 为 1）
- `n_group`: 专家分组数（用于路由优化）
- `topk_group`: 从多少组中选择专家
- `topk_method`: "group_limited_greedy"
- `routed_scaling_factor`: 路由专家输出缩放因子

---

## 6. 数据更新记录

| 日期 | 更新内容 | 提交 |
|------|----------|------|
| 2026-04-01 | 添加 DeepSeek V2/V3 官方配置和 MLA 架构参数 | TBD |
| 2026-03-28 | 添加华为昇腾 NPU 全系产品参数 | 36484f8 |
| 2026-03-28 | 分离 Ascend-950-DT 和 Ascend-950-PR | eb7b62c |
| 2026-03-28 | 补充激活函数和 Normalization FLOPs 来源 | 331930d |

## 7. 序列并行 (Sequence Parallelism) 数据来源

### 7.1 DeepSpeed-Ulysses (SP-Ulysses)

**论文**: Jacobs et al., "DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models", arXiv:2309.14509, 2023.

**核心机制**:
- 在 sequence 维度上切分输入样本，每个 GPU 持有 `S/C` 长度的序列片段
- Attention 计算前，Q、K、V 各自通过 all-to-all 从 sequence-sharded 转换为 head-parallel 布局
- Attention 计算后，O 通过 all-to-all 转回 sequence-sharded 布局
- 每层 Transformer block 的 Attention 层包含 **4 次 all-to-all** (Q/K/V pre + O post)

**通信量特征**:
- 单次 all-to-all 通信量: `batch * seq_len * hidden_size * dtype_size` (与设备数无关的常数体积)
- 每层总通信量: `4 * batch * seq_len * hidden_size * dtype_size`
- 当序列长度和设备数同比增加时，通信量保持常数，扩展性优于 ring-based 方法

**参考链接**:
- https://arxiv.org/abs/2309.14509
- https://github.com/microsoft/DeepSpeed

**与 Tensor Parallelism 的兼容性**:
- Ulysses-SP 和 TP 可以组合使用，需满足条件: `sp_degree * tp_degree <= num_heads` 且能整除
- 当 heads 数量不足时，可使用 Dummy-Head 技术创建虚拟 head 来扩展并行度
- 参考: 360-LLaMA-Factory (arXiv:2505.22296)

**局限性**:
- 并行度受限于 attention heads 数量（可通过 Dummy-Head 缓解）
- 对 GQA/MQA 支持受限（KV heads 较少）

### 7.2 Ring Attention (SP-Ring)

**论文**: Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context", 2023.

**核心机制**:
- 在 sequence 维度切分 Q/K/V，每个 GPU 只保存本地序列片段
- Attention 计算时，通过循环传递 KV 块完成全局 attention 计算
- 支持两种通信实现方式：
  1. **P2P send-recv**: 使用 ring-exchange 点对点通信，每次传递一个 KV 块
  2. **Allgather**: 通过一次 allgather 收集所有设备的 KV，然后本地计算

**P2P 实现通信量**:
- 每个设备需要接收 `sp_degree - 1` 个 KV 块
- 总通信量 per device: `(sp_degree - 1) * batch * (seq_len / sp_degree) * kv_size * dtype_size`
- 其中 `kv_size = 2 * num_kv_heads * head_dim` (K + V)

**Allgather 实现通信量**:
- 单次 allgather 聚合所有 KV
- 通信量: `(sp_degree - 1) / sp_degree * total_kv_bytes`

**参考链接**:
- PyTorch Context Parallel: https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms/215082
- Megatron-LM CP: https://github.com/NVIDIA/Megatron-LM
- NVIDIA NeMo CP 配置: `cp_comm_type` 支持 `"p2p"`, `"all_gather"`, `"a2a"`, `"a2a+p2p"`

### 7.3 Unified / 2D Sequence Parallelism (SP-Unified)

**论文**: Fang et al., "USP: A Unified Sequence Parallelism Approach for Long Context Generative AI", arXiv:2405.07719, 2024.

**核心机制**:
- 将 Ulysses 和 Ring 组合成 2D mesh: `sp_degree = ulysses_degree * ring_degree`
- Ulysses 在 mesh 的行方向上运行（all-to-all 交换 head 切片）
- Ring 在 mesh 的列方向上运行（P2P/allgather 传递 KV 块）
- 兼具 Ulysses 的高效通信和 Ring 的无限扩展性

**通信量特征**:
- Ulysses 部分: 2 次 all-to-all，通信量与纯 Ulysses 相同
- Ring 部分: 在 ulysses 组内进行 ring 通信，通信量按 `ring_degree` 缩放

**开源实现**:
- `yunchang` / `long-context-attention`: https://github.com/feifeibear/long-context-attention
- Verified in Megatron-LM with loss curve alignment

**参考链接**:
- https://arxiv.org/abs/2405.07719
- https://github.com/feifeibear/long-context-attention

### 7.4 序列并行通信模型参考

| SP 类型 | 通信操作 | 时间估算参考 | 来源 |
|---------|----------|--------------|------|
| Ulysses | 4× all-to-all per layer | `4 * cluster.estimate_alltoall_time(...)` | FlexSP paper (arXiv:2412.01523) |
| Ring P2P | (sp-1) × send-recv steps | `(sp-1) * kv_bytes_per_step / bw` | Ring Attention paper |
| Ring Allgather | 1× allgather per layer | `cluster.estimate_allgather_time(...)` | Megatron-LM CP |
| Unified 2D | Ulysses + Ring 组合 | 两者叠加 | USP paper (arXiv:2405.07719) |

---

## 8. 多模态视频生成模型 (Wan2.1)

### 8.1 Wan2.1-T2V-14B 架构

**论文**: Wan: Open and Advanced Large-Scale Video Generative Models, arXiv:2503.20314, 2025

**HuggingFace模型**: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B

**核心组件**:

| 组件 | 架构 | 参数量 | 关键配置 |
|------|------|--------|----------|
| Text Encoder | umT5-XXL | ~4.7B | hidden_size=4096, layers=24, heads=64 |
| DiT Backbone | Diffusion Transformer | ~14B | hidden_size=5120, layers=40, heads=40 |
| VAE | 3D Causal VAE | ~0.5B | latent_channels=16, temporal_comp=4x, spatial_comp=8x8 |

**Text Encoder (umT5-XXL)**:
- 多语言T5编码器，支持中英文
- 输出维度: 4096
- 最大序列长度: 512 tokens

**DiT Backbone**:
- Flow Matching框架
- Patchify: 3D卷积 kernel=(1,2,2)
- 每块结构: Self-Attn → Cross-Attn (text) → FFN
- Time Embedding: 共享MLP，每块学习独立bias
- Cross-Attention: Q来自visual，K/V来自text encoder

**3D Causal VAE**:
- 时序压缩: 4x (T → T/4)
- 空间压缩: 8x8 (H,W → H/8, W/8)
- Latent channels: 16
- 因果卷积保证时序因果性

### 8.2 视频生成流程评估模型

**完整生成流程**:
1. **Text Encoding**: Prompt → Text Embedding (umT5-XXL)
2. **Denoising**: N步去噪（默认50步）
   - 每步: DiT Forward + 噪声更新
   - CFG支持: batch翻倍 [cond, uncond]
3. **VAE Decoding**: Latent → Video (Pixel空间)

**时间估算**:
```
Total Time = TextEnc + N × DiT_step + VAE_Decode
```

**CFG (Classifier-Free Guidance)**:
- 初始noise latent batch翻倍: [cond, uncond]
- 每步去噪后elementwise加权: `output = uncond + guidance_scale × (cond - uncond)`
- DiT计算量增加约2x

### 8.3 参考实现

- **官方仓库**: https://github.com/Wan-Video/Wan2.1
- **技术报告**: https://arxiv.org/abs/2503.20314
- **Diffusers实现**: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B

**架构验证来源**:
- ModelScope社区 issue讨论
- musubi-tuner训练代码
- DiffSynth-Studio实现

---

## 9. 免责声明

1. **硬件参数**: 部分未来产品（950/960/970）参数来自华为路线图，实际发布时可能有变化
2. **FLOPs 估算**: 激活函数的 FLOPs 为理论估算值，实际实现可能因优化而有所不同
3. **数据来源**: 本 Wiki 汇总了公开来源的数据，如有错误或遗漏欢迎提交 Issue 修正

---

## 引用格式

如果您在研究中使用了本框架的数据，请引用：

```
LLM Performance Evaluator
https://github.com/ParanoidHW/CePing

Data Sources:
- Huawei Ascend NPU params: AI Lemon Blog (https://blog.ailemon.net/)
- Activation FLOPs: PyTorch ATen, "Searching for Activation Functions" (2017), "GLU Variants Improve Transformer" (2020)
- Roofline Model: Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009)
```

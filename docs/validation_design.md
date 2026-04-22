# 配置有效性验证设计文档

## 1. 背景

### 1.1 问题场景
- DSv3 168GB 单卡权重超过 80GB 设备内存
- 并行度乘积不等于卡数导致切分失败

### 1.2 验证必要性
- 避免无效配置浪费评估时间
- 提前发现配置错误，减少用户困惑
- 保证评估结果可落地执行

## 2. 验证条件来源与门槛

### 2.1 并行策略验证

#### 2.1.1 分层并行策略验证（新增）

**来源**：
- DeepSeek-V3、Mixtral 等 MoE 模型的分层切分策略
- Megatron-LM、DeepSpeed 的 MoE 实现
- Expert 权重切分：`{0: "ep", 2: "tp"}` - 按 expert 维度 EP 切分，按 intermediate 维度 TP 切分

**原理**：
- Attention 层和 MoE/FFN 层可以采用独立的 TP 策略
- `tp_degree` 用于 Attention 层的 TP 切分
- `expert_tp_degree` 用于 MoE Expert 内部的 TP 切分（默认等于 `tp_degree`）
- `ep_degree` 用于 Expert 并行

**验证条件**：
- Attention 部分：`tp_degree × dp_degree × pp_degree × sp_degree = num_gpus`
- MoE 部分：`expert_tp_degree × ep_degree × dp_degree × pp_degree × sp_degree = num_gpus`
- 如果 `expert_tp_degree == tp_degree`（未设置），两条规则等价
- 如果 `expert_tp_degree != tp_degree`（分层切分），两条规则独立验证

**示例**：
- 8卡集群，Attention TP=8，MoE ETP=2×EP=4：
  - Attention: 8×1×1×1 = 8 ✅
  - MoE: 2×4×1×1×1 = 8 ✅
- 8卡集群，Attention TP=8，MoE ETP=4×EP=2：
  - Attention: 8×1×1×1 = 8 ✅
  - MoE: 4×2×1×1×1 = 8 ✅
- 8卡集群，Attention TP=4，MoE ETP=2×EP=2：
  - Attention: 4×1×1×1 = 4 ❌（不足）
  - MoE: 2×2×1×1×1 = 4 ❌（不足）

**配置建议**：
- DeepSeek-V3 典型配置（64卡）：
  - Attention: TP=64, 或 TP=32×PP=2
  - MoE: ETP=8×EP=8×DP=1, 或 ETP=4×EP=16
- Mixtral-8×7B（8卡）：
  - Attention: TP=8
  - MoE: ETP=2×EP=4（每个 TP 组内 4 个 Expert 分片）

#### 2.1.2 并行度乘积 = 卡数（已更新为分层验证）

**来源**：
- 并行切分理论：不同层可使用不同并行策略
- 参考：Megatron-LM MoE, DeepSpeed-MoE 官方文档

**门槛**（分层验证）：
- Attention 部分乘积必须精确等于集群总卡数
- MoE 部分乘积必须精确等于集群总卡数
- 不允许有余（资源浪费）或不足（切分失败）

**示例**（分层切分）：
- 8卡集群：TP=8, ETP=2, EP=4 → Attention: 8=8 ✅, MoE: 2×4=8 ✅
- 8卡集群：TP=8, ETP=4, EP=2 → Attention: 8=8 ✅, MoE: 4×2=8 ✅
- 8卡集群：TP=8, ETP=8, EP=1 → Attention: 8=8 ✅, MoE: 8×1=8 ✅（uniform TP）

#### 2.1.3 EP 与 Expert TP 关系（性能建议，已降级为 WARNING）

**来源**：
- Expert Parallelism 与 Expert Tensor Parallelism 的关系
- Expert 权重切分：`{0: "ep", 2: "tp"}`
- 参考：DeepSpeed-MoE, MegaBlocks

**门槛**（性能建议，WARNING 级别）：
- EP 不应显著超过 `expert_tp_degree × 2`
- 原 `EP ≤ TP` 约束已改为性能建议
- 分层切分场景下，EP 可以独立于 Attention TP

**示例**：
- ETP=2, EP=4 → ratio=2，正常 ✅
- ETP=2, EP=8 → ratio=4，WARNING ⚠️（可能影响性能，但不是硬性错误）
- TP=8, ETP=2, EP=16 → Attention TP=8, MoE ETP×EP=32，分层切分 ✅

**说明**：
此验证已从 ERROR 改为 WARNING，因为：
1. 分层切分场景下 EP 可以独立于 Attention TP
2. 性能可能受影响，但不会导致切分失败
3. 用户可以根据实际需求选择最优配置

#### 2.1.4 各并行度 ≥ 1

**来源**：
- 并行度代表切分数量，必须为正整数

**门槛**：
- tp_degree ≥ 1
- pp_degree ≥ 1
- dp_degree ≥ 1
- ep_degree ≥ 1
- sp_degree ≥ 1
- expert_tp_degree ≥ 1

### 2.2 模型规格验证

#### 2.2.1 vocab_size 可被 TP 整除

**来源**：
- Embedding 层按 vocab_size 维度切分
- TP 切分要求每个分片大小相等
- 参考：Megatron-LM embedding parallelism

**门槛**：
- vocab_size % tp_degree == 0

**示例**：
- vocab_size=32000, TP=8 → 32000/8=4000 tokens/GPU ✅
- vocab_size=129280, TP=8 → 129280/8=16160 tokens/GPU ✅（DSv3）
- vocab_size=32000, TP=7 → 32000/7≈4571.4 ❌（不整除）

#### 2.2.2 hidden_size 可被 TP 整除

**来源**：
- Attention、FFN 的权重按 hidden 维度切分
- 列切分（column parallel）要求 hidden_size 可被 TP 整除

**门槛**：
- hidden_size % tp_degree == 0

**示例**：
- hidden_size=4096, TP=8 → 4096/8=512/GPU ✅
- hidden_size=7168, TP=8 → 7168/8=896/GPU ✅（DSv3）
- hidden_size=4096, TP=3 → 4096/3≈1365.3 ❌

#### 2.2.3 num_heads 可被 TP 整除

**来源**：
- Attention 的 head 数量需要均匀分配到 TP 卡
- 每个 GPU 需要完整的 head 数量

**门槛**：
- num_heads % tp_degree == 0

**示例**：
- num_heads=32, TP=8 → 32/8=4 heads/GPU ✅
- num_heads=128, TP=8 → 128/8=16 heads/GPU ✅（DSv3）
- num_heads=32, TP=5 → 32/5=6.4 ❌

#### 2.2.4 intermediate_size 可被 TP 整除

**来源**：
- FFN 的 intermediate 维度按 TP 切分

**门槛**：
- intermediate_size % tp_degree == 0

**示例**：
- intermediate_size=11008, TP=8 → 11008/8=1376/GPU ✅（Llama-7B）
- intermediate_size=18432, TP=8 → 18432/8=2304/GPU ✅

#### 2.2.5 num_kv_heads 可被 TP 整除（GQA）

**来源**：
- Grouped Query Attention 的 KV head 数量
- 参考：Llama-2/3 GQA 设计

**门槛**：
- num_kv_heads % tp_degree == 0
- 特例：num_kv_heads=1 时需要特殊处理（MQA）

**示例**：
- Llama-70B: num_heads=64, num_kv_heads=8, TP=8 → 8/8=1 KV_head/GPU ✅

### 2.3 序列切分验证

#### 2.3.1 Sequence Parallelism (SP)

**来源**：
- Megatron-LM Paper (2022): "Reducing Activation Recomputation in Large Transformer Models"
- SP 将序列维度按 TP 组切分，减少激活内存

**原理**：
- SP 是 TP 的扩展，激活按 seq_len 维度切分
- 通信模式：Ring-based allgather + reduce_scatter
- SP degree 不能超过 TP degree

**验证条件**：
- sp_degree ≤ tp_degree
- seq_len % sp_degree == 0

**示例**：
- seq_len=4096, TP=8, SP=8 → 每卡 512 tokens ✅
- seq_len=8192, TP=8, SP=4 → 每卡 2048 tokens ✅
- seq_len=4096, TP=4, SP=8 → SP 超出 TP ❌

**内存影响**：
- SP 启用时激活内存减少：activation_memory /= sp_degree

#### 2.3.2 Megatron-SP

**来源**：
- Megatron-LM 官方实现
- SP_degree = TP_degree 的特殊配置

**原理**：
- Megatron-SP 是 SP 的默认配置
- 所有 TP 卡参与序列切分
- 通信效率最优

**验证条件**：
- megatron_sp_enabled = True 时：
  - sp_degree = tp_degree（强制）
  - seq_len % tp_degree == 0

**示例**：
- TP=8, SP=8, megatron_sp=True → ✅
- TP=8, SP=4, megatron_sp=True → ❌（SP 应等于 TP）

#### 2.3.3 Ring Attention

**来源**：
- Ring Attention (2023): "Ring Attention with Blockwise Transformers"
- 用于超长序列训练（seq_len > 100K）

**原理**：
- Blockwise 计算，在 TP 组内 ring 传递 KV blocks
- 内存占用固定（只存储当前 block）
- 支持 seq_len > 100K 的长序列

**验证条件**：
- ring_attention_enabled = True 时：
  - 必须启用 SP
  - block_size 需要合理设置（通常 512-2048）
  - block_size ≤ seq_len / sp_degree

**示例**：
- seq_len=100K, TP=8, block_size=2048 → 100K/8=12.5K tokens/GPU, block=2K ✅
- seq_len=50K, TP=4, block_size=512 → 50K/4=12.5K tokens/GPU, block=512 ✅

**内存影响**：
- Ring Attention 激活内存 ≈ block_size × hidden_size × batch_size
- 相比普通 SP，内存减少 10-100x

#### 2.3.4 Context Parallelism (CP)

**来源**：
- Llama 3 Technical Report
- 用于长上下文推理（seq_len > 128K）

**原理**：
- CP 独立于 TP，专门用于长上下文
- 将上下文按 CP_degree 切分到多个卡
- KV cache 按 CP 切分

**验证条件**：
- cp_degree ≥ 1
- seq_len % cp_degree == 0
- cp_degree 可以大于 tp_degree（CP 独立）

**示例**：
- seq_len=128K, CP=8 → 每卡 16K context ✅
- seq_len=200K, CP=16 → 每卡 12.5K context ✅

**内存影响**：
- KV cache 内存 /= cp_degree
- 长上下文推理必须使用 CP 或 Ring Attention

#### 2.3.5 Ulysses Sequence Parallelism

**来源**：
- Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models", arXiv:2309.14509, 2023
- 与 Megatron-SP 的区别：Ulysses 使用 all-to-all 通信，Megatron-SP 使用 all-gather + reduce-scatter

**原理**：
- 在 sequence 维度上切分输入样本，每个 GPU 持有 `seq_len / ulysses_degree` 长度的序列片段
- Attention 计算前，Q、K、V 各自通过 all-to-all 从 sequence-sharded 转换为 head-parallel 布局
- Attention 计算后，Output 通过 all-to-all 转回 sequence-sharded 布局
- 每层 Transformer block 的 Attention 层包含 **4 次 all-to-all** (Q/K/V pre-attention + O post-attention)

**验证条件**：
- ulysses_degree ≥ 1
- seq_len % ulysses_degree == 0
- 独立使用时（无 TP）：`ulysses_degree <= num_heads`
- 与 TP 组合时：`ulysses_degree * tp_degree <= num_heads` 且 `num_heads % (ulysses_degree * tp_degree) == 0`

**通信开销**：
- 单次 all-to-all 通信量：`batch_size * seq_len * hidden_size * dtype_size`（与设备数无关的常数体积）
- 每层总通信量：`4 * batch_size * seq_len * hidden_size * dtype_size`
- 当序列长度和设备数同比增加时，通信量保持常数，扩展性优于 ring-based 方法

**与 Megatron-SP 的区别**：
| 特性 | Ulysses-SP | Megatron-SP |
|------|-----------|-------------|
| 通信模式 | all-to-all | all-gather + reduce-scatter |
| 与 TP 关系 | 可独立使用 | sp_degree = tp_degree（强制绑定） |
| 通信量与 seq_len | 常数（seq_len 增加时不增） | 线性增加 |
| 并行度限制 | 受 num_heads 限制 | 无特殊限制 |

**示例**：
- seq_len=64K, Ulysses=8, num_heads=32 → 每卡 8K tokens，每卡 4 heads ✅
- seq_len=128K, Ulysses=8, TP=4, num_heads=64 → Ulysses×TP=32 ≤ 64 heads ✅
- seq_len=64K, Ulysses=16, num_heads=32 → 16 > 32 ❌（Ulysses 超过 head 数）
- seq_len=64K, Ulysses=4, TP=16, num_heads=32 → 4×16=64 > 32 ❌（Ulysses×TP 超过 head 数）

**局限性**：
- 并行度受限于 attention heads 数量（可通过 Dummy-Head 技术扩展，参考 arXiv:2505.22296）
- 对 GQA/MQA 支持受限（KV heads 较少时需要特殊处理）

#### 2.3.6 Ulysses + Ring Attention 混合（Unified 2D-SP）

**来源**：
- Fang et al., "USP: A Unified Sequence Parallelism Approach for Long Context Generative AI", arXiv:2405.07719, 2024
- 开源实现：https://github.com/feifeibear/long-context-attention

**原理**：
- 将 Ulysses 和 Ring 组合成 2D mesh：`sp_degree = ulysses_degree * ring_degree`
- Ulysses 在 mesh 的行方向上运行（all-to-all 交换 head 切片）
- Ring 在 mesh 的列方向上运行（P2P/allgather 传递 KV 块）
- 兼具 Ulysses 的高效通信和 Ring 的无限扩展性

**验证条件**：
- `sp_degree = ulysses_degree * ring_degree`
- `seq_len % sp_degree == 0`
- `ulysses_degree * tp_degree <= num_heads` 且 `num_heads % (ulysses_degree * tp_degree) == 0`
- `ulysses_degree >= 1`, `ring_degree >= 1`
- `ulysses_degree * ring_degree * tp_degree * pp_degree * dp_degree * ep_degree = num_gpus`

**通信开销叠加**：
- Ulysses 部分（每层）：
  - 4 次 all-to-all，通信量：`4 * batch_size * seq_len * hidden_size * dtype_size`
  - 时间：`4 * alltoall_time`
- Ring 部分（每层）：
  - P2P 模式：`(ring_degree - 1) * kv_block_bytes / bandwidth`
  - Allgather 模式：`allgather_time`
- 总时间：`ulysses_time + ring_time`

**示例**：
- seq_len=1M, Ulysses=4, Ring=8, num_heads=64, TP=1 → sp=32，每卡 31K tokens ✅
- seq_len=512K, Ulysses=8, Ring=4, num_heads=128, TP=4 → Ulysses×TP=32 ≤ 128 ✅
- seq_len=1M, Ulysses=16, Ring=16, num_heads=64 → Ulysses=16 > 64 heads ❌
- seq_len=256K, Ulysses=4, Ring=6 → sp=24, 256K/24≈10.67K ❌（不整除）

**内存影响**：
- 激活内存减少：`activation_memory /= sp_degree`
- Ring 部分只需存储当前 block：`block_size * hidden_size * batch_size`

#### 2.3.7 Ulysses + Ring + TP 三维混合

**来源**：
- USP 论文扩展方案
- 最复杂的长序列训练方案，同时利用三种并行

**原理**：
- Ulysses 在 head 维度切分（all-to-all）
- Ring 在 sequence 维度进一步切分（P2P/allgather）
- TP 在 hidden 维度切分（all-reduce）
- 三者正交但需要满足 head 分配约束

**验证条件**：
- `ulysses_degree * ring_degree * tp_degree <= num_gpus`
- `ulysses_degree * tp_degree <= num_heads` 且 `num_heads % (ulysses_degree * tp_degree) == 0`
- `seq_len % (ulysses_degree * ring_degree) == 0`
- `hidden_size % tp_degree == 0`
- `vocab_size % tp_degree == 0`（embedding 层）
- 各并行度优先级：TP（最高带宽）> Ulysses（node 内）> Ring（可跨 node）

**切分维度关系**：
```
总 heads 分配：
  - Ulysses heads: num_heads / ulysses_degree
  - TP heads per Ulysses rank: num_heads / (ulysses_degree * tp_degree)

序列切分：
  - Ring sequence chunks: seq_len / ring_degree
  - Ulysses further splits: seq_len / (ulysses_degree * ring_degree)

Hidden 切分：
  - TP hidden per rank: hidden_size / tp_degree
```

**示例**：
- num_heads=128, Ulysses=4, Ring=8, TP=8, seq_len=1M, hidden=8192 →
  - Ulysses×TP=32 ≤ 128 ✅
  - seq_len % 32 = 1M % 32 = 0 ✅
  - hidden % TP = 8192 % 8 = 0 ✅
- num_heads=64, Ulysses=8, TP=16 → Ulysses×TP=128 > 64 ❌
- seq_len=512K, Ulysses=4, Ring=5 → 512K % 20 = 0 ✅（若整除）

**通信开销**：
- Ulysses all-to-all：`4 * alltoall_time * num_layers`
- Ring P2P/allgather：`(ring_degree - 1) * p2p_time * num_layers`
- TP all-reduce：`2 * allreduce_time * num_layers`（forward + backward）

#### 2.3.8 Ulysses + TP 混合

**来源**：
- DeepSpeed-Ulysses 论文
- 360-LLaMA-Factory 实践（arXiv:2505.22296）

**原理**：
- Ulysses 与 TP 可以组合，两者正交但共享 head 切分约束
- Ulysses 按序列维度切分，通过 all-to-all 在 head 维度重组
- TP 按 hidden 维度切分，每个 head 内部再切分权重
- Head 分配：`num_heads = ulysses_heads * tp_heads`

**验证条件**：
- `ulysses_degree * tp_degree <= num_heads`
- `num_heads % (ulysses_degree * tp_degree) == 0`
- `seq_len % ulysses_degree == 0`
- `hidden_size % tp_degree == 0`
- `vocab_size % tp_degree == 0`
- `intermediate_size % tp_degree == 0`

**与独立 Ulysses 的区别**：
- 独立 Ulysses：`ulysses_degree <= num_heads`，无 hidden 切分
- Ulysses + TP：`ulysses_degree * tp_degree <= num_heads`，同时切分 hidden

**Dummy-Head 技术**（当 heads 不足时）：
- 通过创建虚拟 head 来扩展并行度
- 参考：360-LLaMA-Factory (arXiv:2505.22296)
- 适用场景：`num_heads < ulysses_degree * tp_degree` 时

**示例**：
- num_heads=64, hidden=4096, Ulysses=4, TP=8, seq_len=256K →
  - Ulysses×TP=32 ≤ 64 ✅
  - 每卡 heads: 64/32=2
  - seq_len 切分: 256K/4=64K per Ulysses rank
  - hidden 切分: 4096/8=512 per TP rank ✅
- num_heads=32, Ulysses=8, TP=8 → Ulysses×TP=64 > 32 ❌
  - 可用 Dummy-Head 扩展到 64 heads
- num_heads=128, hidden=8192, Ulysses=2, TP=64 →
  - Ulysses×TP=128 ≤ 128 ✅
  - hidden: 8192/64=128 per rank ✅

**通信开销对比**：
| 配置 | Ulysses 通信 | TP 通信 |
|------|-------------|---------|
| 纯 Ulysses | 4× all-to-all | 无 |
| Ulysses + TP | 4× all-to-all | 2× all-reduce |
| 额外开销 | head 切分重组 | hidden 切分同步 |

### 2.4 Pipeline 切分验证

#### 2.4.1 Virtual Pipeline Parallelism (VPP)

**来源**：
- Megatron-LM: "Interleaved Pipeline Schedule"
- 减少 pipeline bubble 的优化方案

**原理**：
- VPP 在每个物理 stage 上运行多个 virtual stages
- Interleaved schedule 减少 bubble ratio
- bubble_ratio = (pp_degree - 1) / (pp_degree × vpp_degree)

**验证条件**：
- vpp_degree ≥ 1
- num_layers % (pp_degree × vpp_degree) == 0
- num_micro_batches ≥ pp_degree × vpp_degree × 2（推荐）

**示例**：
- num_layers=32, PP=4, VPP=2 → 32/(4×2)=4 layers/virtual-stage ✅
- num_layers=80, PP=8, VPP=2 → 80/(8×2)=5 layers/virtual-stage ✅（Llama-70B）
- num_layers=61, PP=2, VPP=2 → 61/(2×2)=15.25 ❌（DSv3 不整除）

**配置建议**：
- VPP=2 时 bubble ratio 减少到 25%（相比 PP=4 的 75%）
- VPP=4 时 bubble ratio 减少到 12.5%
- 但 VPP 增加 schedule 复杂度

#### 2.4.2 num_micro_batches 验证

**来源**：
- Pipeline 并行调度理论
- 1F1B schedule 需要 micro_batches ≥ pp_degree

**原理**：
- Pipeline schedule 类型：
  - GPipe: num_micro_batches ≥ pp_degree × 2（减小 bubble）
  - 1F1B: num_micro_batches ≥ pp_degree
  - Interleaved (VPP): num_micro_batches ≥ pp_degree × vpp_degree × 2

**验证条件**：
- pipeline_schedule = "gpipe" → num_micro_batches ≥ pp_degree × 2
- pipeline_schedule = "1f1b" → num_micro_batches ≥ pp_degree
- pipeline_schedule = "interleaved" → num_micro_batches ≥ pp_degree × vpp_degree × 2

**示例**：
- PP=4, schedule="1f1b", micro_batches=4 → ✅（刚好填满）
- PP=4, schedule="gpipe", micro_batches=16 → ✅（bubble=12.5%）
- PP=8, VPP=2, schedule="interleaved", micro_batches=32 → ✅（bubble=14%）

### 2.5 内存容量验证

#### 2.5.1 权重内存 ≤ 设备内存

**来源**：
- 权重是静态内存，必须完整加载到 GPU
- 不开启 ZeRO/offload 时，权重不可拆分

**门槛**：
- weight_memory_physical ≤ device_memory_gb

**示例**：
- DSv3 权重=168GB（TP=8时每卡21GB），H100=80GB ✅
- DSv3 权重=168GB（TP=1时每卡168GB），H100=80GB ❌

**计算公式**：
```python
weight_memory_per_gpu = total_weight / (tp_degree * pp_degree * ep_degree)
```

#### 2.5.2 权重+激活+梯度+优化器 ≤ 设备内存

**来源**：
- 训练时需要同时存储：
  - 权重（静态）
  - 激活（动态，依赖 batch_size）
  - 梯度（静态，=权重大小）
  - 优化器状态（静态，Adam=权重×2）

**门槛**：
- total_memory = weight + activation + gradient + optimizer
- total_memory ≤ device_memory_gb

**示例**：
- DSv3 TP=8:
  - weight=21GB
  - gradient=21GB
  - optimizer=42GB（Adam）
  - activation=10GB（batch=32）
  - total=94GB，H100=80GB ❌

**优化建议**：
- 启用 activation checkpointing：激活内存减少 70-80%
- 启用 ZeRO-1/2/3：优化器/梯度/权重内存分摊到 DP 组
- 启用 offload：权重/优化器卸载到 CPU

#### 2.5.3 激活内存阈值（警告）

**来源**：
- 激活内存过大可能影响训练稳定性
- 通常建议激活内存 < 50% 设备内存

**门槛**（警告级别）：
- activation_memory > device_memory_gb × 0.5 → 警告

**建议**：
- 启用 activation checkpointing
- 减小 batch_size
- 减小 seq_len

## 3. 验证实现架构

### 3.1 验证层级

| 层级 | 验证时机 | 验证内容 | 实现位置 |
|------|----------|----------|----------|
| 前端 | 用户输入 | 数值范围、并行度乘积 | app.js |
| 后端策略 | 接收请求 | 并行策略有效性 | strategy_validator.py |
| 后端模型 | 创建模型 | 模型参数切分可行性 | model_validator.py |
| 后端内存 | 评估后 | 内存容量限制 | memory_validator.py |

### 3.2 验证结果处理

**错误级别**：
- error: 必须修复，否则评估无法进行
- warning: 建议优化，评估可继续但结果可能不可落地

**错误响应**（分层切分场景）：
```json
{
  "success": false,
  "validation": {
    "has_errors": true,
    "errors": [
      {
        "level": "error",
        "category": "strategy",
        "code": "ATTN_PARALLEL_PRODUCT_MISMATCH",
        "message": "Attention parallel product (4) ≠ GPU count (8)",
        "suggestion": "Adjust parallel degrees so tp × dp × pp × sp = 8",
        "details": {
          "tp_degree": 4,
          "dp_degree": 1,
          "pp_degree": 1,
          "sp_degree": 1,
          "attn_product": 4,
          "num_gpus": 8,
          "layered_parallelism": true
        }
      }
    ]
  }
}
```

**警告响应**（EP 性能建议）：
```json
{
  "success": true,
  "validation": {
    "has_errors": false,
    "has_warnings": true,
    "warnings": [
      {
        "level": "warning",
        "category": "strategy",
        "code": "EP_EXCEEDS_EXPERT_TP_WARNING",
        "message": "EP (16) significantly exceeds expert_tp (2), may impact performance",
        "suggestion": "Consider setting EP <= expert_tp × 2"
      }
    ]
  }
}
```

## 4. 特殊场景处理

### 4.1 MQA/GQA 场景

**问题**：num_kv_heads=1（MQA）或 num_kv_heads < TP

**处理**：
- MQA（num_kv_heads=1）需要复制 KV cache 到所有 TP 卡
- 验证时允许 num_kv_heads < TP，但需标注通信开销增加

### 4.2 ZeRO 场景

**问题**：ZeRO-1/2/3 改变内存需求计算

**处理**：
- ZeRO-1：优化器状态分摊到 DP 组
- ZeRO-2：梯度+优化器分摊
- ZeRO-3：权重+梯度+优化器分摊
- 内存验证需要考虑 ZeRO stage

### 4.3 Offload 场景

**问题**：权重/优化器卸载到 CPU

**处理**：
- Offload 时权重内存不计入 GPU 内存
- 需要额外验证 CPU 内存是否足够

### 4.4 Ulysses-SP 与 GQA/MQA 场景

**问题**：Ulysses-SP 需要 head 数量支持，但 GQA/MQA 的 KV heads 较少

**处理**：
- MQA（num_kv_heads=1）：Ulysses 无法直接使用，需要复制 KV heads
- GQA（num_kv_heads < num_heads）：
  - Ulysses 按查询 heads 切分
  - KV heads 需要特殊处理（复制或共享）
- 验证条件：`ulysses_degree <= num_kv_heads`（推荐）或使用 Dummy-Head 扩展

**示例**：
- Llama-2-70B: num_heads=64, num_kv_heads=8, GQA ratio=8
  - Ulysses=8 时每卡 1 KV head ✅
  - Ulysses=16 时 KV heads 不足，需要复制 ❌（或使用 Dummy-Head）

### 4.5 多模态生成场景

#### 4.5.1 CFGP（Cross-Modal Flow-based Generation Pipeline）

**来源**：
- Wan2.1: "Open and Advanced Large-Scale Video Generative Models", arXiv:2503.20314, 2025
- DiT 架构的扩散模型（Stable Diffusion 3, Flux, Wan2.1）

**架构组成**：
- Text Encoder：umT5-XXL 或类似多语言编码器
- DiT Backbone：Diffusion Transformer 主干网络
- VAE Encoder/Decoder：图像/视频的压缩解压缩

**切分策略**：
- **Text Encoder**：可独立切分，类似标准 Transformer
- **DiT Backbone**：
  - Self-Attention：可使用 Ulysses/Ring SP（按 token/patch 切分）
  - Cross-Attention（text conditioning）：K/V 来自 text encoder，需要同步
- **VAE**：
  - Encoder/Decoder 通常不切分（计算量较小）
  - 或按 spatial 维度切分（类似图像 patch）

**验证条件**：
- DiT num_heads >= ulysses_degree * tp_degree
- num_patches 或 seq_len（视觉 tokens）可被 SP degree 整除
- text_seq_len 通常固定（512），不受 SP 影响
- CFG（Classifier-Free Guidance）batch 翻倍：实际 batch = config_batch × 2

**示例**：
- Wan2.1-T2V-14B: DiT hidden=5120, layers=40, heads=40
  - Ulysses=4, TP=8 → Ulysses×TP=32 ≤ 40 ✅
  - video tokens: (T/4) × (H/8) × (W/8) × 16 channels
  - seq_len = video_tokens 需满足整除条件

#### 4.5.2 DiT/Vision Transformer 切分

**来源**：
- DiT: "Scalable Diffusion Models with Transformers", Peebles & Xie, 2023
- Flux、Stable Diffusion 3 等基于 DiT 的生成模型

**原理**：
- 图像通过 Patchify 转换为 token 序列：`num_patches = (H/patch_size) × (W/patch_size)`
- 类似 seq_len，可按 SP degree 切分
- 图像分辨率需要满足 patch 切分约束

**验证条件**：
- `image_height % patch_size == 0`
- `image_width % patch_size == 0`
- `num_patches % sp_degree == 0` 或
  - `image_height % sp_degree == 0`（按行切分）
  - `image_width % sp_degree == 0`（按列切分）
- DiT num_heads >= ulysses_degree * tp_degree（同 Ulysses 约束）

**切分方式选择**：
| 方式 | 条件 | 适用场景 |
|------|------|----------|
| 按序列切分 | num_patches % sp == 0 | 通用 |
| 按行切分 | H % sp == 0 | 保持图像结构 |
| 按列切分 | W % sp == 0 | 保持图像结构 |
| 2D 切分 | H % h_sp == 0, W % w_sp == 0 | 大分辨率图像 |

**示例**：
- 1024×1024 图像，patch_size=16 → num_patches=64×64=4096
  - SP=8 → 4096/8=512 patches/GPU ✅
  - 按行切分：1024/8=128 rows/GPU，每行 64 patches ✅
- 512×512 图像，patch_size=8 → num_patches=64×64=4096
  - SP=16 → 4096/16=256 patches/GPU ✅
- 768×512 图像，patch_size=16 → num_patches=48×32=1536
  - SP=8 → 1536/8=192 patches/GPU ✅
  - SP=10 → 1536/10=153.6 ❌（不整除）

#### 4.5.3 视频/3D 内容切分

**来源**：
- Wan2.1 视频生成模型
- 3D VAE 的时空压缩

**原理**：
- 视频 tokens = temporal_frames × spatial_patches
- 可按时间维度、空间维度或混合切分
- 3D VAE 压缩比：temporal_compress × spatial_compress

**验证条件**：
- `num_frames % temporal_sp_degree == 0`（按时间切分）
- `num_spatial_patches % spatial_sp_degree == 0`（按空间切分）
- 总 tokens = `(num_frames/temporal_comp) × (H/spatial_comp) × (W/spatial_comp)`
- 整除条件：`total_tokens % sp_degree == 0`

**示例**：
- 5秒视频，24fps → 120 frames，temporal_comp=4 → 30 temporal tokens
- 1080p（1920×1080），spatial_comp=8×8 → 240×135=32400 spatial tokens
- 总 tokens = 30 × 32400 = 972000
- SP=32 → 972000/32=30375 tokens/GPU ✅

**内存影响**：
- 视频 tokens 激活内存 = batch × num_frames × H × W × channels × dtype_size
- VAE 压缩后内存 /= (temporal_comp × spatial_comp²)

## 5. 参考文献

1. Megatron-LM: Efficient Large-Scale Transformer Model Training (2020)
2. Megatron-LM: Reducing Activation Recomputation in Large Transformer Models (2022)
3. DeepSpeed: System Optimizations Enable Training Deep Learning Models
4. DeepSpeed-MoE: Advancing Mixture-of-Experts Inference
5. Ring Attention: Ring Attention with Blockwise Transformers (2023)
6. Llama 2 Open Foundation and Fine-Tuned Chat Models
7. Llama 3 Technical Report (2024)
8. DeepSeek-V3 Technical Report
9. DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models, arXiv:2309.14509, 2023
10. USP: A Unified Sequence Parallelism Approach for Long Context Generative AI, arXiv:2405.07719, 2024
11. DiT: Scalable Diffusion Models with Transformers, Peebles & Xie, 2023
12. Wan: Open and Advanced Large-Scale Video Generative Models, arXiv:2503.20314, 2025
13. 360-LLaMA-Factory: Dummy-Head Ulysses, arXiv:2505.22296
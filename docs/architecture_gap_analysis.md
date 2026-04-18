# 架构差距分析报告

## 分析日期
2026-04-18

## 分析范围
对照用户需求，系统分析当前架构的满足情况。

## 用户需求清单
1. 配置项：场景/模型/集群环境/并行切分/输入输出规格
2. 场景分类：训练（预训练/微调/RL后训练）、推理（token自回归/扩散/卷积）
3. 推理调度：PD分离、混布、DiT/VAE分离
4. 评估指标：样本吞吐、TTFT、TPOT、TPS、QPS、MFU
5. 分解呈现：模块/阶段/子模块的计算/通信/内存
6. 管线预设与模型预设正交，可配置

---

## 1. 配置项支持情况

### 1.1 已支持 ✅

| 配置项 | 实现位置 | 说明 |
|-------|---------|------|
| **场景配置** | `llm_perf/scenarios/base.py` | ScenarioType枚举 |
| **模型配置** | `configs/models/*.yaml` | YAML preset格式 |
| **集群环境** | `llm_perf/hardware/` | Device、Cluster、Topology |
| **并行切分** | `llm_perf/strategy/base.py` | TP/PP/DP/EP/SP/CP |
| **输入输出规格** | `param_schema` | training/inference参数schema |

---

## 2. 场景细分支持情况

### 2.1 已支持 ✅

| 场景类型 | Workload配置 |
|---------|-------------|
| **训练** | `configs/workloads/base/training.yaml` |
| **RL-PPO** | `configs/workloads/autoregressive/rl-ppo.yaml` |
| **RL-GRPO** | `configs/workloads/autoregressive/rl-grpo.yaml` |
| **Token自回归推理** | `configs/workloads/autoregressive/autoregressive-inference.yaml` |
| **扩散推理** | `configs/workloads/iterative/diffusion-pipeline.yaml` |
| **推测解码** | `configs/workloads/autoregressive/speculative-decoding.yaml` |

### 2.2 部分支持 ⚠️

| 场景类型 | 缺失部分 |
|---------|---------|
| **卷积推理** | 缺少统一的conv推理workload |
| **预训练 vs 微调** | 共用training.yaml，无法区分差异 |

---

## 3. 推理调度支持情况

### 3.1 已支持 ✅

| 调度策略 | 实现位置 |
|---------|---------|
| **PD分离** | `llm_perf/legacy/scenarios/pd_disagg.py` |

### 3.2 部分支持 ⚠️

| 调度策略 | 缺失部分 |
|---------|---------|
| **DiT/VAE分离** | pipeline级分离存在，节点级分离缺失 |

### 3.3 不支持 ❌

| 调度策略 | 说明 |
|---------|------|
| **混布** | 无混布评估代码 |

---

## 4. 评估指标支持情况

### 4.1 已支持 ✅

| 评估指标 | 实现位置 |
|---------|---------|
| **样本吞吐** | `UnifiedResult.throughput['samples_per_sec']` |
| **TTFT** | `LLMInferenceResult.prefill_time_sec` |
| **TPOT** | `LLMInferenceResult.decode_time_per_step_sec` |
| **TPS** | `LLMInferenceResult.decode_tokens_per_sec` |
| **Peak Memory** | `UnifiedResult.peak_memory_gb` |
| **FLOPs** | `PhaseResult.flops` |

### 4.2 不支持 ❌

| 评估指标 | 说明 |
|---------|------|
| **QPS** | 无Queries Per Second指标 |
| **MFU** | 无Model FLOPs Utilization指标 |

---

## 5. 分解呈现支持情况

### 5.1 已支持 ✅

| 分解层级 | 实现位置 |
|---------|---------|
| **阶段分解** | `UnifiedResult.phases` |
| **组件分解** | `PhaseResult.component` |
| **内存分解** | `PhaseResult.memory_gb` |
| **通信分解** | `_estimate_comm_time()` |
| **时间分解** | `breakdown['time_breakdown']` |

### 5.2 部分支持 ⚠️

| 分解层级 | 缺失部分 |
|---------|---------|
| **子模块分解** | 未按emb/attn/ffn细分 |
| **Kernel级分解** | kernel信息不完整 |

### 5.3 不支持 ❌

| 分解层级 | 说明 |
|---------|------|
| **通信详细分解** | 缺少AllReduce/AllGather/ReduceScatter分解 |

---

## 6. 管线预设独立性

### 6.1 已支持 ✅

| 特性 | 实现位置 |
|-----|---------|
| **Workload与Model分离** | `WorkloadConfig.component_mapping` |
| **ComputePattern抽象** | `ComputePattern枚举` |
| **infer_workload()独立性** | 基于workload配置，不依赖model_type |

---

## 总结

### 功能满足度统计

| 类别 | 已支持 ✅ | 部分支持 ⚠️ | 不支持 ❌ |
|-----|---------|-----------|---------|
| **配置项** | 5 | 0 | 0 |
| **场景分类** | 6 | 2 | 0 |
| **推理调度** | 1 | 1 | 1 |
| **评估指标** | 6 | 0 | 2 |
| **分解呈现** | 5 | 2 | 1 |
| **管线独立性** | 3 | 0 | 0 |
| **总计** | **20** | **5** | **4** |

### 满足度：74.1%

---

## 关键缺失功能

### P1 - 高优先级
1. **MFU指标**
2. **子模块分解**
3. **混布评估**

### P2 - 中优先级
1. **QPS指标**
2. **通信详细分解**
3. **卷积推理场景**

### P3 - 低优先级
1. **预训练 vs 微调区分**
2. **DiT/VAE节点级分离**
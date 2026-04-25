# 前端设计方案文档

CePing 项目前端功能已完善，本文档整理了整体设计方案，方便通过 md 文档即可了解前端架构。

## 1. 前端整体架构

```
前端架构:
├── HTML 结构 (web/templates/index.html)
│   ├── Header
│   ├── 配置区域
│   │   ├── Workload 配置 (左上)
│   │   ├── 集群拓扑配置 (右上)
│   │   ├── 模型配置 (第二行，全宽)
│   │   └── 并行切分配置 (第三行，全宽)
│   ├── 评估按钮
│   └── 结果显示区域
│       ├── 性能指标汇总
│       ├── 性能分解表格
│       ├── 子模块分解表格
│       ├── 内存分解表格
│       └── 时间占比可视化
│
├── JavaScript 逻辑 (web/static/js/app.js)
│   ├── 预设模型配置 (modelPresets)
│   ├── 表单事件处理
│   ├── API 调用 (fetch /api/evaluate)
│   └── 结果渲染函数
│       ├── displayResults()
│       ├── renderBreakdown()
│       ├── renderInferenceBreakdown()
│       ├── renderDetailedBreakdown()
│       ├── updateModelSpecDisplay()
│       └── renderValidationErrors()
│
└── CSS 样式 (web/static/css/style.css)
    ├── Grid 布局
    ├── 表格样式
    ├── 进度条样式
    └── 响应式设计
```

## 2. 配置区域详解

### 2.1 Workload 配置

支持五种计算场景：

| 场景 | 说明 | 场景参数 |
|------|------|----------|
| training | 训练 | batch_size, micro_batch_size, seq_len, num_steps |
| inference | 推理 | batch_size, input_tokens, output_tokens |
| pd_disagg | 推理 PD 分离 | batch_size, input_tokens, output_tokens, prefill_devices, decode_devices |
| rl_training | RL 后训练 | batch_size, seq_len, num_rollouts, ppo_epochs |
| diffusion | 多模态生成 | batch_size, diffusion_steps, generation_mode, 输入输出配置 |

**Diffusion 生成模式**:
- T2I (文本生成图像): prompt_tokens + output_image (宽/高)
- T2V (文本生成视频): prompt_tokens + output_video (宽/高/帧数)
- I2I (图像生成图像): input_image (宽/高) + output_image (宽/高)
- I2V (图像生成视频): input_image (宽/高) + output_video (宽/高/帧数)
- V2V (视频生成视频): input_video (宽/高/帧数) + output_video (宽/高/帧数)

### 2.2 集群拓扑配置

| 字段 | 类型 | 说明 | 后端对应参数 |
|------|------|------|--------------|
| device_vendor | select | 设备厂商 (NVIDIA/AMD/Huawei) | - |
| device_model | select | 设备型号 | device |
| total_devices | number | 总设备数 | cluster.total_devices |
| topology_type | select | 拓扑类型 | topology.type |

**拓扑类型参数**:

| 拓扑类型 | 参数 |
|----------|------|
| 2-Tier Simple | intra_node_bw_gbps, inter_node_bw_gbps |
| 3-Tier Clos | node_bw_gbps, rack_bw_gbps, cluster_bw_gbps |
| Fat-Tree | edge_bw_gbps, agg_bw_gbps, core_bw_gbps, oversubscription |
| CloudMatrix Supernode | num_npus, ub_bw_gbps, rdma_bw_gbps |

### 2.3 模型配置

| 字段 | 类型 | 说明 | 后端对应参数 |
|------|------|------|--------------|
| model_preset | select | 预设模型 | model.preset |
| model_type | select | 模型类型 (dense/sparse) | model.sparse_type |
| hidden_size | number | 隐藏层大小 | model.hidden_size |
| num_layers | number | 层数 | model.num_layers |
| num_heads | number | 注意力头数 | model.num_attention_heads |
| dtype | select | 数据类型 (fp16/bf16/fp8) | model.dtype |
| num_experts | number | MoE 专家数 (MoE 专用) | model.num_experts |
| experts_per_token | number | Top-K (MoE 专用) | model.num_experts_per_token |

**模型规格显示**:
- 选择预设后自动显示模型规格参数表格
- 显示内容：参数量、层数、隐藏层、头数、KV头、Head维、FFN类型、专家数

### 2.4 并行切分配置

#### 单实例并行策略 (training/inference/diffusion)

| 字段 | 类型 | 说明 | 后端对应参数 |
|------|------|------|--------------|
| tp_degree | number | Tensor Parallel | strategy.tp |
| pp_degree | number | Pipeline Parallel | strategy.pp |
| vpp_degree | number | Virtual Pipeline Parallel | strategy.vpp |
| ep_degree | number | Expert Parallel | strategy.ep |
| ulysses_degree | number | Ulysses Sequence Parallel | strategy.ulysses_degree |
| ring_degree | number | Ring Sequence Parallel (含 CP) | strategy.ring_degree |
| pipeline_schedule | select | PP 调度策略 (1f1b/gpipe/interleaved) | strategy.pipeline_schedule |
| megatron_sp_enabled | checkbox | Megatron Sequence Parallel | strategy.megatron_sp_enabled |
| dp_degree | readonly | Data Parallel (自动计算) | strategy.dp |

**DP 自动计算**: `dp = total_devices / (tp * pp * ep * ulysses * ring)`

#### PD-Disagg 双实例策略

Prefill 实例和 Decode 实例分别配置独立的并行策略：
- TP_p, PP_p, EP_p, Ulysses_p, Ring_p, DP_p (自动)
- TP_d, PP_d, EP_d, Ulysses_d, Ring_d, DP_d (自动)

#### RL-Training 双阶段策略

训练阶段和推理阶段分别配置并行策略：
- TP_train, PP_train, VPP_train, EP_train, Ulysses_train, Ring_train, DP_train (自动)
- TP_infer, PP_infer, VPP_infer, EP_infer, Ulysses_infer, Ring_infer, DP_infer (自动)

#### 内存优化选项 (仅训练场景)

| 字段 | 类型 | 说明 | 后端对应参数 |
|------|------|------|--------------|
| activation_checkpointing | checkbox | Activation Checkpointing | strategy.activation_checkpointing |
| zero_stage | select | ZeRO Stage (0/1/2/3) | strategy.zero_stage |

## 3. 评估结果展示

### 3.1 性能指标汇总

#### 训练场景

| 指标 | 说明 | 数据来源 |
|------|------|----------|
| Tokens/sec | 吞吐量 | result.throughput.tokens_per_sec |
| Samples/sec | 样本吞吐量 | result.throughput.samples_per_sec |
| Time/Step | 每步时间 | result.time.time_per_step_sec |
| Memory/device | 单设备显存 | result.memory.memory_per_device_gb |

#### 推理场景

| 指标 | 说明 | 数据来源 |
|------|------|----------|
| Decode TPS | 解码吞吐量 | result.decode.tps |
| TTFT | Time To First Token | result.prefill.ttft_sec |
| TPOT | Time Per Output Token | result.decode.tpot_sec |
| Overall TPS | 整体吞吐量 | result.end_to_end.overall_tps |
| Total Time | 总时间 | result.end_to_end.total_time_sec |
| Memory/device | 单设备显存 | result.memory.memory_per_device_gb |
| KV Cache | KV Cache 内存 | result.memory.kv_cache_gb |

#### Diffusion 场景

| 指标 | 说明 | 数据来源 |
|------|------|----------|
| Total Time | 总生成时间 | result.total_time_sec |
| Peak Memory | 最大显存 | result.peak_memory_gb |
| Pixels/sec |像素吞吐量 | result.throughput.pixels_per_sec |
| Inference Steps | 推理步数 | params.num_inference_steps |

### 3.2 性能分解表格

#### 训练性能分解

| 类别 | 时间 | 占比 | 数据来源 |
|------|------|------|----------|
| Compute | xxx ms | xx% | breakdown.time_breakdown.compute_sec |
| Backward | xxx ms | xx% | breakdown.time_breakdown.backward_sec |
| Optimizer | xxx ms | xx% | breakdown.time_breakdown.optimizer_sec |
| Communication | xxx ms | xx% | breakdown.time_breakdown.communication_sec |
| Memory Wait | xxx ms | xx% | breakdown.time_breakdown.memory_sec |

#### 推理性能分解

| 类别 | 时间 | 占比 | 数据来源 |
|------|------|------|----------|
| Prefill | xxx ms | xx% | breakdown.inference_breakdown.prefill_sec |
| Decode | xxx ms | xx% | breakdown.inference_breakdown.decode_sec |
| Communication | xxx ms | xx% | breakdown.inference_breakdown.communication_sec |
| KV Cache Access | xxx ms | xx% | breakdown.inference_breakdown.kv_cache_sec |

#### Diffusion 组件耗时分解

| 组件 | 时间 | 占比 | 数据来源 |
|------|------|------|----------|
| Text Encoder (encode) | xxx ms | xx% | phases.encode.total_time_sec |
| DiT Denoising (N steps) | xxx s | xx% | phases.denoise.total_time_sec |
| VAE Decoder (decode) | xxx ms | xx% | phases.decode.total_time_sec |

### 3.3 子模块分解表格

| 子模块类型 | 计算时间(ms) | 计算占比 | 通信时间(ms) | 通信占比 | 内存(GB) |
|------------|--------------|----------|--------------|----------|----------|
| embedding | xxx | xx% | xxx | xx% | xxx |
| transformer_block | xxx | xx% | xxx | xx% | xxx |
| - attention | xxx | xx% | xxx | xx% | xxx |
| - ffn/moe | xxx | xx% | xxx | xx% | xxx |
| lm_head | xxx | xx% | xxx | xx% | xxx |
| **总计** | xxx ms | 100% | xxx ms | 100% | xxx |

**嵌套分解**: transformer_block 内部显示 attention 和 ffn/moe 的详细分解

### 3.4 内存分解表格

#### 按类型分解

| 内存类型 | 大小 | 数据来源 |
|----------|------|----------|
| weight | xxx GB | detailed.memory.by_type.weight |
| gradient | xxx GB | detailed.memory.by_type.gradient |
| optimizer | xxx GB | detailed.memory.by_type.optimizer |
| activation | xxx GB | detailed.memory.by_type.activation |
| **总计** | xxx GB | detailed.memory.by_type.total |

#### 按子模型分解

| 子模型 | 总内存 | 数据来源 |
|--------|--------|----------|
| model_name | xxx GB | detailed.memory.by_submodel.activations_gb |

### 3.5 通信分解

#### 按并行方式分解

| 并行类型 | 通信量 | 时间 | 数据来源 |
|----------|--------|------|----------|
| TP | xxx GB | xxx ms | communication.by_parallelism.tp |
| PP | xxx GB | xxx ms | communication.by_parallelism.pp |
| DP | xxx GB | xxx ms | communication.by_parallelism.dp |

#### 按通信原语分解

| 原语类型 | 通信量(GB) | 时间(ms) | 数据来源 |
|----------|------------|----------|----------|
| all_reduce | xxx | xxx | communication.by_operation.all_reduce |
| all_gather | xxx | xxx | communication.by_operation.all_gather |
| reduce_scatter | xxx | xxx | communication.by_operation.reduce_scatter |

**嵌套原语分解**: 显示每种原语在不同并行类型 (TP/PP/DP) 下的详细分解

### 3.6 时间占比可视化

**进度条显示**:
- 计算时间: 绿色 (#4CAF50)
- 通信时间: 橙色 (#FF9800)

显示内容:
- 总时间 (ms)
- 计算时间 (ms)
- 通信时间 (ms)
- 计算占比 (%)
- 通信占比 (%)

## 4. 前端渲染函数详解

### 4.1 displayResults(result)

**功能**: 根据场景类型渲染评估结果

**输入**: 
- `result`: API 返回的评估结果对象

**处理流程**:
1. 判断场景类型 (training/inference/diffusion)
2. 判断 pipeline 类型 (diffusion-video)
3. 根据场景调用相应的渲染函数:
   - Diffusion: 渲染 phases 分解 + 组件耗时
   - Training: 渲染性能指标 + renderBreakdown + renderDetailedBreakdown
   - Inference: 渲染推理指标 + renderInferenceBreakdown + renderDetailedBreakdown

### 4.2 renderBreakdown(breakdown)

**功能**: 渲染训练性能分解表格

**输入**: `breakdown.time_breakdown`
- compute_sec, compute_percent
- backward_sec, backward_percent
- optimizer_sec, optimizer_percent
- communication_sec, communication_percent
- memory_sec, memory_percent

**输出**: 性能分解 HTML 表格

### 4.3 renderInferenceBreakdown(breakdown)

**功能**: 渲染推理性能分解表格

**输入**: `breakdown.inference_breakdown`
- prefill_sec, prefill_percent
- decode_sec, decode_percent, decode_per_token_sec
- communication_sec, communication_percent
- kv_cache_sec, kv_cache_percent (可选)

**输出**: 推理性能分解 HTML 表格

### 4.4 renderDetailedBreakdown(detailed)

**功能**: 渲染详细分解表格

**输入**: `result.detailed_breakdown`
- `memory.by_type`: 内存按类型分解
- `memory.by_submodel`: 内存按子模型分解
- `by_submodule_type`: 子模块分解 (含嵌套)
- `communication.by_parallelism`: 通信按并行方式分解
- `communication.by_operation`: 通信按原语分解

**处理**:
1. 计算总计算时间: `totalComputeTime`
2. 计算总通信时间: `totalCommTime`
3. 计算各子模块占比: `computePct = time / total × 100`
4. 渲染嵌套分解: attention + ffn/moe
5. 渲染进度条可视化

**输出**: 
- 内存分解表格 (按类型 + 按子模型)
- 子模块分解表格 + 时间占比进度条
- 通信分解表格 (按并行方式 + 按原语)
- 子模型详情

### 4.5 updateModelSpecDisplay(config, presetKey)

**功能**: 渲染模型规格参数表格

**输入**: 预设模型配置
- name, params (估算), num_layers, hidden_size
- num_heads, num_kv_heads, head_dim
- num_experts, num_experts_per_tok

**处理**:
1. 估算模型参数量: embedParams + attnParams + ffnParams
2. 计算 KV 头数和 Head 维度
3. 判断 FFN 类型 (Dense/MoE)

**输出**: 模型规格参数紧凑表格 (左侧显示)

### 4.6 renderValidationErrors(validation)

**功能**: 渲染配置验证错误/警告

**输入**: `validation`
- `errors`: 错误列表 (category, message, suggestion)
- `warnings`: 警告列表 (category, message, suggestion)

**输出**: 验证错误/警告 HTML 容器

**错误分类**:
- strategy: 并行策略
- model: 模型规格
- sequence: 序列切分
- memory: 内存容量
- special: 特殊场景

## 5. 后端数据结构对应

### 5.1 API 接口

#### 主评估接口

```
POST /api/evaluate
请求：
{
    "cluster": {
        "topology": "2-Tier Simple",
        "total_devices": 64
    },
    "model": {
        "preset": "llama-7b",
        "type": "llama",
        "sparse_type": "dense",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "dtype": "fp16"
    },
    "device": "A100-80GB",
    "num_devices": 64,
    "topology": {
        "type": "2-Tier Simple",
        "intra_node_bw_gbps": 900,
        "inter_node_bw_gbps": 200
    },
    "workload": {
        "scenario": "training",
        "batch_size": 32,
        "micro_batch_size": 1,
        "seq_len": 4096,
        "num_steps": 1000
    },
    "strategy": {
        "tp": 8,
        "pp": 1,
        "vpp": 1,
        "dp": 8,
        "ep": 1,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "megatron_sp_enabled": false,
        "pipeline_schedule": "1f1b",
        "activation_checkpointing": false,
        "zero_stage": 0
    }
}

响应：
{
    "success": true,
    "result": {
        "throughput": {
            "tokens_per_sec": 12345,
            "samples_per_sec": 12.3
        },
        "time": {
            "time_per_step_sec": 0.123
        },
        "memory": {
            "memory_per_device_gb": 45.6
        },
        "breakdown": {
            "time_breakdown": {
                "compute_sec": 0.1,
                "backward_sec": 0.02,
                "optimizer_sec": 0.001,
                "communication_sec": 0.002,
                "memory_sec": 0.0
            }
        },
        "detailed_breakdown": {
            "memory": {
                "by_type": {
                    "weight": 10.5,
                    "gradient": 10.5,
                    "optimizer": 21.0,
                    "activation": 3.6,
                    "total": 45.6
                },
                "by_submodel": {
                    "embedding": {"activations_gb": 0.1},
                    "transformer": {"activations_gb": 3.5}
                }
            },
            "by_submodule_type": {
                "embedding": {
                    "compute": {"time_sec": 0.001},
                    "communication": {"time_sec": 0.0},
                    "memory": {"activations_gb": 0.1}
                },
                "transformer_block": {
                    "compute": {"time_sec": 0.098},
                    "communication": {"time_sec": 0.002},
                    "memory": {"activations_gb": 3.5},
                    "nested_breakdown": {
                        "attention": {...},
                        "ffn": {...}
                    }
                }
            },
            "communication": {
                "by_parallelism": {
                    "tp": {"total_bytes": 1e9, "total_time_sec": 0.002},
                    "dp": {"total_bytes": 2e9, "total_time_sec": 0.001}
                },
                "by_operation": {
                    "all_reduce": {
                        "total_bytes": 1e9,
                        "total_time_sec": 0.002,
                        "by_ptype": {
                            "tp": {...},
                            "dp": {...}
                        }
                    }
                }
            }
        }
    },
    "validation": {
        "errors": [],
        "warnings": []
    }
}
```

#### Diffusion Pipeline 接口

```
POST /api/evaluate/pipeline/diffusion-video
响应：
{
    "success": true,
    "result": {
        "total_time_sec": 12.3,
        "peak_memory_gb": 45.6,
        "throughput": {
            "pixels_per_sec": 12345678
        },
        "phases": [
            {"name": "encode", "total_time_sec": 0.5, ...},
            {"name": "denoise", "total_time_sec": 11.5, "repeat_count": 50},
            {"name": "decode", "total_time_sec": 0.3}
        ],
        "params": {
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "use_cfg": true
        }
    }
}
```

### 5.2 数据字段对应表

| 前端显示 | 后端字段 | 说明 |
|----------|----------|------|
| 计算时间 | compute.time_sec | 子模块计算时间 |
| 计算占比 | compute.time_sec / total × 100 | 前端计算 |
| 通信时间 | communication.time_sec | 子模块通信时间 |
| 通信占比 | communication.time_sec / total × 100 | 前端计算 |
| 内存 | memory.activations_gb | 后端别名 |
| 嵌套分解 | nested_breakdown | transformer_block 内部 |
| 性能分解 | time_breakdown / inference_breakdown | 总体分解 |
| 通信量 | total_bytes / 1e9 | 转换为 GB |
| 通信时间 | total_time_sec * 1000 | 转换为 ms |
| 原语分解 | by_operation | 通信原语类型 |
| 并行类型分解 | by_ptype | 原语下的并行类型 |

## 6. 前端交互逻辑

### 6.1 初始化流程

```
init() {
    loadData()              // 加载设备、模型、拓扑预设
    setupEventListeners()   // 注册事件监听器
    updateDeviceModels()    // 初始化设备型号选择
    updateTopologyParams()  // 初始化拓扑参数
    loadModelPreset()       // 加载默认模型预设
    calculateDP()           // 计算初始 DP
    switchWorkloadScenario('training')  // 初始化场景
}
```

### 6.2 事件处理

| 事件 | 处理函数 | 说明 |
|------|----------|------|
| model_preset change | loadModelPreset | 加载预设配置，更新 UI |
| model_type change | updateModelTypeUI | 显示/隐藏 MoE 字段 |
| device_vendor change | updateDeviceModels | 更新设备型号列表 |
| topology_type change | updateTopologyParams | 更新拓扑参数表单 |
| workload_scenario change | switchWorkloadScenario | 切换场景配置和并行策略 |
| evaluate_btn click | evaluate | 执行评估 |
| 并行参数 input | calculateDP/calculatePDDP/calculateRLDP | 自动计算 DP |
| RL phase tab click | switchRLPhase | 切换训练/推理阶段显示 |
| Diffusion mode tab click | updateDiffusionModeUI | 切换生成模式 UI |

### 6.3 配置收集

**collectConfig()**: 收集所有配置参数构建请求对象

1. 收集拓扑配置 (根据 topology_type)
2. 收集模型配置 (预设 + 用户输入)
3. 收集 workload 配置 (根据 scenario)
4. 收集并行策略 (根据 scenario: 单实例/PD-Disagg/RL-Training)
5. 添加内存优化选项 (仅训练)

### 6.4 配置验证

**前端验证** (validateConfigBeforeSubmit):
- 并行度乘积 = 总设备数
- DP ≥ 1

**后端验证** (validation):
- errors: 策略冲突、内存不足等
- warnings: 性能建议、参数不合理等

### 6.5 DP 自动计算逻辑

```
calculateDP() {
    product = tp * pp * ep * ulysses * ring
    dp = total_devices / product
    
    if (product > total_devices) {
        显示错误: "无法计算 DP"
    } else if (!Number.isInteger(dp)) {
        显示警告: "DP 非整数"
    }
    
    return Math.floor(dp)
}
```

## 7. 新增特性

### 7.1 多场景支持

- **Training**: 传统训练场景
- **Inference**: 推理场景 (TTFT/TPOT/TPS)
- **PD-Disagg**: Prefill-Decode 分离架构
- **RL-Training**: RL 后训练 (双阶段策略)
- **Diffusion**: 多模态生成 (5种生成模式)

### 7.2 双策略/双阶段配置

- PD-Disagg: Prefill 实例 + Decode 实例独立策略
- RL-Training: 训练阶段 + 推理阶段独立策略
- 独立 DP 计算，独立拓扑设备数

### 7.3 详细分解展示

- **子模块分解**: embedding/transformer_block/lm_head
- **嵌套分解**: transformer_block → attention + ffn/moe
- **内存分解**: 按类型 (weight/gradient/optimizer/activation) + 按子模型
- **通信分解**: 按并行方式 (TP/PP/DP) + 按原语 (all_reduce/all_gather/reduce_scatter)
- **原语嵌套**: 每种原语下的不同并行类型分解

### 7.4 时间占比可视化

- 进度条显示计算/通信占比
- 绿色计算 + 橙色通信
- 同时显示绝对时间和百分比

### 7.5 模型规格显示

- 选择预设后自动显示规格表格
- 紧凑布局，左侧显示
- 显示参数量估算、层数、隐藏层、头数、KV头、Head维、FFN类型、专家数

### 7.6 配置验证反馈

- 前端实时验证 (DP 计算)
- 后端验证反馈 (errors/warnings)
- 分类显示 (strategy/model/sequence/memory/special)
- 提供修复建议

### 7.7 设备厂商支持

- NVIDIA: A100, H100, H200 等
- AMD: MI300X 等
- Huawei Ascend: 910B 等

### 7.8 拓扑类型支持

- 2-Tier Simple: 节点内 + 节点间带宽
- 3-Tier Clos: 节点 + 机架 + 集群带宽
- Fat-Tree: Edge + Aggregation + Core 带宽
- CloudMatrix Supernode: 超节点架构

## 8. 代码结构索引

### 8.1 HTML 关键元素 ID

| 元素 ID | 说明 |
|---------|------|
| workload-scenario | 计算场景选择 |
| model-preset | 模型预设选择 |
| model-type | 模型类型选择 |
| device-vendor | 设备厂商选择 |
| device-model | 设备型号选择 |
| topology-type | 拓扑类型选择 |
| total-devices | 总设备数 |
| tp-degree, pp-degree, ep-degree | 并行度参数 |
| ulysses-degree, ring-degree | SP 参数 |
| dp-degree-display | DP 显示 (自动计算) |
| evaluate-btn | 评估按钮 |
| results | 结果显示区域 |
| results-content | 结果内容区域 |

### 8.2 JavaScript 核心函数

| 函数名 | 文件位置 | 说明 |
|--------|----------|------|
| init | app.js:50 | 初始化 |
| loadData | app.js:60 | 加载预设数据 |
| collectConfig | app.js:671 | 收集配置 |
| evaluate | app.js:604 | 执行评估 |
| displayResults | app.js:836 | 显示结果 |
| renderBreakdown | app.js:1159 | 渲染训练分解 |
| renderInferenceBreakdown | app.js:1199 | 渲染推理分解 |
| renderDetailedBreakdown | app.js:976 | 渲染详细分解 |
| calculateDP | app.js:121 | 计算 DP |
| switchWorkloadScenario | app.js:160 | 切换场景 |

### 8.3 CSS 关键样式

| 样式类 | 说明 |
|--------|------|
| .card | 配置卡片 |
| .grid | Grid 布局 |
| .form-group | 表单组 |
| .form-row | 表单行 (两列) |
| .parallel-grid-wide | 并行参数网格 |
| .result-grid | 结果指标网格 |
| .result-card | 结果指标卡片 |
| .result-card.highlight | 高亮指标卡片 |
| .breakdown-table | 分解表格 |
| .time-breakdown-bar | 时间占比进度条 |
| .compute-bar | 计算进度条 (绿色) |
| .comm-bar | 通信进度条 (橙色) |
| .validation-error | 验证错误样式 |
| .validation-warning | 验证警告样式 |

## 9. 设计原则

### 9.1 UI 设计原则

- **扁平化设计**: 无 3D 效果，简洁明了
- **Grid 布局**: 自动响应式，灵活适配
- **紧凑显示**: 模型规格紧凑表格，输入参数紧凑布局
- **颜色语义化**: 绿色计算、橙色通信、红色错误、黄色警告
- **自动计算提示**: readonly 显示 + hint 提示

### 9.2 交互设计原则

- **实时验证**: 输入时即时反馈错误
- **智能联动**: 场景切换自动切换配置和策略
- **预设优先**: 预设选择自动填充参数
- **错误提示**: 提供修复建议
- **进度反馈**: 评估按钮 spinner 动画

### 9.3 数据展示原则

- **双重显示**: 绝对值 + 占比百分比
- **嵌套分解**: 层级展示，缩进显示子项
- **分类展示**: 多维度分解 (按类型、按子模型、按并行方式、按原语)
- **可视化**: 进度条直观显示占比
- **总计汇总**: 每个表格显示总计行

## 10. 扩展点

### 10.1 新增场景

1. HTML: 添加 workload-params-{scenario} 区域
2. HTML: 添加 parallel-{scenario} 策略区域 (如需)
3. JavaScript: collectWorkloadConfig 添加 scenario 分支
4. JavaScript: switchWorkloadScenario 添加场景逻辑
5. JavaScript: displayResults 添加场景渲染逻辑

### 10.2 新增模型预设

1. 后端: 添加预设配置到 model_presets.json
2. 前端: 自动通过 /api/model/presets 加载

### 10.3 新增设备

1. 后端: 添加设备配置到 devices.json
2. 前端: 自动通过 /api/devices 加载

### 10.4 新增拓扑

1. 后端: 添加拓扑配置到 topology_presets
2. 前端: updateTopologyParams 添加参数渲染逻辑

### 10.5 新增并行策略参数

1. HTML: 添加 input 元素到 parallel-grid-wide
2. JavaScript: collectConfig 添加参数收集
3. JavaScript: calculateDP 添加到乘积计算

### 10.6 新增分解维度

1. 后端: detailed_breakdown 添加新维度
2. 前端: renderDetailedBreakdown 添加渲染逻辑

## 11. 解耦设计原则

前端渲染采用**完全解耦设计**，所有类别/类型通过数据自动发现，无需硬编码任何列表。

详细文档见 `docs/architecture_decoupling.md`。

### 11.1 数据驱动原则

**后端数据结构**：所有分解使用字典结构，自动包含所有类别

```python
{
    "time_breakdown": {
        "compute_sec": 0.1,
        "backward_sec": 0.02,
        "new_category_sec": 0.05  # 新增类别自动出现
    },
    "by_submodule_type": {
        "embedding": {...},
        "transformer_block": {...},
        "new_type": {...}  # 新增类型自动出现
    }
}
```

**前端渲染**：使用 `Object.keys()` 自动获取所有字段

```javascript
const allTypes = Object.keys(bySubmoduleType);  // 自动发现所有类型
const allSecKeys = Object.keys(tb).filter(k => k.endsWith('_sec'));  // 自动发现所有时间类别
```

### 11.2 内存分解解耦

**位置**：app.js:983

```javascript
const memByType = detailed.memory?.by_type || {};
const { total, ...breakdownItems } = memByType;

// 自动发现所有类型（非硬编码）
const allTypes = Object.keys(breakdownItems).filter(k => !k.startsWith('total'));

// 优先顺序排列（仅排序，不限制）
const orderedPriority = ['weight', 'gradient', 'optimizer', 'activation'];
const orderedTypes = [
    ...orderedPriority.filter(t => breakdownItems[t] !== undefined),
    ...allTypes.filter(t => !orderedPriority.includes(t))  # 新类型自动追加
];
```

**效果**：新增内存类型（如 `kv_cache`）自动显示

### 11.3 子模块分解解耦

**位置**：app.js:1004

```javascript
const bySubmoduleType = detailed.by_submodule_type || {};

// 自动遍历所有类型
for (const [submoduleType, data] of Object.entries(bySubmoduleType)) {
    // embedding, transformer_block, lm_head, new_type...
    
    // 自动遍历嵌套类型
    if (data.nested_breakdown) {
        for (const [nestedType, nestedData] of Object.entries(data.nested_breakdown)) {
            // attention, ffn, moe, linear_attention, full_attention, new_nested...
        }
    }
}
```

**效果**：
- 新增子模块类型（如 `new_attention`）自动显示
- 新增嵌套类型（如 `full_attention`）自动显示

### 11.4 时间分解解耦

**位置**：app.js:1175

```javascript
const tb = breakdown.time_breakdown || {};
const allSecKeys = Object.keys(tb).filter(k => k.endsWith('_sec'));

// 自动发现所有时间类别
// compute_sec, backward_sec, optimizer_sec, communication_sec, memory_sec, new_category_sec
```

**效果**：新增时间类别自动显示

### 11.5 通信分解解耦

**位置**：app.js:1076+

```javascript
const commByPara = detailed.communication?.by_parallelism || {};
const commByOp = detailed.communication?.by_operation || {};

// 自动遍历所有并行类型
for (const [paraType, data] of Object.entries(commByPara)) {
    // tp, pp, dp, ep, ulysses, ring, new_parallelism...
}

// 自动遍历所有原语类型
for (const [opType, data] of Object.entries(commByOp)) {
    // all_reduce, all_gather, reduce_scatter, all_to_all, new_op...
}
```

**效果**：
- 新增通信域（如 `ep`）自动显示
- 新增通信原语（如 `all_to_all`）自动显示

### 11.6 所有分解类型解耦状态

| 分解类型 | 解耦状态 | 实现方式 | 位置 |
|---------|---------|---------|------|
| 耗时分解 | ✅ 已解耦 | Object.keys(time_breakdown) | app.js:1175 |
| 推理分解 | ✅ 已解耦 | Object.keys(inference_breakdown) | app.js:1216 |
| 子模块分解 | ✅ 已解耦 | Object.keys(bySubmoduleType) | app.js:1004 |
| 嵌套分解 | ✅ 已解耦 | Object.keys(nested_breakdown) | app.js:1031 |
| 内存类型分解 | ✅ 已解耦 | Object.keys(memory_by_type) | app.js:983 |
| 通信域分解 | ✅ 已解耦 | Object.keys(communication_by_domain) | app.js:1076+ |
| 通信原语分解 | ✅ 已解耦 | Object.keys(communication_by_op) | app.js:1076+ |

### 11.7 新增模型天然兼容

**无需修改前端代码**：

1. 后端添加 `_submodule_name` 属性
2. 数据结构自动包含新类型
3. 前端 `Object.keys()` 自动发现

**示例**：

```python
class ShardedNewAttention(ShardedModule):
    _submodule_name = "new_attention"  # 显式声明
```

```javascript
// 前端自动显示
for (const [type, data] of Object.entries(bySubmoduleType)) {
    // → embedding, transformer_block, lm_head, new_attention...
}
```

### 11.8 结构签名缓存机制

**问题**：Qwen3.5 混合注意力（linear + full）所有 Block 签名相同，只分析一次

**解决**：`_compute_structure_signature` 添加 `layer_type` 属性

```python
config_attrs = [
    "hidden_size", "num_heads", ...,
    "layer_type",  # 新增！区分 linear vs full
]
```

**效果**：
- linear_attention Block 和 full_attention Block 结构签名不同
- 分别缓存和分析
- 前端显示完整分解：linear_attention + attention (full) + moe
# LLM Performance Evaluator

大模型训推性能评估工具 - 在给定策略下评估模型训练和推理的理论性能。

## 功能特性

- **模型支持**: Llama、MoE (Mixture of Experts) 等基于 Transformer 的模型
- **硬件建模**: 支持 H100、A100、MI300X 等 GPU 的预设配置
- **并行策略**: 支持 TP、PP、DP、EP、SP、CP 等并行方式的灵活组合
- **性能评估**: 
  - 训练: 吞吐量 (tokens/sec)、显存占用、通信开销占比
  - 推理: TTFT (首token延迟)、TPOT (单token生成延迟)、TPS (吞吐)
- **详细分解**: 计算开销、通信开销、内存占用的时延分解

## 架构设计

```
llm_perf/
├── models/          # 模型定义 (Llama, MoE)
├── hardware/        # 硬件抽象 (Device, Cluster)
├── kernels/         # Kernel 评估 (Compute, Communication)
├── strategy/        # 切分策略 (TP, PP, DP, EP)
├── analyzer/        # 性能分析器 (Training, Inference)
├── reporter/        # 报告生成 (Table, JSON, HTML)
└── cli/             # 命令行接口
```

### 核心模块说明

#### 1. Models (`llm_perf/models/`)
- `base.py`: 基础模型类和层配置
- `llama.py`: Llama 模型实现
- `moe.py`: MoE 模型实现 (支持 EP)

#### 2. Hardware (`llm_perf/hardware/`)
- `device.py`: 单卡 GPU 配置 (算力、带宽、显存)
- `cluster.py`: 集群拓扑和通信带宽建模

#### 3. Kernels (`llm_perf/kernels/`)
- `compute.py`: 算子性能评估 (GEMM, Attention, Activation)
  - 基于 Roofline 模型
  - 支持 FLOPs 和内存带宽建模
- `communication.py`: 通信算子评估 (AllReduce, AllGather, AllToAll)
  - 支持 Ring/Tree 算法建模
  - 区分机内/机间带宽

#### 4. Strategy (`llm_perf/strategy/`)
- `base.py`: 并行策略配置 (TP/PP/DP/EP 度数)
- `planner.py`: 策略规划器 (自动搜索最优策略)

#### 5. Analyzer (`llm_perf/analyzer/`)
- `training.py`: 训练性能分析
- `inference.py`: 推理性能分析 (Prefill + Decode)
- `breakdown.py`: 性能分解 (计算/通信/内存)

#### 6. Reporter (`llm_perf/reporter/`)
- `table.py`: 控制台表格报告
- `json_reporter.py`: JSON 格式输出
- `html_reporter.py`: HTML 可视化报告

## 快速开始

### 安装

```bash
pip install -e .
```

### 运行示例

```bash
# 运行所有示例
python run_eval.py all

# 运行特定示例
python run_eval.py training
python run_eval.py inference
python run_eval.py moe

# 保存 JSON 结果
python run_eval.py all --json
```

### CLI 使用

```bash
# 评估训练性能
llm-perf evaluate \
    --model-config llm_perf/configs/model_llama7b.json \
    --hardware-config llm_perf/configs/hardware_h100_8gpu.json \
    --strategy-config llm_perf/configs/strategy_tp8.json \
    --mode training \
    --batch-size 32 \
    --seq-len 4096 \
    --json \
    --output training_result.json

# 评估推理性能
llm-perf evaluate \
    --model-config llm_perf/configs/model_llama7b.json \
    --hardware-config llm_perf/configs/hardware_h100_8gpu.json \
    --strategy-config llm_perf/configs/strategy_tp8.json \
    --mode inference \
    --batch-size 8 \
    --prompt-len 1024 \
    --generation-len 128 \
    --html \
    --output inference_result.html

# 对比多个策略
llm-perf compare \
    --model-config llm_perf/configs/model_llama70b.json \
    --hardware-config llm_perf/configs/hardware_h100_8gpu.json \
    --strategy-configs llm_perf/configs/strategy_tp8.json llm_perf/configs/strategy_tp4_dp2.json \
    --mode training \
    --metric throughput
```

## 配置文件格式

### 模型配置 (`model_*.json`)

```json
{
  "type": "llama",
  "name": "llama-7b",
  "vocab_size": 32000,
  "hidden_size": 4096,
  "num_layers": 32,
  "num_attention_heads": 32,
  "intermediate_size": 11008,
  "max_seq_len": 4096,
  "dtype": "fp16"
}
```

### 硬件配置 (`hardware_*.json`)

```json
{
  "device_preset": "H100-SXM-80GB",
  "num_devices": 8,
  "devices_per_node": 8,
  "intra_node_bw_gbps": 900,
  "inter_node_bw_gbps": 400
}
```

支持的 device preset:
- `H100-SXM-80GB`
- `H100-NVL-94GB`
- `H200-SXM-141GB`
- `A100-SXM-40GB`
- `A100-SXM-80GB`
- `MI300X`
- `L40S`

### 策略配置 (`strategy_*.json`)

```json
{
  "tp": 8,
  "pp": 1,
  "dp": 1,
  "ep": 1,
  "activation_checkpointing": true,
  "sequence_parallel": true,
  "zero_stage": 1
}
```

## 输出示例

### 控制台输出

```
================================================================================
                              TRAINING PERFORMANCE
================================================================================

[Throughput]
  Samples/sec                        0.50
  Tokens/sec                    4.10K tokens/s

[Time]
  Time per step                   64.00 s
  Time to solution              17.78 hours

[Memory]
  Memory per GPU                  76.23 GB

[Breakdown]
================================================================================
PERFORMANCE BREAKDOWN
================================================================================

Total Time: 64000.00 ms
Throughput: 2048.00 tokens/sec

--- Time Breakdown ---
  Compute:        51200.00 ms ( 80.0%)
  Communication:  12800.00 ms ( 20.0%)
  Memory Wait:        0.00 ms (  0.0%)

--- Memory Breakdown ---
  Peak Memory:       79933.15 MB
  Activations:        4096.00 MB
  Parameters:        70000.00 MB

--- Communication Details ---
  tensor_parallel: 12800.00 ms

--- Top Time-Consuming Layers ---
  layer_0_down_proj                        2048.00 ms
  layer_0_up_proj                          1536.00 ms
================================================================================
```

## 扩展开发

### 添加新的模型类型

```python
from llm_perf.models.base import BaseModel, ModelConfig, LayerConfig

class MyModelConfig(ModelConfig):
    custom_param: int = 0

class MyModel(BaseModel):
    def build_layers(self) -> List[LayerConfig]:
        # Implement layer building logic
        pass
```

### 添加新的 Kernel 评估

```python
from llm_perf.kernels.compute import ComputeKernel, ComputeKernelConfig

# Register custom matmul kernel
kernel = compute_registry.get_or_create_matmul(m=1024, n=4096, k=4096, dtype="fp16")
```

### 添加自定义通信模式

```python
from llm_perf.kernels.communication import CommKernelRegistry

# Create custom all-to-all for MoE
comm_kernel = comm_registry.create_ep_alltoall(
    layer_name="moe_layer_0",
    token_bytes=8192,
    ep_ranks=[0, 1, 2, 3]
)
```

## 算法说明

### Roofline 模型

算子性能使用 Roofline 模型估算:

```
achievable_flops = min(
    peak_flops,
    arithmetic_intensity * memory_bandwidth
)
```

其中 `arithmetic_intensity = flops / bytes_accessed`

### 通信时间估算

- **AllReduce (Ring)**: `2 * (n-1) * data_size / (n * bandwidth)`
- **AllGather**: 约为 AllReduce 的一半
- **AllToAll**: `(n-1)/n * data_size / bandwidth`

### 内存估算

- **Parameters**: `params * dtype_size / tp_degree`
- **Activations**: 取决于 batch size 和 sequence length
- **Gradients**: 与 parameters 相同 (ZeRO 可减少)
- **Optimizer States**: Adam 需要 2x parameters (fp32)
- **KV Cache**: `2 * num_layers * num_kv_heads * head_dim * seq_len * batch * dtype_size`

## 路线图

- [ ] 更精确的 FlashAttention kernel 建模
- [ ] 支持更多的并行策略 (FSDP, 3D Parallelism)
- [ ] 自动策略搜索 (DP + Genetic Algorithm)
- [ ] 集成实际的 kernel benchmark 数据
- [ ] 支持更多的模型架构 (GPT-NeoX, Falcon, etc.)
- [ ] 可视化工具 (roofline plot, timeline)

## License

MIT

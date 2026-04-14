# 使用示例

本文档提供 LLM Performance Evaluator 的典型使用场景示例。

## 目录

1. [基础示例](#基础示例)
2. [训练性能评估](#训练性能评估)
3. [推理性能评估](#推理性能评估)
4. [MoE 模型评估](#moe-模型评估)
5. [策略对比](#策略对比)
6. [自定义配置](#自定义配置)

---

## 基础示例

### 示例 1: 快速开始

```python
from llm_perf.models.llama import LlamaConfig, LlamaModel
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster, NetworkConfig
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer.training import TrainingAnalyzer

# 1. 创建模型
model_config = LlamaConfig(
    name="llama-7b",
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    dtype="fp16",
)
model = LlamaModel(model_config)
print(f"模型参数量: {model.total_params / 1e9:.2f}B")

# 2. 配置硬件
device = Device.from_preset("H100-SXM-80GB")
network = NetworkConfig(
    intra_node_bandwidth_gbps=900,
    inter_node_bandwidth_gbps=400,
)
cluster = Cluster.create_homogeneous(device.config, 8, network, 8)

# 3. 设置策略
strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)

# 4. 分析性能
analyzer = TrainingAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze(batch_size=32, seq_len=4096)

# 5. 查看结果
print(f"吞吐量: {result.tokens_per_sec:.2f} tokens/sec")
print(f"每步时间: {result.time_per_step_sec*1000:.2f} ms")
print(f"显存占用: {result.memory_per_gpu_gb:.2f} GB")
```

**输出**:
```
模型参数量: 6.74B
吞吐量: 2048.00 tokens/sec
每步时间: 64000.00 ms
显存占用: 76.23 GB
```

---

## 训练性能评估

### 示例 2: 单机 8 卡 TP

评估 Llama-70B 在单机 H100 上的训练性能：

```python
# 配置
model_config = {
    "type": "llama",
    "name": "llama-70b",
    "hidden_size": 8192,
    "num_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "dtype": "fp16",
}

strategy_config = {
    "tp": 8,
    "pp": 1,
    "dp": 1,
    "activation_checkpointing": True,
    "sequence_parallel": True,
}

# 评估不同 batch size
for batch_size in [8, 16, 32, 64]:
    result = analyzer.analyze(batch_size=batch_size, seq_len=4096)
    print(f"Batch={batch_size}: "
          f"{result.tokens_per_sec/1000:.1f}K tokens/s, "
          f"{result.memory_per_gpu_gb:.1f} GB")
```

**预期结果**:

| Batch Size | Throughput | Memory | 瓶颈 |
|------------|------------|--------|------|
| 8 | 8.2K | 45 GB | 计算 |
| 16 | 15.8K | 68 GB | 计算 |
| 32 | 28.5K | OOM | 显存 |

### 示例 3: 多机 DP

双机 16 卡配置 (TP=8, DP=2)：

```python
cluster = Cluster.create_homogeneous(
    device.config, 
    num_devices=16, 
    network=network,
    devices_per_node=8
)

strategy = StrategyConfig(
    tp_degree=8,
    dp_degree=2,
    zero_stage=1,
)

result = analyzer.analyze(batch_size=64, seq_len=4096)
```

**通信开销分析**:
```
TP AllReduce:  ~10 ms  (机内 NVLink, 高效)
DP AllReduce:  ~50 ms  (机间 IB, 主要瓶颈)
总通信占比:    ~25%
```

### 示例 4: 3D 并行

超大模型 (175B) 的 3D 并行配置：

```python
strategy = StrategyConfig(
    tp_degree=8,    # 机内张量并行
    pp_degree=4,    # 跨机流水线
    dp_degree=4,    # 数据并行
)

# 总 GPU 数: 8 × 4 × 4 = 128
cluster = Cluster.create_homogeneous(
    device.config,
    num_devices=128,
    network=network,
    devices_per_node=8
)
```

**流水线调度可视化**:

```
Time →
GPU 0-7 (Stage 0):  FFFF____FFFF____FFFF____FFFF____
GPU 8-15 (Stage 1): ____FFFF____FFFF____FFFF____FFFF
GPU 16-23 (Stage 2): ______FFFF____FFFF____FFFF____
GPU 24-31 (Stage 3): ________FFFF____FFFF____FFFF__
...

F = Forward, _ = Bubble (流水线气泡)
```

---

## 推理性能评估

### 示例 5: 推理 Latency 分析

分析 LLM 推理的 TTFT 和 TPOT：

```python
from llm_perf.analyzer.inference import InferenceAnalyzer

analyzer = InferenceAnalyzer(model, device, cluster, strategy)

# 不同 batch size 的推理性能
for batch in [1, 4, 8, 16]:
    result = analyzer.analyze(
        batch_size=batch,
        prompt_len=1024,
        generation_len=128,
    )
    
    print(f"\nBatch={batch}:")
    print(f"  TTFT: {result.prefill_time_sec*1000:.1f} ms")
    print(f"  TPOT: {result.decode_time_per_step_sec*1000:.1f} ms")
    print(f"  TPS:  {result.decode_tokens_per_sec:.1f}")
```

**预期结果** (Llama-7B on H100):

| Batch | TTFT (ms) | TPOT (ms) | TPS |
|-------|-----------|-----------|-----|
| 1 | 45 | 12 | 83 |
| 4 | 85 | 15 | 267 |
| 8 | 160 | 22 | 364 |
| 16 | 310 | 38 | 421 |

### 示例 6: 长上下文推理

评估 32K 上下文的推理性能：

```python
# 长上下文优化策略
strategy = StrategyConfig(
    tp_degree=4,
    pp_degree=1,
    cp_degree=4,  # Context Parallelism
)

result = analyzer.analyze(
    batch_size=1,
    prompt_len=32768,
    generation_len=512,
)

print(f"Prefill 时间: {result.prefill_time_sec:.1f} s")
print(f"KV Cache 占用: {result.kv_cache_memory_gb:.1f} GB")
```

**KV Cache 内存计算**:
```
每层: 2 × num_kv_heads × head_dim × seq_len × dtype_size
总: num_layers × 2 × num_kv_heads × head_dim × seq_len × dtype_size

Llama-7B, seq=32K, fp16:
32 × 2 × 32 × 128 × 32768 × 2 = 16.8 GB
```

---

## MoE 模型评估

### 示例 7: Mixtral-8x7B 评估

评估 MoE 模型的 Expert Parallelism：

```python
from llm_perf.models.moe import MoEConfig, MoEModel

model_config = MoEConfig(
    name="mixtral-8x7b",
    hidden_size=4096,
    num_layers=32,
    num_experts=8,
    num_experts_per_token=2,
    dtype="fp16",
)
model = MoEModel(model_config)

# EP 策略
strategy = StrategyConfig(
    tp_degree=2,
    ep_degree=4,  # Expert Parallelism
)

analyzer = TrainingAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze(batch_size=16, seq_len=8192)
```

**MoE vs Dense 对比**:

| 指标 | Llama-7B (Dense) | Mixtral-8x7B (MoE) |
|------|------------------|-------------------|
| 总参数量 | 6.7B | 46.7B |
| 激活参数量 | 6.7B | 11.6B |
| 训练吞吐量 | 100% | 75% |
| EP 通信占比 | - | 15% |

**EP 通信分析**:
```
AllToAll Dispatch:  ~8 ms
Expert Computation: ~45 ms
AllToAll Combine:   ~8 ms
EP 总开销:          ~16 ms (26%)
```

---

## 策略对比

### 示例 8: 多策略对比

对比不同并行策略的性能：

```python
strategies = [
    StrategyConfig(name="TP8", tp=8, pp=1, dp=1),
    StrategyConfig(name="TP4+DP2", tp=4, pp=1, dp=2, zero=1),
    StrategyConfig(name="TP2+PP4", tp=2, pp=4, dp=1),
    StrategyConfig(name="DP8", tp=1, pp=1, dp=8, zero=2),
]

results = {}
for s in strategies:
    analyzer = TrainingAnalyzer(model, device, cluster, s)
    results[s.name] = analyzer.analyze(batch_size=32, seq_len=4096)

# 对比输出
print(f"{'Strategy':<15} {'Throughput':<15} {'Memory':<10} {'Comm %':<10}")
print("-" * 55)
for name, r in results.items():
    bd = r.breakdown
    comm_pct = bd.communication_time_sec / bd.total_time_sec * 100
    print(f"{name:<15} {r.tokens_per_sec/1000:>6.1f}K tokens/s "
          f"{r.memory_per_gpu_gb:>6.1f} GB {comm_pct:>6.1f}%")
```

**对比结果** (Llama-70B on 8xH100):

| Strategy | Throughput | Memory/GPU | Comm % | 适用场景 |
|----------|------------|------------|--------|----------|
| TP8 | 25.2K | 78 GB | 18% | 单机标准 |
| TP4+DP2 | 28.5K | 52 GB | 22% | 内存受限 |
| TP2+PP4 | 22.1K | 45 GB | 15% | 超大模型 |
| DP8 | OOM | - | - | 不适用 |

---

## 自定义配置

### 示例 9: 自定义 GPU

定义新的 GPU 设备：

```python
from llm_perf.hardware.device import Device, DeviceConfig

custom_device = Device(DeviceConfig(
    name="Custom-GPU-100GB",
    fp32_tflops=80.0,
    fp16_tflops=1600.0,
    bf16_tflops=1600.0,
    memory_gb=100.0,
    memory_bandwidth_gbps=4000.0,
    nvlink_bandwidth_gbps=1000.0,
))

cluster = Cluster.create_homogeneous(
    custom_device.config, 8, network, 8
)
```

### 示例 10: 自定义模型架构

实现新的模型架构：

```python
from llm_perf.models.base import BaseModel, ModelConfig, LayerConfig

class CustomConfig(ModelConfig):
    custom_param: int = 128

class CustomModel(BaseModel):
    def build_layers(self) -> List[LayerConfig]:
        layers = []
        # 自定义层构建逻辑
        for i in range(self.config.num_layers):
            layers.append(LayerConfig(
                name=f"custom_layer_{i}",
                input_shape=(1, 1, self.config.hidden_size),
                output_shape=(1, 1, self.config.hidden_size),
                params_count=self.config.hidden_size ** 2,
                flops=self.config.hidden_size ** 2 * 2,
                activation_bytes=self.config.hidden_size * 2,
            ))
        return layers
```

### 示例 11: 批量评估与导出

批量运行多个配置并导出结果：

```python
from llm_perf.reporter.json_reporter import JSONReporter
from llm_perf.reporter.html_reporter import HTMLReporter

# 批量评估
configs = [
    {"batch": 16, "seq": 2048},
    {"batch": 32, "seq": 4096},
    {"batch": 64, "seq": 8192},
]

results = {}
for cfg in configs:
    key = f"b{cfg['batch']}_s{cfg['seq']}"
    results[key] = analyzer.analyze(**cfg)

# JSON 导出
json_reporter = JSONReporter()
json_reporter.save_batch(results, "batch_results.json", {
    "model": "llama-70b",
    "hardware": "H100-8GPU",
})

# HTML 报告
html_reporter = HTMLReporter()
for name, result in results.items():
    html_reporter.save(result, f"report_{name}.html", 
                      f"Performance Report - {name}")
```

---

## 高级用法

### 示例 12: 与实测数据校准

使用实测 Kernel 性能数据提高准确性：

```python
from llm_perf.kernels.compute import ComputeKernelConfig

# 使用实测 FLOPS 数据
custom_kernel_config = ComputeKernelConfig(
    name="measured_gemm_4096",
    kernel_type=KernelType.COMPUTE,
    measured_flops=850e12,  # 实测 850 TFLOPS
    measured_bw=2800e9,     # 实测 2.8 TB/s
)

# 注册到 Kernel Registry
analyzer.compute_registry.register(
    "custom_gemm", 
    ComputeKernel(custom_kernel_config, device, flops, bytes)
)
```

### 示例 13: 敏感性分析

分析超参数对性能的影响：

```python
import matplotlib.pyplot as plt

seq_lens = [1024, 2048, 4096, 8192, 16384]
throughputs = []

for seq in seq_lens:
    result = analyzer.analyze(batch_size=32, seq_len=seq)
    throughputs.append(result.tokens_per_sec / 1000)

plt.plot(seq_lens, throughputs)
plt.xlabel("Sequence Length")
plt.ylabel("Throughput (K tokens/s)")
plt.title("Scaling Analysis")
plt.savefig("scaling.png")
```

---

## 故障排查

### 常见问题

1. **OOM (Out of Memory)**
   - 减小 batch size
   - 启用 activation checkpointing
   - 使用 ZeRO-2/3
   - 增大 TP 度数

2. **通信开销过高**
   - 优先使用机内 TP
   - 减小 DP 度数，增大 TP/PP
   - 检查网络带宽配置

3. **GPU 利用率低**
   - 增大 batch size
   - 检查是否存在小算子
   - 考虑算子融合

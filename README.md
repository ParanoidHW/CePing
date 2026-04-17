# LLM Performance Evaluator

大模型训推性能评估工具 - 在给定策略下评估模型训练和推理的理论性能。

## 核心特性

### 🚀 一行代码评估性能

```python
from llm_perf.modeling import create_model_from_config
from llm_perf.analyzer.training import TrainingAnalyzer

model = create_model_from_config({"preset": "llama-7b"})
result = TrainingAnalyzer(model, device, cluster, strategy).analyze(batch_size=32)
print(f"吞吐量: {result.tokens_per_sec:.1f} tokens/s")
```

### 🧠 支持主流模型架构

| 模型类型 | 支持模型 | 特性 |
|---------|---------|------|
| **Dense** | Llama-7B/13B/70B, GPT | GQA、RoPE、SwiGLU |
| **MoE** | Mixtral-8x7B, DeepSeek-V3 | Expert Parallelism、共享专家 |
| **Vision** | ResNet, VAE | CNN、图像分类/生成 |
| **Video** | Wan2.1-T2V | 文本编码器 + DiT + 3D VAE |

### 🖥️ 多硬件平台支持

| 平台 | 设备 | 特性 |
|------|------|------|
| **NVIDIA** | H100/H200, A100, L40S | Tensor Core / CUDA Core 分离 |
| **AMD** | MI300X | Matrix Core / Stream Processor 分离 |
| **华为** | Ascend 910B/C, 950/960/970 | CUBE Core / VECTOR Core 分离 |

### ⚡ 全套并行策略

- **TP**: 张量并行
- **PP**: 流水线并行
- **DP**: 数据并行 + ZeRO-1/2/3
- **EP**: 专家并行
- **SP/CP**: 序列并行、上下文并行

### 📊 详细性能分解

```
Total Time: 64000.00 ms
├── Compute:     51200.00 ms (80.0%)
├── Communication: 12800.00 ms (20.0%)
└── Memory Wait:     0.00 ms ( 0.0%)

Memory Breakdown:
├── Parameters:    70.00 GB
├── Activations:    4.00 GB
└── Gradients:     70.00 GB
```

### 🌐 网络拓扑建模

- **2-Tier Simple**: 机内 NVLink + 机间 IB
- **3-Tier Clos**: Leaf-Spine-Core 三级交换
- **Fat-Tree**: 数据中心胖树拓扑
- **CloudMatrix 超节点**: 华为 384 NPU 全对等

---

## 快速开始

### 安装

```bash
pip install -e .
```

### 方式一：Web 可视化界面（推荐）

```bash
pip install flask flask-cors cryptography
cd web && python app.py
# 访问 https://localhost:8443
```

### 方式二：Python API

#### 训练性能评估

```python
from llm_perf.modeling import create_model_from_config
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer.training import TrainingAnalyzer

# 创建模型（使用 preset）
model = create_model_from_config({"preset": "llama-7b"})

# 配置硬件
device = Device.from_preset("H100-SXM-80GB")
cluster = Cluster.create_homogeneous(device.config, num_devices=8)

# 设置策略
strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)

# 评估性能
analyzer = TrainingAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze(batch_size=32, seq_len=4096)

# 输出结果
print(f"吞吐量: {result.tokens_per_sec:.1f} tokens/s")
print(f"显存: {result.memory_per_gpu_gb:.2f} GB")
print(f"通信占比: {result.breakdown.communication_time_sec / result.breakdown.total_time_sec * 100:.1f}%")
```

#### 推理性能评估

```python
from llm_perf.analyzer.inference import InferenceAnalyzer

analyzer = InferenceAnalyzer(model, device, cluster, strategy)
result = analyzer.analyze(
    batch_size=8,
    prompt_len=1024,
    generation_len=128,
)

print(f"TTFT: {result.prefill_time_sec*1000:.1f} ms")
print(f"TPOT: {result.decode_time_per_step_sec*1000:.1f} ms")
print(f"TPS:  {result.decode_tokens_per_sec:.1f} tokens/s")
```

#### MoE 模型评估

```python
# 使用 DeepSeek-V3 preset
model = create_model_from_config({"preset": "deepseek-v3"})

# EP 策略
strategy = StrategyConfig(
    tp_degree=4,
    ep_degree=8,
    num_experts=256,
)

result = analyzer.analyze(batch_size=16, seq_len=8192)
print(f"EP 通信占比: {result.ep_communication_ratio:.1f}%")
```

### 方式三：高级工具 API

#### Evaluator - 一行代码快速评估

```python
from llm_perf.app import Evaluator

evaluator = Evaluator()

# 训练评估 - 使用 preset
result = evaluator.evaluate_training(
    model="llama-7b",
    hardware="h100_8gpu",
    strategy="tp8",
    batch_size=32,
)
print(f"吞吐量: {result.tokens_per_sec:.1f} tokens/s")

# 推理评估 - 使用 preset
result = evaluator.evaluate_inference(
    model="llama-70b",
    hardware="h200_8gpu",
    strategy="tp8",
    batch_size=8,
    prompt_len=1024,
    generation_len=128,
)
print(f"TTFT: {result.prefill_time_sec*1000:.1f} ms")
print(f"TPS:  {result.decode_tokens_per_sec:.1f}")

# 策略对比
comparison = evaluator.compare_strategies(
    model="llama-7b",
    hardware="h100_8gpu",
    strategies=["tp1", "tp2", "tp4", "tp8"],
    mode="training",
    batch_size=32,
)
print(f"最佳策略: {comparison['best_strategy']}")
```

#### StrategyOptimizer - 自动搜索最优策略

```python
from llm_perf.app import StrategyOptimizer, StrategyConstraints, OptimizeObjective

optimizer = StrategyOptimizer()

# 设置约束
constraints = StrategyConstraints(
    max_gpus=8,
    max_memory_gb=80,
    require_tp=True,
)

# 搜索最优策略 (Grid搜索)
result = optimizer.search_best_strategy(
    model="llama-70b",
    hardware="h100_8gpu",
    mode="training",
    constraints=constraints,
    objective=OptimizeObjective.THROUGHPUT,
)
print(f"最优策略: TP={result.best_strategy.tp_degree}")
print(f"吞吐量: {result.best_metric:.1f} samples/s")
print(f"搜索时间: {result.search_time_sec:.2f}s")

# 对比特定策略
comparison = optimizer.compare_strategies(
    model="llama-7b",
    hardware="h100_8gpu",
    strategies=["tp8", "tp4_dp2", "tp2_pp4"],
    mode="training",
    batch_size=32,
)
```

#### BatchOptimizer - 在约束下找最大 Batch

```python
from llm_perf.app import BatchOptimizer, LatencyBudget

optimizer = BatchOptimizer()

# 训练场景：内存约束下找最大 batch
result = optimizer.find_max_batch(
    model="llama-7b",
    hardware="h100_8gpu",
    strategy="tp4",
    mode="training",
    memory_budget_gb=80,
)
print(f"最大 Batch: {result.best_batch_size}")
print(f"停止原因: {result.reason}")

# 推理场景：TTFT 约束
result = optimizer.find_max_batch(
    model="llama-7b",
    hardware="h100_8gpu",
    strategy="tp4",
    mode="inference",
    latency_budget=LatencyBudget(ttft_budget_ms=50.0),
)

# 推理场景：TPOT 约束
result = optimizer.find_max_batch(
    model="llama-7b",
    hardware="h100_8gpu",
    strategy="tp4",
    mode="inference",
    latency_budget=LatencyBudget(tpot_budget_ms=2.0),
)

# 推理场景：总延迟约束
result = optimizer.find_max_batch(
    model="llama-7b",
    hardware="h100_8gpu",
    strategy="tp4",
    mode="inference",
    latency_budget=LatencyBudget(
        total_latency_budget_ms=500.0,
        generation_len=128,
    ),
)

# 综合优化：策略 + Batch 联合搜索
result = optimizer.find_max_tps(
    model="llama-7b",
    hardware="h100_8gpu",
    mode="training",
)
print(f"最优配置: {result['best_strategy']}, batch={result['best_batch_size']}")
```

### 方式四：CLI 命令行

```bash
# 训练评估
llm-perf evaluate \
    --model llama-7b \
    --device H100-SXM-80GB \
    --num-devices 8 \
    --tp 8 \
    --mode training \
    --batch-size 32 \
    --seq-len 4096

# 推理评估
llm-perf evaluate \
    --model llama-70b \
    --device H200-SXM-141GB \
    --num-devices 16 \
    --tp 8 --dp 2 \
    --mode inference \
    --batch-size 8 \
    --prompt-len 1024 \
    --generation-len 128

# 策略对比
llm-perf compare \
    --model llama-70b \
    --device H100 \
    --strategies tp8 tp4_dp2 tp2_pp4 \
    --metric throughput
```

---

## 使用 Presets 快速配置

### 模型 Presets

```python
from llm_perf.modeling import get_model_presets, create_model_from_config

# 查看所有 presets
presets = get_model_presets()
# {'llama-7b', 'llama-70b', 'deepseek-v3', 'mixtral-8x7b', 'resnet50', ...}

# 使用 preset 创建模型
model = create_model_from_config({"preset": "llama-7b"})
model = create_model_from_config({"preset": "deepseek-v3"})
model = create_model_from_config({"preset": "mixtral-8x7b"})
```

### 硬件 Presets

```python
from llm_perf.hardware.device import Device

device = Device.from_preset("H100-SXM-80GB")
device = Device.from_preset("H200-SXM-141GB")
device = Device.from_preset("MI300X")
device = Device.from_preset("Ascend-910C")
```

### 策略 Presets

```python
from llm_perf.utils.config_loader import ConfigLoader

strategy = ConfigLoader.load_strategy_config("tp8")
strategy = ConfigLoader.load_strategy_config("tp4_dp2")
strategy = ConfigLoader.load_strategy_config("megatron_sp")
```

---

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

[Memory]
  Memory per GPU                  76.23 GB

[Breakdown]
  Compute:        51200.00 ms ( 80.0%)
  Communication:  12800.00 ms ( 20.0%)
  
--- Top Time-Consuming Layers ---
  layer_0_down_proj    2048.00 ms
  layer_0_up_proj      1536.00 ms
================================================================================
```

---

## 架构设计

```
llm_perf/
├── modeling/         # PyTorch-like 建模框架
│   ├── models.py     # Llama, DeepSeek, MoE
│   ├── layers.py     # ShardedAttention, ShardedFFN, ShardedMoE
│   ├── mla.py        # Multi-head Latent Attention
│   ├── registry.py   # Presets & create_model_from_config
│   └── vision_models.py  # ResNet, VAE
├── hardware/         # 硬件抽象
│   ├── device.py     # GPU/NPU 配置
│   ├── cluster.py    # 集群拓扑
│   └── topology.py   # 网络层级建模
├── kernels/          # Kernel 评估
│   ├── compute.py    # GEMM, Attention, Activation
│   └── communication.py  # AllReduce, AllToAll
├── strategy/         # 并行策略
├── analyzer/         # 性能分析器
│   ├── training.py   # 训练分析
│   └── inference.py  # 推理分析 (Prefill + Decode)
├── reporter/         # 报告生成
└── web/              # Web 可视化界面
```

---

## 扩展开发

### 添加新模型

```python
from llm_perf.modeling import ShardedModule, ShardedTensor, ModelingRegistry

class MyModel(ShardedModule):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.weight = ShardedTensor(
            shape=(hidden_size, hidden_size),
            shardable={0: "tp"},  # 支持 TP 分片
            dtype="fp16",
        )
    
    def forward(self, x):
        return x @ self.weight

ModelingRegistry().register("my_model", MyModel)
```

---

## 路线图

- [ ] FlashAttention kernel 精确建模
- [ ] FSDP、Ulysses-SP、Unified-SP 支持
- [ ] 自动策略搜索 (遗传算法)
- [ ] Kernel benchmark 数据集成
- [ ] Roofline plot 可视化
- [ ] 多模态理解、Agentic、RL训练场景

## License

MIT
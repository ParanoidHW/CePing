# LLM Performance Evaluator

大模型训推性能评估工具 - 在给定策略下评估模型训练和推理的理论性能。

## 功能特性

### 🧠 模型支持
- **Dense 模型**: Llama、LLaMA-2、LLaMA-3 等 Transformer 架构
- **MoE 模型**: Mixtral 8x7B、DeepSeek-V3、Qwen-MoE 等专家混合模型
- **自定义配置**: 灵活调整 hidden size、layers、attention heads、experts 等参数

### 🖥️ 硬件建模
- **NVIDIA GPU**: H100/H200、A100、L40S（支持 Tensor Core / CUDA Core 分离）
- **AMD GPU**: MI300X（支持 Matrix Core / Stream Processor 分离）
- **华为昇腾 NPU**: Ascend 910A/B/C、950/960/970（支持 CUBE Core / VECTOR Core 分离）
- **精确算力建模**: 基于 Roofline 模型，区分矩阵运算和向量运算单元

### 🌐 网络拓扑（新特性）
- **2-Tier Simple**: 机内 NVLink + 机间 IB/Ethernet
- **3-Tier Clos**: Leaf-Spine-Core 三级交换架构
- **Fat-Tree**: 数据中心级胖树拓扑，支持超配配置
- **CloudMatrix 超节点**: 华为 384 NPU 全对等超节点（arXiv:2506.12708）
- **分层带宽建模**: 支持带宽随拓扑层级递减的真实场景

### ⚡ 并行策略
- **TP (Tensor Parallelism)**: 张量并行，支持 Megatron 风格
- **PP (Pipeline Parallelism)**: 流水线并行，支持 1F1B 调度
- **DP (Data Parallelism)**: 数据并行，支持 ZeRO-1/2/3
- **EP (Expert Parallelism)**: 专家并行，针对 MoE 模型优化
- **SP/CP**: 序列并行、上下文并行支持

### 📊 性能评估
- **训练评估**: 
  - 吞吐量: samples/sec、tokens/sec
  - 显存占用: 参数、梯度、优化器状态、激活值
  - 通信开销: AllReduce/AllToAll 时间分解
- **推理评估**:
  - TTFT (Time To First Token): 首 token 延迟
  - TPOT (Time Per Output Token): 每 token 生成时间
  - TPS (Tokens Per Second): 吞吐率
  - KV Cache 显存占用分析

### 🌟 交互式 Web 界面（新特性）
- **可视化配置**: 通过网页交互式选择模型、硬件、拓扑、策略
- **实时评估**: 即时获取性能评估结果和分解
- **HTTPS 支持**: 本地安全服务，支持自签名证书
- **扁平化设计**: 简约现代的 UI 风格

### 📚 完整文档
- **架构文档**: 系统设计和模块说明
- **数据来源 Wiki**: 硬件参数、FLOPs 计算、参考资料汇总
- **拓扑文档**: Clos/Fat-Tree/CloudMatrix 网络建模详解
- **示例代码**: 多种使用场景的示例配置

## 架构设计

```
llm_perf/
├── models/          # 模型定义 (Llama, MoE)
├── hardware/        # 硬件抽象 (Device, Cluster, Topology)
├── kernels/         # Kernel 评估 (Compute, Communication)
├── strategy/        # 切分策略 (TP, PP, DP, EP)
├── analyzer/        # 性能分析器 (Training, Inference)
├── reporter/        # 报告生成 (Table, JSON, HTML)
├── cli/             # 命令行接口
└── web/             # Web 可视化界面 (HTTPS)
    ├── app.py       # Flask 后端服务
    ├── static/      # CSS/JS 前端资源
    └── templates/   # HTML 模板
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

### Web 可视化界面（推荐）

启动本地 HTTPS Web 服务，通过浏览器交互式配置和评估：

```bash
# 安装依赖
pip install flask flask-cors cryptography

# 启动 HTTPS 服务（默认 https://localhost:8443）
cd web
python app.py

# 或使用 HTTP 模式
python app.py --http
```

访问 https://localhost:8443，即可使用可视化界面：
- 选择模型预设（Llama-7B/70B, Mixtral-8x7B, DeepSeek-V3）
- 配置硬件（NVIDIA/AMD/Huawei GPU）
- 设置网络拓扑（2-Tier/3-Tier Clos/Fat-Tree/CloudMatrix）
- 调整并行策略（TP/PP/DP/EP）
- 实时获取性能评估结果

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

## Web 服务详细说明

### 环境搭建

#### 1. 安装 Python 依赖

```bash
# 基础依赖
pip install flask flask-cors

# SSL 证书生成（可选，用于 HTTPS）
pip install cryptography

# 或使用 requirements.txt
cd web
pip install -r requirements.txt
```

#### 2. 生成 SSL 证书（HTTPS）

首次运行时会自动生成自签名证书，或手动生成：

```bash
cd web/certs

# 使用 OpenSSL
openssl req -x509 -newkey rsa:2048 -keyout server.key -out server.crt -days 365 -nodes

# 或使用 mkcert（推荐，会生成本地信任证书）
mkcert -install
mkcert localhost 127.0.0.1
```

### 启动服务

```bash
cd web

# HTTPS 模式（默认）
python app.py
# 访问 https://localhost:8443

# 指定端口
python app.py --port 8080

# HTTP 模式（开发调试）
python app.py --http
# 访问 http://localhost:8443

# 绑定所有接口
python app.py --host 0.0.0.0 --port 8080
```

### 功能特性

#### 支持的配置项

| 类别 | 配置项 |
|------|--------|
| **模型** | Llama/Dense, MoE (Mixtral/DeepSeek) |
| **模型参数** | Hidden size, Layers, Attention heads, Experts (MoE) |
| **硬件** | NVIDIA (H100/A100/L40S), AMD (MI300X), Huawei (Ascend 910B/C/950/960/970) |
| **网络拓扑** | 2-Tier Simple, 3-Tier Clos, Fat-Tree, CloudMatrix Supernode |
| **并行策略** | TP (Tensor), PP (Pipeline), DP (Data), EP (Expert) |
| **优化选项** | Activation Checkpointing, ZeRO Stage |
| **训练参数** | Batch size, Sequence length |
| **推理参数** | Batch size, Prompt length, Generation length |

#### 网络拓扑可视化

Web 界面支持配置和可视化多种网络拓扑：

1. **2-Tier Simple**: 机内 NVLink + 机间 IB
2. **3-Tier Clos**: Leaf-Spine-Core 三级交换
3. **Fat-Tree**: 数据中心常用胖树拓扑
4. **CloudMatrix Supernode**: 华为 384 NPU 全对等超节点

### 架构

```
web/
├── app.py              # Flask 后端服务
├── static/
│   ├── css/
│   │   └── style.css   # 扁平化简约样式
│   └── js/
│       └── app.js      # 前端交互逻辑
├── templates/
│   └── index.html      # 主页面
└── certs/              # SSL 证书目录
    ├── server.crt
    └── server.key
```

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/devices` | GET | 获取设备预设列表 |
| `/api/model/presets` | GET | 获取模型预设 |
| `/api/topology/presets` | GET | 获取拓扑预设 |
| `/api/evaluate/training` | POST | 训练性能评估 |
| `/api/evaluate/inference` | POST | 推理性能评估 |

### 示例：通过 API 直接调用

```bash
# 训练评估
curl -X POST https://localhost:8443/api/evaluate/training \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"type": "llama", "hidden_size": 4096, "num_layers": 32, ...},
    "device": "H100-SXM-80GB",
    "num_devices": 8,
    "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 900, ...},
    "strategy": {"tp": 8, "pp": 1, "dp": 1, ...},
    "training": {"batch_size": 32, "seq_len": 4096}
  }'
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
- [ ] 支持更多的并行策略 (FSDP, 3D Parallelism，Ulysses-SP，Unified-SP)
- [ ] 自动策略搜索 (DP + Genetic Algorithm)
- [ ] 集成实际的 kernel benchmark 数据
- [ ] 支持更多的模型架构 (DeepSeek-V3, LongCat-Flash, etc.)
- [ ] 可视化工具 (roofline plot, timeline)
- [ ] 支持更多场景（LLM、多模态理解、多模态生成、Agentic、RL训练）

## License

MIT

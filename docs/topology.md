# 网络拓扑与分层带宽模型

本文档介绍 LLM Performance Evaluator 的分层网络拓扑支持，包括 Clos、Fat-Tree 等多级交换架构。

## 1. 为什么需要分层拓扑

在大规模集群（数百到数千 GPU）中，网络拓扑通常采用分层结构：

```
┌─────────────────────────────────────────────────────────────────┐
│                        Core Switches                            │
│                      (Lowest Bandwidth)                         │
│                         100 Gbps                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│ Spine Switch  │   │ Spine Switch  │   │ Spine Switch  │
│   200 Gbps    │   │   200 Gbps    │   │   200 Gbps    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
    ┌───┴───┐           ┌───┴───┐           ┌───┴───┐
    │       │           │       │           │       │
┌───▼───┐ ┌─▼───┐   ┌───▼───┐ ┌─▼───┐   ┌───▼───┐ ┌─▼───┐
│ Leaf  │ │ Leaf│   │ Leaf  │ │ Leaf│   │ Leaf  │ │ Leaf│
│ 400G  │ │ 400G│   │ 400G  │ │ 400G│   │ 400G  │ │ 400G│
└───┬───┘ └──┬──┘   └───┬───┘ └──┬──┘   └───┬───┘ └──┬──┘
    │        │          │        │          │        │
  ┌─┴─┐    ┌─┴─┐      ┌─┴─┐    ┌─┴─┐      ┌─┴─┐    ┌─┴─┐
  │GPU│    │GPU│      │GPU│    │GPU│      │GPU│    │GPU│
  │x8 │    │x8 │      │x8 │    │x8 │      │x8 │    │x8 │
  └───┘    └───┘      └───┘    └───┘      └───┘    └───┘
```

### 1.1 带宽递减特性

| 层级 | 带宽范围 | 延迟 | 典型技术 |
|------|---------|------|---------|
| **Level 0 (Node)** | 400-900 GB/s | 1 μs | NVLink, HCCS |
| **Level 1 (Rack)** | 100-400 Gbps | 2-5 μs | InfiniBand HDR |
| **Level 2 (Cluster)** | 25-100 Gbps | 5-10 μs | InfiniBand NDR, Ethernet |

## 2. 拓扑类型

### 2.1 Clos 拓扑

Clos 网络是一种多级交换架构，提供非阻塞或近似非阻塞的连接。

**特点**:
- 多级交换机（Leaf → Spine → Core）
- 每级带宽递减
- 良好的扩展性（可支持 1000+ 节点）

**配置示例**:
```python
from llm_perf.hardware.topology import NetworkTopology

topology = NetworkTopology.create_clos_3tier(
    node_bw_gbps=900,      # NVLink within node
    rack_bw_gbps=200,      # Leaf to Spine
    cluster_bw_gbps=100,   # Spine to Core
    devices_per_node=8,
    nodes_per_rack=16,
    racks_per_cluster=32,
)
```

### 2.2 Fat-Tree 拓扑

Fat-Tree 是数据中心常用的拓扑，与 Clos 类似但强调带宽收敛比。

**特点**:
- 边缘带宽 > 汇聚带宽 > 核心带宽
- 常见的超配比例：4:1 或 8:1

**配置示例**:
```python
topology = NetworkTopology.create_fat_tree(
    core_bw_gbps=100,
    agg_bw_gbps=400,
    edge_bw_gbps=800,
    oversubscription=4.0,
)
```

### 2.3 简单 2 层拓扑

适用于小规模集群（< 100 节点）。

```python
topology = NetworkTopology.create_2tier_simple(
    intra_node_bw_gbps=900,
    inter_node_bw_gbps=200,
)
```

## 3. 通信评估模型

### 3.1 AllReduce 分层算法

在 Clos 拓扑中，AllReduce 采用分层算法：

1. **Intra-Node (Level 0)**: Ring AllReduce within NVLink domain
2. **Intra-Rack (Level 1)**: Tree reduction across leaf switches
3. **Inter-Rack (Level 2)**: Ring across spine/core

```
Time = T_intra_node + T_intra_rack + T_inter_rack

Where:
  T_intra_node = 2*(n0-1)/n0 * data / bw0 + n0*lat0
  T_intra_rack = 2*(n1-1)/n1 * data / bw1 + n1*lat1
  T_inter_rack = 2*(n2-1)/n2 * data / bw2 + n2*lat2
```

### 3.2 带宽瓶颈分析

对于跨越多层的通信，瓶颈由最低带宽决定：

```python
# Example: 256 GPUs across 3 tiers
bw_node = 900 GB/s     # NVLink
bw_rack = 200 Gbps     # IB HDR
bw_cluster = 100 Gbps  # IB NDR

# AllReduce time is dominated by inter-rack communication
# when data size is large
```

## 4. 配置示例

### 4.1 JSON 配置

```json
{
  "device_preset": "H100-SXM-80GB",
  "num_devices": 256,
  "devices_per_node": 8,
  "topology": {
    "name": "clos_3tier_256gpu",
    "topology_type": "clos",
    "levels": [
      {
        "name": "node",
        "level": 0,
        "bandwidth_gbps": 900,
        "latency_us": 1.0,
        "devices_per_group": 8
      },
      {
        "name": "rack",
        "level": 1,
        "bandwidth_gbps": 200,
        "latency_us": 2.0,
        "oversubscription_ratio": 4.0,
        "devices_per_group": 128
      },
      {
        "name": "cluster",
        "level": 2,
        "bandwidth_gbps": 100,
        "latency_us": 5.0,
        "devices_per_group": 256
      }
    ]
  }
}
```

### 4.2 Python API

```python
from llm_perf.hardware import Cluster, NetworkTopology
from llm_perf.hardware.topology import ClosTopologyBuilder

# Method 1: Manual configuration
topology = NetworkTopology.create_clos_3tier(
    node_bw_gbps=900,
    rack_bw_gbps=200,
    cluster_bw_gbps=100,
)

cluster = Cluster.create_from_preset(
    preset_name="H100-SXM-80GB",
    num_devices=256,
    topology=topology,
    devices_per_node=8,
)

# Method 2: Automatic Clos calculation
cluster = Cluster.create_with_clos_topology(
    preset_name="H100-SXM-80GB",
    num_devices=256,
    host_bw_gbps=200,
    leaf_bw_gbps=200,
    spine_bw_gbps=200,
    switch_radix=32,
    oversubscription=4.0,
)

# Estimate AllReduce time
comm_time = cluster.estimate_allreduce_time(
    num_bytes=1_000_000_000,  # 1 GB
    participating_ranks=list(range(256))
)
print(f"AllReduce time: {comm_time*1000:.2f} ms")
```

## 5. 性能对比

### 5.1 不同拓扑的 AllReduce 性能

假设：256 GPUs, 1GB data, H100

| 拓扑 | 配置 | 估计时间 | 说明 |
|------|------|---------|------|
| **Full Mesh** | 900 GB/s uniform | 5.7 ms | 理论最优，不可扩展 |
| **2-Tier** | 900/200 GB/s | 25.6 ms | 小型集群 (<100节点) |
| **Clos 3-Tier** | 900/200/100 | 48.2 ms | 大规模集群 (256+节点) |
| **Fat-Tree 4:1** | 800/400/100 | 52.3 ms | 典型数据中心配置 |

### 5.2 带宽利用率

```
Effective Bandwidth = Data Size / Communication Time

For 3-tier Clos:
  - Intra-node: ~80-90% efficiency (NVLink)
  - Intra-rack: ~70-80% efficiency (IB)
  - Inter-rack: ~60-70% efficiency (routing overhead)
```

## 6. 最佳实践

### 6.1 拓扑选择指南

| 集群规模 | 推荐拓扑 | 说明 |
|---------|---------|------|
| 8-64 GPUs | 2-Tier | 单层交换机即可覆盖 |
| 64-256 GPUs | Clos 2-Tier | Leaf-Spine 架构 |
| 256-1024 GPUs | Clos 3-Tier | 需 Core 层扩展 |
| 1024+ GPUs | Fat-Tree + DDC | 超大规模数据中心 |

### 6.2 超配比例选择

| 工作负载 | 推荐超配比例 | 说明 |
|---------|-------------|------|
| 训练 (AllReduce) | 1:1 或 2:1 | 通信密集型，需要高带宽 |
| 推理 (PP) | 4:1 | 通信较少，可容忍超配 |
| MoE (EP) | 2:1 | AllToAll 需要较好带宽 |

### 6.3 性能优化建议

1. **拓扑感知调度**: 将 TP 组放在同一节点（Level 0）
2. **分层通信**: 使用 hierarchical AllReduce 算法
3. **通信压缩**: 在跨机通信时使用 FP8/INT8 压缩
4. **重叠优化**: 计算与跨机通信（Level 2+）重叠

## 7. 参考资料

### 7.1 学术论文

1. **Clos Network**: "A Study of Non-blocking Switching Networks" (Clos, 1953)
2. **Fat-Tree**: "Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing" (Leiserson, 1985)
3. **AllReduce Algorithms**: "Bandwidth Optimal All-reduce Algorithms for Clusters" (Thakur et al.)
4. **Hierarchical Collective**: "Optimization of Collective Communication Operations in MPICH" (Thakur et al., 2005)

### 7.2 工业实践

- **NVIDIA DGX SuperPOD**: Uses 2-tier Fat-Tree with InfiniBand
- **Google TPU Pod**: 2D Torus + Switch hybrid topology
- **AWS Trainium**: Custom Clos variant for training

### 7.3 网络技术

- **InfiniBand**: HDR (200 Gbps), NDR (400 Gbps)
- **NVLink**: NVLink 3.0 (600 GB/s), NVLink 4.0 (900 GB/s)
- **Ethernet**: 400GbE, 800GbE for scale-out networks

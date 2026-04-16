"""Base strategy classes."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class ParallelType(Enum):
    """Types of parallelism."""
    TENSOR = "tp"          # Tensor Parallelism
    PIPELINE = "pp"        # Pipeline Parallelism
    DATA = "dp"            # Data Parallelism
    EXPERT = "ep"          # Expert Parallelism (MoE)
    SEQUENCE = "sp"        # Sequence Parallelism
    CONTEXT = "cp"         # Context Parallelism


class SPType(Enum):
    """Types of sequence parallelism."""
    ULYSSES = "ulysses"
    RING_P2P = "ring_p2p"
    RING_ALLGATHER = "ring_allgather"
    UNIFIED_2D = "unified_2d"
    MEGATRON = "megatron"


@dataclass
class ParallelConfig:
    """Configuration for a specific parallelism type."""
    enabled: bool = False
    degree: int = 1  # Parallelism degree (e.g., TP=4 means 4 GPUs)
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Complete parallelism strategy configuration."""
    
    # Model configuration reference
    model_name: str = ""
    
    # Parallelism degrees
    tp_degree: int = 1   # Tensor Parallelism
    pp_degree: int = 1   # Pipeline Parallelism
    dp_degree: int = 1   # Data Parallelism
    ep_degree: int = 1   # Expert Parallelism
    sp_degree: int = 1   # Sequence Parallelism
    cp_degree: int = 1   # Context Parallelism
    
    # Sequence parallelism configuration
    sp_type: SPType = SPType.ULYSSES
    ulysses_degree: int = 1
    ring_degree: int = 1
    
    # Scheduling options
    pipeline_schedule: str = "1f1b"  # 1F1B, GPipe, etc.
    micro_batch_size: int = 1
    
    # Optimization flags
    activation_checkpointing: bool = False
    sequence_parallel: bool = False
    use_megatron: bool = True
    
    # Memory optimization
    zero_stage: int = 0  # ZeRO stage (0-3)
    
    @property
    def world_size(self) -> int:
        """Total number of GPUs needed."""
        return (
            self.tp_degree * 
            self.pp_degree * 
            self.dp_degree * 
            self.ep_degree * 
            self.sp_degree
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "parallelism": {
                "tp": self.tp_degree,
                "pp": self.pp_degree,
                "dp": self.dp_degree,
                "ep": self.ep_degree,
                "sp": self.sp_degree,
                "cp": self.cp_degree,
            },
            "sequence_parallelism": {
                "sp_type": self.sp_type.value,
                "ulysses_degree": self.ulysses_degree,
                "ring_degree": self.ring_degree,
            },
            "scheduling": {
                "pipeline_schedule": self.pipeline_schedule,
                "micro_batch_size": self.micro_batch_size,
            },
            "optimization": {
                "activation_checkpointing": self.activation_checkpointing,
                "sequence_parallel": self.sequence_parallel,
                "use_megatron": self.use_megatron,
                "zero_stage": self.zero_stage,
            },
            "world_size": self.world_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Create from dictionary."""
        parallel = data.get("parallelism", {})
        scheduling = data.get("scheduling", {})
        optimization = data.get("optimization", {})
        sp_config = data.get("sequence_parallelism", {})
        
        sp_type_str = sp_config.get("sp_type", "ulysses")
        try:
            sp_type = SPType(sp_type_str)
        except ValueError:
            sp_type = SPType.ULYSSES
        
        return cls(
            model_name=data.get("model_name", ""),
            tp_degree=parallel.get("tp", 1),
            pp_degree=parallel.get("pp", 1),
            dp_degree=parallel.get("dp", 1),
            ep_degree=parallel.get("ep", 1),
            sp_degree=parallel.get("sp", 1),
            cp_degree=parallel.get("cp", 1),
            sp_type=sp_type,
            ulysses_degree=sp_config.get("ulysses_degree", 1),
            ring_degree=sp_config.get("ring_degree", 1),
            pipeline_schedule=scheduling.get("pipeline_schedule", "1f1b"),
            micro_batch_size=scheduling.get("micro_batch_size", 1),
            activation_checkpointing=optimization.get("activation_checkpointing", False),
            sequence_parallel=optimization.get("sequence_parallel", False),
            use_megatron=optimization.get("use_megatron", True),
            zero_stage=optimization.get("zero_stage", 0),
        )

    def get_communication_domain_mapping(
        self,
        devices_per_node: int,
        nodes_per_rack: int = 16,
        total_devices: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        根据通信域优先级划分 ranks 到物理拓扑位置。

        划分顺序：TP(node内) > EP(node/rack内) > Ulysses(node内) > Ring(rack内) > DP(跨节点) > PP(跨节点)

        Args:
            devices_per_node: 每节点设备数
            nodes_per_rack: 每机架节点数（默认16）
            total_devices: 总设备数（默认为 world_size）

        Returns:
            {
                "tp": {"degree": ..., "topology_level": 0, "bandwidth_domain": "intra_node"},
                "ep": {"degree": ..., "topology_level": 0, "bandwidth_domain": "intra_node"},
                "ulysses": {"degree": ..., "topology_level": 0, "bandwidth_domain": "intra_node"},
                "ring": {"degree": ..., "topology_level": 1, "bandwidth_domain": "intra_rack"},
                "dp": {"degree": ..., "topology_level": 2, "bandwidth_domain": "inter_rack"},
                "pp": {"degree": ..., "topology_level": 2, "bandwidth_domain": "inter_rack"},
            }
        """
        total_devices = total_devices or self.world_size
        devices_per_rack = devices_per_node * nodes_per_rack

        # 计算每个 PP stage 的设备数（包含 TP x EP x SP）
        stage_size = self.tp_degree * self.ep_degree * self.sp_degree

        # 计算每个 DP replica 的设备数（包含完整的 TP x PP x EP x SP）
        replica_size = stage_size * self.pp_degree

        result = {}

        # TP 通信域映射
        # TP 优先放在同一个 node 内
        tp_topology_level = 0 if self.tp_degree <= devices_per_node else 1
        tp_bandwidth_domain = "intra_node" if self.tp_degree <= devices_per_node else "intra_rack"
        result["tp"] = {
            "degree": self.tp_degree,
            "topology_level": tp_topology_level,
            "bandwidth_domain": tp_bandwidth_domain,
            "ranks": list(range(self.tp_degree)),  # Logical ranks for communication
            "description": "Tensor Parallelism - highest bandwidth, within node preferred",
        }

        # EP 通信域映射
        # EP 优先放在 node 内，如果超过则扩展到 rack
        ep_topology_level = 0 if self.ep_degree <= devices_per_node else (
            1 if self.ep_degree <= devices_per_rack else 2
        )
        ep_bandwidth_domain = (
            "intra_node" if self.ep_degree <= devices_per_node else
            "intra_rack" if self.ep_degree <= devices_per_rack else
            "inter_rack"
        )
        result["ep"] = {
            "degree": self.ep_degree,
            "topology_level": ep_topology_level,
            "bandwidth_domain": ep_bandwidth_domain,
            "ranks": list(range(self.ep_degree)),  # Logical ranks for communication
            "description": "Expert Parallelism - high bandwidth for MoE",
        }

        # Ulysses SP 通信域映射
        ulysses_degree = self.ulysses_degree if self.sp_type == SPType.ULYSSES else 1
        ulysses_topology_level = 0 if ulysses_degree <= devices_per_node else 1
        ulysses_bandwidth_domain = "intra_node" if ulysses_degree <= devices_per_node else "intra_rack"
        result["ulysses"] = {
            "degree": ulysses_degree,
            "topology_level": ulysses_topology_level,
            "bandwidth_domain": ulysses_bandwidth_domain,
            "ranks": list(range(ulysses_degree)),  # Logical ranks for communication
            "description": "Ulysses Sequence Parallelism - within node preferred",
        }

        # Ring SP 通信域映射
        ring_degree = self.ring_degree if self.sp_type in (SPType.RING_P2P, SPType.RING_ALLGATHER) else 1
        ring_topology_level = 1 if ring_degree > 1 else 0  # Ring 可以跨 rack
        ring_bandwidth_domain = "intra_rack" if ring_degree > 1 else "intra_node"
        result["ring"] = {
            "degree": ring_degree,
            "topology_level": ring_topology_level,
            "bandwidth_domain": ring_bandwidth_domain,
            "ranks": list(range(ring_degree)),  # Logical ranks for communication
            "description": "Ring Sequence Parallelism - can span racks",
        }

        # SP 综合信息
        result["sp"] = {
            "degree": self.sp_degree,
            "sp_type": self.sp_type.value,
            "topology_level": max(ulysses_topology_level, ring_topology_level),
            "bandwidth_domain": ulysses_bandwidth_domain if self.sp_type == SPType.ULYSSES else ring_bandwidth_domain,
            "ranks": list(range(self.sp_degree)),  # Logical ranks for communication
            "description": f"Sequence Parallelism ({self.sp_type.value})",
        }

        # DP 通信域映射
        # DP 跨节点分布，每个 DP replica 包含完整的 TP+PP+EP+SP 组
        dp_topology_level = 2 if self.dp_degree > nodes_per_rack else (
            1 if self.dp_degree > 1 else 0
        )
        dp_bandwidth_domain = (
            "inter_rack" if self.dp_degree > nodes_per_rack else
            "intra_rack" if self.dp_degree > 1 else
            "intra_node"
        )
        result["dp"] = {
            "degree": self.dp_degree,
            "topology_level": dp_topology_level,
            "bandwidth_domain": dp_bandwidth_domain,
            "ranks": list(range(self.dp_degree)),  # Logical ranks for communication
            "description": "Data Parallelism - spans across nodes/racks",
        }

        # PP 通信域映射
        # PP 跨节点分布，每个 PP stage 包含完整的 TP+EP+SP 组
        pp_topology_level = 2 if self.pp_degree > nodes_per_rack else (
            1 if self.pp_degree > 1 else 0
        )
        pp_bandwidth_domain = (
            "inter_rack" if self.pp_degree > nodes_per_rack else
            "intra_rack" if self.pp_degree > 1 else
            "intra_node"
        )
        result["pp"] = {
            "degree": self.pp_degree,
            "topology_level": pp_topology_level,
            "bandwidth_domain": pp_bandwidth_domain,
            "ranks": list(range(self.pp_degree)),  # Logical ranks for communication
            "description": "Pipeline Parallelism - spans across nodes for stage isolation",
        }

        return result

    def get_rank_assignment(
        self,
        devices_per_node: int,
        nodes_per_rack: int = 16,
        total_devices: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        生成详细的 rank 分配方案，将逻辑 rank 映射到物理拓扑位置。

        Args:
            devices_per_node: 每节点设备数
            nodes_per_rack: 每机架节点数
            total_devices: 总设备数

        Returns:
            {
                "total_devices": int,
                "topology": {"devices_per_node": int, "nodes_per_rack": int, "devices_per_rack": int},
                "parallel_groups": {
                    "tp_groups": [[r1, r2, ...], ...],  # TP 分组
                    "ep_groups": [...],
                    "sp_groups": [...],
                    "dp_groups": [...],
                    "pp_groups": [...],
                },
                "rank_to_position": {rank: {"node": int, "rack": int, ...}},
            }
        """
        total_devices = total_devices or self.world_size
        devices_per_rack = devices_per_node * nodes_per_rack

        # 计算每个 PP stage 的设备数
        stage_size = self.tp_degree * self.ep_degree * self.sp_degree

        # 计算每个 DP replica 的设备数
        replica_size = stage_size * self.pp_degree

        # 构建 TP 分组
        tp_groups = []
        num_stages = self.pp_degree
        num_replicas = self.dp_degree

        for pp_stage in range(num_stages):
            for dp_replica in range(num_replicas):
                # 每个 TP group 的起始 rank
                base_rank = pp_stage * stage_size + dp_replica * replica_size
                tp_group = list(range(base_rank, base_rank + self.tp_degree))
                tp_groups.append(tp_group)

        # 构建 EP 分组（在每个 TP group 内）
        ep_groups = []
        for tp_group in tp_groups:
            if self.ep_degree > 1:
                for i in range(0, len(tp_group), self.ep_degree):
                    ep_group = tp_group[i:i + self.ep_degree]
                    ep_groups.append(ep_group)
            else:
                ep_groups.append(tp_group[:1] if tp_group else [])

        # 构建 SP 分组
        sp_groups = []
        if self.sp_type == SPType.ULYSSES:
            # Ulysses SP 与 TP 共享通信域
            sp_groups = tp_groups.copy() if self.ulysses_degree > 1 else [[r] for r in range(total_devices)]
        elif self.sp_type in (SPType.RING_P2P, SPType.RING_ALLGATHER):
            # Ring SP 可以更大范围
            for pp_stage in range(num_stages):
                for dp_replica in range(num_replicas):
                    base_rank = pp_stage * stage_size + dp_replica * replica_size
                    ring_group = list(range(base_rank, base_rank + self.ring_degree))
                    sp_groups.append(ring_group)
        else:
            sp_groups = [[r] for r in range(total_devices)]

        # 构建 DP 分组
        dp_groups = []
        for pp_stage in range(num_stages):
            # 同一个 PP stage 内，按 TP+EP+SP 位置分组
            for local_idx in range(stage_size):
                dp_group = []
                for dp_replica in range(num_replicas):
                    rank = pp_stage * stage_size + dp_replica * replica_size + local_idx
                    if rank < total_devices:
                        dp_group.append(rank)
                dp_groups.append(dp_group)

        # 构建 PP 分组
        pp_groups = []
        for dp_replica in range(num_replicas):
            for local_idx in range(stage_size):
                pp_group = []
                for pp_stage in range(num_stages):
                    rank = pp_stage * stage_size + dp_replica * replica_size + local_idx
                    if rank < total_devices:
                        pp_group.append(rank)
                pp_groups.append(pp_group)

        # 构建 rank 到物理位置的映射
        rank_to_position = {}
        for rank in range(total_devices):
            node = rank // devices_per_node
            rack = rank // devices_per_rack
            position_in_node = rank % devices_per_node
            position_in_rack = rank % devices_per_rack

            # 计算在并行组中的位置
            pp_stage = rank // stage_size % self.pp_degree
            dp_replica = rank // replica_size

            rank_to_position[rank] = {
                "node": node,
                "rack": rack,
                "position_in_node": position_in_node,
                "position_in_rack": position_in_rack,
                "pp_stage": pp_stage if self.pp_degree > 1 else 0,
                "dp_replica": dp_replica if self.dp_degree > 1 else 0,
                "tp_rank": rank % self.tp_degree if self.tp_degree > 1 else 0,
            }

        return {
            "total_devices": total_devices,
            "topology": {
                "devices_per_node": devices_per_node,
                "nodes_per_rack": nodes_per_rack,
                "devices_per_rack": devices_per_rack,
            },
            "parallel_groups": {
                "tp_groups": tp_groups,
                "ep_groups": ep_groups,
                "sp_groups": sp_groups,
                "dp_groups": dp_groups,
                "pp_groups": pp_groups,
            },
            "rank_to_position": rank_to_position,
        }


class ParallelStrategy:
    """
    Represents a parallel execution strategy.
    
    This class tracks how different parts of the model are distributed
    across devices and what communication patterns are needed.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self._layer_assignment: Dict[str, int] = {}  # layer -> pp_stage
        self._tensor_sharding: Dict[str, int] = {}   # layer -> tp_group
        self._expert_assignment: Dict[str, int] = {} # expert -> ep_group
    
    def assign_layer_to_stage(self, layer_name: str, pp_stage: int):
        """Assign a layer to a pipeline stage."""
        self._layer_assignment[layer_name] = pp_stage
    
    def get_layer_stage(self, layer_name: str) -> int:
        """Get pipeline stage for a layer."""
        return self._layer_assignment.get(layer_name, 0)
    
    def is_tp_enabled(self) -> bool:
        """Check if tensor parallelism is enabled."""
        return self.config.tp_degree > 1
    
    def is_pp_enabled(self) -> bool:
        """Check if pipeline parallelism is enabled."""
        return self.config.pp_degree > 1
    
    def is_dp_enabled(self) -> bool:
        """Check if data parallelism is enabled."""
        return self.config.dp_degree > 1
    
    def is_ep_enabled(self) -> bool:
        """Check if expert parallelism is enabled."""
        return self.config.ep_degree > 1
    
    def get_tp_group(self, rank: int) -> List[int]:
        """Get all ranks in the same TP group as given rank."""
        if not self.is_tp_enabled():
            return [rank]
        
        # TP groups are formed within each DP group
        tp_size = self.config.tp_degree
        
        # Find which DP replica this rank belongs to
        dp_replica = rank // tp_size
        
        # Ranks in the same TP group
        start = dp_replica * tp_size
        return list(range(start, start + tp_size))
    
    def get_dp_group(self, rank: int) -> List[int]:
        """Get all ranks in the same DP group as given rank."""
        if not self.is_dp_enabled():
            return [rank]
        
        tp_size = self.config.tp_degree
        world_size = self.config.world_size
        
        # DP groups span across TP groups
        # For each TP position, collect ranks across DP replicas
        tp_position = rank % tp_size
        return list(range(tp_position, world_size, tp_size))
    
    def get_pp_group(self, rank: int) -> List[int]:
        """Get all ranks in the same PP group (all stages) as given rank."""
        if not self.is_pp_enabled():
            return [rank]
        
        # PP groups are formed within each TP+DP combination
        # This is a simplification - actual PP groups depend on implementation
        pp_size = self.config.pp_degree
        
        # Find position within PP stage
        ranks_per_stage = self.config.world_size // pp_size
        
        # All ranks in all stages of this pipeline
        result = []
        for s in range(pp_size):
            start = s * ranks_per_stage
            result.extend(range(start, start + ranks_per_stage))
        return result
    
    def get_ep_group(self, rank: int) -> List[int]:
        """Get all ranks in the same EP group as given rank."""
        if not self.is_ep_enabled():
            return [rank]
        
        # EP groups typically span across TP
        ep_size = self.config.ep_degree
        tp_size = self.config.tp_degree
        
        # Simplified: EP groups are formed within each TP group
        tp_position = rank % tp_size
        start = tp_position * ep_size
        return list(range(start, start + ep_size))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "layer_assignment": self._layer_assignment,
            "tensor_sharding": self._tensor_sharding,
            "expert_assignment": self._expert_assignment,
        }

"""Base strategy classes."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    pass


class ParallelType(Enum):
    """Types of parallelism."""

    TENSOR = "tp"  # Tensor Parallelism
    PIPELINE = "pp"  # Pipeline Parallelism
    DATA = "dp"  # Data Parallelism
    EXPERT = "ep"  # Expert Parallelism (MoE)
    SEQUENCE = "sp"  # Sequence Parallelism
    CONTEXT = "cp"  # Context Parallelism


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
    """Complete parallelism strategy configuration.
    
    Layered Parallelism Support:
    - Attention layers: use tp_degree for tensor parallelism
    - MoE/FFN layers: can use independent expert_tp_degree + ep_degree
    
    Validation rules:
    - Attention part: tp_degree × dp_degree × pp_degree × sp_degree = world_size
    - MoE part: expert_tp_degree × ep_degree × dp_degree × pp_degree × sp_degree = world_size
    - If expert_tp_degree is not set, defaults to tp_degree (uniform TP)
    """

    model_name: str = ""

    tp_degree: int = 1
    pp_degree: int = 1
    dp_degree: int = 1
    ep_degree: int = 1
    sp_degree: int = 1
    cp_degree: int = 1

    expert_tp_degree: Optional[int] = None

    sp_type: SPType = SPType.ULYSSES
    ulysses_degree: int = 1
    ring_degree: int = 1

    pipeline_schedule: str = "1f1b"
    micro_batch_size: int = 1

    kv_separate_allgather: bool = False

    activation_checkpointing: bool = False
    sequence_parallel: bool = False
    use_megatron: bool = True

    zero_stage: int = 0

    scheduler: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-init processing.
        
        - Initialize scheduler if not provided
        - Set expert_tp_degree default to tp_degree if not specified
        """
        if self.scheduler is None:
            self.scheduler = {}
        if self.expert_tp_degree is None:
            self.expert_tp_degree = self.tp_degree

    @property
    def world_size(self) -> int:
        """Total number of GPUs needed for Attention part.
        
        Uses tp_degree for Attention layers.
        """
        return self.tp_degree * self.pp_degree * self.dp_degree * self.sp_degree

    @property
    def moe_world_size(self) -> int:
        """Total number of GPUs needed for MoE part.
        
        Uses expert_tp_degree for MoE/FFN layers.
        If expert_tp_degree == tp_degree, this equals world_size (uniform TP).
        """
        effective_expert_tp = self.expert_tp_degree or self.tp_degree
        return effective_expert_tp * self.ep_degree * self.pp_degree * self.dp_degree * self.sp_degree

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
                "expert_tp": self.expert_tp_degree,
            },
            "sequence_parallelism": {
                "sp_type": self.sp_type.value,
                "ulysses_degree": self.ulysses_degree,
                "ring_degree": self.ring_degree,
                "kv_separate_allgather": self.kv_separate_allgather,
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
            "scheduler": self.scheduler if isinstance(self.scheduler, dict) else {},
            "world_size": self.world_size,
            "moe_world_size": self.moe_world_size,
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
            expert_tp_degree=parallel.get("expert_tp"),
            sp_type=sp_type,
            ulysses_degree=sp_config.get("ulysses_degree", 1),
            ring_degree=sp_config.get("ring_degree", 1),
            kv_separate_allgather=sp_config.get("kv_separate_allgather", False),
            pipeline_schedule=scheduling.get("pipeline_schedule", "1f1b"),
            micro_batch_size=scheduling.get("micro_batch_size", 1),
            activation_checkpointing=optimization.get("activation_checkpointing", False),
            sequence_parallel=optimization.get("sequence_parallel", False),
            use_megatron=optimization.get("use_megatron", True),
            zero_stage=optimization.get("zero_stage", 0),
            scheduler=data.get("scheduler", {}),
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

        嵌套 rank 公式:
        rank = dp_replica * replica_size + pp_stage * stage_size + in_stage_rank
        其中:
        - replica_size = TP * EP * SP * PP
        - stage_size = TP * EP * SP
        - in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos

        Args:
            devices_per_node: 每节点设备数
            nodes_per_rack: 每机架节点数（默认16）
            total_devices: 总设备数（默认为 world_size）

        Returns:
            {
                "tp": {"degree": ..., "groups": [[...], ...], "topology_level": ..., ...},
                "dp": {"degree": ..., "groups": [[...], ...], ...},
                "pp": {"degree": ..., "groups": [[...], ...], ...},
                ...
            }
        """
        total_devices = total_devices or self.world_size
        devices_per_rack = devices_per_node * nodes_per_rack

        # 计算每个 PP stage 的设备数（包含 TP x EP x SP）
        stage_size = self.tp_degree * self.ep_degree * self.sp_degree

        # 计算每个 DP replica 的设备数（包含完整的 TP x PP x EP x SP）
        replica_size = stage_size * self.pp_degree

        # 获取各并行维度的分组
        rank_assignment = self._build_parallel_groups(total_devices, stage_size, replica_size)

        result = {}

        # TP 通信域映射
        # TP 优先放在同一个 node 内
        tp_topology_level = 0 if self.tp_degree <= devices_per_node else 1
        tp_bandwidth_domain = "intra_node" if self.tp_degree <= devices_per_node else "intra_rack"
        result["tp"] = {
            "degree": self.tp_degree,
            "topology_level": tp_topology_level,
            "bandwidth_domain": tp_bandwidth_domain,
            "groups": rank_assignment["tp_groups"],
            "description": "Tensor Parallelism - highest bandwidth, within node preferred",
        }

        # EP 通信域映射
        # EP 优先放在 node 内，如果超过则扩展到 rack
        ep_topology_level = (
            0 if self.ep_degree <= devices_per_node else (1 if self.ep_degree <= devices_per_rack else 2)
        )
        ep_bandwidth_domain = (
            "intra_node"
            if self.ep_degree <= devices_per_node
            else "intra_rack"
            if self.ep_degree <= devices_per_rack
            else "inter_rack"
        )
        result["ep"] = {
            "degree": self.ep_degree,
            "topology_level": ep_topology_level,
            "bandwidth_domain": ep_bandwidth_domain,
            "groups": rank_assignment["ep_groups"],
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
            "groups": rank_assignment["ulysses_groups"],
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
            "groups": rank_assignment["ring_groups"],
            "description": "Ring Sequence Parallelism - can span racks",
        }

        # SP 综合信息
        result["sp"] = {
            "degree": self.sp_degree,
            "sp_type": self.sp_type.value,
            "topology_level": max(ulysses_topology_level, ring_topology_level),
            "bandwidth_domain": ulysses_bandwidth_domain if self.sp_type == SPType.ULYSSES else ring_bandwidth_domain,
            "groups": rank_assignment["sp_groups"],
            "description": f"Sequence Parallelism ({self.sp_type.value})",
        }

        # DP 通信域映射
        # DP 跨节点分布，每个 DP replica 包含完整的 TP+PP+EP+SP 组
        dp_topology_level = 2 if self.dp_degree > nodes_per_rack else (1 if self.dp_degree > 1 else 0)
        dp_bandwidth_domain = (
            "inter_rack" if self.dp_degree > nodes_per_rack else "intra_rack" if self.dp_degree > 1 else "intra_node"
        )
        result["dp"] = {
            "degree": self.dp_degree,
            "topology_level": dp_topology_level,
            "bandwidth_domain": dp_bandwidth_domain,
            "groups": rank_assignment["dp_groups"],
            "description": "Data Parallelism - spans across nodes/racks",
        }

        # PP 通信域映射
        # PP 跨节点分布，每个 PP stage 包含完整的 TP+EP+SP 组
        pp_topology_level = 2 if self.pp_degree > nodes_per_rack else (1 if self.pp_degree > 1 else 0)
        pp_bandwidth_domain = (
            "inter_rack" if self.pp_degree > nodes_per_rack else "intra_rack" if self.pp_degree > 1 else "intra_node"
        )
        result["pp"] = {
            "degree": self.pp_degree,
            "topology_level": pp_topology_level,
            "bandwidth_domain": pp_bandwidth_domain,
            "groups": rank_assignment["pp_groups"],
            "description": "Pipeline Parallelism - spans across nodes for stage isolation",
        }

        return result

    def _build_parallel_groups(
        self,
        total_devices: int,
        stage_size: int,
        replica_size: int,
    ) -> Dict[str, List[List[int]]]:
        """
        构建各并行维度的 rank 分组。

        嵌套 rank 公式:
        rank = dp_replica * replica_size + pp_stage * stage_size + in_stage_rank
        其中:
        - stage_size = TP * EP * SP
        - replica_size = stage_size * PP
        - in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos

        Args:
            total_devices: 总设备数
            stage_size: 每个 PP stage 的设备数 (TP * EP * SP)
            replica_size: 每个 DP replica 的设备数 (TP * EP * SP * PP)

        Returns:
            {
                "tp_groups": [[r1, r2, ...], ...],
                "ep_groups": [...],
                "sp_groups": [...],
                "ulysses_groups": [...],
                "ring_groups": [...],
                "dp_groups": [...],
                "pp_groups": [...],
            }
        """
        tp_size = self.tp_degree
        ep_size = self.ep_degree
        sp_size = self.sp_degree
        pp_size = self.pp_degree
        dp_size = self.dp_degree

        # TP groups: 每个 (DP replica, PP stage) 组合一个 TP group
        # 在每个 stage 内，连续的 tp_size 个 rank 组成一个 TP group
        tp_groups = []
        for dp_replica in range(dp_size):
            for pp_stage in range(pp_size):
                base = dp_replica * replica_size + pp_stage * stage_size
                # 在每个 stage 内，TP ranks 按 tp_pos * (ep_size * sp_size) 间隔
                # 但更简单的方式是：按 tp_pos 分组
                for ep_pos in range(ep_size):
                    for sp_pos in range(sp_size):
                        tp_group = []
                        for tp_pos in range(tp_size):
                            in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos
                            rank = base + in_stage_rank
                            if rank < total_devices:
                                tp_group.append(rank)
                        if len(tp_group) == tp_size:
                            tp_groups.append(tp_group)
                # 如果 ep_size == 1 and sp_size == 1，上面的循环会退化成简单的连续 ranks
        # 简化版本：当 ep_size == 1 and sp_size == 1 时，TP groups 是连续的
        if ep_size == 1 and sp_size == 1:
            tp_groups = []
            for dp_replica in range(dp_size):
                for pp_stage in range(pp_size):
                    base = dp_replica * replica_size + pp_stage * stage_size
                    tp_group = list(range(base, base + tp_size))
                    if all(r < total_devices for r in tp_group):
                        tp_groups.append(tp_group)

        # EP groups: 每个 (DP replica, PP stage, TP position, SP position) 组合一个 EP group
        ep_groups = []
        for dp_replica in range(dp_size):
            for pp_stage in range(pp_size):
                base = dp_replica * replica_size + pp_stage * stage_size
                for tp_pos in range(tp_size):
                    for sp_pos in range(sp_size):
                        ep_group = []
                        for ep_pos in range(ep_size):
                            in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos
                            rank = base + in_stage_rank
                            if rank < total_devices:
                                ep_group.append(rank)
                        if len(ep_group) == ep_size:
                            ep_groups.append(ep_group)
        # 简化版本：当 sp_size == 1 且 tp_size == 1 时
        if sp_size == 1 and tp_size == 1:
            ep_groups = []
            for dp_replica in range(dp_size):
                for pp_stage in range(pp_size):
                    base = dp_replica * replica_size + pp_stage * stage_size
                    ep_group = list(range(base, base + ep_size))
                    if all(r < total_devices for r in ep_group):
                        ep_groups.append(ep_group)

        # SP groups (general): 每个 (DP replica, PP stage, TP position, EP position) 组合一个 SP group
        sp_groups = []
        for dp_replica in range(dp_size):
            for pp_stage in range(pp_size):
                base = dp_replica * replica_size + pp_stage * stage_size
                for tp_pos in range(tp_size):
                    for ep_pos in range(ep_size):
                        sp_group = []
                        for sp_pos in range(sp_size):
                            in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos
                            rank = base + in_stage_rank
                            if rank < total_devices:
                                sp_group.append(rank)
                        if len(sp_group) == sp_size:
                            sp_groups.append(sp_group)
        # 简化版本
        if tp_size == 1 and ep_size == 1:
            sp_groups = []
            for dp_replica in range(dp_size):
                for pp_stage in range(pp_size):
                    base = dp_replica * replica_size + pp_stage * stage_size
                    sp_group = list(range(base, base + sp_size))
                    if all(r < total_devices for r in sp_group):
                        sp_groups.append(sp_group)

        # Ulysses SP groups (当 sp_type == ULYSSES 时使用)
        ulysses_degree = self.ulysses_degree if self.sp_type == SPType.ULYSSES else 1
        ulysses_groups = []
        if ulysses_degree > 1:
            # Ulysses 与 TP 共享通信域，每个 TP group 内的连续 ranks
            for tp_group in tp_groups if tp_groups else [[i] for i in range(total_devices)]:
                for i in range(0, len(tp_group), ulysses_degree):
                    ulysses_group = tp_group[i : i + ulysses_degree]
                    if len(ulysses_group) == ulysses_degree:
                        ulysses_groups.append(ulysses_group)
        else:
            ulysses_groups = [[r] for r in range(total_devices)]

        # Ring SP groups (当 sp_type in (RING_P2P, RING_ALLGATHER) 时使用)
        ring_degree = self.ring_degree if self.sp_type in (SPType.RING_P2P, SPType.RING_ALLGATHER) else 1
        ring_groups = []
        if ring_degree > 1:
            for dp_replica in range(dp_size):
                for pp_stage in range(pp_size):
                    base = dp_replica * replica_size + pp_stage * stage_size
                    ring_group = list(range(base, base + ring_degree))
                    if all(r < total_devices for r in ring_group):
                        ring_groups.append(ring_group)
        else:
            ring_groups = [[r] for r in range(total_devices)]

        # DP groups: 每个 (TP position, EP position, SP position, PP stage) 组合一个 DP group
        # DP 同一位置跨 DP replicas 的 ranks
        dp_groups = []
        in_group_size = ep_size * sp_size
        for pp_stage in range(pp_size):
            for tp_pos in range(tp_size):
                for in_group_pos in range(in_group_size):
                    ep_pos = in_group_pos // sp_size
                    sp_pos = in_group_pos % sp_size
                    dp_group = []
                    for dp_replica in range(dp_size):
                        in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos
                        rank = dp_replica * replica_size + pp_stage * stage_size + in_stage_rank
                        if rank < total_devices:
                            dp_group.append(rank)
                    if dp_group:
                        dp_groups.append(dp_group)

        # PP groups: 每个 (TP position, EP position, SP position, DP replica) 组合一个 PP group
        # PP 同一位置跨 PP stages 的 ranks
        pp_groups = []
        for dp_replica in range(dp_size):
            for tp_pos in range(tp_size):
                for ep_pos in range(ep_size):
                    for sp_pos in range(sp_size):
                        pp_group = []
                        for pp_stage in range(pp_size):
                            in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos
                            rank = dp_replica * replica_size + pp_stage * stage_size + in_stage_rank
                            if rank < total_devices:
                                pp_group.append(rank)
                        if pp_group:
                            pp_groups.append(pp_group)

        return {
            "tp_groups": tp_groups,
            "ep_groups": ep_groups,
            "sp_groups": sp_groups,
            "ulysses_groups": ulysses_groups,
            "ring_groups": ring_groups,
            "dp_groups": dp_groups,
            "pp_groups": pp_groups,
        }

    def get_rank_assignment(
        self,
        devices_per_node: int,
        nodes_per_rack: int = 16,
        total_devices: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        生成详细的 rank 分配方案，将逻辑 rank 映射到物理拓扑位置。

        嵌套 rank 公式:
        rank = dp_replica * replica_size + pp_stage * stage_size + in_stage_rank
        其中:
        - stage_size = TP * EP * SP
        - replica_size = stage_size * PP
        - in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos

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
                    "ulysses_groups": [...],
                    "ring_groups": [...],
                    "dp_groups": [...],
                    "pp_groups": [...],
                },
                "rank_to_position": {rank: {"node": int, "rack": int, ...}},
                "nested_rank_formula": "...",  # 嵌套公式说明
            }
        """
        total_devices = total_devices or self.world_size
        devices_per_rack = devices_per_node * nodes_per_rack

        # 计算每个 PP stage 的设备数
        stage_size = self.tp_degree * self.ep_degree * self.sp_degree

        # 计算每个 DP replica 的设备数
        replica_size = stage_size * self.pp_degree

        # 构建并行分组
        parallel_groups = self._build_parallel_groups(total_devices, stage_size, replica_size)

        # 构建 rank 到物理位置的映射
        rank_to_position = {}
        for rank in range(total_devices):
            # 物理拓扑位置
            node = rank // devices_per_node
            rack = rank // devices_per_rack
            position_in_node = rank % devices_per_node
            position_in_rack = rank % devices_per_rack

            # 计算在并行组中的位置
            dp_replica = rank // replica_size
            remaining = rank % replica_size
            pp_stage = remaining // stage_size
            in_stage_rank = remaining % stage_size

            # 计算 TP/EP/SP 位置
            in_group_size = self.ep_degree * self.sp_degree
            tp_pos = in_stage_rank // in_group_size
            in_group_remaining = in_stage_rank % in_group_size
            ep_pos = in_group_remaining // self.sp_degree
            sp_pos = in_group_remaining % self.sp_degree

            rank_to_position[rank] = {
                "node": node,
                "rack": rack,
                "position_in_node": position_in_node,
                "position_in_rack": position_in_rack,
                "dp_replica": dp_replica,
                "pp_stage": pp_stage,
                "tp_position": tp_pos,
                "ep_position": ep_pos,
                "sp_position": sp_pos,
            }

        return {
            "total_devices": total_devices,
            "topology": {
                "devices_per_node": devices_per_node,
                "nodes_per_rack": nodes_per_rack,
                "devices_per_rack": devices_per_rack,
            },
            "parallel_groups": parallel_groups,
            "rank_to_position": rank_to_position,
            "nested_rank_formula": (
                "rank = dp_replica * replica_size + pp_stage * stage_size + in_stage_rank\n"
                "其中:\n"
                f"  stage_size = {stage_size} (TP={self.tp_degree} * EP={self.ep_degree} * SP={self.sp_degree})\n"
                f"  replica_size = {replica_size} (stage_size * PP={self.pp_degree})\n"
                "  in_stage_rank = tp_pos * ep_size * sp_size + ep_pos * sp_size + sp_pos"
            ),
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
        self._tensor_sharding: Dict[str, int] = {}  # layer -> tp_group
        self._expert_assignment: Dict[str, int] = {}  # expert -> ep_group

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

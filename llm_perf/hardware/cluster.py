"""Cluster and network topology definitions."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING
from .device import Device, DeviceConfig
from .topology import NetworkTopology, TopologyLevel, TopologyType

if TYPE_CHECKING:
    from ..strategy.base import StrategyConfig


@dataclass
class NetworkConfig:
    """
    Network configuration for cluster interconnect.
    
    Deprecated: Use NetworkTopology instead for hierarchical topologies.
    This class is kept for backward compatibility.
    """
    
    # Intra-node (within server) bandwidth
    intra_node_bandwidth_gbps: float = 1.0 # NVLink / Infinity Fabric
    intra_node_latency_us: float = 1.0
    
    # Inter-node (between servers) bandwidth
    inter_node_bandwidth_gbps: float = 1.0 # IB / RoCE / etc
    inter_node_latency_us: float = 2.0
    
    # Topology
    topology: str = "full_mesh"  # full_mesh, switch, dragonfly, etc
    
    # Oversubscription ratio
    oversubscription_ratio: float = 1.0  # 1.0 = no oversubscription
    
    def to_topology(self) -> NetworkTopology:
        """Convert legacy NetworkConfig to NetworkTopology."""
        return NetworkTopology.create_2tier_simple(
            intra_node_bw_gbps=self.intra_node_bandwidth_gbps,
            inter_node_bw_gbps=self.inter_node_bandwidth_gbps,
        )


class Cluster:
    """Represents a cluster of compute devices with hierarchical topology."""
    
    def __init__(
        self,
        devices: List[Device],
        topology: Union[NetworkTopology, NetworkConfig],
        devices_per_node: int = 8,
        # Hierarchical grouping for multi-tier topologies
        node_grouping: Optional[List[List[int]]] = None,
        rack_grouping: Optional[List[List[int]]] = None,
    ):
        """
        Initialize cluster with hierarchical topology.
        
        Args:
            devices: List of devices in the cluster
            topology: NetworkTopology or NetworkConfig defining hierarchical bandwidth
            devices_per_node: Number of devices per physical node
            node_grouping: Optional explicit node to device mapping
            rack_grouping: Optional rack to node mapping for multi-tier
        """
        self.devices = devices
        if isinstance(topology, NetworkConfig):
            topology = topology.to_topology()
        self.topology = topology
        self.devices_per_node = devices_per_node
        
        self.num_devices = len(devices)
        self.num_nodes = (self.num_devices + devices_per_node - 1) // devices_per_node
        
        # Build hierarchical groupings
        self._device_groups = self._build_device_groups(node_grouping, rack_grouping)
    
    def _build_device_groups(
        self,
        node_grouping: Optional[List[List[int]]],
        rack_grouping: Optional[List[List[int]]],
    ) -> List[List[List[int]]]:
        """
        Build hierarchical device groupings for topology traversal.
        
        Returns list of groupings at each level:
            [Level0_groups, Level1_groups, ...]
        """
        groups = []
        
        # Level 0: Node-level grouping
        if node_grouping:
            groups.append(node_grouping)
        else:
            # Auto-generate node groups
            node_groups = []
            for i in range(0, self.num_devices, self.devices_per_node):
                node_devices = list(range(i, min(i + self.devices_per_node, self.num_devices)))
                node_groups.append(node_devices)
            groups.append(node_groups)
        
        # Level 1+: Additional groupings from topology levels
        if rack_grouping:
            groups.append(rack_grouping)
        elif len(self.topology.levels) > 2:
            # Auto-generate based on topology level definitions
            for level in self.topology.levels[1:-1]:
                if level.devices_per_group < self.num_devices:
                    rack_groups = []
                    dpg = level.devices_per_group
                    for i in range(0, self.num_devices, dpg):
                        rack_devices = list(range(i, min(i + dpg, self.num_devices)))
                        rack_groups.append(rack_devices)
                    groups.append(rack_groups)
        
        return groups
    
    @classmethod
    def create_homogeneous(
        cls,
        device_config: DeviceConfig,
        num_devices: int,
        topology: Union[NetworkTopology, NetworkConfig],
        devices_per_node: int = 8,
    ) -> "Cluster":
        """Create a homogeneous cluster with identical devices."""
        devices = [Device(device_config) for _ in range(num_devices)]
        return cls(devices, topology, devices_per_node)
    
    @classmethod
    def create_from_preset(
        cls,
        preset_name: str,
        num_devices: int,
        topology: Union[NetworkTopology, NetworkConfig],
        devices_per_node: int = 8,
    ) -> "Cluster":
        """Create cluster from device preset."""
        device = Device.from_preset(preset_name)
        return cls.create_homogeneous(
            device.config, num_devices, topology, devices_per_node
        )
    
    @classmethod
    def create_with_clos_topology(
        cls,
        preset_name: str,
        num_devices: int,
        host_bw_gbps: float = 200.0,
        leaf_bw_gbps: float = 200.0,
        spine_bw_gbps: float = 200.0,
        switch_radix: int = 32,
        oversubscription: float = 4.0,
        devices_per_node: int = 8,
    ) -> "Cluster":
        """
        Create cluster with automatically calculated Clos topology.
        
        Args:
            preset_name: Device preset name
            num_devices: Total number of devices
            host_bw_gbps: Bandwidth from host to leaf switch
            leaf_bw_gbps: Bandwidth from leaf to spine
            spine_bw_gbps: Bandwidth from spine to core
            switch_radix: Number of ports per switch
            oversubscription: Oversubscription ratio at aggregation layer
            devices_per_node: Devices per physical node
        """
        from .topology import ClosTopologyBuilder
        
        topology = ClosTopologyBuilder.create_from_device_count(
            num_devices=num_devices,
            switch_radix=switch_radix,
            host_bw_gbps=host_bw_gbps,
            leaf_bw_gbps=leaf_bw_gbps,
            spine_bw_gbps=spine_bw_gbps,
            oversubscription=oversubscription,
        )
        
        return cls.create_from_preset(
            preset_name, num_devices, topology, devices_per_node
        )
    
    def get_device(self, rank: int) -> Device:
        """Get device by rank."""
        return self.devices[rank]
    
    def get_node_for_rank(self, rank: int) -> int:
        """Get node index for a device rank."""
        return rank // self.devices_per_node
    
    def are_on_same_node(self, rank1: int, rank2: int) -> bool:
        """Check if two ranks are on the same node."""
        return self.get_node_for_rank(rank1) == self.get_node_for_rank(rank2)
    
    def _find_topology_level(self, rank1: int, rank2: int) -> Tuple[TopologyLevel, int]:
        """
        Find the topology level and distance for communication between ranks.
        
        Returns:
            (topology_level, distance_in_devices)
        """
        if rank1 == rank2:
            return None, 0
        
        # Check each level of grouping
        for level_idx, groups in enumerate(self._device_groups):
            # Find which groups the ranks belong to at this level
            group1 = None
            group2 = None
            for g_idx, group in enumerate(groups):
                if rank1 in group:
                    group1 = g_idx
                if rank2 in group:
                    group2 = g_idx
            
            if group1 is not None and group2 is not None:
                if group1 != group2:
                    # Communication crosses this level
                    topo_level = self.topology.get_level_for_distance(level_idx)
                    distance = abs(group2 - group1) * len(groups[group1])
                    return topo_level, distance
        
        # If no crossing found, communication is within the finest grouping (same node)
        if self.topology.levels:
            if 0 <= rank1 < self.num_devices and 0 <= rank2 < self.num_devices:
                return min(self.topology.levels, key=lambda l: l.level), 0
            # Fallback for out-of-bounds ranks to preserve backward compatibility
            return max(self.topology.levels, key=lambda l: l.level), self.num_devices
        return None, self.num_devices
    
    def get_bandwidth_between(self, rank1: int, rank2: int) -> float:
        """
        Get bandwidth between two ranks in GB/s, considering topology hierarchy.
        
        Returns:
            Bandwidth in GB/s
        """
        if rank1 == rank2:
            return float('inf')
        
        level, _ = self._find_topology_level(rank1, rank2)
        if level:
            return level.bandwidth_gbps
        
        # Fallback for legacy compatibility
        if self.are_on_same_node(rank1, rank2):
            # Try to find node-level bandwidth
            for lvl in self.topology.levels:
                if lvl.level == 0:
                    return lvl.bandwidth_gbps
        else:
            for lvl in self.topology.levels:
                if lvl.level > 0:
                    return lvl.bandwidth_gbps
        
        return 0.0
    
    def get_latency_between(self, rank1: int, rank2: int) -> float:
        """
        Get latency between two ranks in microseconds, considering topology.
        
        Returns:
            Latency in microseconds
        """
        if rank1 == rank2:
            return 0.0
        
        level, _ = self._find_topology_level(rank1, rank2)
        if level:
            return level.latency_us
        
        return 2.0  # Default fallback
    
    def estimate_allreduce_time(
        self,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> float:
        """
        Estimate all-reduce communication time with topology awareness.
        
        In hierarchical topologies (Clos), collective operations may use
        different algorithms at different levels (e.g., ring within node,
        tree across nodes).
        
        Args:
            num_bytes: Size of data per rank
            participating_ranks: List of participating ranks
        
        Returns:
            Estimated time in seconds
        """
        n = len(participating_ranks)
        if n <= 1:
            return 0.0
        
        # Analyze the topology levels needed
        levels_needed = set()
        for i in range(n):
            for j in range(i+1, n):
                level, _ = self._find_topology_level(
                    participating_ranks[i], participating_ranks[j]
                )
                if level:
                    levels_needed.add(level.level)
        
        # Estimate time for each level
        total_time = 0.0
        sorted_levels = sorted(levels_needed)
        
        for level_idx in sorted_levels:
            level = self.topology.get_level_for_distance(level_idx)
            if not level:
                continue
            
            # Bandwidth at this level
            bw = level.bandwidth_gbps * 1e9  # Convert to bytes/s
            lat = level.latency_us * 1e-6    # Convert to seconds
            
            # Ring all-reduce at this level
            # Data transferred at this level depends on how many ranks cross it
            ranks_at_level = []
            for rank in participating_ranks:
                # Determine if this rank communicates at this level
                for other in participating_ranks:
                    if rank != other:
                        lvl, _ = self._find_topology_level(rank, other)
                        if lvl and lvl.level == level_idx:
                            ranks_at_level.append(rank)
                            break
            
            n_level = len(set(ranks_at_level))
            if n_level <= 1:
                continue
            
            # Ring algorithm: 2*(n-1) steps
            data_per_step = num_bytes / n_level
            num_steps = 2 * (n_level - 1)
            
            transfer_time = num_steps * data_per_step / bw
            latency_time = n_level * lat
            
            # Add to total (some overlap possible, but conservative estimate)
            total_time += (transfer_time + latency_time) * 0.8  # 80% efficiency
        
        return max(total_time, 0.0)
    
    def estimate_allgather_time(
        self,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> float:
        """
        Estimate all-gather communication time with topology awareness.

        Args:
            num_bytes: Size of data per rank (output will be n * num_bytes)
            participating_ranks: List of participating ranks

        Returns:
            Estimated time in seconds
        """
        n = len(participating_ranks)
        if n <= 1:
            return 0.0

        # All-gather is approximately half of all-reduce in ring algorithm
        return self.estimate_allreduce_time(num_bytes, participating_ranks) / 2

    def estimate_reducescatter_time(
        self,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> float:
        """
        Estimate reduce-scatter communication time with topology awareness.

        Reduce-scatter is equivalent to all-gather in reverse, with similar
        communication complexity. In ring algorithm, it's approximately
        half of all-reduce.

        Args:
            num_bytes: Size of data per rank (output will be num_bytes / n)
            participating_ranks: List of participating ranks

        Returns:
            Estimated time in seconds
        """
        n = len(participating_ranks)
        if n <= 1:
            return 0.0

        # Reduce-scatter is approximately half of all-reduce
        return self.estimate_allreduce_time(num_bytes, participating_ranks) / 2
    
    def estimate_alltoall_time(
        self,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> float:
        """
        Estimate all-to-all communication time with topology awareness.

        Args:
            num_bytes: Size of data per rank
            participating_ranks: List of participating ranks

        Returns:
            Estimated time in seconds
        """
        n = len(participating_ranks)
        if n <= 1:
            return 0.0

        # Find the highest topology level needed
        max_level = 0
        for i in range(n):
            for j in range(i+1, n):
                level, _ = self._find_topology_level(
                    participating_ranks[i], participating_ranks[j]
                )
                if level:
                    max_level = max(max_level, level.level)

        # Use bandwidth from highest level
        level = self.topology.get_level_for_distance(max_level)
        if level:
            avg_bw = level.bandwidth_gbps * 1e9
        else:
            # Average across levels
            avg_bw = sum(l.bandwidth_gbps for l in self.topology.levels) / len(self.topology.levels) * 1e9

        # Total data moved per rank in all-to-all
        total_data = num_bytes * (n - 1) / n

        return total_data / avg_bw

    def build_communication_domain_groups(
        self,
        strategy: "StrategyConfig",
    ) -> Dict[str, List[List[int]]]:
        """
        Build communication domain groups based on strategy and topology.

        This method computes the actual physical rank assignments for each
        parallelism domain, considering the cluster topology.

        Args:
            strategy: StrategyConfig defining parallelism degrees

        Returns:
            Dict with keys: "tp_groups", "pp_groups", "dp_groups", "ep_groups", "sp_groups"
            Each contains a list of rank groups for that parallelism type.
        """
        result = {}

        world_size = strategy.world_size
        if world_size > self.num_devices:
            world_size = self.num_devices

        # TP groups: ranks within the same TP group
        if strategy.tp_degree > 1:
            tp_groups = []
            # TP groups are formed within each DP replica
            tp_size = strategy.tp_degree
            num_tp_groups = world_size // tp_size
            for g in range(num_tp_groups):
                start = g * tp_size
                tp_groups.append(list(range(start, start + tp_size)))
            result["tp_groups"] = tp_groups

        # PP groups: ranks in the same PP position across stages
        if strategy.pp_degree > 1:
            pp_groups = []
            pp_size = strategy.pp_degree
            ranks_per_stage = world_size // pp_size
            # Each PP group contains all stages for a given position
            for pos in range(ranks_per_stage):
                pp_group = []
                for stage in range(pp_size):
                    rank = stage * ranks_per_stage + pos
                    pp_group.append(rank)
                pp_groups.append(pp_group)
            result["pp_groups"] = pp_groups

        # DP groups: ranks in the same DP position across replicas
        if strategy.dp_degree > 1:
            dp_groups = []
            tp_size = strategy.tp_degree
            pp_size = strategy.pp_degree
            dp_size = strategy.dp_degree
            # Ranks in the same TP+PP position but different DP replicas
            ranks_per_dp = world_size // dp_size
            for pos in range(ranks_per_dp):
                dp_group = []
                for dp in range(dp_size):
                    rank = dp * ranks_per_dp + pos
                    dp_group.append(rank)
                dp_groups.append(dp_group)
            result["dp_groups"] = dp_groups

        # EP groups: ranks in the same EP domain (for MoE)
        if strategy.ep_degree > 1:
            ep_groups = []
            ep_size = strategy.ep_degree
            num_ep_groups = world_size // ep_size
            for g in range(num_ep_groups):
                start = g * ep_size
                ep_groups.append(list(range(start, start + ep_size)))
            result["ep_groups"] = ep_groups

        # SP groups: ranks in the same SP domain
        if strategy.sp_degree > 1:
            sp_groups = []
            sp_size = strategy.sp_degree
            num_sp_groups = world_size // sp_size
            for g in range(num_sp_groups):
                start = g * sp_size
                sp_groups.append(list(range(start, start + sp_size)))
            result["sp_groups"] = sp_groups

        return result

    def get_bandwidth_for_communication_domain(
        self,
        domain_type: str,
        strategy: Optional["StrategyConfig"] = None,
    ) -> float:
        """
        根据通信域类型返回对应的带宽（GB/s）。

        Args:
            domain_type: 通信域类型，可以是：
                - "tp": Tensor Parallelism
                - "ep": Expert Parallelism
                - "sp_ulysses": Ulysses Sequence Parallelism
                - "sp_ring": Ring Sequence Parallelism
                - "sp": General Sequence Parallelism
                - "dp": Data Parallelism
                - "pp": Pipeline Parallelism
            strategy: Optional StrategyConfig for degree-based bandwidth lookup

        Returns:
            Bandwidth in GB/s (gigabytes per second)
        """
        # 通信域带宽层级映射
        # 优先级：TP(高带宽) > EP > Ulysses > Ring > DP(低带宽) > PP
        bandwidth_by_domain = {
            "tp": 0,          # Level 0: intra_node (NVLink)
            "ep": 0,          # Level 0 or 1: depends on ep_degree
            "sp_ulysses": 0,  # Level 0: intra_node
            "sp_ring": 1,     # Level 1: intra_rack
            "sp": 0,          # Default: Level 0
            "dp": 1,          # Level 1 or 2: intra_rack or inter_rack
            "pp": 1,          # Level 1 or 2: intra_rack or inter_rack
        }

        level_idx = bandwidth_by_domain.get(domain_type, 0)

        # 如果提供了 strategy，可以根据并行度动态调整拓扑层级
        if strategy:
            devices_per_node = self.devices_per_node
            devices_per_rack = devices_per_node * 16  # 默认16节点/rack

            if domain_type == "tp":
                level_idx = 0 if strategy.tp_degree <= devices_per_node else 1
            elif domain_type == "ep":
                level_idx = 0 if strategy.ep_degree <= devices_per_node else (
                    1 if strategy.ep_degree <= devices_per_rack else 2
                )
            elif domain_type == "sp_ulysses":
                ulysses_degree = strategy.ulysses_degree if strategy.sp_type.value == "ulysses" else 1
                level_idx = 0 if ulysses_degree <= devices_per_node else 1
            elif domain_type == "sp_ring":
                ring_degree = strategy.ring_degree if strategy.sp_type.value in ("ring_p2p", "ring_allgather") else 1
                level_idx = 1 if ring_degree > 1 else 0
            elif domain_type == "sp":
                level_idx = 0 if strategy.sp_degree <= devices_per_node else 1
            elif domain_type == "dp":
                level_idx = 2 if strategy.dp_degree > 16 else (1 if strategy.dp_degree > 1 else 0)
            elif domain_type == "pp":
                level_idx = 2 if strategy.pp_degree > 16 else (1 if strategy.pp_degree > 1 else 0)

        # 查找对应的拓扑层级带宽
        for level in self.topology.levels:
            if level.level == level_idx:
                return level.bandwidth_gbps

        # 如果找不到精确匹配，找最接近的层级
        if self.topology.levels:
            # 按 level 排序，找最接近的
            sorted_levels = sorted(self.topology.levels, key=lambda l: l.level)
            for level in sorted_levels:
                if level.level <= level_idx:
                    continue
                # 返回不超过请求层级的最高层级的带宽
                prev_level = sorted_levels[sorted_levels.index(level) - 1] if sorted_levels.index(level) > 0 else level
                return prev_level.bandwidth_gbps
            # 如果请求层级比所有定义的层级都高，返回最高层级的带宽
            return sorted_levels[-1].bandwidth_gbps

        # 默认 fallback
        return 100.0

    def get_bandwidth_for_topology_level(
        self,
        topology_level: str,
        devices_per_group: int = 8,
    ) -> float:
        """
        根据拓扑层级名称获取带宽。

        Args:
            topology_level: 层级名称 ("node", "rack", "inter_node", "cluster", "intra_node", "intra_rack", "inter_rack")
            devices_per_group: 该层级设备数（用于 fallback）

        Returns:
            Bandwidth in GB/s (gigabytes per second)
        """
        # 拓扑层级名称映射
        level_map = {
            "node": 0,
            "intra_node": 0,
            "rack": 1,
            "intra_rack": 1,
            "inter_node": 1,
            "cluster": 2,
            "inter_rack": 2,
        }

        level_idx = level_map.get(topology_level, 0)

        # 查找对应的拓扑层级带宽
        for level in self.topology.levels:
            if level.level == level_idx:
                return level.bandwidth_gbps

        # Fallback: 按 devices_per_group 查找
        for level in self.topology.levels:
            if level.devices_per_group >= devices_per_group:
                return level.bandwidth_gbps

        # Final fallback
        if self.topology.levels:
            return max(self.topology.levels, key=lambda l: l.bandwidth_gbps).bandwidth_gbps

        return 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_devices": self.num_devices,
            "num_nodes": self.num_nodes,
            "devices_per_node": self.devices_per_node,
            "device": self.devices[0].to_dict() if self.devices else None,
            "topology": self.topology.to_dict(),
        }

"""Cluster and network topology definitions."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from .device import Device, DeviceConfig
from .topology import NetworkTopology, TopologyLevel, TopologyType


@dataclass
class NetworkConfig:
    """
    Network configuration for cluster interconnect.
    
    Deprecated: Use NetworkTopology instead for hierarchical topologies.
    This class is kept for backward compatibility.
    """
    
    # Intra-node (within server) bandwidth
    intra_node_bandwidth_gbps: float  # NVLink / Infinity Fabric
    intra_node_latency_us: float = 1.0
    
    # Inter-node (between servers) bandwidth
    inter_node_bandwidth_gbps: float  # IB / RoCE / etc
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
        topology: NetworkTopology,
        devices_per_node: int = 8,
        # Hierarchical grouping for multi-tier topologies
        node_grouping: Optional[List[List[int]]] = None,
        rack_grouping: Optional[List[List[int]]] = None,
    ):
        """
        Initialize cluster with hierarchical topology.
        
        Args:
            devices: List of devices in the cluster
            topology: NetworkTopology defining hierarchical bandwidth
            devices_per_node: Number of devices per physical node
            node_grouping: Optional explicit node to device mapping
            rack_grouping: Optional rack to node mapping for multi-tier
        """
        self.devices = devices
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
        topology: NetworkTopology,
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
        topology: NetworkTopology,
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
        
        # Highest level needed
        if self.topology.levels:
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_devices": self.num_devices,
            "num_nodes": self.num_nodes,
            "devices_per_node": self.devices_per_node,
            "device": self.devices[0].to_dict() if self.devices else None,
            "topology": self.topology.to_dict(),
        }

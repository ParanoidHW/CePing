"""Network topology definitions with hierarchical bandwidth support (Clos, Fat-Tree, etc.)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class TopologyType(Enum):
    """Network topology types."""

    FULL_MESH = "full_mesh"
    SWITCH = "switch"
    CLOS = "clos"  # Clos/Fat-Tree topology
    DRAGONFLY = "dragonfly"
    TORUS = "torus"
    HYPERCUBE = "hypercube"
    SUPERNODE = "supernode"  # Fully peer-to-peer supernode (e.g., Huawei CloudMatrix)
    CUSTOM = "custom"


@dataclass
class TopologyLevel:
    """
    Represents one level in a hierarchical topology.

    In a Clos topology, levels are ordered from closest to devices (Level 0)
    to furthest (highest level). Bandwidth typically decreases with level.

    Example for 3-tier Clos:
        Level 0: Node-local (NVLink/PCIe) - 900 GB/s
        Level 1: Rack-local (Leaf switch) - 200 GB/s
        Level 2: Cluster (Spine switch) - 100 GB/s
    """

    name: str  # e.g., "node", "rack", "cluster", "dc"
    level: int  # 0 = closest to device, higher = further
    bandwidth_gbps: float  # Bandwidth at this level
    latency_us: float = 1.0
    oversubscription_ratio: float = 1.0
    # Grouping information
    devices_per_group: int = 1  # How many devices share this level

    def __post_init__(self):
        if self.level < 0:
            raise ValueError(f"Level must be non-negative, got {self.level}")


@dataclass
class NetworkTopology:
    """
    Hierarchical network topology definition.

    Supports Clos, Fat-Tree, and other multi-tier topologies where bandwidth
    varies depending on which "tier" or "level" the communication traverses.
    """

    name: str = "default"
    topology_type: TopologyType = TopologyType.CLOS

    # Ordered list of levels from closest (0) to furthest
    levels: List[TopologyLevel] = field(default_factory=list)

    # Legacy support for simple 2-tier (intra/inter node) topologies
    intra_node_bandwidth_gbps: Optional[float] = None
    intra_node_latency_us: float = 1.0
    inter_node_bandwidth_gbps: Optional[float] = None
    inter_node_latency_us: float = 2.0

    def __post_init__(self):
        """Initialize levels from legacy config if not provided."""
        if not self.levels and self.intra_node_bandwidth_gbps is not None:
            # Create 2-level topology from legacy config
            self.levels = [
                TopologyLevel(
                    name="node",
                    level=0,
                    bandwidth_gbps=self.intra_node_bandwidth_gbps,
                    latency_us=self.intra_node_latency_us,
                    devices_per_group=8,  # Default
                ),
                TopologyLevel(
                    name="inter_node",
                    level=1,
                    bandwidth_gbps=self.inter_node_bandwidth_gbps or self.intra_node_bandwidth_gbps,
                    latency_us=self.inter_node_latency_us,
                    devices_per_group=999999,  # All devices
                ),
            ]

    @classmethod
    def create_clos_3tier(
        cls,
        node_bw_gbps: float,  # Level 0: NVLink/PCIe within node
        rack_bw_gbps: float,  # Level 1: Leaf switch within rack
        cluster_bw_gbps: float,  # Level 2: Spine switch across cluster
        devices_per_node: int = 8,
        nodes_per_rack: int = 16,
        racks_per_cluster: int = 32,
    ) -> "NetworkTopology":
        """
        Create a standard 3-tier Clos topology.

        Bandwidth hierarchy: node > rack > cluster
        """
        return cls(
            name="clos_3tier",
            topology_type=TopologyType.CLOS,
            levels=[
                TopologyLevel(
                    name="node",
                    level=0,
                    bandwidth_gbps=node_bw_gbps,
                    latency_us=1.0,
                    devices_per_group=devices_per_node,
                ),
                TopologyLevel(
                    name="rack",
                    level=1,
                    bandwidth_gbps=rack_bw_gbps,
                    latency_us=2.0,
                    devices_per_group=devices_per_node * nodes_per_rack,
                ),
                TopologyLevel(
                    name="cluster",
                    level=2,
                    bandwidth_gbps=cluster_bw_gbps,
                    latency_us=5.0,
                    devices_per_group=devices_per_node * nodes_per_rack * racks_per_cluster,
                ),
            ],
        )

    @classmethod
    def create_fat_tree(
        cls,
        core_bw_gbps: float,  # Core switches (highest level, lowest bw per path)
        agg_bw_gbps: float,  # Aggregation switches
        edge_bw_gbps: float,  # Edge switches (closest to hosts)
        oversubscription: float = 4.0,  # 4:1 is common
    ) -> "NetworkTopology":
        """
        Create a Fat-Tree topology (common in data centers).

        Note: In Fat-Tree, higher levels have more aggregate bandwidth,
        but per-flow bandwidth may be limited by edge links.
        """
        return cls(
            name="fat_tree",
            topology_type=TopologyType.CLOS,
            levels=[
                TopologyLevel(
                    name="edge",
                    level=0,
                    bandwidth_gbps=edge_bw_gbps,
                    latency_us=1.0,
                    oversubscription_ratio=oversubscription,
                ),
                TopologyLevel(
                    name="aggregation",
                    level=1,
                    bandwidth_gbps=agg_bw_gbps,
                    latency_us=2.0,
                    oversubscription_ratio=oversubscription,
                ),
                TopologyLevel(
                    name="core",
                    level=2,
                    bandwidth_gbps=core_bw_gbps,
                    latency_us=3.0,
                    oversubscription_ratio=1.0,  # Fully provisioned at core
                ),
            ],
        )

    @classmethod
    def create_2tier_simple(
        cls,
        intra_node_bw_gbps: float,
        inter_node_bw_gbps: float,
        devices_per_node: int = 8,
    ) -> "NetworkTopology":
        """Create simple 2-tier topology (backward compatible)."""
        return cls(
            name="2tier_simple",
            topology_type=TopologyType.SWITCH,
            levels=[
                TopologyLevel(
                    name="node",
                    level=0,
                    bandwidth_gbps=intra_node_bw_gbps,
                    latency_us=1.0,
                    devices_per_group=devices_per_node,
                ),
                TopologyLevel(
                    name="inter_node",
                    level=1,
                    bandwidth_gbps=inter_node_bw_gbps,
                    latency_us=2.0,
                    devices_per_group=999999,
                ),
            ],
        )

    @classmethod
    def create_cloudmatrix_supernode(
        cls,
        num_npus: int = 384,
        num_cpus: int = 192,
        ub_bw_gbps: float = 3136.0,  # ~400GB/s x 8 = 3.1TB/s for 910C
        ub_latency_us: float = 2.0,
        rdma_bw_gbps: float = 400.0,  # 400Gbps RoCE for scale-out
        rdma_latency_us: float = 5.0,
    ) -> "NetworkTopology":
        """
        Create Huawei CloudMatrix 384 supernode topology.

        Based on: "Serving Large Language Models on Huawei CloudMatrix384"
        arXiv:2506.12708 (June 2025)

        Architecture:
        - 384 Ascend 910C NPUs + 192 Kunpeng CPUs
        - Fully peer-to-peer non-blocking all-to-all topology via Unified Bus (UB)
        - Each 910C provides >392GB/s unidirectional UB bandwidth
        - Inter-node bandwidth degradation < 3%, latency increase < 1μs
        - Three network planes: UB (intra-supernode), RDMA (inter-supernode), VPC (datacenter)

        This topology treats all 384 NPUs as a single flat fully-connected domain
        with uniform high bandwidth, unlike hierarchical Clos topologies.

        Args:
            num_npus: Number of Ascend 910C NPUs (default 384)
            num_cpus: Number of Kunpeng CPUs (default 192)
            ub_bw_gbps: Unified Bus bandwidth per NPU in Gbps (default 3136 Gbps = 392 GB/s)
            ub_latency_us: UB latency in microseconds (default 2.0)
            rdma_bw_gbps: RDMA bandwidth for scale-out in Gbps (default 400)
            rdma_latency_us: RDMA latency in microseconds (default 5.0)

        Returns:
            NetworkTopology configured for CloudMatrix supernode
        """
        # Calculate aggregate UB bandwidth for the supernode
        # In CloudMatrix, all NPUs are fully connected via UB switches
        # Effective bisection bandwidth is maintained via non-blocking topology

        return cls(
            name=f"cloudmatrix_{num_npus}npu",
            topology_type=TopologyType.SUPERNODE,
            levels=[
                TopologyLevel(
                    name="ub_plane",
                    level=0,
                    bandwidth_gbps=ub_bw_gbps,  # Per-link bandwidth to UB switch
                    latency_us=ub_latency_us,
                    oversubscription_ratio=1.0,  # Non-blocking
                    devices_per_group=num_npus,  # All NPUs in one UB domain
                ),
                TopologyLevel(
                    name="rdma_plane",
                    level=1,
                    bandwidth_gbps=rdma_bw_gbps,  # RoCE for scale-out
                    latency_us=rdma_latency_us,
                    oversubscription_ratio=1.0,
                    devices_per_group=999999,  # External communication
                ),
            ],
        )

    def get_level_for_distance(self, distance: int) -> TopologyLevel:
        """
        Get the topology level based on communication distance.

        Args:
            distance: Number of devices between source and destination
                     (in terms of grouping, not absolute rank difference)

        Returns:
            The appropriate topology level
        """
        for level in sorted(self.levels, key=lambda l: l.level):
            if distance < level.devices_per_group or level.devices_per_group >= 999999:
                return level
        return self.levels[-1] if self.levels else None

    def get_bandwidth_and_latency(
        self, source_rank: int, dest_rank: int, device_groups: List[List[int]]
    ) -> Tuple[float, float]:
        """
        Get bandwidth and latency between two ranks considering topology.

        Args:
            source_rank: Source device rank
            dest_rank: Destination device rank
            device_groups: List of device groupings at each level
                          e.g., [[0-7], [8-15], ...] for nodes

        Returns:
            (bandwidth_gbps, latency_us)
        """
        if source_rank == dest_rank:
            return float("inf"), 0.0

        # Determine which level the communication traverses
        for i, group in enumerate(device_groups):
            source_group = None
            dest_group = None
            for j, g in enumerate(group):
                if source_rank in g:
                    source_group = j
                if dest_rank in g:
                    dest_group = j

            if source_group != dest_group:
                # Communication crosses this level
                level = self.get_level_for_distance(i)
                return level.bandwidth_gbps, level.latency_us

        # Default to highest level if no match
        if self.levels:
            highest = max(self.levels, key=lambda l: l.level)
            return highest.bandwidth_gbps, highest.latency_us

        return 0.0, 0.0

    def get_bottleneck_bandwidth(self, ranks: List[int]) -> float:
        """
        Get the bottleneck bandwidth for a group of ranks.

        In collective operations, the bottleneck is determined by the
        minimum bandwidth along the communication path.

        Args:
            ranks: List of participating ranks

        Returns:
            Bottleneck bandwidth in GB/s
        """
        if len(ranks) <= 1:
            return float("inf")

        # For simplicity, find the highest level needed
        max_distance = len(ranks)  # Conservative estimate
        level = self.get_level_for_distance(max_distance)

        return level.bandwidth_gbps if level else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "topology_type": self.topology_type.value,
            "levels": [
                {
                    "name": l.name,
                    "level": l.level,
                    "bandwidth_gbps": l.bandwidth_gbps,
                    "latency_us": l.latency_us,
                    "oversubscription_ratio": l.oversubscription_ratio,
                    "devices_per_group": l.devices_per_group,
                }
                for l in self.levels
            ],
        }


class ClosTopologyBuilder:
    """
    Builder for creating Clos topology configurations.

    Clos topology characteristics:
    - Multiple stages (tiers) of switches
    - Each stage has fixed radix (number of ports)
    - Bandwidth decreases as you go up the hierarchy
    - Provides non-blocking or near-non-blocking connectivity
    """

    @staticmethod
    def calculate_clos_parameters(
        num_devices: int,
        switch_radix: int = 32,  # Typical: 32 or 64 port switches
    ) -> Dict[str, Any]:
        """
        Calculate Clos topology parameters for given number of devices.

        Returns dict with:
            - num_tiers: Number of switch tiers needed
            - switches_per_tier: List of switch counts per tier
            - devices_per_leaf: Number of devices connected to each leaf switch
        """
        if num_devices <= switch_radix:
            # Single tier (just edge switches)
            return {
                "num_tiers": 1,
                "switches_per_tier": [num_devices],
                "devices_per_leaf": 1,
            }

        # Two-tier (leaf-spine)
        devices_per_leaf = switch_radix // 2  # Half for hosts, half for uplinks
        num_leaf_switches = (num_devices + devices_per_leaf - 1) // devices_per_leaf
        num_spine_switches = min(num_leaf_switches, switch_radix)

        if num_leaf_switches <= switch_radix:
            return {
                "num_tiers": 2,
                "switches_per_tier": [num_leaf_switches, num_spine_switches],
                "devices_per_leaf": devices_per_leaf,
            }

        # Three-tier (leaf-spine-core)
        num_leaf_groups = (num_leaf_switches + switch_radix - 1) // switch_radix
        num_spine_per_group = switch_radix // 2
        num_core_switches = num_spine_per_group

        return {
            "num_tiers": 3,
            "switches_per_tier": [num_leaf_switches, num_leaf_groups * num_spine_per_group, num_core_switches],
            "devices_per_leaf": devices_per_leaf,
        }

    @classmethod
    def create_from_device_count(
        cls,
        num_devices: int,
        switch_radix: int = 32,
        host_bw_gbps: float = 200.0,  # Host to leaf
        leaf_bw_gbps: float = 200.0,  # Leaf to spine (may be oversubscribed)
        spine_bw_gbps: float = 200.0,  # Spine to core
        oversubscription: float = 4.0,  # Typical: 4:1 oversubscription at leaf
    ) -> NetworkTopology:
        """
        Create a Clos topology based on device count.

        Automatically calculates the number of tiers needed.
        """
        params = cls.calculate_clos_parameters(num_devices, switch_radix)
        num_tiers = params["num_tiers"]

        if num_tiers == 1:
            # Single tier - all devices connected to same switch
            return NetworkTopology.create_2tier_simple(
                intra_node_bw_gbps=host_bw_gbps,
                inter_node_bw_gbps=host_bw_gbps,
            )

        elif num_tiers == 2:
            # Two-tier: leaf-spine
            return NetworkTopology(
                name=f"clos_2tier_{num_devices}devices",
                topology_type=TopologyType.CLOS,
                levels=[
                    TopologyLevel(
                        name="leaf",
                        level=0,
                        bandwidth_gbps=host_bw_gbps,
                        latency_us=1.0,
                        oversubscription_ratio=1.0,
                        devices_per_group=params["devices_per_leaf"],
                    ),
                    TopologyLevel(
                        name="spine",
                        level=1,
                        bandwidth_gbps=leaf_bw_gbps / oversubscription,
                        latency_us=2.0,
                        oversubscription_ratio=oversubscription,
                        devices_per_group=num_devices,
                    ),
                ],
            )

        else:  # num_tiers == 3
            # Three-tier: leaf-spine-core
            dpl = params["devices_per_leaf"]
            leaf_per_pod = switch_radix // 2
            devices_per_pod = dpl * leaf_per_pod

            return NetworkTopology(
                name=f"clos_3tier_{num_devices}devices",
                topology_type=TopologyType.CLOS,
                levels=[
                    TopologyLevel(
                        name="leaf",
                        level=0,
                        bandwidth_gbps=host_bw_gbps,
                        latency_us=1.0,
                        devices_per_group=dpl,
                    ),
                    TopologyLevel(
                        name="spine",
                        level=1,
                        bandwidth_gbps=leaf_bw_gbps / oversubscription,
                        latency_us=2.0,
                        oversubscription_ratio=oversubscription,
                        devices_per_group=devices_per_pod,
                    ),
                    TopologyLevel(
                        name="core",
                        level=2,
                        bandwidth_gbps=spine_bw_gbps,
                        latency_us=5.0,
                        devices_per_group=num_devices,
                    ),
                ],
            )

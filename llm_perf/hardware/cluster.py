"""Cluster and network topology definitions."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from .device import Device, DeviceConfig


@dataclass
class NetworkConfig:
    """Network configuration for cluster interconnect."""
    
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


class Cluster:
    """Represents a cluster of compute devices."""
    
    def __init__(
        self,
        devices: List[Device],
        network: NetworkConfig,
        devices_per_node: int = 8,
    ):
        self.devices = devices
        self.network = network
        self.devices_per_node = devices_per_node
        
        self.num_devices = len(devices)
        self.num_nodes = (self.num_devices + devices_per_node - 1) // devices_per_node
    
    @classmethod
    def create_homogeneous(
        cls,
        device_config: DeviceConfig,
        num_devices: int,
        network: NetworkConfig,
        devices_per_node: int = 8,
    ) -> "Cluster":
        """Create a homogeneous cluster with identical devices."""
        devices = [Device(device_config) for _ in range(num_devices)]
        return cls(devices, network, devices_per_node)
    
    @classmethod
    def create_from_preset(
        cls,
        preset_name: str,
        num_devices: int,
        network: NetworkConfig,
        devices_per_node: int = 8,
    ) -> "Cluster":
        """Create cluster from device preset."""
        device = Device.from_preset(preset_name)
        return cls.create_homogeneous(
            device.config, num_devices, network, devices_per_node
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
    
    def get_bandwidth_between(self, rank1: int, rank2: int) -> float:
        """
        Get bandwidth between two ranks in GB/s.
        
        Returns:
            Bandwidth in GB/s
        """
        if self.are_on_same_node(rank1, rank2):
            return self.network.intra_node_bandwidth_gbps
        else:
            return self.network.inter_node_bandwidth_gbps
    
    def get_latency_between(self, rank1: int, rank2: int) -> float:
        """
        Get latency between two ranks in microseconds.
        
        Returns:
            Latency in microseconds
        """
        if self.are_on_same_node(rank1, rank2):
            return self.network.intra_node_latency_us
        else:
            return self.network.inter_node_latency_us
    
    def estimate_allreduce_time(
        self,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> float:
        """
        Estimate all-reduce communication time.
        
        Uses ring algorithm estimation:
        time = 2 * (n-1)/n * data_size / bandwidth + n * latency
        
        Args:
            num_bytes: Size of data per rank
            participating_ranks: List of participating ranks
        
        Returns:
            Estimated time in seconds
        """
        n = len(participating_ranks)
        if n <= 1:
            return 0.0
        
        # Assume homogeneous network, use average bandwidth
        total_bw = sum(
            self.get_bandwidth_between(participating_ranks[i], participating_ranks[(i+1) % n])
            for i in range(n)
        ) / n
        
        avg_latency = sum(
            self.get_latency_between(participating_ranks[i], participating_ranks[(i+1) % n])
            for i in range(n)
        ) / n
        
        # Ring all-reduce: 2*(n-1) steps, each moving (num_bytes/n) data
        # With bandwidth-optimal algorithm
        data_per_step = num_bytes / n
        num_steps = 2 * (n - 1)
        
        # Convert GB/s to bytes/s
        bw_bytes_per_sec = total_bw * 1e9
        latency_sec = avg_latency * 1e-6
        
        transfer_time = num_steps * data_per_step / bw_bytes_per_sec
        latency_time = n * latency_sec
        
        return transfer_time + latency_time
    
    def estimate_allgather_time(
        self,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> float:
        """
        Estimate all-gather communication time.
        
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
        Estimate all-to-all communication time (used in MoE).
        
        Args:
            num_bytes: Size of data per rank
            participating_ranks: List of participating ranks
        
        Returns:
            Estimated time in seconds
        """
        n = len(participating_ranks)
        if n <= 1:
            return 0.0
        
        # Each rank sends (n-1)/n of its data to other ranks
        # On average, half goes to local node, half to remote
        
        avg_bw = (
            self.network.intra_node_bandwidth_gbps * 0.5 +
            self.network.inter_node_bandwidth_gbps * 0.5
        )
        
        # Total data moved per rank
        total_data = num_bytes * (n - 1) / n
        
        bw_bytes_per_sec = avg_bw * 1e9
        return total_data / bw_bytes_per_sec
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_devices": self.num_devices,
            "num_nodes": self.num_nodes,
            "devices_per_node": self.devices_per_node,
            "device": self.devices[0].to_dict() if self.devices else None,
            "network": {
                "intra_node_bandwidth_gbps": self.network.intra_node_bandwidth_gbps,
                "intra_node_latency_us": self.network.intra_node_latency_us,
                "inter_node_bandwidth_gbps": self.network.inter_node_bandwidth_gbps,
                "inter_node_latency_us": self.network.inter_node_latency_us,
                "topology": self.network.topology,
                "oversubscription_ratio": self.network.oversubscription_ratio,
            }
        }

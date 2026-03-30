"""Communication kernel evaluation."""

from typing import Dict, Optional, List, Any, Union
import math

from .base import Kernel, KernelConfig, KernelType
from ..hardware.cluster import Cluster
from ..hardware.topology import NetworkTopology


class CommKernel(Kernel):
    """Communication kernel for collective operations with hierarchical topology support."""
    
    def __init__(
        self,
        config: KernelConfig,
        cluster: Cluster,
        collective_type: str,  # "allreduce", "allgather", "alltoall", etc.
        num_bytes: int,
        participating_ranks: List[int],
    ):
        super().__init__(config)
        self.cluster = cluster
        self.collective_type = collective_type
        self.num_bytes = num_bytes
        self.participating_ranks = participating_ranks
    
    def _get_average_bandwidth(self) -> float:
        """
        Get average bandwidth across topology levels.
        
        For hierarchical topologies (Clos), returns weighted average
        considering how many ranks communicate at each level.
        """
        n = len(self.participating_ranks)
        if n <= 1:
            return float('inf')
        
        # Check if using new hierarchical topology
        topology = self.cluster.topology
        
        # Count communications at each level
        level_bandwidths = []
        for i in range(n):
            for j in range(i + 1, n):
                bw = self.cluster.get_bandwidth_between(
                    self.participating_ranks[i],
                    self.participating_ranks[j]
                )
                level_bandwidths.append(bw)
        
        if level_bandwidths:
            # Use harmonic mean for bandwidth (more accurate for bottlenecks)
            return len(level_bandwidths) / sum(1.0 / bw for bw in level_bandwidths if bw > 0)
        
        # Fallback: use topology levels
        if topology.levels:
            return sum(l.bandwidth_gbps for l in topology.levels) / len(topology.levels)
        
        return 100.0  # Default fallback
    
    def estimate_time(
        self,
        input_shape: tuple = None,
        output_shape: tuple = None,
        dtype: str = None,
        **kwargs
    ) -> float:
        """
        Estimate communication time for collective operation.
        
        Uses topology-aware estimation from Cluster class for
        hierarchical topologies (Clos, Fat-Tree).
        
        Returns:
            Time in seconds
        """
        if self.config.measured_bw:
            # Use measured bandwidth if available
            return self.num_bytes / self.config.measured_bw
        
        # Use cluster's topology-aware estimation methods
        if self.collective_type == "allreduce":
            return self.cluster.estimate_allreduce_time(
                self.num_bytes,
                self.participating_ranks
            )
        elif self.collective_type == "allgather":
            return self.cluster.estimate_allgather_time(
                self.num_bytes,
                self.participating_ranks
            )
        elif self.collective_type == "alltoall":
            return self.cluster.estimate_alltoall_time(
                self.num_bytes,
                self.participating_ranks
            )
        elif self.collective_type == "broadcast":
            # Simple broadcast to all ranks
            n = len(self.participating_ranks)
            if n <= 1:
                return 0.0
            # Tree-based broadcast: log2(n) steps
            steps = math.ceil(math.log2(n))
            avg_bw = self._get_average_bandwidth() * 1e9  # Convert to bytes/s
            return steps * self.num_bytes / avg_bw
        elif self.collective_type == "reduce_scatter":
            # Similar to allreduce but in opposite direction
            return self.cluster.estimate_allreduce_time(
                self.num_bytes,
                self.participating_ranks
            ) / 2
        else:
            # Default: simple bandwidth model
            avg_bw = self._get_average_bandwidth() * 1e9
            return self.num_bytes / avg_bw
    
    def estimate_memory(
        self,
        input_shape: tuple = None,
        output_shape: tuple = None,
        dtype: str = None,
        **kwargs
    ) -> int:
        """Estimate communication buffer memory."""
        # Communication usually needs buffer space
        if self.collective_type in ["allreduce", "reduce_scatter"]:
            return self.num_bytes  # In-place or single buffer
        elif self.collective_type in ["allgather", "alltoall"]:
            return self.num_bytes * len(self.participating_ranks)
        else:
            return self.num_bytes * 2


class CommKernelRegistry:
    """Registry for communication kernels."""
    
    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self._kernels: Dict[str, CommKernel] = {}
    
    def create_allreduce(
        self,
        name: str,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> CommKernel:
        """Create an all-reduce kernel."""
        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMMUNICATION,
        )
        kernel = CommKernel(
            config, self.cluster, "allreduce", num_bytes, participating_ranks
        )
        self._kernels[name] = kernel
        return kernel
    
    def create_allgather(
        self,
        name: str,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> CommKernel:
        """Create an all-gather kernel."""
        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMMUNICATION,
        )
        kernel = CommKernel(
            config, self.cluster, "allgather", num_bytes, participating_ranks
        )
        self._kernels[name] = kernel
        return kernel
    
    def create_alltoall(
        self,
        name: str,
        num_bytes: int,
        participating_ranks: List[int],
    ) -> CommKernel:
        """Create an all-to-all kernel."""
        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMMUNICATION,
        )
        kernel = CommKernel(
            config, self.cluster, "alltoall", num_bytes, participating_ranks
        )
        self._kernels[name] = kernel
        return kernel
    
    def create_tp_allreduce(
        self,
        layer_name: str,
        param_bytes: int,
        tp_ranks: List[int],
    ) -> CommKernel:
        """
        Create tensor parallelism all-reduce kernel.
        
        In TP, after each linear layer that is split across devices,
        we need all-reduce to aggregate results.
        """
        name = f"tp_allreduce_{layer_name}"
        return self.create_allreduce(name, param_bytes, tp_ranks)
    
    def create_dp_allreduce(
        self,
        layer_name: str,
        grad_bytes: int,
        dp_ranks: List[int],
    ) -> CommKernel:
        """
        Create data parallelism all-reduce kernel.
        
        In DP, gradients need to be all-reduced across data parallel ranks.
        """
        name = f"dp_allreduce_{layer_name}"
        return self.create_allreduce(name, grad_bytes, dp_ranks)
    
    def create_ep_alltoall(
        self,
        layer_name: str,
        token_bytes: int,
        ep_ranks: List[int],
    ) -> CommKernel:
        """
        Create expert parallelism all-to-all kernel.
        
        In EP, tokens are dispatched to experts via all-to-all.
        """
        name = f"ep_alltoall_{layer_name}_dispatch"
        return self.create_alltoall(name, token_bytes, ep_ranks)
    
    def create_ep_alltoall_combine(
        self,
        layer_name: str,
        token_bytes: int,
        ep_ranks: List[int],
    ) -> CommKernel:
        """Create EP all-to-all for combining expert outputs."""
        name = f"ep_alltoall_{layer_name}_combine"
        return self.create_alltoall(name, token_bytes, ep_ranks)
    
    def create_pp_p2p(
        self,
        stage: int,
        activation_bytes: int,
        src_rank: int,
        dst_rank: int,
    ) -> CommKernel:
        """
        Create pipeline parallelism P2P send/recv kernel.
        
        In PP, activations are sent between stages.
        """
        name = f"pp_p2p_stage{stage}"
        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMMUNICATION,
        )
        # P2P uses point-to-point bandwidth
        ranks = [src_rank, dst_rank]
        kernel = CommKernel(
            config, self.cluster, "broadcast", activation_bytes, ranks
        )
        self._kernels[name] = kernel
        return kernel
    
    def get(self, name: str) -> Optional[CommKernel]:
        """Get a kernel by name."""
        return self._kernels.get(name)
    
    def list_kernels(self) -> list:
        """List all registered kernel names."""
        return list(self._kernels.keys())
    
    def estimate_tp_communication(
        self,
        model_params: int,
        dtype: str,
        tp_degree: int,
        seq_len: int,
        batch_size: int,
    ) -> Dict[str, float]:
        """
        Estimate total tensor parallelism communication overhead.
        
        Returns:
            Dict with time breakdown
        """
        from ..utils.constants import DTYPE_SIZES
        
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        # In Megatron-style TP, each layer has 2 all-reduces (attention + MLP)
        # Each all-reduce handles the full activation size
        
        # Assuming hidden_size from typical models
        hidden_size = 4096  # Could be parameterized
        
        # Activation bytes per layer
        activation_bytes = batch_size * seq_len * hidden_size * dtype_size
        
        # Number of transformer layers (typical)
        num_layers = 32
        
        # 2 all-reduces per layer (attention + MLP)
        total_bytes = activation_bytes * 2 * num_layers
        
        # Create a sample TP group
        tp_ranks = list(range(tp_degree))
        
        time = self.cluster.estimate_allreduce_time(total_bytes, tp_ranks)
        
        return {
            "total_bytes": total_bytes,
            "estimated_time_sec": time,
            "overhead_percent": 0.0,  # Will be calculated relative to compute
        }
    
    def estimate_dp_communication(
        self,
        model_params: int,
        dtype: str,
        dp_degree: int,
    ) -> Dict[str, float]:
        """
        Estimate total data parallelism communication overhead.
        
        Returns:
            Dict with time breakdown
        """
        from ..utils.constants import DTYPE_SIZES
        
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        grad_bytes = model_params * dtype_size
        
        dp_ranks = list(range(dp_degree))
        time = self.cluster.estimate_allreduce_time(grad_bytes, dp_ranks)
        
        return {
            "grad_bytes": grad_bytes,
            "estimated_time_sec": time,
            "overhead_percent": 0.0,
        }
    
    def estimate_ep_communication(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype: str,
        ep_degree: int,
        num_experts: int,
        num_experts_per_token: int,
    ) -> Dict[str, float]:
        """
        Estimate expert parallelism communication overhead.
        
        Returns:
            Dict with time breakdown
        """
        from ..utils.constants import DTYPE_SIZES
        
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        # Tokens dispatched per device
        # In EP, each token goes to num_experts_per_token experts
        # On average, each expert gets (batch * seq * num_experts_per_token / num_experts) tokens
        
        tokens_per_expert = batch_size * seq_len * num_experts_per_token / num_experts
        token_bytes = int(tokens_per_expert * hidden_size * dtype_size)
        
        # All-to-all for dispatch and combine
        ep_ranks = list(range(ep_degree))
        dispatch_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
        combine_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
        
        return {
            "dispatch_bytes": token_bytes,
            "combine_bytes": token_bytes,
            "dispatch_time_sec": dispatch_time,
            "combine_time_sec": combine_time,
            "total_time_sec": dispatch_time + combine_time,
        }

"""Base analyzer with shared estimation logic."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING

from ..models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig, SPType
from ..kernels.compute import ComputeKernelRegistry
from ..kernels.communication import CommKernelRegistry
from ..utils.constants import DTYPE_SIZES

if TYPE_CHECKING:
    pass


class BaseAnalyzer(ABC):
    """Abstract base class for performance analyzers.

    Provides shared estimation logic for:
    - Communication time (TP/PP/DP/EP/SP)
    - Memory estimation framework
    - Compute kernel retrieval

    Subclasses implement:
    - analyze(): Main analysis method
    - _estimate_compute_time(): Compute time estimation
    - Specific result types
    """

    def __init__(
        self,
        model: BaseModel,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize analyzer.

        Args:
            model: Model to analyze
            device: Device configuration
            cluster: Cluster configuration
            strategy: Parallelism strategy
        """
        self.model = model
        self.device = device
        self.cluster = cluster
        self.strategy = strategy

        self.compute_registry = ComputeKernelRegistry(device)
        self.comm_registry = CommKernelRegistry(cluster)

        self._scheduler_model = None

        # Cache communication domain mapping
        self._comm_domain_cache: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def scheduler_model(self):
        """Get scheduler model (lazy initialization to avoid circular import)."""
        if self._scheduler_model is None:
            from ..scheduler.base import SchedulerModel, SchedulerConfig

            scheduler_config = SchedulerConfig.from_dict(self.strategy.scheduler)
            self._scheduler_model = SchedulerModel(scheduler_config)
        return self._scheduler_model

    def _get_communication_domain_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get cached communication domain mapping."""
        if self._comm_domain_cache is None:
            self._comm_domain_cache = self.strategy.get_communication_domain_mapping(
                devices_per_node=self.cluster.devices_per_node,
                nodes_per_rack=self.cluster.num_nodes,
                total_devices=self.cluster.num_devices,
            )
        return self._comm_domain_cache

    def _get_communication_bandwidth(self, domain_type: str) -> float:
        """Get bandwidth for a specific communication domain type.

        Args:
            domain_type: Communication domain type ("tp", "pp", "dp", "ep", "sp",
                        "ulysses", "ring")

        Returns:
            Bandwidth in Gbps (gigabits per second)
        """
        comm_domain = self._get_communication_domain_mapping()

        if domain_type in comm_domain:
            domain_info = comm_domain[domain_type]
            bandwidth_domain = domain_info.get("bandwidth_domain", "inter_node")
            devices_per_group = domain_info.get("devices_per_group", 1)
            return self.cluster.get_bandwidth_for_topology_level(bandwidth_domain, devices_per_group)

        # Fallback: use domain_type directly with strategy
        return self.cluster.get_bandwidth_for_communication_domain(domain_type, self.strategy)

    def _get_communication_ranks(self, domain_type: str) -> List[int]:
        """Get ranks for a specific communication domain type.

        Args:
            domain_type: Communication domain type ("tp", "pp", "dp", "ep", "sp",
                        "ulysses", "ring")

        Returns:
            List of ranks in the domain
        """
        comm_domain = self._get_communication_domain_mapping()

        if domain_type in comm_domain:
            return comm_domain[domain_type].get("ranks", [])

        # Fallback: return logical ranks
        degree_map = {
            "tp": self.strategy.tp_degree,
            "pp": self.strategy.pp_degree,
            "dp": self.strategy.dp_degree,
            "ep": self.strategy.ep_degree,
            "sp": self.strategy.sp_degree,
        }
        degree = degree_map.get(domain_type, 1)
        return list(range(degree)) if degree > 1 else []

    @abstractmethod
    def analyze(self, **kwargs) -> Any:
        """Analyze performance. Subclasses implement specific logic."""
        pass

    # ==================== Communication Estimation ====================

    def _estimate_tp_communication_time(
        self,
        activation_bytes: int,
        num_layers: int,
    ) -> Tuple[float, float]:
        """Estimate Tensor Parallelism communication time.

        Args:
            activation_bytes: Activation size per token
            num_layers: Number of layers

        Returns:
            Tuple of (total_time, per_layer_time)
        """
        if self.strategy.tp_degree <= 1:
            return 0.0, 0.0

        # 2 all-reduces per layer (forward + backward for training)
        # For inference, only 1 all-reduce per layer
        tp_bytes = activation_bytes * 2 * num_layers

        # Use correct rank mapping from communication domain
        tp_ranks = self._get_communication_ranks("tp")
        if not tp_ranks:
            tp_ranks = list(range(self.strategy.tp_degree))

        tp_time = self.cluster.estimate_allreduce_time(tp_bytes, tp_ranks)

        per_layer_time = tp_time / num_layers
        return tp_time, per_layer_time

    def _estimate_pp_communication_time(
        self,
        activation_bytes: int,
        num_micro_batches: int = 1,
    ) -> float:
        """Estimate Pipeline Parallelism communication time.

        Args:
            activation_bytes: Activation size to transfer
            num_micro_batches: Number of micro-batches

        Returns:
            Total PP communication time
        """
        if self.strategy.pp_degree <= 1:
            return 0.0

        # P2P communication for activation transfer between pipeline stages
        # PP typically uses inter-node bandwidth (cross-node communication)
        pp_bw_gbps = self._get_communication_bandwidth("pp")
        pp_time = activation_bytes / (pp_bw_gbps * 1e9)
        pp_time *= num_micro_batches

        return pp_time

    def _estimate_dp_communication_time(
        self,
        grad_bytes: int,
    ) -> float:
        """Estimate Data Parallelism communication time.

        Args:
            grad_bytes: Gradient size to all-reduce

        Returns:
            Total DP communication time
        """
        if self.strategy.dp_degree <= 1:
            return 0.0

        # ZeRO reduces communication
        zero_factor = {0: 1.0, 1: 1.0, 2: 1.0 / self.strategy.dp_degree, 3: 0}
        effective_bytes = grad_bytes * zero_factor.get(self.strategy.zero_stage, 1.0)

        # Use correct rank mapping from communication domain
        dp_ranks = self._get_communication_ranks("dp")
        if not dp_ranks:
            dp_ranks = list(range(self.strategy.dp_degree))

        dp_time = self.cluster.estimate_allreduce_time(effective_bytes, dp_ranks)

        return dp_time

    def _estimate_ep_communication_time(
        self,
        token_bytes: int,
    ) -> Tuple[float, float]:
        """Estimate Expert Parallelism communication time.

        Args:
            token_bytes: Token activation size

        Returns:
            Tuple of (total_time, dispatch_time)
        """
        if self.strategy.ep_degree <= 1:
            return 0.0, 0.0

        # Use correct rank mapping from communication domain
        ep_ranks = self._get_communication_ranks("ep")
        if not ep_ranks:
            ep_ranks = list(range(self.strategy.ep_degree))

        # Dispatch and combine (2 all-to-alls)
        dispatch_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
        combine_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)

        total_time = dispatch_time + combine_time
        return total_time, dispatch_time

    def _estimate_sp_communication_time(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> float:
        """Estimate Sequence Parallelism communication time.

        Supports ulysses-SP, ring-SP (p2p/allgather), and unified-2D-SP.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_size: Hidden dimension
            dtype_size: Data type size in bytes
            num_layers: Number of layers
            num_kv_heads: Number of KV heads
            head_dim: Head dimension

        Returns:
            Total SP communication time
        """
        sp_degree = self.strategy.sp_degree
        if sp_degree <= 1:
            return 0.0

        sp_type = self.strategy.sp_type

        # Get correct rank mapping from communication domain
        sp_ranks = self._get_communication_ranks("sp")
        if not sp_ranks:
            sp_ranks = list(range(sp_degree))

        # Activation bytes for all-to-all (Ulysses)
        activation_bytes = batch_size * seq_len * hidden_size * dtype_size

        # KV bytes per step for Ring
        kv_bytes_per_step = batch_size * (seq_len // sp_degree) * num_kv_heads * head_dim * 2 * dtype_size

        total_time = 0.0

        if sp_type == SPType.ULYSSES:
            # Ulysses-SP: All-to-all for sequence parallelism
            # Forward: 4 all-to-all per attention layer (Q, K, V pre-attention + O post-attention)
            # Backward: 4 all-to-all per attention layer (gradients flow in reverse)
            # Total: 8 all-to-all per layer
            alltoall_time = self.cluster.estimate_alltoall_time(activation_bytes, sp_ranks)
            total_time = alltoall_time * 8 * num_layers

        elif sp_type == SPType.RING_P2P:
            # Ring-P2P: (sp_degree - 1) P2P steps per layer
            # Forward: (sp_degree - 1) steps for KV transmission
            # Backward: (sp_degree - 1) steps for gradient transmission
            # Total: 2 * (sp_degree - 1) steps per layer
            if sp_degree > 1:
                avg_bw = self._get_average_bandwidth(sp_ranks)
                step_time = kv_bytes_per_step / (avg_bw * 1e9)
                total_time = step_time * (sp_degree - 1) * 2 * num_layers

        elif sp_type == SPType.RING_ALLGATHER:
            # Ring-AllGather: AllGather for KV aggregation
            # Forward: 1 or 2 allgather depending on kv_separate_allgather config
            #   - kv_separate_allgather=False: K+V一起传输，1个 AllGather
            #   - kv_separate_allgather=True: K、V分开传输，2个 AllGather
            # Backward: ReduceScatter (AllGather 的逆操作)
            # Total per layer:
            #   - kv_separate_allgather=False: 1 AG (forward) + 1 RS (backward) = 2 ops
            #   - kv_separate_allgather=True: 2 AG (forward) + 2 RS (backward) = 4 ops

            kv_bytes_per_block = kv_bytes_per_step

            num_forward_ag = 2 if self.strategy.kv_separate_allgather else 1
            num_backward_rs = num_forward_ag

            allgather_bytes = kv_bytes_per_block * sp_degree
            allgather_time = self.cluster.estimate_allgather_time(allgather_bytes, sp_ranks)
            reducescatter_time = self.cluster.estimate_reducescatter_time(allgather_bytes, sp_ranks)

            total_time = (allgather_time * num_forward_ag + reducescatter_time * num_backward_rs) * num_layers

        elif sp_type == SPType.UNIFIED_2D:
            ulysses_degree, ring_degree = self._resolve_2d_sp_config(sp_degree)

            # Use correct rank mapping from communication domain for ulysses and ring
            ulysses_ranks = self._get_communication_ranks("ulysses")
            if not ulysses_ranks:
                ulysses_ranks = list(range(ulysses_degree))

            ring_ranks = self._get_communication_ranks("ring")
            if not ring_ranks:
                ring_ranks = list(range(ring_degree))

            # Ulysses part: 8 all-to-all per layer (4 forward + 4 backward)
            ulysses_time = self.cluster.estimate_alltoall_time(activation_bytes, ulysses_ranks) * 8 * num_layers

            # Ring part
            ring_time = 0.0
            if ring_degree > 1:
                # Forward: (ring_degree - 1) steps
                # Backward: (ring_degree - 1) steps
                # Total: 2 * (ring_degree - 1) steps per layer
                ring_kv_bytes = batch_size * (seq_len // sp_degree) * num_kv_heads * head_dim * 2 * dtype_size
                avg_bw = self._get_average_bandwidth(ring_ranks)
                ring_step_time = ring_kv_bytes / (avg_bw * 1e9)
                ring_time = ring_step_time * (ring_degree - 1) * 2 * num_layers

            total_time = ulysses_time + ring_time

        elif sp_type == SPType.MEGATRON:
            # Megatron-SP: ReduceScatter + AllGather per layer
            # Communication volume: 2 * activation_bytes per layer (same as TP AllReduce)
            # But memory is sharded by sp_degree
            rs_time = self.cluster.estimate_reducescatter_time(activation_bytes, sp_ranks)
            ag_time = self.cluster.estimate_allgather_time(activation_bytes, sp_ranks)
            # Forward: 2 ops (1 rs + 1 ag)
            # Backward: 2 ops (1 rs + 1 ag, reverse direction)
            # Total: 4 ops per layer (2 rs + 2 ag)
            total_time = (rs_time + ag_time) * 4 * num_layers

        return total_time

    def _get_average_bandwidth(self, ranks: list) -> float:
        """Get average bandwidth between ranks.

        Args:
            ranks: List of rank indices

        Returns:
            Average bandwidth in Gbps
        """
        bw_values = []
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                bw = self.cluster.get_bandwidth_between(ranks[i], ranks[j])
                if bw > 0:
                    bw_values.append(bw)

        if not bw_values:
            return 100.0  # Default fallback

        # Harmonic mean for bandwidth
        return len(bw_values) / sum(1.0 / bw for bw in bw_values)

    def _resolve_2d_sp_config(self, sp_degree: int) -> Tuple[int, int]:
        """Resolve 2D SP configuration.

        Args:
            sp_degree: Total SP degree

        Returns:
            Tuple of (ulysses_degree, ring_degree)
        """
        ulysses_degree = self.strategy.ulysses_degree or 1
        ring_degree = self.strategy.ring_degree or 1

        if ulysses_degree * ring_degree != sp_degree:
            # Auto-partition: prefer larger ulysses if possible
            ulysses_degree = min(sp_degree, 8)
            ring_degree = sp_degree // ulysses_degree
            while ulysses_degree * ring_degree != sp_degree and ring_degree > 1:
                ulysses_degree -= 1
                ring_degree = sp_degree // ulysses_degree

        return ulysses_degree, ring_degree

    # ==================== Effective Dimension Helpers ====================

    def _get_effective_seq_len(self, seq_len: int) -> int:
        """获取考虑 SP 后的有效 seq_len"""
        return seq_len // max(self.strategy.sp_degree, 1)

    def _get_effective_num_heads(self, num_heads: int) -> int:
        """获取考虑 TP 后的有效 attention heads"""
        return num_heads // max(self.strategy.tp_degree, 1)

    def _get_effective_intermediate_size(self, intermediate_size: int) -> int:
        """获取考虑 TP 后的有效 FFN intermediate size"""
        return intermediate_size // max(self.strategy.tp_degree, 1)

    def _get_effective_num_layers(self) -> int:
        """获取考虑 PP 后的有效层数（每 stage）"""
        return self.model.config.num_layers // max(self.strategy.pp_degree, 1)

    def _get_effective_num_experts(self, num_experts: int) -> int:
        """获取考虑 EP 后的有效 expert 数（每 GPU）"""
        return num_experts // max(self.strategy.ep_degree, 1)

    def _get_total_gpus(self) -> int:
        """获取总 GPU 数 = tp × pp × dp × ep × sp"""
        return self.strategy.world_size

    # ==================== Memory Estimation Helpers ====================

    def _get_base_param_memory(self) -> int:
        """Get base parameter memory (before sharding).

        Returns:
            Parameter memory in bytes
        """
        dtype_size = DTYPE_SIZES.get(self.model.config.dtype, 2)
        return self.model.total_params * dtype_size

    def _get_sharded_param_memory(self) -> int:
        """Get parameter memory after TP sharding.

        Returns:
            Sharded parameter memory in bytes
        """
        param_memory = self._get_base_param_memory()
        param_memory //= self.strategy.tp_degree
        return param_memory

    def _is_distributed(self) -> bool:
        """Check if running in distributed mode.

        Returns:
            True if any parallelism degree > 1
        """
        return self.strategy.tp_degree > 1 or self.strategy.dp_degree > 1 or self.strategy.pp_degree > 1

    def _apply_memory_calibration(self, memory_bytes: int) -> int:
        """Apply memory calibration factors.

        Args:
            memory_bytes: Base memory estimate

        Returns:
            Calibrated memory estimate
        """
        calib = self.model.config.memory_calibration
        return calib.apply(memory_bytes, self._is_distributed())

    # ==================== Kernel Helpers ====================

    def _get_matmul_kernel(self, m: int, n: int, k: int, dtype: str):
        """Get matmul kernel for given dimensions.

        Args:
            m: First dimension
            n: Second dimension
            k: Third dimension
            dtype: Data type

        Returns:
            ComputeKernel or None
        """
        return self.compute_registry.get_or_create_matmul(m, n, k, dtype)

    def _get_attention_kernel(self, batch: int, seq_len: int, heads: int, head_dim: int, dtype: str):
        """Get attention kernel for given dimensions.

        Args:
            batch: Batch size
            seq_len: Sequence length
            heads: Number of heads
            head_dim: Head dimension
            dtype: Data type

        Returns:
            ComputeKernel or None
        """
        name = f"flash_attn_{batch}_{seq_len}_{heads}_{head_dim}_{dtype}"
        return self.compute_registry.get(name)

    def _get_norm_kernel(self, hidden_size: int, dtype: str, norm_type: str = "rmsnorm"):
        """Get normalization kernel.

        Args:
            hidden_size: Hidden dimension
            dtype: Data type
            norm_type: Normalization type

        Returns:
            ComputeKernel or None
        """
        name = f"{norm_type}_{hidden_size}_{dtype}"
        return self.compute_registry.get(name)

    def _get_activation_kernel(self, num_elements: int, activation: str, dtype: str):
        """Get activation function kernel.

        Args:
            num_elements: Number of elements
            activation: Activation name
            dtype: Data type

        Returns:
            ComputeKernel or None
        """
        return self.compute_registry.get_or_create_activation(num_elements, activation, dtype)

    def _apply_scheduler_features(
        self,
        compute_time: float,
        comm_time: float,
        memory_bytes: int,
    ):
        """Apply scheduler features to performance estimates.

        Args:
            compute_time: Compute time in seconds
            comm_time: Communication time in seconds
            memory_bytes: Memory usage in bytes

        Returns:
            SchedulerResult with optimized values
        """
        from ..scheduler.base import SchedulerResult

        result = SchedulerResult(
            compute_time=compute_time,
            comm_time=comm_time,
            memory_bytes=memory_bytes,
        )
        return self.scheduler_model.apply_all(result)

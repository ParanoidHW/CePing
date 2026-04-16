"""Base analyzer with shared estimation logic."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from ..models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig, SPType
from ..kernels.compute import ComputeKernelRegistry
from ..kernels.communication import CommKernelRegistry
from ..utils.constants import DTYPE_SIZES


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

        # P2P communication for activation transfer
        pp_time = activation_bytes / (self.cluster.network.intra_node_bandwidth_gbps * 1e9)
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
        sp_ranks = list(range(sp_degree))

        # Activation bytes for all-to-all (Ulysses)
        activation_bytes = batch_size * seq_len * hidden_size * dtype_size

        # KV bytes per step for Ring
        kv_bytes_per_step = (
            batch_size * (seq_len // sp_degree) *
            num_kv_heads * head_dim * 2 * dtype_size
        )

        total_time = 0.0

        if sp_type == SPType.ULYSSES:
            # 4 all-to-all per attention layer
            alltoall_time = self.cluster.estimate_alltoall_time(
                activation_bytes, sp_ranks
            )
            total_time = alltoall_time * 4 * num_layers

        elif sp_type == SPType.RING_P2P:
            if sp_degree > 1:
                avg_bw = self._get_average_bandwidth(sp_ranks)
                step_time = kv_bytes_per_step / (avg_bw * 1e9)
                total_time = step_time * (sp_degree - 1) * num_layers

        elif sp_type == SPType.RING_ALLGATHER:
            allgather_time = self.cluster.estimate_allgather_time(
                kv_bytes_per_step * sp_degree, sp_ranks
            )
            total_time = allgather_time * num_layers

        elif sp_type == SPType.UNIFIED_2D:
            ulysses_degree, ring_degree = self._resolve_2d_sp_config(sp_degree)
            ulysses_ranks = list(range(ulysses_degree))
            ring_ranks = list(range(ring_degree))

            # Ulysses part
            ulysses_time = self.cluster.estimate_alltoall_time(
                activation_bytes, ulysses_ranks
            ) * 4 * num_layers

            # Ring part
            ring_time = 0.0
            if ring_degree > 1:
                ring_kv_bytes = (
                    batch_size * (seq_len // sp_degree) *
                    num_kv_heads * head_dim * 2 * dtype_size
                )
                avg_bw = self._get_average_bandwidth(ring_ranks)
                ring_step_time = ring_kv_bytes / (avg_bw * 1e9)
                ring_time = ring_step_time * (ring_degree - 1) * num_layers

            total_time = ulysses_time + ring_time

        elif sp_type == SPType.MEGATRON:
            # Megatron-SP: ReduceScatter + AllGather per layer
            # Communication volume: 2 * activation_bytes per layer (same as TP AllReduce)
            # But memory is sharded by sp_degree
            rs_time = self.cluster.estimate_reducescatter_time(
                activation_bytes, sp_ranks
            )
            ag_time = self.cluster.estimate_allgather_time(
                activation_bytes, sp_ranks
            )
            # 2 communication ops per layer (forward pass)
            total_time = (rs_time + ag_time) * 2 * num_layers

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
        return (
            self.strategy.tp_degree > 1
            or self.strategy.dp_degree > 1
            or self.strategy.pp_degree > 1
        )

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
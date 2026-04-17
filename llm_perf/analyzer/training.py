"""Training performance analyzer."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from llm_perf.modeling import ShardedModule
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from llm_perf.kernels.compute import ComputeKernelRegistry
from llm_perf.kernels.communication import CommKernelRegistry
from llm_perf.utils.constants import DTYPE_SIZES

from .base import BaseAnalyzer, BaseResult, PerformanceBreakdown


@dataclass
class TrainingResult(BaseResult):
    """Result of training performance analysis."""

    samples_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    time_per_step_sec: float = 0.0
    memory_per_gpu_gb: float = 0.0
    breakdown: PerformanceBreakdown = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "throughput": {
                "samples_per_sec": self.samples_per_sec,
                "tokens_per_sec": self.tokens_per_sec,
            },
            "time": {
                "time_per_step_sec": self.time_per_step_sec,
                "time_per_step_ms": self.time_per_step_sec * 1000,
            },
            "memory": {
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
            },
            "breakdown": self.breakdown.to_dict() if self.breakdown else None,
        }


class TrainingAnalyzer(BaseAnalyzer):
    """Analyzes training performance for ShardedModule models."""

    def __init__(
        self,
        model: ShardedModule,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        super().__init__(model, device, cluster, strategy)
        self.compute_registry = ComputeKernelRegistry(device)
        self.comm_registry = CommKernelRegistry(cluster)

    def analyze(
        self,
        batch_size: int,
        seq_len: int,
    ) -> TrainingResult:
        """Analyze training performance.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            TrainingResult with performance metrics
        """
        model_info = self._get_model_info()
        parallel_degrees = self._get_parallel_degrees()
        dtype = self.model.dtype if hasattr(self.model, "dtype") else "fp16"
        dtype_bytes = DTYPE_SIZES.get(dtype, 2)

        hidden_size = self.model.hidden_size if hasattr(self.model, "hidden_size") else 4096
        num_layers = self.model.num_layers if hasattr(self.model, "num_layers") else 32

        effective_hidden = hidden_size // parallel_degrees["tp"]
        effective_batch = batch_size // parallel_degrees["dp"]

        compute_time = self._estimate_compute_time(
            batch_size=effective_batch,
            seq_len=seq_len,
            hidden_size=effective_hidden,
            num_layers=num_layers,
            dtype=dtype,
        )

        comm_time = self._estimate_communication_time(
            batch_size=effective_batch,
            seq_len=seq_len,
            hidden_size=hidden_size,
            parallel_degrees=parallel_degrees,
        )

        total_time = compute_time + comm_time
        memory_gb = self._estimate_memory(
            batch_size=effective_batch,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dtype_bytes=dtype_bytes,
            parallel_degrees=parallel_degrees,
        )

        samples_per_sec = batch_size / total_time if total_time > 0 else 0
        tokens_per_sec = batch_size * seq_len / total_time if total_time > 0 else 0

        breakdown = PerformanceBreakdown(
            compute_time_sec=compute_time,
            communication_time_sec=comm_time,
            total_time_sec=total_time,
        )

        return TrainingResult(
            samples_per_sec=samples_per_sec,
            tokens_per_sec=tokens_per_sec,
            time_per_step_sec=total_time,
            memory_per_gpu_gb=memory_gb,
            breakdown=breakdown,
        )

    def _estimate_compute_time(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dtype: str,
    ) -> float:
        """Estimate compute time."""
        total_time = 0.0

        for layer_idx in range(num_layers):
            m = batch_size * seq_len
            k = hidden_size
            n = hidden_size * 4

            matmul_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
            layer_time = matmul_kernel.estimate_time((m, k), (m, n), dtype)
            layer_time *= 6

            m_attn = batch_size * seq_len
            n_attn = hidden_size * 3
            attn_kernel = self.compute_registry.get_or_create_matmul(m_attn, n_attn, hidden_size, dtype)
            layer_time += attn_kernel.estimate_time((m_attn, hidden_size), (m_attn, n_attn), dtype) * 2

            m_flash = batch_size * seq_len
            flash_kernel = self.compute_registry.get_or_create_matmul(m_flash, hidden_size, hidden_size, dtype)
            if flash_kernel:
                layer_time += flash_kernel.estimate_time(
                    (m_flash, hidden_size),
                    (m_flash, hidden_size),
                    dtype,
                )

            total_time += layer_time

        return total_time

    def _estimate_communication_time(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        parallel_degrees: Dict[str, int],
    ) -> float:
        """Estimate communication time."""
        total_time = 0.0
        tp = parallel_degrees["tp"]

        if tp > 1:
            data_size = batch_size * seq_len * hidden_size * 2
            allreduce_kernel = self.comm_registry.create_allreduce("tp_allreduce", data_size, list(range(tp)))
            total_time += allreduce_kernel.estimate_time() * 32

        return total_time

    def _estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dtype_bytes: float,
        parallel_degrees: Dict[str, int],
    ) -> float:
        """Estimate memory per GPU in GB."""
        tp = parallel_degrees["tp"]
        dp = parallel_degrees["dp"]

        params_per_gpu = self._count_params() // tp // dp
        params_memory = params_per_gpu * dtype_bytes

        activations_per_layer = batch_size * seq_len * hidden_size * dtype_bytes
        activations_total = activations_per_layer * num_layers * 2 // tp

        gradients = params_per_gpu * dtype_bytes

        total_bytes = params_memory + activations_total + gradients
        total_gb = total_bytes / 1e9

        return total_gb

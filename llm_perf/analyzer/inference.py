"""Inference performance analyzer."""

from dataclasses import dataclass
from typing import Dict, Any

from llm_perf.modeling import ShardedModule
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from llm_perf.kernels.compute import ComputeKernelRegistry
from llm_perf.kernels.communication import CommKernelRegistry
from llm_perf.utils.constants import DTYPE_SIZES

from .base import BaseAnalyzer, BaseResult, PerformanceBreakdown


@dataclass
class InferenceResult(BaseResult):
    """Result of inference performance analysis."""

    prefill_time_sec: float = 0.0
    decode_time_per_step_sec: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    memory_per_gpu_gb: float = 0.0
    breakdown: PerformanceBreakdown = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prefill": {
                "time_sec": self.prefill_time_sec,
                "time_ms": self.prefill_time_sec * 1000,
                "tokens_per_sec": self.prefill_tokens_per_sec,
            },
            "decode": {
                "time_per_step_sec": self.decode_time_per_step_sec,
                "time_per_step_ms": self.decode_time_per_step_sec * 1000,
                "tokens_per_sec": self.decode_tokens_per_sec,
            },
            "memory": {
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
            },
            "breakdown": self.breakdown.to_dict() if self.breakdown else None,
        }


class InferenceAnalyzer(BaseAnalyzer):
    """Analyzes inference performance for ShardedModule models."""

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
        prompt_len: int,
        generation_len: int,
    ) -> InferenceResult:
        """Analyze inference performance.

        Args:
            batch_size: Batch size
            prompt_len: Prompt length
            generation_len: Generation length

        Returns:
            InferenceResult with performance metrics
        """
        parallel_degrees = self._get_parallel_degrees()
        dtype = self.model.dtype if hasattr(self.model, "dtype") else "fp16"

        hidden_size = self.model.hidden_size if hasattr(self.model, "hidden_size") else 4096
        num_layers = self.model.num_layers if hasattr(self.model, "num_layers") else 32

        effective_batch = batch_size // parallel_degrees["dp"]
        effective_hidden = hidden_size // parallel_degrees["tp"]

        prefill_time = self._estimate_prefill_time(
            batch_size=effective_batch,
            prompt_len=prompt_len,
            hidden_size=effective_hidden,
            num_layers=num_layers,
            dtype=dtype,
            parallel_degrees=parallel_degrees,
        )

        decode_time = self._estimate_decode_time(
            batch_size=effective_batch,
            hidden_size=effective_hidden,
            num_layers=num_layers,
            dtype=dtype,
            parallel_degrees=parallel_degrees,
        )

        prefill_tokens = batch_size * prompt_len
        decode_tokens_per_step = batch_size

        prefill_tps = prefill_tokens / prefill_time if prefill_time > 0 else 0
        decode_tps = decode_tokens_per_step / decode_time if decode_time > 0 else 0

        memory_gb = self._estimate_memory(
            batch_size=effective_batch,
            seq_len=prompt_len + generation_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dtype_bytes=DTYPE_SIZES.get(dtype, 2),
            parallel_degrees=parallel_degrees,
            is_inference=True,
        )

        breakdown = PerformanceBreakdown(
            compute_time_sec=prefill_time + decode_time * generation_len,
            total_time_sec=prefill_time + decode_time * generation_len,
        )

        return InferenceResult(
            prefill_time_sec=prefill_time,
            decode_time_per_step_sec=decode_time,
            prefill_tokens_per_sec=prefill_tps,
            decode_tokens_per_sec=decode_tps,
            memory_per_gpu_gb=memory_gb,
            breakdown=breakdown,
        )

    def _estimate_prefill_time(
        self,
        batch_size: int,
        prompt_len: int,
        hidden_size: int,
        num_layers: int,
        dtype: str,
        parallel_degrees: Dict[str, int],
    ) -> float:
        """Estimate prefill phase time."""
        compute_time = self._estimate_layer_time(batch_size, prompt_len, hidden_size, dtype) * num_layers

        comm_time = self._estimate_comm_time(batch_size, prompt_len, hidden_size, parallel_degrees) * num_layers

        return compute_time + comm_time

    def _estimate_decode_time(
        self,
        batch_size: int,
        hidden_size: int,
        num_layers: int,
        dtype: str,
        parallel_degrees: Dict[str, int],
    ) -> float:
        """Estimate decode phase time per step."""
        seq_len = 1
        compute_time = self._estimate_layer_time(batch_size, seq_len, hidden_size, dtype) * num_layers

        comm_time = self._estimate_comm_time(batch_size, seq_len, hidden_size, parallel_degrees) * num_layers

        return compute_time + comm_time

    def _estimate_layer_time(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype: str,
    ) -> float:
        """Estimate single layer time."""
        m = batch_size * seq_len
        k = hidden_size

        matmul_kernel = self.compute_registry.get_or_create_matmul(m, k * 4, k, dtype)
        time = matmul_kernel.estimate_time((m, k), (m, k * 4), dtype) * 3

        attn_kernel = self.compute_registry.get_or_create_matmul(m, k * 3, k, dtype)
        time += attn_kernel.estimate_time((m, k), (m, k * 3), dtype) * 2

        return time

    def _estimate_comm_time(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        parallel_degrees: Dict[str, int],
    ) -> float:
        """Estimate communication time."""
        tp = parallel_degrees["tp"]
        if tp <= 1:
            return 0.0

        data_size = batch_size * seq_len * hidden_size * 2
        kernel = self.comm_registry.create_allreduce("tp_allreduce", data_size, list(range(tp)))
        return kernel.estimate_time()

    def _estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dtype_bytes: float,
        parallel_degrees: Dict[str, int],
        is_inference: bool = False,
    ) -> float:
        """Estimate memory per GPU in GB."""
        tp = parallel_degrees["tp"]

        params_per_gpu = self._count_params() // tp
        params_memory = params_per_gpu * dtype_bytes

        kv_cache = batch_size * seq_len * hidden_size * dtype_bytes * num_layers * 2 // tp

        total_bytes = params_memory + kv_cache
        return total_bytes / 1e9

"""Inference performance analyzer."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from .models.base import BaseModel
from .models.sharding import ShardedLayerConfig
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig
from ..kernels.compute import ComputeKernelRegistry
from ..kernels.communication import CommKernelRegistry
from ..kernels.backend.registry import KernelBackendRegistry
from ..utils.constants import DTYPE_SIZES, PHASE_PREFILL, PHASE_DECODE
from .breakdown import PerformanceBreakdown, LayerBreakdown, KernelBreakdown
from .detailed_breakdown import DetailedPerformanceResult
from .breakdown_generator import BreakdownGenerator
from .result_base import BaseResult
from .memory import MemoryEstimator
from .communication import CommunicationEstimator


@dataclass
class InferenceResult(BaseResult):
    """Result of inference performance analysis."""

    prefill_time_sec: float
    decode_time_per_step_sec: float
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    total_time_sec: float
    total_tokens: int
    memory_per_gpu_gb: float
    kv_cache_memory_gb: float
    prefill_tokens_per_sec_per_gpu: float = 0.0
    decode_tokens_per_sec_per_gpu: float = 0.0
    total_gpus: int = 1
    prefill_breakdown: PerformanceBreakdown = None
    decode_breakdown: PerformanceBreakdown = None
    detailed_breakdown: Optional[DetailedPerformanceResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "prefill": {
                "ttft_sec": self.prefill_time_sec,
                "ttft_ms": self.prefill_time_sec * 1000,
                "tokens_per_sec": self.prefill_tokens_per_sec,
                "tokens_per_sec_per_gpu": self.prefill_tokens_per_sec_per_gpu,
            },
            "decode": {
                "tpot_sec": self.decode_time_per_step_sec,
                "tpot_ms": self.decode_time_per_step_sec * 1000,
                "tps": self.decode_tokens_per_sec,
                "tps_per_gpu": self.decode_tokens_per_sec_per_gpu,
            },
            "end_to_end": {
                "total_time_sec": self.total_time_sec,
                "total_tokens": self.total_tokens,
                "overall_tps": self.total_tokens / self.total_time_sec,
            },
            "memory": {
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
                "kv_cache_gb": self.kv_cache_memory_gb,
            },
            "parallelism": {
                "total_gpus": self.total_gpus,
            },
            "prefill_breakdown": self.prefill_breakdown.to_dict() if self.prefill_breakdown else None,
            "decode_breakdown": self.decode_breakdown.to_dict() if self.decode_breakdown else None,
        }
        if self.detailed_breakdown is not None:
            result["detailed_breakdown"] = self.detailed_breakdown.to_dict()
        return result


class InferenceAnalyzer:
    """Analyzes inference performance for prefill and decode phases."""

    def __init__(
        self,
        model: BaseModel,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
        use_sharded_layers: bool = True,
    ):
        self.model = model
        self.device = device
        self.cluster = cluster
        self.strategy = strategy
        self.use_sharded_layers = use_sharded_layers

        self.compute_registry = ComputeKernelRegistry(device)
        self.comm_registry = CommKernelRegistry(cluster)

        self.memory_estimator = MemoryEstimator(model, device, cluster, strategy)
        self.comm_estimator = CommunicationEstimator(model, device, cluster, strategy)

        self._sharded_layers: Optional[List[ShardedLayerConfig]] = None
        self._backend_registry = KernelBackendRegistry()
        self._backend = self._backend_registry.get_backend("theory")

    def _get_sharded_layers(self) -> List[ShardedLayerConfig]:
        """Get cached sharded layers."""
        if self._sharded_layers is None and self.use_sharded_layers:
            try:
                self._sharded_layers = self.model.build_sharded_layers(self.strategy)
            except Exception:
                self.use_sharded_layers = False
        return self._sharded_layers or []

    def analyze(
        self,
        batch_size: int,
        prompt_len: int,
        generation_len: int,
    ) -> InferenceResult:
        """Analyze inference performance."""
        prefill_time = self._estimate_prefill_time(batch_size, prompt_len)
        decode_time_per_step = self._estimate_decode_time(batch_size)

        memory_bytes, kv_cache_bytes = self.memory_estimator.estimate_inference_memory(
            batch_size, prompt_len + generation_len
        )

        prefill_tokens = batch_size * prompt_len
        decode_tokens = batch_size * generation_len

        prefill_tps = prefill_tokens / prefill_time if prefill_time > 0 else 0
        decode_tps = batch_size / decode_time_per_step if decode_time_per_step > 0 else 0

        total_gpus = self._get_total_gpus()
        prefill_tps_per_gpu = prefill_tps / total_gpus if total_gpus > 0 else prefill_tps
        decode_tps_per_gpu = decode_tps / total_gpus if total_gpus > 0 else decode_tps

        total_time = prefill_time + decode_time_per_step * generation_len

        prefill_breakdown = self._build_prefill_breakdown(batch_size, prompt_len, prefill_time)
        decode_breakdown = self._build_decode_breakdown(batch_size, decode_time_per_step)

        detailed_breakdown = self._generate_detailed_breakdown(
            prefill_time, decode_time_per_step, generation_len, memory_bytes
        )

        return InferenceResult(
            prefill_time_sec=prefill_time,
            decode_time_per_step_sec=decode_time_per_step,
            prefill_tokens_per_sec=prefill_tps,
            decode_tokens_per_sec=decode_tps,
            total_time_sec=total_time,
            total_tokens=prefill_tokens + decode_tokens,
            memory_per_gpu_gb=memory_bytes / 1024 / 1024 / 1024,
            kv_cache_memory_gb=kv_cache_bytes / 1024 / 1024 / 1024,
            prefill_breakdown=prefill_breakdown,
            decode_breakdown=decode_breakdown,
            detailed_breakdown=detailed_breakdown,
            prefill_tokens_per_sec_per_gpu=prefill_tps_per_gpu,
            decode_tokens_per_sec_per_gpu=decode_tps_per_gpu,
            total_gpus=total_gpus,
        )

    def _generate_detailed_breakdown(
        self, prefill_time: float, decode_time_per_step: float, generation_len: int, memory_bytes: int
    ) -> DetailedPerformanceResult:
        """Generate detailed performance breakdown for inference."""
        generator = BreakdownGenerator(self.model, self.device, self.cluster, self.strategy, is_training=False)

        submodel = generator.generate_submodel_breakdown(
            model_name=self.model.config.name or "model",
            model_type=getattr(self.model.config, "model_type", None) or "transformer",
            compute_time_sec=prefill_time + decode_time_per_step * generation_len,
            num_iterations=1,
        )

        from .detailed_breakdown import MemoryBreakdown, CommunicationBreakdown

        by_block_type: Dict[str, Dict[Any, int]] = {}
        for block in submodel.blocks:
            if block.block_type not in by_block_type:
                by_block_type[block.block_type] = {}
            for mem_type, bytes_val in block.memory_by_type.items():
                if mem_type not in by_block_type[block.block_type]:
                    by_block_type[block.block_type][mem_type] = 0
                by_block_type[block.block_type][mem_type] += bytes_val

        memory_breakdown = MemoryBreakdown(
            by_type=submodel.memory_by_type,
            by_submodel={submodel.model_name: submodel.memory_by_type},
            by_block_type=by_block_type,
        )

        comm_breakdown = CommunicationBreakdown(
            by_type=submodel.comm_by_parallelism,
            by_submodel={submodel.model_name: [op for ops in submodel.comm_by_parallelism.values() for op in ops]},
        )

        total_time = prefill_time + decode_time_per_step * generation_len
        return DetailedPerformanceResult(
            total_time_sec=total_time,
            throughput=generation_len / total_time if total_time > 0 else 0,
            submodels=[submodel],
            memory=memory_breakdown,
            communication=comm_breakdown,
        )

    def _estimate_prefill_time(self, batch_size: int, seq_len: int) -> float:
        """Estimate prefill phase time (prompt processing)."""
        dtype = self.model.config.dtype
        total_time = 0.0

        effective_seq_len = self._get_effective_seq_len(seq_len)
        effective_num_layers = self._get_effective_num_layers()

        for layer in self.model.layers:
            kernel = self._get_compute_kernel_for_phase(layer, batch_size, effective_seq_len, dtype, PHASE_PREFILL)
            if kernel:
                time = kernel.estimate_time(layer.input_shape, layer.output_shape, dtype)
                total_time += time

        original_num_layers = self.model.config.num_layers
        if original_num_layers > 0:
            total_time = total_time * effective_num_layers / original_num_layers

        comm_time, _ = self.comm_estimator.estimate_inference_communication(PHASE_PREFILL)

        overlap_factor = 0.8
        total_time = max(total_time, comm_time * (1 - overlap_factor) + max(total_time, comm_time * overlap_factor))

        return total_time

    def _estimate_decode_time(self, batch_size: int) -> float:
        """Estimate decode phase time (single token generation)."""
        dtype = self.model.config.dtype
        total_time = 0.0

        effective_num_layers = self._get_effective_num_layers()
        seq_len = 1

        for layer in self.model.layers:
            kernel = self._get_compute_kernel_for_phase(layer, batch_size, seq_len, dtype, PHASE_DECODE)
            if kernel:
                time = kernel.estimate_time(layer.input_shape, layer.output_shape, dtype)
                total_time += time

        original_num_layers = self.model.config.num_layers
        if original_num_layers > 0:
            total_time = total_time * effective_num_layers / original_num_layers

        kv_read_time = self._estimate_kv_cache_read_time(batch_size)
        total_time += kv_read_time

        comm_time, _ = self.comm_estimator.estimate_inference_communication(PHASE_DECODE)

        overlap_factor = 0.7
        total_time = max(total_time, comm_time * (1 - overlap_factor) + max(total_time, comm_time * overlap_factor))

        return total_time

    def _estimate_kv_cache_read_time(self, batch_size: int) -> float:
        """Estimate time to read KV cache during decode."""
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        _, effective_num_kv_heads = self._get_effective_num_heads()
        effective_num_layers = self._get_effective_num_layers()

        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // self.model.config.num_attention_heads

        kv_per_token = 2 * effective_num_kv_heads * head_dim * dtype_size

        avg_seq_len = self.model.config.max_seq_len // 2
        total_kv_bytes = batch_size * avg_seq_len * effective_num_layers * kv_per_token

        mem_bw = self.device.get_memory_bw_gbps() * 1e9
        return total_kv_bytes / mem_bw

    def _get_effective_seq_len(self, seq_len: int) -> int:
        """Get sequence length after SP/CP sharding."""
        seq_parallel_degree = max(self.strategy.sp_degree, self.strategy.cp_degree)
        if seq_parallel_degree > 1:
            return max(1, seq_len // seq_parallel_degree)
        return seq_len

    def _get_effective_num_heads(self) -> tuple:
        """Get attention heads after TP sharding."""
        tp_degree = self.strategy.tp_degree
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = self.model.config.num_key_value_heads or num_heads

        if tp_degree > 1:
            effective_num_heads = max(1, num_heads // tp_degree)
            effective_num_kv_heads = max(1, num_kv_heads // tp_degree)
            return effective_num_heads, effective_num_kv_heads
        return num_heads, num_kv_heads

    def _get_effective_num_layers(self) -> int:
        """Get number of layers for current PP stage."""
        pp_degree = self.strategy.pp_degree
        num_layers = self.model.config.num_layers

        if pp_degree > 1:
            return max(1, num_layers // pp_degree)
        return num_layers

    def _get_effective_intermediate_size(self) -> int:
        """Get FFN intermediate size after TP sharding."""
        tp_degree = self.strategy.tp_degree
        intermediate_size = self.model.config.intermediate_size

        if intermediate_size > 0 and tp_degree > 1:
            return max(1, intermediate_size // tp_degree)
        return intermediate_size

    def _get_total_gpus(self) -> int:
        """Get total number of GPUs used."""
        return self.strategy.world_size

    def _get_compute_kernel_for_phase(self, layer, batch_size: int, seq_len: int, dtype: str, phase: str):
        """Get appropriate compute kernel for a layer and phase."""
        name = layer.name
        hidden_size = self.model.config.hidden_size

        effective_num_heads, _ = self._get_effective_num_heads()
        head_dim = hidden_size // self.model.config.num_attention_heads
        effective_intermediate_size = self._get_effective_intermediate_size()

        if "proj" in name or "up" in name or "gate" in name or "down" in name:
            m = batch_size * seq_len
            k = layer.input_shape[-1] if layer.input_shape else hidden_size
            n = layer.output_shape[-1] if layer.output_shape else effective_intermediate_size
            if effective_intermediate_size > 0:
                n = effective_intermediate_size
            return self.compute_registry.get_or_create_matmul(m, n, k, dtype)
        elif "attention" in name:
            effective_head_dim = head_dim
            if phase == PHASE_PREFILL:
                return self.compute_registry.get(
                    f"flash_attn_{batch_size}_{seq_len}_{effective_num_heads}_{effective_head_dim}_{dtype}"
                )
            else:
                return self.compute_registry.get(
                    f"flash_attn_{batch_size}_1_{effective_num_heads}_{effective_head_dim}_{dtype}"
                )
        elif "norm" in name:
            return self.compute_registry.get(f"rmsnorm_{layer.input_shape[-1]}_{dtype}")
        elif "swiglu" in name or "activation" in name:
            return self.compute_registry.get(f"swiglu_{layer.input_shape[-1]}_{dtype}")

        return None

    def _build_prefill_breakdown(self, batch_size: int, seq_len: int, total_time: float) -> PerformanceBreakdown:
        """Build detailed breakdown for prefill phase."""
        return self._build_phase_breakdown(batch_size, seq_len, total_time, PHASE_PREFILL)

    def _build_decode_breakdown(self, batch_size: int, total_time: float) -> PerformanceBreakdown:
        """Build detailed breakdown for decode phase."""
        return self._build_phase_breakdown(batch_size, 1, total_time, PHASE_DECODE)

    def _build_phase_breakdown(
        self, batch_size: int, seq_len: int, total_time: float, phase: str
    ) -> PerformanceBreakdown:
        """Build detailed breakdown for a phase."""
        layer_breakdowns = []
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        compute_time = 0.0

        for layer in self.model.layers:
            kernels = []

            kernel = self._get_compute_kernel_for_phase(layer, batch_size, seq_len, dtype, phase)
            if kernel:
                ktime = kernel.estimate_time(layer.input_shape, layer.output_shape, dtype)
                kernels.append(
                    KernelBreakdown(
                        name=layer.name + f"_{phase}",
                        kernel_type="compute",
                        time_sec=ktime,
                        memory_bytes=kernel.estimate_memory(layer.input_shape, layer.output_shape, dtype),
                        flops=layer.flops,
                    )
                )
                compute_time += ktime

            layer_breakdowns.append(LayerBreakdown(name=layer.name, kernels=kernels))

        comm_time, comm_breakdown = self.comm_estimator.estimate_inference_communication(phase)

        tokens = batch_size * seq_len
        throughput = tokens / total_time if total_time > 0 else 0

        return PerformanceBreakdown(
            total_time_sec=total_time,
            throughput=throughput,
            compute_time_sec=compute_time,
            communication_time_sec=comm_time,
            memory_time_sec=0.0,
            layers=layer_breakdowns,
            peak_memory_bytes=0,
            activation_memory_bytes=self.model.activation_memory * batch_size,
            parameter_memory_bytes=self.model.total_params * dtype_size,
            comm_breakdown=comm_breakdown,
        )

"""Training performance analyzer."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..models.base import BaseModel, SubmoduleType
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig
from ..kernels.compute import ComputeKernelRegistry
from ..kernels.communication import CommKernelRegistry
from ..utils.constants import DTYPE_SIZES
from .breakdown import PerformanceBreakdown, LayerBreakdown, KernelBreakdown
from .detailed_breakdown import DetailedPerformanceResult
from .breakdown_generator import BreakdownGenerator
from .result_base import BaseResult
from .memory import MemoryEstimator
from .communication import CommunicationEstimator


@dataclass
class TrainingResult(BaseResult):
    """Result of training performance analysis."""

    samples_per_sec: float
    tokens_per_sec: float
    time_per_step_sec: float
    time_to_solution_sec: float
    memory_per_gpu_gb: float
    samples_per_sec_per_gpu: float = 0.0
    tokens_per_sec_per_gpu: float = 0.0
    breakdown: PerformanceBreakdown = None
    detailed_breakdown: Optional[DetailedPerformanceResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "throughput": {
                "samples_per_sec": self.samples_per_sec,
                "tokens_per_sec": self.tokens_per_sec,
                "samples_per_sec_per_gpu": self.samples_per_sec_per_gpu,
                "tokens_per_sec_per_gpu": self.tokens_per_sec_per_gpu,
            },
            "time": {
                "time_per_step_sec": self.time_per_step_sec,
                "time_per_step_ms": self.time_per_step_sec * 1000,
                "time_to_solution_sec": self.time_to_solution_sec,
            },
            "memory": {
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
            },
            "breakdown": self.breakdown.to_dict() if self.breakdown else None,
        }
        if self.detailed_breakdown is not None:
            result["detailed_breakdown"] = self.detailed_breakdown.to_dict()
        return result


class TrainingAnalyzer:
    """Analyzes training performance."""

    def __init__(
        self,
        model: BaseModel,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        self.model = model
        self.device = device
        self.cluster = cluster
        self.strategy = strategy

        self.compute_registry = ComputeKernelRegistry(device)
        self.comm_registry = CommKernelRegistry(cluster)

        self.memory_estimator = MemoryEstimator(model, device, cluster, strategy)
        self.comm_estimator = CommunicationEstimator(model, device, cluster, strategy)

    def analyze(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int = 1000,
    ) -> TrainingResult:
        """Analyze training performance."""
        local_batch_size = batch_size // self.strategy.dp_degree

        compute_time = self._estimate_compute_time(local_batch_size, seq_len)

        comm_time, comm_breakdown = self.comm_estimator.estimate_training_communication(
            seq_len, self.strategy.micro_batch_size
        )

        memory_bytes = self.memory_estimator.estimate_training_memory(local_batch_size, seq_len)

        overlap_factor = 0.7
        effective_comm_time = comm_time * (1 - overlap_factor)
        time_per_step = compute_time + effective_comm_time

        samples_per_sec = batch_size / time_per_step
        tokens_per_sec = samples_per_sec * seq_len

        total_gpus = self._get_total_gpus()
        samples_per_sec_per_gpu = samples_per_sec / total_gpus
        tokens_per_sec_per_gpu = tokens_per_sec / total_gpus

        breakdown = self._build_breakdown(compute_time, comm_time, comm_breakdown, memory_bytes)

        detailed_breakdown = self._generate_detailed_breakdown(compute_time, comm_time, memory_bytes, batch_size)

        return TrainingResult(
            samples_per_sec=samples_per_sec,
            tokens_per_sec=tokens_per_sec,
            time_per_step_sec=time_per_step,
            time_to_solution_sec=time_per_step * num_steps,
            memory_per_gpu_gb=memory_bytes / 1024 / 1024 / 1024,
            breakdown=breakdown,
            detailed_breakdown=detailed_breakdown,
            samples_per_sec_per_gpu=samples_per_sec_per_gpu,
            tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
        )

    def _generate_detailed_breakdown(
        self, compute_time: float, comm_time: float, memory_bytes: int, batch_size: int = 1
    ) -> DetailedPerformanceResult:
        """Generate detailed performance breakdown."""
        generator = BreakdownGenerator(self.model, self.device, self.cluster, self.strategy)

        submodel = generator.generate_submodel_breakdown(
            model_name=self.model.config.name or "model",
            model_type=getattr(self.model.config, "model_type", None) or "transformer",
            compute_time_sec=compute_time,
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

        return DetailedPerformanceResult(
            total_time_sec=compute_time + comm_time,
            throughput=batch_size / (compute_time + comm_time) if (compute_time + comm_time) > 0 else 0,
            submodels=[submodel],
            memory=memory_breakdown,
            communication=comm_breakdown,
        )

    def _estimate_compute_time(self, batch_size: int, seq_len: int) -> float:
        """Estimate compute time for one training step."""
        dtype = self.model.config.dtype

        effective_seq_len = self._get_effective_seq_len(seq_len)
        effective_num_heads = self._get_effective_num_heads()
        effective_intermediate_size = self._get_effective_intermediate_size()
        effective_num_layers = self._get_effective_num_layers()
        effective_hidden_size = self._get_effective_hidden_size()

        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // self.model.config.num_attention_heads

        forward_time = 0.0
        backward_time = 0.0

        attention_layers = 0
        ffn_layers = 0

        for layer in self.model.layers[:effective_num_layers]:
            layer_forward_time = 0.0
            layer_backward_time = 0.0

            if layer.submodule_type == SubmoduleType.ATTENTION or "attention" in layer.name:
                attention_layers += 1

                m = batch_size * effective_seq_len
                n = 3 * effective_hidden_size
                k = hidden_size
                qkv_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
                layer_forward_time += qkv_kernel.estimate_time((m, k), (m, n), dtype)
                layer_backward_time += qkv_kernel.estimate_backward_time((m, k), (m, n), dtype)

                attn_kernel = self.compute_registry.get(
                    f"flash_attn_{batch_size}_{effective_seq_len}_{effective_num_heads}_{head_dim}_{dtype}"
                )
                if attn_kernel:
                    layer_forward_time += attn_kernel.estimate_time(
                        (batch_size, effective_seq_len, effective_num_heads * head_dim),
                        (batch_size, effective_seq_len, hidden_size),
                        dtype,
                    )
                    layer_backward_time += attn_kernel.estimate_backward_time(
                        (batch_size, effective_seq_len, effective_num_heads * head_dim),
                        (batch_size, effective_seq_len, hidden_size),
                        dtype,
                    )
                else:
                    attention_flops = (
                        2 * batch_size * effective_seq_len * effective_num_heads * effective_seq_len * head_dim
                    )
                    effective_tflops = self.device.get_compute_tflops(dtype) * 0.5
                    layer_forward_time += attention_flops / (effective_tflops * 1e12)
                    layer_backward_time += attention_flops * 2 / (effective_tflops * 1e12)

                m = batch_size * effective_seq_len
                n = hidden_size
                k = effective_hidden_size
                o_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
                layer_forward_time += o_kernel.estimate_time((m, k), (m, n), dtype)
                layer_backward_time += o_kernel.estimate_backward_time((m, k), (m, n), dtype)

            elif layer.submodule_type == SubmoduleType.FFN or "ffn" in layer.name or "proj" in layer.name:
                ffn_layers += 1

                m = batch_size * effective_seq_len
                n = effective_intermediate_size
                k = hidden_size
                up_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
                layer_forward_time += up_kernel.estimate_time((m, k), (m, n), dtype) * 2
                layer_backward_time += up_kernel.estimate_backward_time((m, k), (m, n), dtype) * 2

                activation_bytes = (
                    batch_size * effective_seq_len * effective_intermediate_size * DTYPE_SIZES.get(dtype, 2)
                )
                layer_forward_time += activation_bytes / (self.device.get_memory_bw_gbps() * 1e9)
                layer_backward_time += activation_bytes * 1.5 / (self.device.get_memory_bw_gbps() * 1e9)

                m = batch_size * effective_seq_len
                n = hidden_size
                k = effective_intermediate_size
                down_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
                layer_forward_time += down_kernel.estimate_time((m, k), (m, n), dtype)
                layer_backward_time += down_kernel.estimate_backward_time((m, k), (m, n), dtype)

            forward_time += layer_forward_time
            backward_time += layer_backward_time

        total_time = forward_time + backward_time

        if self.strategy.dp_degree > 1 or self.strategy.zero_stage > 0:
            effective_params = self.model.total_params // self.strategy.tp_degree
            if self.strategy.pp_degree > 1:
                effective_params = effective_params // self.strategy.pp_degree

            if self.strategy.zero_stage >= 1:
                effective_params = effective_params // self.strategy.dp_degree

            opt_ops = effective_params * 3
            opt_time = opt_ops / (self.device.get_memory_bw_gbps() * 1e9)
            total_time += opt_time

        return total_time

    def _get_effective_seq_len(self, seq_len: int) -> int:
        """Get effective sequence length after SP sharding."""
        if self.strategy.sp_degree > 1:
            return seq_len // self.strategy.sp_degree
        return seq_len

    def _get_effective_num_heads(self) -> int:
        """Get effective number of attention heads after TP sharding."""
        num_heads = self.model.config.num_attention_heads
        if self.strategy.tp_degree > 1:
            return num_heads // self.strategy.tp_degree
        return num_heads

    def _get_effective_num_kv_heads(self) -> int:
        """Get effective number of KV heads after TP sharding."""
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        if self.strategy.tp_degree > 1:
            return max(1, num_kv_heads // self.strategy.tp_degree)
        return num_kv_heads

    def _get_effective_intermediate_size(self) -> int:
        """Get effective intermediate size after TP sharding."""
        intermediate_size = self.model.config.intermediate_size
        if intermediate_size == 0:
            intermediate_size = self.model.config.hidden_size * 4
        if self.strategy.tp_degree > 1:
            return intermediate_size // self.strategy.tp_degree
        return intermediate_size

    def _get_effective_hidden_size(self) -> int:
        """Get effective hidden size after TP sharding."""
        hidden_size = self.model.config.hidden_size
        if self.strategy.tp_degree > 1:
            return hidden_size // self.strategy.tp_degree
        return hidden_size

    def _get_effective_num_layers(self) -> int:
        """Get effective number of layers after PP sharding."""
        num_layers = self.model.config.num_layers
        if self.strategy.pp_degree > 1:
            return num_layers // self.strategy.pp_degree
        return num_layers

    def _get_total_gpus(self) -> int:
        """Get total number of GPUs used in training."""
        return (
            self.strategy.tp_degree
            * self.strategy.pp_degree
            * self.strategy.dp_degree
            * self.strategy.sp_degree
            * self.strategy.ep_degree
        )

    def _get_compute_kernel(self, layer, batch_size: int, seq_len: int, dtype: str):
        """Get appropriate compute kernel for a layer."""
        name = layer.name

        if "proj" in name or "up" in name or "gate" in name or "down" in name:
            m = batch_size * seq_len
            n = layer.output_shape[-1]
            k = layer.input_shape[-1]
            return self.compute_registry.get_or_create_matmul(m, n, k, dtype)
        elif "attention" in name:
            return self.compute_registry.get(f"flash_attn_{batch_size}_{seq_len}_32_128_{dtype}")
        elif "norm" in name:
            return self.compute_registry.get(f"rmsnorm_{layer.input_shape[-1]}_{dtype}")
        elif "swiglu" in name or "activation" in name:
            return self.compute_registry.get(f"swiglu_{layer.input_shape[-1]}_{dtype}")

        return None

    def _build_breakdown(
        self,
        compute_time: float,
        comm_time: float,
        comm_breakdown: Dict[str, float],
        memory_bytes: int,
    ) -> PerformanceBreakdown:
        """Build detailed performance breakdown."""
        layer_breakdowns = []
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        for layer in self.model.layers:
            kernels = []

            kernel = self._get_compute_kernel(layer, 1, 4096, dtype)
            if kernel:
                kernels.append(
                    KernelBreakdown(
                        name=layer.name + "_compute",
                        kernel_type="compute",
                        time_sec=kernel.estimate_time(layer.input_shape, layer.output_shape, dtype),
                        memory_bytes=kernel.estimate_memory(layer.input_shape, layer.output_shape, dtype),
                        flops=layer.flops,
                    )
                )

            layer_breakdowns.append(LayerBreakdown(name=layer.name, kernels=kernels))

        return PerformanceBreakdown(
            total_time_sec=compute_time + comm_time,
            throughput=0.0,
            compute_time_sec=compute_time,
            communication_time_sec=comm_time,
            memory_time_sec=0.0,
            layers=layer_breakdowns,
            peak_memory_bytes=memory_bytes,
            activation_memory_bytes=self.model.activation_memory,
            parameter_memory_bytes=self.model.total_params * dtype_size,
            comm_breakdown=comm_breakdown,
        )

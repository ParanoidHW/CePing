"""Unified Analyzer implementation."""

from typing import Any, Dict, List, Union, Optional

from llm_perf.modeling import ShardedModule
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from llm_perf.kernels.compute import ComputeKernelRegistry
from llm_perf.kernels.communication import CommKernelRegistry
from llm_perf.utils.constants import DTYPE_SIZES

from .base import (
    Phase,
    PhaseResult,
    UnifiedResult,
    WorkloadConfig,
    WorkloadType,
    ComputeType,
    ThroughputMetric,
)
from .presets import get_workload, infer_workload


class UnifiedAnalyzer:
    """Unified performance analyzer for all workload types.

    Supports:
    - LLM training/inference
    - Diffusion training/inference
    - MoE training/inference
    - Mixed workloads (speculative decoding, RL PPO, etc.)
    - Custom workload configurations
    """

    def __init__(
        self,
        model: Union[ShardedModule, Dict[str, ShardedModule]],
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

    def analyze(
        self,
        workload: Union[str, WorkloadConfig],
        **kwargs,
    ) -> UnifiedResult:
        """Analyze performance for a workload.

        Args:
            workload: Workload preset name or custom WorkloadConfig
            **kwargs: Dynamic parameters (batch_size, seq_len, generation_len, etc.)

        Returns:
            UnifiedResult with performance metrics
        """
        if isinstance(workload, str):
            workload = get_workload(workload)

        resolved_params = {**workload.default_params, **kwargs}

        phase_results = self._analyze_phases(workload.phases, resolved_params)

        total_time = sum(p.total_time_sec for p in phase_results)
        peak_memory = max(p.memory_gb for p in phase_results) if phase_results else 0.0

        throughput = self._calculate_throughput(phase_results, workload, resolved_params)

        return UnifiedResult(
            workload_name=workload.name,
            workload_type=workload.workload_type,
            phases=phase_results,
            total_time_sec=total_time,
            peak_memory_gb=peak_memory,
            throughput=throughput,
            params=resolved_params,
            metadata={
                "device": self.device.config.name,
                "num_devices": self.cluster.num_devices,
                "strategy": self.strategy.to_dict(),
            },
        )

    def _analyze_phases(
        self,
        phases: List[Phase],
        params: Dict[str, Any],
    ) -> List[PhaseResult]:
        """Analyze all phases in a workload."""
        results = []

        for phase in phases:
            component = self._get_component(phase.component)

            repeat_count = self._resolve_param(phase.repeat, params)
            seq_len_factor = self._resolve_param(phase.seq_len_factor, params)

            single_time, memory, flops = self._estimate_phase(component, phase, params, seq_len_factor)

            total_time = single_time * repeat_count

            results.append(
                PhaseResult(
                    name=phase.name,
                    component=phase.component,
                    compute_type=phase.compute_type,
                    single_time_sec=single_time,
                    repeat_count=repeat_count,
                    total_time_sec=total_time,
                    memory_gb=memory,
                    flops=flops,
                )
            )

        return results

    def _get_component(self, component_name: str) -> ShardedModule:
        """Get model component by name."""
        if isinstance(self.model, dict):
            if component_name not in self.model:
                raise KeyError(f"Component '{component_name}' not found in model dict")
            return self.model[component_name]
        else:
            return self.model

    def _resolve_param(self, value: Union[int, str, float], params: Dict[str, Any]) -> Any:
        """Resolve dynamic parameter.

        Supports:
        - int/float: direct value
        - str: parameter name from params dict
        - str expression like "1/seq_len": evaluate expression
        """
        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            if "/" in value:
                parts = value.split("/")
                numerator = float(parts[0])
                denominator_param = parts[1].strip()
                denominator = params.get(denominator_param, 1)
                return numerator / denominator if denominator > 0 else 0.0
            else:
                return params.get(value, 1)

        return value

    def _estimate_phase(
        self,
        component: ShardedModule,
        phase: Phase,
        params: Dict[str, Any],
        seq_len_factor: float,
    ) -> tuple:
        """Estimate single phase execution.

        Returns:
            (time_sec, memory_gb, flops)
        """
        dtype = getattr(component, "dtype", "fp16")
        dtype_bytes = DTYPE_SIZES.get(dtype, 2)

        hidden_size = getattr(component, "hidden_size", 4096)
        num_layers = getattr(component, "num_layers", 32)

        batch_size = params.get("batch_size", 1)

        model_type = self._get_model_type(component)

        if model_type == "llm":
            seq_len = params.get("seq_len", params.get("prompt_len", 2048))
            effective_seq_len = int(seq_len * seq_len_factor)
            time, flops = self._estimate_llm_compute(
                component, batch_size, effective_seq_len, hidden_size, num_layers, dtype, phase.compute_type
            )
            memory = self._estimate_llm_memory(
                batch_size, effective_seq_len, hidden_size, num_layers, dtype_bytes, phase.compute_type
            )

        elif model_type == "dit":
            num_frames = params.get("num_frames", 81)
            height = params.get("height", 720)
            width = params.get("width", 1280)

            time, flops = self._estimate_dit_compute(
                component, batch_size, num_frames, height, width, dtype, phase.compute_type, params
            )
            memory = self._estimate_dit_memory(
                batch_size, num_frames, height, width, dtype_bytes, phase.compute_type, params
            )

        elif model_type == "vae":
            num_frames = params.get("num_frames", 81)
            height = params.get("height", 720)
            width = params.get("width", 1280)

            time, flops = self._estimate_vae_compute(
                component, batch_size, num_frames, height, width, dtype, phase.compute_type
            )
            memory = self._estimate_vae_memory(batch_size, num_frames, height, width, dtype_bytes)

        elif model_type == "text_encoder":
            seq_len = params.get("prompt_len", 512)
            time, flops = self._estimate_text_encoder_compute(component, batch_size, seq_len, dtype, phase.compute_type)
            memory = self._estimate_text_encoder_memory(batch_size, seq_len, dtype_bytes)

        elif model_type == "resnet":
            image_size = params.get("image_size", 224)
            time, flops = self._estimate_resnet_compute(component, batch_size, image_size, dtype, phase.compute_type)
            memory = self._estimate_resnet_memory(batch_size, image_size, dtype_bytes, phase.compute_type)

        else:
            seq_len = params.get("seq_len", params.get("prompt_len", 2048))
            time, flops = self._estimate_llm_compute(
                component, batch_size, seq_len, hidden_size, num_layers, dtype, phase.compute_type
            )
            memory = self._estimate_llm_memory(
                batch_size, seq_len, hidden_size, num_layers, dtype_bytes, phase.compute_type
            )

        return time, memory, flops

    def _get_model_type(self, component: ShardedModule) -> str:
        """Get model type from component."""
        name = getattr(component, "_name", "")
        class_name = type(component).__name__.lower()

        if "dit" in class_name or "dit" in name:
            return "dit"
        elif "vae" in class_name or "vae" in name:
            return "vae"
        elif "textencoder" in class_name or "text_encoder" in name or "t5" in name:
            return "text_encoder"
        elif "resnet" in class_name or "resnet" in name:
            return "resnet"
        elif "llama" in class_name or "deepseek" in class_name or "moe" in class_name:
            return "llm"
        else:
            return "llm"

    def _estimate_llm_compute(
        self,
        component: ShardedModule,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dtype: str,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate LLM compute time and FLOPs."""
        parallel_degrees = self._get_parallel_degrees()
        tp = parallel_degrees["tp"]
        dp = parallel_degrees["dp"]

        effective_batch = batch_size // dp
        effective_hidden = hidden_size // tp

        forward_time = self._estimate_layer_time(effective_batch, seq_len, effective_hidden, dtype) * num_layers

        forward_flops = self._estimate_llm_flops(batch_size, seq_len, hidden_size, num_layers)

        if compute_type == ComputeType.FORWARD:
            comm_time = self._estimate_comm_time(batch_size, seq_len, hidden_size, tp, num_layers)
            return forward_time + comm_time, forward_flops

        elif compute_type == ComputeType.BACKWARD:
            backward_time = forward_time * 2.0
            backward_flops = forward_flops * 2.0
            comm_time = self._estimate_comm_time(batch_size, seq_len, hidden_size, tp, num_layers)
            return backward_time + comm_time, backward_flops

        elif compute_type == ComputeType.OPTIMIZER:
            optimizer_time = forward_time * 1.5
            return optimizer_time, 0.0

        return forward_time, forward_flops

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

    def _estimate_llm_flops(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
    ) -> float:
        """Estimate LLM FLOPs."""
        m = batch_size * seq_len

        ffn_flops = m * hidden_size * hidden_size * 4 * 2
        attn_flops = m * hidden_size * hidden_size * 3 * 2
        flash_flops = m * hidden_size * hidden_size * 2

        layer_flops = ffn_flops + attn_flops + flash_flops
        return layer_flops * num_layers

    def _estimate_comm_time(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        tp: int,
        num_layers: int,
    ) -> float:
        """Estimate communication time."""
        if tp <= 1:
            return 0.0

        data_size = batch_size * seq_len * hidden_size * 2
        kernel = self.comm_registry.create_allreduce("tp_allreduce", data_size, list(range(tp)))
        return kernel.estimate_time() * num_layers

    def _estimate_llm_memory(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dtype_bytes: float,
        compute_type: ComputeType,
    ) -> float:
        """Estimate LLM memory."""
        parallel_degrees = self._get_parallel_degrees()
        tp = parallel_degrees["tp"]
        dp = parallel_degrees["dp"]

        component = self.model if isinstance(self.model, ShardedModule) else self.model.get("main")
        params_per_gpu = (
            self._count_params(component) // tp // dp if component else hidden_size * num_layers * 12 // tp // dp
        )
        params_memory = params_per_gpu * dtype_bytes

        if compute_type == ComputeType.FORWARD:
            activations = batch_size * seq_len * hidden_size * dtype_bytes * num_layers * 2 // tp
            return (params_memory + activations) / 1e9

        elif compute_type == ComputeType.BACKWARD:
            activations = batch_size * seq_len * hidden_size * dtype_bytes * num_layers * 4 // tp
            gradients = params_per_gpu * dtype_bytes
            return (params_memory + activations + gradients) / 1e9

        elif compute_type == ComputeType.OPTIMIZER:
            optimizer_states = params_per_gpu * dtype_bytes * 2
            return (params_memory + optimizer_states) / 1e9

        return params_memory / 1e9

    def _estimate_dit_compute(
        self,
        component: ShardedModule,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: str,
        compute_type: ComputeType,
        params: Dict[str, Any],
    ) -> tuple:
        """Estimate DiT compute time and FLOPs."""
        use_cfg = params.get("use_cfg", True)
        effective_batch = batch_size * 2 if use_cfg else batch_size

        hidden_size = getattr(component, "hidden_size", 4096)
        num_layers = getattr(component, "num_layers", 32)

        latent_t = (num_frames - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8

        seq_len = latent_t * latent_h * latent_w

        forward_time = self._estimate_layer_time(effective_batch, seq_len, hidden_size, dtype) * num_layers

        forward_flops = self._estimate_llm_flops(effective_batch, seq_len, hidden_size, num_layers)

        if compute_type == ComputeType.FORWARD:
            return forward_time, forward_flops

        elif compute_type == ComputeType.BACKWARD:
            backward_time = forward_time * 2.0
            backward_flops = forward_flops * 2.0
            return backward_time, backward_flops

        elif compute_type == ComputeType.OPTIMIZER:
            optimizer_time = forward_time * 1.2
            return optimizer_time, 0.0

        return forward_time, forward_flops

    def _estimate_dit_memory(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype_bytes: float,
        compute_type: ComputeType,
        params: Dict[str, Any],
    ) -> float:
        """Estimate DiT memory."""
        use_cfg = params.get("use_cfg", True)
        effective_batch = batch_size * 2 if use_cfg else batch_size

        hidden_size = getattr(
            self._get_component("dit") if isinstance(self.model, dict) else self.model, "hidden_size", 4096
        )
        num_layers = getattr(
            self._get_component("dit") if isinstance(self.model, dict) else self.model, "num_layers", 32
        )

        latent_t = (num_frames - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8

        seq_len = latent_t * latent_h * latent_w

        parallel_degrees = self._get_parallel_degrees()
        tp = parallel_degrees["tp"]

        dit_component = self.model if isinstance(self.model, ShardedModule) else self.model.get("dit")
        params_per_gpu = (
            self._count_params(dit_component) // tp if dit_component else hidden_size * num_layers * 12 // tp
        )
        params_memory = params_per_gpu * dtype_bytes

        if compute_type == ComputeType.FORWARD:
            activations = effective_batch * seq_len * hidden_size * dtype_bytes * num_layers * 2 // tp
            return (params_memory + activations) / 1e9

        elif compute_type == ComputeType.BACKWARD:
            activations = effective_batch * seq_len * hidden_size * dtype_bytes * num_layers * 4 // tp
            gradients = params_per_gpu * dtype_bytes
            return (params_memory + activations + gradients) / 1e9

        elif compute_type == ComputeType.OPTIMIZER:
            optimizer_states = params_per_gpu * dtype_bytes * 2
            return (params_memory + optimizer_states) / 1e9

        return params_memory / 1e9

    def _estimate_vae_compute(
        self,
        component: ShardedModule,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: str,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate VAE compute time and FLOPs."""
        pixels = num_frames * height * width

        latent_channels = getattr(component, "latent_channels", 16)
        in_channels = getattr(component, "in_channels", 3)

        encoder_flops = pixels * latent_channels * 16
        decoder_flops = pixels * in_channels * 16

        total_flops = encoder_flops + decoder_flops

        tflops = self.device.get_compute_tflops(dtype)
        time = total_flops / (tflops * 1e12) if tflops > 0 else 0.0

        if compute_type == ComputeType.BACKWARD:
            time *= 2.0
            total_flops *= 2.0

        return time, total_flops

    def _estimate_vae_memory(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype_bytes: float,
    ) -> float:
        """Estimate VAE memory."""
        latent_t = (num_frames - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8

        latent_channels = 16
        in_channels = 3

        input_memory = num_frames * height * width * in_channels * dtype_bytes
        latent_memory = latent_t * latent_h * latent_w * latent_channels * dtype_bytes

        vae_component = self.model if isinstance(self.model, ShardedModule) else self.model.get("vae")
        params_memory = self._count_params(vae_component) * dtype_bytes if vae_component else 100e6 * dtype_bytes

        return (params_memory + max(input_memory, latent_memory)) / 1e9

    def _estimate_text_encoder_compute(
        self,
        component: ShardedModule,
        batch_size: int,
        seq_len: int,
        dtype: str,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate text encoder compute time and FLOPs."""
        hidden_size = getattr(component, "hidden_size", 4096)
        num_layers = getattr(component, "num_layers", 32)

        forward_time = self._estimate_layer_time(batch_size, seq_len, hidden_size, dtype) * num_layers
        forward_flops = self._estimate_llm_flops(batch_size, seq_len, hidden_size, num_layers)

        if compute_type == ComputeType.FORWARD:
            return forward_time, forward_flops

        elif compute_type == ComputeType.BACKWARD:
            return forward_time * 2.0, forward_flops * 2.0

        return forward_time, forward_flops

    def _estimate_text_encoder_memory(
        self,
        batch_size: int,
        seq_len: int,
        dtype_bytes: float,
    ) -> float:
        """Estimate text encoder memory."""
        component = self.model if isinstance(self.model, ShardedModule) else self.model.get("text_encoder")
        hidden_size = getattr(component, "hidden_size", 4096) if component else 4096
        num_layers = getattr(component, "num_layers", 32) if component else 32

        params_memory = (
            self._count_params(component) * dtype_bytes if component else hidden_size * num_layers * 12 * dtype_bytes
        )
        activations = batch_size * seq_len * hidden_size * dtype_bytes * num_layers * 2

        return (params_memory + activations) / 1e9

    def _estimate_resnet_compute(
        self,
        component: ShardedModule,
        batch_size: int,
        image_size: int,
        dtype: str,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate ResNet compute time and FLOPs."""
        flops_per_image = 7e9

        total_flops = batch_size * flops_per_image

        tflops = self.device.get_compute_tflops(dtype)
        time = total_flops / (tflops * 1e12) if tflops > 0 else 0.0

        if compute_type == ComputeType.BACKWARD:
            time *= 2.0
            total_flops *= 2.0

        return time, total_flops

    def _estimate_resnet_memory(
        self,
        batch_size: int,
        image_size: int,
        dtype_bytes: float,
        compute_type: ComputeType,
    ) -> float:
        """Estimate ResNet memory."""
        params_memory = 25e6 * dtype_bytes

        activations = batch_size * image_size * image_size * 3 * dtype_bytes * 50

        if compute_type == ComputeType.BACKWARD:
            gradients = params_memory
            return (params_memory + activations * 2 + gradients) / 1e9

        return (params_memory + activations) / 1e9

    def _calculate_throughput(
        self,
        phase_results: List[PhaseResult],
        workload: WorkloadConfig,
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate throughput metrics."""
        total_time = sum(p.total_time_sec for p in phase_results)
        if total_time <= 0:
            return {}

        throughput = {}
        metric = workload.throughput_metric

        batch_size = params.get("batch_size", 1)

        if metric == ThroughputMetric.TOKENS_PER_SEC:
            seq_len = params.get("seq_len", params.get("prompt_len", 512))
            generation_len = params.get("generation_len", 0)
            total_tokens = batch_size * (seq_len + generation_len)
            throughput["tokens_per_sec"] = total_tokens / total_time
            throughput["samples_per_sec"] = batch_size / total_time

        elif metric == ThroughputMetric.SAMPLES_PER_SEC:
            throughput["samples_per_sec"] = batch_size / total_time

        elif metric == ThroughputMetric.PIXELS_PER_SEC:
            num_frames = params.get("num_frames", 81)
            height = params.get("height", 720)
            width = params.get("width", 1280)
            total_pixels = num_frames * height * width
            throughput["pixels_per_sec"] = total_pixels / total_time

        elif metric == ThroughputMetric.VIDEOS_PER_SEC:
            throughput["videos_per_sec"] = batch_size / total_time

        elif metric == ThroughputMetric.IMAGES_PER_SEC:
            throughput["images_per_sec"] = batch_size / total_time

        return throughput

    def _get_parallel_degrees(self) -> Dict[str, int]:
        """Get parallel degrees from strategy."""
        return {
            "tp": self.strategy.tp_degree,
            "pp": self.strategy.pp_degree,
            "dp": self.strategy.dp_degree,
            "ep": self.strategy.ep_degree,
        }

    def _count_params(self, component: Optional[ShardedModule]) -> int:
        """Count parameters in a component."""
        if component is None:
            return 0

        if hasattr(component, "_count_params"):
            return component._count_params()

        hidden_size = getattr(component, "hidden_size", 4096)
        num_layers = getattr(component, "num_layers", 32)

        return hidden_size * num_layers * 12


def analyze_workload(
    model: Union[ShardedModule, Dict[str, ShardedModule]],
    device: Device,
    cluster: Cluster,
    strategy: StrategyConfig,
    workload: Union[str, WorkloadConfig],
    **kwargs,
) -> UnifiedResult:
    """Convenience function to analyze a workload."""
    analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
    return analyzer.analyze(workload, **kwargs)

"""Unified Analyzer implementation.

Analyzes performance based on compute patterns, NOT model types.
"""

from typing import Any, Dict, List, Union, Optional

from llm_perf.modeling import ShardedModule
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from llm_perf.kernels.compute import ComputeKernelRegistry
from llm_perf.kernels.communication import CommKernelRegistry
from llm_perf.kernels.functional import linear, flash_attention, conv3d, KernelResult
from llm_perf.utils.constants import DTYPE_SIZES

from .base import (
    Phase,
    PhaseResult,
    UnifiedResult,
    WorkloadConfig,
    ComputeType,
    ComputePattern,
    ThroughputMetric,
)
from .workload_loader import get_workload


class UnifiedAnalyzer:
    """Unified performance analyzer based on compute patterns.

    Analyzes performance based on:
    - ComputePattern (transformer_block, conv_encoder, etc.)
    - NOT model type (llm, dit, vae, etc.)

    The workload configuration specifies:
    - Phases: sequence of compute operations
    - ComputePattern: how to estimate compute time/memory
    - Component mapping: generic names -> user component names
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
            workload: Workload name, YAML path, or WorkloadConfig
            **kwargs: Dynamic parameters (batch_size, seq_len, generation_len, etc.)

        Returns:
            UnifiedResult with performance metrics
        """
        if isinstance(workload, str):
            workload = get_workload(workload)

        resolved_params = {**workload.default_params, **kwargs}

        phase_results = self._analyze_phases(workload, resolved_params)

        total_time = sum(p.total_time_sec for p in phase_results)
        peak_memory = max(p.memory_gb for p in phase_results) if phase_results else 0.0

        throughput = self._calculate_throughput(phase_results, workload, resolved_params)

        breakdown = self._generate_breakdown(phase_results, workload, total_time, throughput)
        detailed_breakdown = self._generate_detailed_breakdown(phase_results, workload)

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
            breakdown=breakdown,
            detailed_breakdown=detailed_breakdown,
        )

    def _analyze_phases(
        self,
        workload: WorkloadConfig,
        params: Dict[str, Any],
    ) -> List[PhaseResult]:
        """Analyze all phases in a workload."""
        results = []

        for phase in workload.phases:
            user_component_name = workload.resolve_component(phase.component)
            component = self._get_component(user_component_name)

            repeat_count = self._resolve_param(phase.repeat, params)
            seq_len_factor = self._resolve_param(phase.seq_len_factor, params)

            compute_pattern = phase.compute_pattern
            if compute_pattern is None:
                compute_pattern = self._infer_compute_pattern(component)

            single_time, memory, flops = self._estimate_phase(component, phase, compute_pattern, params, seq_len_factor)

            total_time = single_time * repeat_count

            results.append(
                PhaseResult(
                    name=phase.name,
                    component=user_component_name,
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
        """Resolve dynamic parameter."""
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

    def _infer_compute_pattern(self, component: ShardedModule) -> ComputePattern:
        """Infer compute pattern from component attributes.

        Default to transformer_block for most neural network components.
        """
        return ComputePattern.TRANSFORMER_BLOCK

    def _estimate_phase(
        self,
        component: ShardedModule,
        phase: Phase,
        compute_pattern: ComputePattern,
        params: Dict[str, Any],
        seq_len_factor: float,
    ) -> tuple:
        """Estimate single phase execution based on compute pattern.

        Returns:
            (time_sec, memory_gb, flops)
        """
        dtype = getattr(component, "dtype", "fp16")
        dtype_bytes = DTYPE_SIZES.get(dtype, 2)

        batch_size = params.get("batch_size", 1)

        if compute_pattern == ComputePattern.TRANSFORMER_BLOCK:
            return self._estimate_transformer_block(
                component, batch_size, params, seq_len_factor, dtype, dtype_bytes, phase.compute_type
            )

        elif compute_pattern == ComputePattern.CONV_ENCODER:
            return self._estimate_conv_encoder(component, batch_size, params, dtype, dtype_bytes, phase.compute_type)

        elif compute_pattern == ComputePattern.CONV_DECODER:
            return self._estimate_conv_decoder(component, batch_size, params, dtype, dtype_bytes, phase.compute_type)

        elif compute_pattern == ComputePattern.ATTENTION_ONLY:
            return self._estimate_attention_only(
                component, batch_size, params, seq_len_factor, dtype, dtype_bytes, phase.compute_type
            )

        elif compute_pattern == ComputePattern.DENSE_FORWARD:
            return self._estimate_dense_forward(component, batch_size, params, dtype, dtype_bytes, phase.compute_type)

        else:
            return self._estimate_transformer_block(
                component, batch_size, params, seq_len_factor, dtype, dtype_bytes, phase.compute_type
            )

    def _estimate_transformer_block(
        self,
        component: ShardedModule,
        batch_size: int,
        params: Dict[str, Any],
        seq_len_factor: float,
        dtype: str,
        dtype_bytes: float,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate transformer block compute (attention + FFN)."""
        hidden_size = getattr(component, "hidden_size", 4096)
        num_layers = getattr(component, "num_layers", 32)

        seq_len = params.get("seq_len", params.get("prompt_len", 2048))

        num_frames = params.get("num_frames")
        height = params.get("height")
        width = params.get("width")

        if num_frames and height and width:
            latent_t = (num_frames - 1) // 4 + 1
            latent_h = height // 8
            latent_w = width // 8
            seq_len = latent_t * latent_h * latent_w

        use_cfg = params.get("use_cfg", False)
        if use_cfg:
            batch_size = batch_size * 2

        effective_seq_len = int(seq_len * seq_len_factor)

        parallel_degrees = self._get_parallel_degrees()
        tp = parallel_degrees["tp"]
        dp = parallel_degrees["dp"]

        effective_batch = batch_size // dp
        effective_hidden = hidden_size // tp

        if compute_type == ComputeType.FORWARD:
            forward_time, forward_flops = self._estimate_transformer_block_forward(
                effective_batch, effective_seq_len, effective_hidden, dtype, num_layers
            )
            comm_time = self._estimate_comm_time(batch_size, effective_seq_len, hidden_size, tp, num_layers)
            return (
                forward_time + comm_time,
                self._estimate_forward_memory(
                    batch_size, effective_seq_len, hidden_size, num_layers, dtype_bytes, tp, dp, component
                ),
                forward_flops,
            )

        elif compute_type == ComputeType.BACKWARD:
            backward_time, backward_flops = self._estimate_transformer_block_backward(
                effective_batch, effective_seq_len, effective_hidden, dtype, num_layers
            )
            comm_time = self._estimate_comm_time(batch_size, effective_seq_len, hidden_size, tp, num_layers)
            return (
                backward_time + comm_time,
                self._estimate_backward_memory(
                    batch_size, effective_seq_len, hidden_size, num_layers, dtype_bytes, tp, dp, component
                ),
                backward_flops,
            )

        elif compute_type == ComputeType.OPTIMIZER:
            optimizer_factor = params.get("optimizer_factor", 1.5)
            forward_time, forward_flops = self._estimate_transformer_block_forward(
                effective_batch, effective_seq_len, effective_hidden, dtype, num_layers
            )
            optimizer_time = forward_time * optimizer_factor
            return optimizer_time, self._estimate_optimizer_memory(dtype_bytes, tp, dp, component), 0.0

        forward_time, forward_flops = self._estimate_transformer_block_forward(
            effective_batch, effective_seq_len, effective_hidden, dtype, num_layers
        )
        return forward_time, 0.0, forward_flops

    def _estimate_transformer_block_forward(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype: str,
        num_layers: int,
    ) -> tuple:
        """Estimate forward pass using kernel definitions."""
        total_time = 0.0
        total_flops = 0

        for _ in range(num_layers):
            layer_time, layer_flops = self._estimate_transformer_layer_forward(batch_size, seq_len, hidden_size, dtype)
            total_time += layer_time
            total_flops += layer_flops

        return total_time, total_flops

    def _estimate_transformer_block_backward(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype: str,
        num_layers: int,
    ) -> tuple:
        """Estimate backward pass using kernel definitions."""
        total_time = 0.0
        total_flops = 0

        for _ in range(num_layers):
            layer_time, layer_flops = self._estimate_transformer_layer_backward(batch_size, seq_len, hidden_size, dtype)
            total_time += layer_time
            total_flops += layer_flops

        return total_time, total_flops

    def _estimate_transformer_layer_forward(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype: str,
    ) -> tuple:
        """Estimate single transformer layer forward using kernel API."""
        m = batch_size * seq_len

        attention_time, attention_flops = self._estimate_attention_forward(m, hidden_size, dtype)
        ffn_time, ffn_flops = self._estimate_ffn_forward(m, hidden_size, dtype)

        return attention_time + ffn_time, attention_flops + ffn_flops

    def _estimate_transformer_layer_backward(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype: str,
    ) -> tuple:
        """Estimate single transformer layer backward using kernel API."""
        m = batch_size * seq_len

        attention_time, attention_flops = self._estimate_attention_backward(m, hidden_size, dtype)
        ffn_time, ffn_flops = self._estimate_ffn_backward(m, hidden_size, dtype)

        return attention_time + ffn_time, attention_flops + ffn_flops

    def _estimate_attention_forward(
        self,
        m: int,
        hidden_size: int,
        dtype: str,
    ) -> tuple:
        """Estimate attention forward: QKV projections + attention + output projection."""
        total_time = 0.0
        total_flops = 0

        qkv_result = self._estimate_linear_kernel((m, hidden_size), (hidden_size, hidden_size * 3), dtype)
        total_time += self._estimate_kernel_time_from_result(qkv_result, is_forward=True)
        total_flops += qkv_result.flops

        num_heads = 32
        head_dim = hidden_size // num_heads
        batch = m // hidden_size
        seq_len = hidden_size

        attn_result = flash_attention(
            (batch, num_heads, seq_len, head_dim),
            (batch, num_heads, seq_len, head_dim),
            (batch, num_heads, seq_len, head_dim),
            is_causal=True,
            dtype=dtype,
        )
        total_time += self._estimate_kernel_time_from_result(attn_result, is_forward=True)
        total_flops += attn_result.flops

        o_result = self._estimate_linear_kernel((m, hidden_size), (hidden_size, hidden_size), dtype)
        total_time += self._estimate_kernel_time_from_result(o_result, is_forward=True)
        total_flops += o_result.flops

        return total_time, total_flops

    def _estimate_attention_backward(
        self,
        m: int,
        hidden_size: int,
        dtype: str,
    ) -> tuple:
        """Estimate attention backward using kernel backward metrics."""
        total_time = 0.0
        total_flops = 0

        qkv_result = self._estimate_linear_kernel((m, hidden_size), (hidden_size, hidden_size * 3), dtype)
        total_time += self._estimate_kernel_time_from_result(qkv_result, is_forward=False)
        total_flops += qkv_result.flops_backward

        num_heads = 32
        head_dim = hidden_size // num_heads
        batch = m // hidden_size
        seq_len = hidden_size

        attn_result = flash_attention(
            (batch, num_heads, seq_len, head_dim),
            (batch, num_heads, seq_len, head_dim),
            (batch, num_heads, seq_len, head_dim),
            is_causal=True,
            dtype=dtype,
        )
        total_time += self._estimate_kernel_time_from_result(attn_result, is_forward=False)
        total_flops += attn_result.flops_backward

        o_result = self._estimate_linear_kernel((m, hidden_size), (hidden_size, hidden_size), dtype)
        total_time += self._estimate_kernel_time_from_result(o_result, is_forward=False)
        total_flops += o_result.flops_backward

        return total_time, total_flops

    def _estimate_ffn_forward(
        self,
        m: int,
        hidden_size: int,
        dtype: str,
    ) -> tuple:
        """Estimate FFN forward: up projection + activation + down projection."""
        total_time = 0.0
        total_flops = 0

        up_result = self._estimate_linear_kernel((m, hidden_size), (hidden_size * 4, hidden_size), dtype)
        total_time += self._estimate_kernel_time_from_result(up_result, is_forward=True)
        total_flops += up_result.flops

        down_result = self._estimate_linear_kernel((m, hidden_size * 4), (hidden_size, hidden_size * 4), dtype)
        total_time += self._estimate_kernel_time_from_result(down_result, is_forward=True)
        total_flops += down_result.flops

        return total_time, total_flops

    def _estimate_ffn_backward(
        self,
        m: int,
        hidden_size: int,
        dtype: str,
    ) -> tuple:
        """Estimate FFN backward using kernel backward metrics."""
        total_time = 0.0
        total_flops = 0

        up_result = self._estimate_linear_kernel((m, hidden_size), (hidden_size * 4, hidden_size), dtype)
        total_time += self._estimate_kernel_time_from_result(up_result, is_forward=False)
        total_flops += up_result.flops_backward

        down_result = self._estimate_linear_kernel((m, hidden_size * 4), (hidden_size, hidden_size * 4), dtype)
        total_time += self._estimate_kernel_time_from_result(down_result, is_forward=False)
        total_flops += down_result.flops_backward

        return total_time, total_flops

    def _estimate_linear_kernel(
        self,
        input_shape: tuple,
        weight_shape: tuple,
        dtype: str,
    ) -> KernelResult:
        """Get linear kernel result with forward/backward metrics."""
        return linear(input_shape, weight_shape, dtype=dtype)

    def _estimate_kernel_time_from_result(
        self,
        result: KernelResult,
        is_forward: bool = True,
    ) -> float:
        """Estimate kernel execution time from KernelResult."""
        if is_forward:
            flops = result.flops
            bytes_accessed = result.bytes_accessed
        else:
            flops = result.flops_backward
            bytes_accessed = result.bytes_accessed_backward

        tflops = self.device.get_compute_tflops(result.dtype)
        bandwidth_gbps = self.device.get_memory_bw_gbps()

        compute_time = flops / (tflops * 1e12) if tflops > 0 else 0.0
        memory_time = bytes_accessed / (bandwidth_gbps * 1e9) if bandwidth_gbps > 0 else 0.0

        return max(compute_time, memory_time)

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

    def _estimate_forward_memory(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dtype_bytes: float,
        tp: int,
        dp: int,
        component: ShardedModule,
    ) -> float:
        """Estimate forward pass memory."""
        params_per_gpu = self._count_params(component) // tp // dp
        params_memory = params_per_gpu * dtype_bytes

        activations = batch_size * seq_len * hidden_size * dtype_bytes * num_layers * 2 // tp

        return (params_memory + activations) / 1e9

    def _estimate_backward_memory(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dtype_bytes: float,
        tp: int,
        dp: int,
        component: ShardedModule,
    ) -> float:
        """Estimate backward pass memory."""
        params_per_gpu = self._count_params(component) // tp // dp
        params_memory = params_per_gpu * dtype_bytes

        activations = batch_size * seq_len * hidden_size * dtype_bytes * num_layers * 4 // tp
        gradients = params_per_gpu * dtype_bytes

        return (params_memory + activations + gradients) / 1e9

    def _estimate_optimizer_memory(
        self,
        dtype_bytes: float,
        tp: int,
        dp: int,
        component: ShardedModule,
    ) -> float:
        """Estimate optimizer memory."""
        params_per_gpu = self._count_params(component) // tp // dp
        params_memory = params_per_gpu * dtype_bytes

        optimizer_states = params_per_gpu * dtype_bytes * 2

        return (params_memory + optimizer_states) / 1e9

    def _estimate_conv_encoder(
        self,
        component: ShardedModule,
        batch_size: int,
        params: Dict[str, Any],
        dtype: str,
        dtype_bytes: float,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate convolutional encoder compute."""
        image_size = params.get("image_size", 224)
        num_frames = params.get("num_frames", 1)
        height = params.get("height", image_size)
        width = params.get("width", image_size)

        pixels = num_frames * height * width

        flops_per_pixel = 16
        total_flops = pixels * flops_per_pixel * batch_size

        tflops = self.device.get_compute_tflops(dtype)
        time = total_flops / (tflops * 1e12) if tflops > 0 else 0.0

        if compute_type == ComputeType.BACKWARD:
            time *= 2.0
            total_flops *= 2.0

        params_memory = self._count_params(component) * dtype_bytes
        activations = pixels * batch_size * dtype_bytes * 4

        memory = (params_memory + activations) / 1e9
        if compute_type == ComputeType.BACKWARD:
            memory *= 2.0

        return time, memory, total_flops

    def _estimate_conv_decoder(
        self,
        component: ShardedModule,
        batch_size: int,
        params: Dict[str, Any],
        dtype: str,
        dtype_bytes: float,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate convolutional decoder compute using kernel API.

        VAE Decoder structure (from llm_perf/modeling/vision.py:254-333):
        - conv_in: latent_channels → block_out_channels[-1]
        - mid_block_0, mid_block_1: 2 ResBlocks
        - up_blocks: 4 layers, each with 3 ResBlocks + upsampling conv
        - norm_out: GroupNorm
        - conv_out: block_out_channels[0] → out_channels

        ResBlock structure (ShardedResNetBlock3d):
        - norm1 → conv1 (in_ch → out_ch) → norm2 → conv2 (out_ch → out_ch)
        - optional shortcut conv (in_ch → out_ch) if channels differ
        """
        num_frames = params.get("num_frames", 81)
        height = params.get("height", 720)
        width = params.get("width", 1280)

        latent_channels = getattr(component, "latent_channels", 16)
        out_channels = getattr(component, "out_channels", getattr(component, "in_channels", 3))
        block_out_channels = getattr(component, "block_out_channels", (128, 256, 512, 512))

        latent_t = (num_frames - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8

        reverse_channels = list(reversed(block_out_channels))

        total_flops = 0.0

        def calc_conv3d_flops(n, c_in, c_out, d, h, w, kd=3, kh=3, kw=3):
            result = conv3d((n, c_in, d, h, w), (c_out, c_in, kd, kh, kw), dtype=dtype)
            return result.flops

        def calc_resblock_flops(n, in_ch, out_ch, d, h, w):
            flops = 0.0
            flops += calc_conv3d_flops(n, in_ch, out_ch, d, h, w)
            flops += calc_conv3d_flops(n, out_ch, out_ch, d, h, w)
            if in_ch != out_ch:
                flops += calc_conv3d_flops(n, in_ch, out_ch, d, h, w, kd=1, kh=1, kw=1)
            return flops

        n = batch_size
        curr_d = latent_t
        curr_h = latent_h
        curr_w = latent_w

        total_flops += calc_conv3d_flops(n, latent_channels, reverse_channels[0], curr_d, curr_h, curr_w)

        ch = reverse_channels[0]
        total_flops += calc_resblock_flops(n, ch, ch, curr_d, curr_h, curr_w)
        total_flops += calc_resblock_flops(n, ch, ch, curr_d, curr_h, curr_w)

        for i, out_ch in enumerate(reverse_channels):
            in_ch = reverse_channels[i - 1] if i > 0 else out_ch
            for j in range(3):
                block_in_ch = in_ch if j == 0 else out_ch
                total_flops += calc_resblock_flops(n, block_in_ch, out_ch, curr_d, curr_h, curr_w)

            if i < len(reverse_channels) - 1:
                total_flops += calc_conv3d_flops(n, out_ch, out_ch, curr_d, curr_h, curr_w)
                curr_h *= 2
                curr_w *= 2

        total_flops += calc_conv3d_flops(n, reverse_channels[-1], out_channels, curr_d, curr_h, curr_w)

        tflops = self.device.get_compute_tflops(dtype)
        time = total_flops / (tflops * 1e12) if tflops > 0 else 0.0

        if compute_type == ComputeType.BACKWARD:
            time *= 2.0
            total_flops *= 2.0

        latent_memory = latent_t * latent_h * latent_w * latent_channels * dtype_bytes
        output_memory = num_frames * height * width * out_channels * dtype_bytes
        params_memory = self._count_params(component) * dtype_bytes

        memory = (params_memory + max(latent_memory, output_memory)) / 1e9

        return time, memory, total_flops

    def _estimate_attention_only(
        self,
        component: ShardedModule,
        batch_size: int,
        params: Dict[str, Any],
        seq_len_factor: float,
        dtype: str,
        dtype_bytes: float,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate pure attention compute (no FFN)."""
        hidden_size = getattr(component, "hidden_size", 4096)
        num_layers = getattr(component, "num_layers", 32)

        seq_len = params.get("prompt_len", params.get("seq_len", 512))
        effective_seq_len = int(seq_len * seq_len_factor)

        parallel_degrees = self._get_parallel_degrees()
        tp = parallel_degrees["tp"]
        dp = parallel_degrees["dp"]

        effective_batch = batch_size // dp
        effective_hidden = hidden_size // tp

        m = effective_batch * effective_seq_len
        k = effective_hidden

        attn_kernel = self.compute_registry.get_or_create_matmul(m, k * 3, k, dtype)
        forward_time = attn_kernel.estimate_time((m, k), (m, k * 3), dtype) * 2 * num_layers

        attn_flops = m * hidden_size * hidden_size * 3 * 2 * num_layers

        if compute_type == ComputeType.FORWARD:
            return (
                forward_time,
                self._estimate_forward_memory(
                    batch_size, effective_seq_len, hidden_size, num_layers, dtype_bytes, tp, dp, component
                ),
                attn_flops,
            )

        elif compute_type == ComputeType.BACKWARD:
            backward_time = forward_time * 2.0
            backward_flops = attn_flops * 2.0
            return (
                backward_time,
                self._estimate_backward_memory(
                    batch_size, effective_seq_len, hidden_size, num_layers, dtype_bytes, tp, dp, component
                ),
                backward_flops,
            )

        return forward_time, 0.0, attn_flops

    def _estimate_dense_forward(
        self,
        component: ShardedModule,
        batch_size: int,
        params: Dict[str, Any],
        dtype: str,
        dtype_bytes: float,
        compute_type: ComputeType,
    ) -> tuple:
        """Estimate dense/MLP forward compute."""
        input_size = getattr(component, "input_size", 1024)
        output_size = getattr(component, "output_size", 1024)
        hidden_size = getattr(component, "hidden_size", 4096)
        num_layers = getattr(component, "num_layers", 4)

        total_flops = batch_size * input_size * hidden_size * 2 * num_layers
        total_flops += batch_size * hidden_size * output_size * 2

        tflops = self.device.get_compute_tflops(dtype)
        time = total_flops / (tflops * 1e12) if tflops > 0 else 0.0

        if compute_type == ComputeType.BACKWARD:
            time *= 2.0
            total_flops *= 2.0

        params_memory = self._count_params(component) * dtype_bytes
        activations = batch_size * hidden_size * dtype_bytes * num_layers

        memory = (params_memory + activations) / 1e9

        return time, memory, total_flops

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

    def _generate_breakdown(
        self,
        phases: List[PhaseResult],
        workload: WorkloadConfig,
        total_time: float,
        throughput: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate legacy breakdown format for frontend compatibility."""
        compute_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.FORWARD)
        backward_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.BACKWARD)
        optimizer_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.OPTIMIZER)

        layers = []
        for phase in phases:
            kernels = []
            if phase.flops and phase.flops > 0:
                kernels.append(
                    {
                        "name": f"{phase.name}_compute",
                        "kernel_type": "compute",
                        "time_sec": phase.single_time_sec,
                        "flops": phase.flops,
                    }
                )
            layers.append(
                {
                    "name": phase.component,
                    "kernels": kernels,
                    "total_time_ms": phase.total_time_sec * 1000,
                }
            )

        return {
            "overview": {
                "total_time_sec": total_time,
                "throughput": sum(throughput.values()) if throughput else 0,
            },
            "time_breakdown": {
                "compute_sec": compute_time,
                "backward_sec": backward_time,
                "optimizer_sec": optimizer_time,
                "communication_sec": 0,
                "memory_sec": 0,
                "compute_percent": compute_time / total_time * 100 if total_time > 0 else 0,
            },
            "layers": layers,
        }

    def _generate_detailed_breakdown(
        self,
        phases: List[PhaseResult],
        workload: WorkloadConfig,
    ) -> Dict[str, Any]:
        """Generate legacy detailed_breakdown format for frontend compatibility."""
        by_component: Dict[str, Dict[str, float]] = {}
        for phase in phases:
            if phase.component not in by_component:
                by_component[phase.component] = {}
            by_component[phase.component]["activations"] = phase.memory_gb

        total_memory = sum(p.memory_gb for p in phases)

        return {
            "submodels": [
                {
                    "model_name": phase.component,
                    "model_type": phase.compute_type.value,
                    "compute_time_sec": phase.total_time_sec,
                    "memory": {"by_type": {"activations": phase.memory_gb}},
                }
                for phase in phases
            ],
            "memory": {
                "by_type": {"activations": total_memory},
                "by_submodel": by_component,
                "by_block_type": {},
            },
            "communication": {
                "by_parallelism": {},
            },
        }


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

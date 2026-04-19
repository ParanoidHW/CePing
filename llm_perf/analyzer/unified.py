"""Unified Analyzer implementation.

Analyzes performance based on compute patterns, NOT model types.
"""

from typing import Any, Dict, List, Union, Optional, Tuple

from llm_perf.modeling import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor
from llm_perf.modeling.module import ModuleInstance
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from llm_perf.strategy.parallel_context import ParallelContext, SPType, CommDomain
from llm_perf.kernels.compute import ComputeKernelRegistry
from llm_perf.kernels.communication import CommKernelRegistry
from llm_perf.kernels.functional import linear, flash_attention, conv3d, KernelResult
from llm_perf.kernels.backend.theory import TheoryBackend, BackendConfig
from llm_perf.utils.constants import DTYPE_SIZES

from .base import (
    Phase,
    PhaseResult,
    SubmoduleResult,
    UnifiedResult,
    WorkloadConfig,
    ComputeType,
    ComputePattern,
    ThroughputMetric,
    CommunicationBreakdown,
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

    def _create_parallel_context(self, params: Dict[str, Any]) -> ParallelContext:
        """从StrategyConfig创建ParallelContext."""
        sp_type = SPType.NONE
        if getattr(self.strategy, "sequence_parallel", False):
            sp_type = SPType.ULYSSES

        ctx = ParallelContext(
            tp_degree=self.strategy.tp_degree,
            pp_degree=self.strategy.pp_degree,
            ep_degree=self.strategy.ep_degree,
            sp_degree=self.strategy.sp_degree,
            dp_degree=self.strategy.dp_degree,
            sp_type=sp_type,
            dtype=params.get("dtype", "fp16"),
            device=self.device.config,
            activation_checkpointing=getattr(self.strategy, "activation_checkpointing", False),
            zero_stage=getattr(self.strategy, "zero_stage", 0),
        )

        if hasattr(self.cluster, "topology"):
            topology = self.cluster.topology
            ctx.comm_domains = {
                "tp": CommDomain(
                    "tp",
                    ctx.tp_degree,
                    list(range(ctx.tp_degree)),
                    getattr(topology, "intra_node_bandwidth_gbps", 400.0),
                ),
                "dp": CommDomain(
                    "dp",
                    ctx.dp_degree,
                    list(range(ctx.dp_degree)),
                    getattr(topology, "inter_node_bandwidth_gbps", 100.0),
                ),
            }

        return ctx

    def _infer_submodule_type(self, sub_name: str, sub_inst: Any = None) -> str:
        """推断子模块类型，优先检查实际类型."""
        if sub_inst and hasattr(sub_inst, "module") and sub_inst.module:
            module = sub_inst.module
            module_class = type(module).__name__.lower()

            if "attention" in module_class:
                return "attention"
            elif "ffn" in module_class or "mlp" in module_class:
                return "ffn"
            elif "embedding" in module_class:
                return "embedding"
            elif "transformerblock" in module_class or "block" in module_class:
                return "transformer_block"
            elif "rmsnorm" in module_class or "norm" in module_class:
                return "rms_norm"
            elif "lmhead" in module_class:
                return "lm_head"
            elif "moe" in module_class:
                return "moe"
            elif "vae" in module_class:
                return "vae"
            elif "conv" in module_class:
                return "conv"
            elif "dit" in module_class:
                return "dit"

        sub_lower = sub_name.lower()
        if "embedding" in sub_lower or "emb" in sub_lower:
            return "embedding"
        elif "attention" in sub_lower or "attn" in sub_lower:
            return "attention"
        elif "ffn" in sub_lower or "mlp" in sub_lower:
            return "ffn"
        elif "moe" in sub_lower:
            return "moe"
        elif "lm_head" in sub_lower or "output" in sub_lower:
            return "lm_head"
        elif "norm" in sub_lower:
            return "rms_norm"
        elif "conv" in sub_lower:
            return "conv"
        elif "resblock" in sub_lower or "res" in sub_lower:
            return "resblock"
        elif "layer" in sub_lower:
            return "transformer_block"
        else:
            return "unknown"

    def _estimate_submodule_time(self, sub_inst: ModuleInstance) -> float:
        """估算子模块时间."""
        backend_config = BackendConfig(name="theory", device=self.device)
        backend = TheoryBackend(backend_config)
        return sub_inst.estimate_time(backend)

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

        forward_flops = sum(
            p.flops for p in phase_results if p.compute_type == ComputeType.FORWARD and p.flops is not None
        )
        mfu = self._calculate_mfu(int(forward_flops), total_time)

        batch_size = resolved_params.get("batch_size", 1)
        qps = self._calculate_qps(batch_size, total_time)

        comm_breakdown = self._extract_communication_breakdown(phase_results)

        breakdown = self._generate_breakdown(phase_results, workload, total_time, throughput)
        detailed_breakdown = self._generate_detailed_breakdown(phase_results, workload, comm_breakdown)

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
                "tp_degree": self.strategy.tp_degree,
                "pp_degree": self.strategy.pp_degree,
                "dp_degree": self.strategy.dp_degree,
                "ep_degree": self.strategy.ep_degree,
                "kv_cache_gb": 0.0,
            },
            breakdown=breakdown,
            detailed_breakdown=detailed_breakdown,
            mfu=mfu,
            qps=qps,
            communication_breakdown=comm_breakdown,
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

            single_time, memory, flops, submodules = self._estimate_phase(
                component, phase, compute_pattern, params, seq_len_factor
            )

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
                    submodules=submodules,
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

    def _analyze_phase_with_submodules(
        self,
        component: ShardedModule,
        phase: Phase,
        params: Dict[str, Any],
    ) -> Tuple[float, float, int, List[SubmoduleResult]]:
        """使用ShardedModule.bind()机制分析phase和子模块."""
        ctx = self._create_parallel_context(params)

        batch_size = params.get("batch_size", 1)
        seq_len = params.get("seq_len", params.get("prompt_len", 512))
        hidden_size = getattr(component, "hidden_size", 4096)

        compute_pattern = phase.compute_pattern or self._infer_compute_pattern(component)

        if compute_pattern == ComputePattern.CONV_ENCODER:
            num_frames = params.get("num_frames", 81)
            height = params.get("height", 720)
            width = params.get("width", 1280)
            channels = getattr(component, "in_channels", 3)
            input_tensor = ShardedTensor(shape=(batch_size, channels, num_frames, height, width))
        elif compute_pattern == ComputePattern.CONV_DECODER:
            latent_t = (params.get("num_frames", 81) - 1) // 4 + 1
            latent_h = params.get("height", 720) // 8
            latent_w = params.get("width", 1280) // 8
            latent_channels = getattr(component, "latent_channels", 16)
            input_tensor = ShardedTensor(shape=(batch_size, latent_channels, latent_t, latent_h, latent_w))
        elif hasattr(component, "vocab_size"):
            vocab_size = getattr(component, "vocab_size", 32000)
            input_tensor = ShardedTensor(shape=(batch_size, seq_len))
        else:
            input_tensor = ShardedTensor(shape=(batch_size, seq_len, hidden_size))

        try:
            component(input_tensor)
        except Exception:
            pass

        mode = "forward_backward" if phase.compute_type in [ComputeType.BACKWARD, ComputeType.OPTIMIZER] else "forward"
        module_instance = component.bind(ctx, mode=mode)

        if phase.compute_type == ComputeType.FORWARD:
            flops = module_instance.flops_forward_physical
        else:
            flops = module_instance.flops_total_physical

        memory = module_instance.activation_memory_physical / 1e9

        submodules = []
        for sub_name, sub_inst in module_instance._submodule_instances.items():
            submodules.append(
                SubmoduleResult(
                    name=sub_name,
                    submodule_type=self._infer_submodule_type(sub_name, sub_inst),
                    time_sec=self._estimate_submodule_time(sub_inst),
                    flops=sub_inst.flops_forward_physical,
                    memory_gb=sub_inst.activation_memory_physical / 1e9,
                    communication_bytes=sum(op.data_bytes for op in sub_inst.total_comm_ops),
                )
            )

        backend_config = BackendConfig(name="theory", device=self.device)
        backend = TheoryBackend(backend_config)
        time_sec = module_instance.estimate_time(backend)

        return time_sec, memory, flops, submodules

    def _estimate_phase(
        self,
        component: ShardedModule,
        phase: Phase,
        compute_pattern: ComputePattern,
        params: Dict[str, Any],
        seq_len_factor: float,
    ) -> tuple:
        """Estimate single phase execution.

        Primary: use bind() mechanism for accurate estimation
        Fallback: use formula-based estimation if bind fails

        Returns:
            (time_sec, memory_gb, flops, submodules)
        """
        submodules: List[SubmoduleResult] = []

        try:
            bind_time, bind_memory, bind_flops, bind_submodules = self._analyze_phase_with_submodules(
                component, phase, params
            )

            if bind_time > 0 or bind_flops > 0 or len(bind_submodules) > 0:
                submodules = bind_submodules

            if bind_time > 0 and bind_flops > 0:
                return bind_time, bind_memory, bind_flops, submodules

        except Exception:
            pass

        dtype = getattr(component, "dtype", "fp16")
        dtype_bytes = DTYPE_SIZES.get(dtype, 2)
        batch_size = params.get("batch_size", 1)

        if compute_pattern == ComputePattern.TRANSFORMER_BLOCK:
            time_sec, memory_gb, flops = self._estimate_transformer_block(
                component, batch_size, params, seq_len_factor, dtype, dtype_bytes, phase.compute_type
            )
        elif compute_pattern == ComputePattern.CONV_ENCODER:
            time_sec, memory_gb, flops = self._estimate_conv_encoder(
                component, batch_size, params, dtype, dtype_bytes, phase.compute_type
            )
        elif compute_pattern == ComputePattern.CONV_DECODER:
            time_sec, memory_gb, flops = self._estimate_conv_decoder(
                component, batch_size, params, dtype, dtype_bytes, phase.compute_type
            )
        elif compute_pattern == ComputePattern.ATTENTION_ONLY:
            time_sec, memory_gb, flops = self._estimate_attention_only(
                component, batch_size, params, seq_len_factor, dtype, dtype_bytes, phase.compute_type
            )
        elif compute_pattern == ComputePattern.DENSE_FORWARD:
            time_sec, memory_gb, flops = self._estimate_dense_forward(
                component, batch_size, params, dtype, dtype_bytes, phase.compute_type
            )
        else:
            time_sec, memory_gb, flops = self._estimate_transformer_block(
                component, batch_size, params, seq_len_factor, dtype, dtype_bytes, phase.compute_type
            )

        return time_sec, memory_gb, flops, submodules

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

    def _calculate_qps(self, batch_size: int, total_time_sec: float) -> float:
        """计算QPS = batch_size * dp_degree / total_time."""
        dp = self.strategy.dp_degree
        if total_time_sec <= 0:
            return 0.0
        return batch_size * dp / total_time_sec

    def _infer_comm_op_type(self, submodule_type: str) -> str:
        """根据子模块类型推断通信操作类型."""
        type_mapping = {
            "attention": "all_reduce",
            "ffn": "all_reduce",
            "moe": "all_to_all",
            "embedding": "all_gather",
            "lm_head": "reduce_scatter",
        }
        return type_mapping.get(submodule_type, "all_reduce")

    def _extract_communication_breakdown(self, phase_results: List[PhaseResult]) -> CommunicationBreakdown:
        """从PhaseResult.submodules的communication_bytes提取通信分解."""
        comm_ops = {
            "all_reduce": {"total_bytes": 0, "total_time_sec": 0.0},
            "all_gather": {"total_bytes": 0, "total_time_sec": 0.0},
            "reduce_scatter": {"total_bytes": 0, "total_time_sec": 0.0},
            "all_to_all": {"total_bytes": 0, "total_time_sec": 0.0},
        }

        for phase in phase_results:
            for sm in phase.submodules:
                if sm.communication_bytes > 0:
                    op_type = self._infer_comm_op_type(sm.submodule_type)
                    comm_ops[op_type]["total_bytes"] += sm.communication_bytes
                    comm_ops[op_type]["total_time_sec"] += sm.time_sec * 0.1

        return CommunicationBreakdown(**comm_ops)

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

    def _calculate_mfu(self, total_flops: int, total_time_sec: float) -> float:
        """Calculate Model FLOPs Utilization.

        Args:
            total_flops: Total FLOPs executed
            total_time_sec: Total execution time in seconds

        Returns:
            MFU value between 0 and 1
        """
        peak_tflops = self.device.config.fp16_tflops_cube
        num_devices = self.cluster.num_devices

        if peak_tflops <= 0 or total_time_sec <= 0 or total_flops <= 0:
            return 0.0

        theoretical_peak_flops = peak_tflops * 1e12 * num_devices * total_time_sec
        mfu = total_flops / theoretical_peak_flops

        return min(mfu, 1.0)

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

        submodule_breakdown = []
        for phase in phases:
            for sm in phase.submodules:
                submodule_breakdown.append(
                    {
                        "phase": phase.name,
                        "name": sm.name,
                        "type": sm.submodule_type,
                        "time_sec": sm.time_sec,
                        "time_ms": sm.time_sec * 1000,
                        "flops": sm.flops,
                        "flops_gflops": sm.flops / 1e9,
                        "memory_gb": sm.memory_gb,
                        "communication_gb": sm.communication_bytes / 1e9,
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
            "submodules": submodule_breakdown,
        }

    def _generate_detailed_breakdown(
        self,
        phases: List[PhaseResult],
        workload: WorkloadConfig,
        comm_breakdown: Optional[CommunicationBreakdown] = None,
    ) -> Dict[str, Any]:
        """Generate detailed breakdown format for frontend compatibility.

        All memory metrics are per-GPU.
        """
        by_component: Dict[str, Dict[str, float]] = {}
        for phase in phases:
            if phase.component not in by_component:
                by_component[phase.component] = {}
            by_component[phase.component]["activations_gb"] = phase.memory_gb

        by_block_type: Dict[str, Dict[str, Any]] = {}
        for phase in phases:
            for sm in phase.submodules:
                block_type = sm.submodule_type
                if block_type not in by_block_type:
                    by_block_type[block_type] = {
                        "activations_gb": 0.0,
                        "flops": 0,
                        "time_sec": 0.0,
                    }
                by_block_type[block_type]["activations_gb"] += sm.memory_gb
                by_block_type[block_type]["flops"] += sm.flops
                by_block_type[block_type]["time_sec"] += sm.time_sec

        total_memory = sum(p.memory_gb for p in phases)

        comm_breakdown_dict = {}
        if comm_breakdown:
            comm_breakdown_dict = comm_breakdown.to_dict()

        return {
            "submodels": [
                {
                    "model_name": phase.component,
                    "model_type": phase.compute_type.value,
                    "compute_time_sec": phase.total_time_sec,
                    "memory": {"by_type": {"activations_gb": phase.memory_gb}},
                }
                for phase in phases
            ],
            "memory": {
                "by_type": {"activations_gb": total_memory},
                "by_submodel": by_component,
                "by_block_type": by_block_type,
            },
            "communication": {
                "by_parallelism": comm_breakdown_dict,
                "total_bytes": sum(v.get("total_bytes", 0) for v in comm_breakdown_dict.values()),
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

"""Unified Analyzer implementation.

Analyzes performance based on compute patterns, NOT model types.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.device import Device
from llm_perf.kernels.backend.theory import BackendConfig, TheoryBackend
from llm_perf.kernels.communication import CommKernelRegistry
from llm_perf.kernels.compute import ComputeKernelRegistry
from llm_perf.kernels.functional import KernelResult, conv3d, flash_attention, linear
from llm_perf.modeling import ShardedModule
from llm_perf.modeling.module import ModuleInstance
from llm_perf.modeling.tensor import ShardedTensor
from llm_perf.strategy.base import StrategyConfig
from llm_perf.strategy.parallel_context import CommDomain, ParallelContext, SPType
from llm_perf.utils.constants import DTYPE_SIZES

from .base import (
    CommunicationBreakdown,
    ComputePattern,
    ComputeType,
    Phase,
    PhaseResult,
    SubmoduleResult,
    ThroughputMetric,
    UnifiedResult,
    WorkloadConfig,
    WorkloadType,
)
from .breakdown import generate_module_breakdown
from .workload_loader import get_workload

logger = logging.getLogger(__name__)


def infer_submodule_name_from_class(cls_name: str) -> str:
    """从类名推断子模块名称（fallback方案）.

    规则：
    1. 前缀 "Sharded" 移除
    2. 后缀 "Block" 移除
    3. 特殊处理：常见类型直接映射
    4. CamelCase to snake_case（智能处理缩写）
    """
    name = cls_name.replace("Sharded", "")
    if name.endswith("Block"):
        name = name[:-5]
    
    if name == "Attention":
        return "attention"
    elif name == "LinearAttention":
        return "linear_attention"
    elif name == "MLA":
        return "mla"
    elif name == "FFN":
        return "ffn"
    elif name == "MoE":
        return "moe"
    elif name == "Embedding":
        return "embedding"
    elif name == "LMHead":
        return "lm_head"
    elif name in ["RMSNorm", "LayerNorm"]:
        return name.lower().replace("norm", "_norm")
    elif name == "Transformer":
        return "transformer_block"
    
    abbreviations = ["MLA", "FFN", "MoE", "ViT", "DiT", "VAE", "LM", "RMS"]
    
    result = []
    i = 0
    while i < len(name):
        matched_abbr = None
        for abbr in abbreviations:
            if name[i:i+len(abbr)] == abbr:
                matched_abbr = abbr
                break
        
        if matched_abbr:
            if i > 0 and result[-1] != "_":
                result.append("_")
            result.append(matched_abbr.lower())
            i += len(matched_abbr)
        elif name[i].isupper():
            if i > 0 and result[-1] != "_":
                result.append("_")
            result.append(name[i].lower())
            i += 1
        elif name[i] == "_":
            result.append("_")
            i += 1
        elif name[i].isdigit():
            result.append(name[i])
            i += 1
        else:
            result.append(name[i])
            i += 1
    
    snake_name = "".join(result)
    snake_name = re.sub(r"_+", "_", snake_name)
    snake_name = snake_name.strip("_")
    
    return snake_name


def _aggregate_submodel_memory(phases: List[PhaseResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate memory by submodel across all phases.

    Returns the peak memory for each submodel.
    """
    submodel_memory = {}
    for phase in phases:
        component = phase.component
        if component not in submodel_memory:
            submodel_memory[component] = {
                "activations_gb": 0.0,
                "weight_gb": 0.0,
                "gradient_gb": 0.0,
                "optimizer_gb": 0.0,
                "activation_gb": 0.0,
                "peak_memory_gb": 0.0,
            }
        if phase.memory_gb > submodel_memory[component]["peak_memory_gb"]:
            submodel_memory[component]["peak_memory_gb"] = phase.memory_gb
            submodel_memory[component]["weight_gb"] = phase.memory_breakdown.get("weight_gb", 0)
            submodel_memory[component]["gradient_gb"] = phase.memory_breakdown.get("gradient_gb", 0)
            submodel_memory[component]["optimizer_gb"] = phase.memory_breakdown.get("optimizer_gb", 0)
            submodel_memory[component]["activation_gb"] = phase.memory_breakdown.get("activation_gb", 0)
            submodel_memory[component]["activations_gb"] = phase.memory_gb
    return submodel_memory


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

            def _get_level_bandwidth(topology, level_name: str, default: float) -> float:
                """从 topology.levels 获取指定层级的带宽."""
                for level in topology.levels:
                    if level.name == level_name:
                        return level.bandwidth_gbps
                return default

            ctx.comm_domains = {
                "tp": CommDomain(
                    "tp",
                    ctx.tp_degree,
                    list(range(ctx.tp_degree)),
                    _get_level_bandwidth(topology, "node", 400.0),
                ),
                "dp": CommDomain(
                    "dp",
                    ctx.dp_degree,
                    list(range(ctx.dp_degree)),
                    _get_level_bandwidth(topology, "inter_node", 100.0),
                ),
            }

        return ctx

    def _infer_submodule_type(self, sub_name: str, sub_inst: Any = None) -> str:
        """识别子模块类型（解耦设计：优先使用 _submodule_name，fallback从类名推断）."""
        if sub_inst and hasattr(sub_inst, "module") and sub_inst.module:
            module = sub_inst.module
            
            if hasattr(module, "_submodule_name") and module._submodule_name:
                return module._submodule_name
            
            module_class = type(module).__name__
            inferred_name = infer_submodule_name_from_class(module_class)
            
            if inferred_name and inferred_name != "unknown":
                return inferred_name
            
            module_class_lower = module_class.lower()
            if "vae" in module_class_lower:
                return "vae"
            elif "conv" in module_class_lower:
                return "conv"
            elif "resnet" in module_class_lower or "resblock" in module_class_lower:
                return "resblock"
            elif "dit" in module_class_lower:
                return "dit"

        sub_lower = sub_name.lower()
        if sub_lower.startswith("layers.") or sub_lower.startswith("blocks."):
            return "transformer_block"
        elif "embedding" in sub_lower or "emb" in sub_lower:
            return "embedding"
        elif "attention" in sub_lower or "attn" in sub_lower:
            return "attention"
        elif "linear_attention" in sub_lower:
            return "linear_attention"
        elif "mla" in sub_lower:
            return "mla"
        elif "ffn" in sub_lower or "mlp" in sub_lower:
            return "ffn"
        elif "moe" in sub_lower:
            return "moe"
        elif "lm_head" in sub_lower or "output" in sub_lower:
            return "lm_head"
        elif "norm" in sub_lower:
            if "rms" in sub_lower:
                return "rms_norm"
            elif "layer" in sub_lower:
                return "layer_norm"
            else:
                return "rms_norm"
        elif "conv" in sub_lower:
            return "conv"
        elif "resblock" in sub_lower or "res" in sub_lower:
            return "resblock"
        else:
            return "unknown"

    def _estimate_submodule_time(self, sub_inst: ModuleInstance) -> float:
        """估算子模块计算时间（不包括通信）."""
        backend_config = BackendConfig(name="theory", device=self.device)
        backend = TheoryBackend(backend_config)
        return sub_inst.estimate_compute_time_only(backend)

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
        logger.info(
            f"[ANALYZE_START] workload={workload.name}, "
            f"phases=[{[p.name for p in workload.phases]}], "
            f"batch_size={resolved_params.get('batch_size')}, "
            f"seq_len={resolved_params.get('seq_len')}"
        )

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

        breakdown = self._generate_breakdown(phase_results, workload, total_time, throughput, comm_breakdown)
        detailed_breakdown = self._generate_detailed_breakdown(phase_results, workload, comm_breakdown)

        global_comm_bytes = 0
        if comm_breakdown:
            global_comm_bytes = comm_breakdown.total_bytes

        module_breakdown = generate_module_breakdown(phase_results, workload.workload_type.value, global_comm_bytes)

        total_kv_cache_gb = sum(
            p.memory_breakdown.get("kv_cache_gb", 0.0) for p in phase_results
        )

        topology_dict = None
        if hasattr(self.cluster, "topology") and self.cluster.topology:
            topology_dict = self.cluster.topology.to_dict()

        logger.debug(
            f"[UNIFIED_RESULT] weight_gb={detailed_breakdown['memory']['by_type']['weight']:.2f}"
        )
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
                "kv_cache_gb": total_kv_cache_gb,
                "topology": topology_dict,
            },
            breakdown=breakdown,
            detailed_breakdown=detailed_breakdown,
            mfu=mfu,
            qps=qps,
            communication_breakdown=comm_breakdown,
            module_breakdown=module_breakdown,
        )

    def _analyze_phases(
        self,
        workload: WorkloadConfig,
        params: Dict[str, Any],
    ) -> List[PhaseResult]:
        """Analyze all phases in a workload."""
        results = []

        for phase in workload.phases:
            logger.debug(
                f"[PHASE_START] phase={phase.name}, "
                f"component={phase.component}, "
                f"compute_type={phase.compute_type.value}, "
                f"repeat={phase.repeat}"
            )
            user_component_name = workload.resolve_component(phase.component)
            component = self._get_component(user_component_name)

            repeat_count = self._resolve_param(phase.repeat, params)
            seq_len_factor = self._resolve_param(phase.seq_len_factor, params)

            compute_pattern = phase.compute_pattern
            if compute_pattern is None:
                compute_pattern = self._infer_compute_pattern(component)

            single_time, memory, flops, submodules = self._estimate_phase(
                component, phase, compute_pattern, params, seq_len_factor, workload
            )

            framework_overhead_bytes = self._calculate_framework_overhead(phase, params, component)
            
            kv_cache_time_sec = 0.0
            if framework_overhead_bytes > 0:
                bandwidth_gbps = self.device.get_memory_bw_gbps()
                memory_bandwidth = bandwidth_gbps * 1e9
                kv_cache_time_sec = framework_overhead_bytes / memory_bandwidth
                logger.debug(
                    f"[KV_CACHE_TIME] phase={phase.name}, "
                    f"framework_overhead_bytes={framework_overhead_bytes:.2e}, "
                    f"bandwidth_gbps={bandwidth_gbps:.2f}, "
                    f"kv_cache_time_sec={kv_cache_time_sec:.6f}"
                )
            
            single_time += kv_cache_time_sec

            total_time = single_time * repeat_count
            framework_overhead_gb = framework_overhead_bytes / 1e9

            memory_breakdown = {
                "weight_gb": 0.0,
                "gradient_gb": 0.0,
                "optimizer_gb": 0.0,
                "activation_gb": memory,
                "kv_cache_gb": framework_overhead_gb,
                "kv_cache_time_sec": kv_cache_time_sec,
            }

            # Only add gradient/optimizer for training (backward phase)
            is_training_backward = workload.workload_type == WorkloadType.TRAINING and phase.compute_type in [
                ComputeType.BACKWARD,
                ComputeType.OPTIMIZER,
            ]

            for sub in submodules:
                memory_breakdown["weight_gb"] += sub.weight_memory_gb
                if is_training_backward:
                    memory_breakdown["gradient_gb"] += sub.gradient_memory_gb
                    memory_breakdown["optimizer_gb"] += sub.optimizer_memory_gb

            logger.debug(
                f"[PHASE_MEMORY] phase={phase.name}, "
                f"weight_gb={memory_breakdown['weight_gb']:.4f}, "
                f"num_submodules={len(submodules)}"
            )

            # Calculate total memory
            total_memory_gb = sum(memory_breakdown.values())

            logger.debug(
                f"[PHASE_RESULT] name={phase.name}, "
                f"memory_gb={total_memory_gb:.4f}, "
                f"weight_gb={memory_breakdown['weight_gb']:.4f}"
            )
            results.append(
                PhaseResult(
                    name=phase.name,
                    component=user_component_name,
                    compute_type=phase.compute_type,
                    single_time_sec=single_time,
                    repeat_count=repeat_count,
                    total_time_sec=total_time,
                    memory_gb=total_memory_gb,
                    memory_breakdown=memory_breakdown,
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

    def _set_kv_seq_len(self, component: ShardedModule, kv_seq_len: int) -> None:
        """Recursively set _kv_seq_len for all attention submodules.

        Args:
            component: ShardedModule (may contain attention submodules)
            kv_seq_len: KV sequence length for decode phase
        """
        from llm_perf.modeling.layers import ShardedAttention
        from llm_perf.modeling.mla import ShardedMLA

        if isinstance(component, (ShardedAttention, ShardedMLA)):
            component._kv_seq_len = kv_seq_len

        for submodule in component._submodules.values():
            self._set_kv_seq_len(submodule, kv_seq_len)

    def _calculate_framework_overhead(
        self,
        phase: Phase,
        params: Dict[str, Any],
        component: ShardedModule,
    ) -> float:
        """Calculate framework scheduling overhead (e.g., KV cache read/write).

        Only applies to decode phase in inference workloads.

        Returns:
            Framework overhead in bytes
        """
        if phase.name != "decode":
            return 0

        batch_size = params.get("batch_size", 1)
        prompt_len = params.get("prompt_len", 512)
        generated_tokens = params.get("generated_tokens", 0)

        kv_seq_len = prompt_len + generated_tokens

        num_heads = getattr(component, "num_heads", 32)
        num_kv_heads = getattr(component, "num_kv_heads", num_heads)
        head_dim = getattr(component, "head_dim", getattr(component, "hidden_size", 4096) // num_heads)
        num_layers = getattr(component, "num_layers", 32)
        dtype = getattr(component, "dtype", "fp16")

        dtype_size = DTYPE_SIZES.get(dtype, 2)

        kv_cache_read = batch_size * kv_seq_len * num_kv_heads * head_dim * dtype_size * 2
        kv_cache_write = batch_size * 1 * num_kv_heads * head_dim * dtype_size * 2

        total_overhead = (kv_cache_read + kv_cache_write) * num_layers

        return total_overhead

    def _compute_structure_signature(self, sub_inst: ModuleInstance) -> str:
        """计算子模块结构签名（类名+关键配置参数）.

        相同签名的子模块具有相同的计算特征，评估一次即可复用结果。
        """
        module = sub_inst.module
        class_name = type(module).__name__

        config_attrs = [
            "hidden_size",
            "num_heads",
            "head_dim",
            "intermediate_size",
            "num_experts",
            "num_experts_per_token",
            "vocab_size",
            "embedding_dim",
            "dtype",
            "num_key_value_heads",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "v_head_dim",
            "layer_type",
        ]
        config_values = []
        for attr in config_attrs:
            if hasattr(module, attr):
                val = getattr(module, attr)
                config_values.append(f"{attr}={val}")

        return f"{class_name}:{','.join(config_values)}"

    def _analyze_phase_with_submodules(
        self,
        component: ShardedModule,
        phase: Phase,
        params: Dict[str, Any],
        workload: WorkloadConfig,
    ) -> Tuple[float, float, int, List[SubmoduleResult]]:
        """使用ShardedModule.bind()机制分析phase和子模块.

        支持嵌套子模块分析：transformer_block 内部分解为 attention 和 ffn。
        """
        ctx = self._create_parallel_context(params)
        batch_size = params.get("batch_size", 1)

        hidden_size = getattr(component, "hidden_size", 4096)
        compute_pattern = phase.compute_pattern or self._infer_compute_pattern(component)

        is_diffusion_model = hasattr(component, "image_height") and hasattr(component, "image_width")

        if phase.name == "prefill":
            seq_len = params.get("prompt_len", params.get("seq_len", 512))
        elif phase.name == "decode":
            seq_len = 1
            prompt_len = params.get("prompt_len", 512)
            generated_tokens = params.get("generated_tokens", 0)
            kv_seq_len = prompt_len + generated_tokens
        elif is_diffusion_model and compute_pattern == ComputePattern.TRANSFORMER_BLOCK:
            height = params.get("height", 720)
            width = params.get("width", 1280)
            vae_compression_ratio = 16
            latent_h = height // vae_compression_ratio
            latent_w = width // vae_compression_ratio
            seq_len = latent_h * latent_w
        else:
            seq_len = params.get("seq_len", params.get("prompt_len", 512))

        if phase.name == "decode":
            self._set_kv_seq_len(component, kv_seq_len)

        logger.debug(
            f"[SUBMODULE_ANALYZE] tp={ctx.tp_degree}, pp={ctx.pp_degree}, "
            f"dp={ctx.dp_degree}, ep={ctx.ep_degree}, dtype={ctx.dtype}, "
            f"batch_size={batch_size}, seq_len={seq_len}, phase={phase.name}, "
            f"is_diffusion_model={is_diffusion_model}"
        )

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
            input_tensor = ShardedTensor(shape=(batch_size, seq_len))
        elif is_diffusion_model:
            latent_channels = getattr(component, "latent_channels", 16)
            input_tensor = ShardedTensor(shape=(batch_size, seq_len, latent_channels))
        else:
            input_tensor = ShardedTensor(shape=(batch_size, seq_len, hidden_size))

        try:
            if hasattr(component, "text_dim") and hasattr(component, "freq_dim"):
                text_dim = component.text_dim
                freq_dim = component.freq_dim
                text_embed = ShardedTensor(shape=(batch_size, 256, text_dim))
                time_embed = ShardedTensor(shape=(batch_size, freq_dim))
                component(input_tensor, text_embed, time_embed)
            elif is_diffusion_model:
                timestep_dim = getattr(component, "freq_dim", None)
                if timestep_dim is None and hasattr(component, "timestep_in_weight"):
                    timestep_dim = component.timestep_in_weight.shape[0]
                if timestep_dim is None:
                    timestep_dim = 256
                timestep = ShardedTensor(shape=(batch_size, timestep_dim))
                component(input_tensor, timestep)
            else:
                component(input_tensor)
        except Exception:
            pass

        backend_config = BackendConfig(name="theory", device=self.device)
        backend = TheoryBackend(backend_config)

        is_training = workload.workload_type == WorkloadType.TRAINING

        # For time estimation:
        # - forward phase: use forward mode (time = forward_time)
        # - backward phase: use forward_backward mode (time = forward + backward = 3x forward_time)
        time_mode = (
            "forward_backward" if phase.compute_type in [ComputeType.BACKWARD, ComputeType.OPTIMIZER] else "forward"
        )

        # For memory estimation (training workload):
        # - forward phase: use forward_backward mode to accumulate all activations for backward pass
        # - backward/optimizer phase: same as time_mode
        memory_mode = (
            "forward_backward"
            if is_training or phase.compute_type in [ComputeType.BACKWARD, ComputeType.OPTIMIZER]
            else "forward"
        )

        # Bind for time estimation
        time_instance = component.bind(ctx, mode=time_mode)
        time_sec = time_instance.estimate_time(backend)

        # Bind for memory/flops estimation (may use different mode)
        module_instance = component.bind(ctx, mode=memory_mode)

        if phase.compute_type == ComputeType.FORWARD:
            flops = module_instance.flops_forward_physical
        elif phase.compute_type == ComputeType.BACKWARD:
            flops = module_instance.flops_backward_physical
        else:
            flops = 0

        memory = module_instance.activation_memory_physical / 1e9

        signature_groups: Dict[str, List[Tuple[str, ModuleInstance]]] = {}
        for sub_name, sub_inst in module_instance._submodule_instances.items():
            sig = self._compute_structure_signature(sub_inst)
            if sig not in signature_groups:
                signature_groups[sig] = []
            signature_groups[sig].append((sub_name, sub_inst))

        logger.debug(
            f"[STRUCTURE_GROUPS] {len(signature_groups)} unique structures, "
            f"total {len(module_instance._submodule_instances)} submodules"
        )

        signature_cache: Dict[str, SubmoduleResult] = {}
        for sig, instances in signature_groups.items():
            first_name, first_inst = instances[0]
            cached = self._evaluate_single_submodule(first_name, first_inst, phase)
            signature_cache[sig] = cached
            logger.debug(
                f"[STRUCTURE_CACHE] signature={sig}, instances={len(instances)}, first={first_name}"
            )

        submodules = []
        for sig, instances in signature_groups.items():
            cached = signature_cache[sig]
            for sub_name, sub_inst in instances:
                submodules.append(
                    SubmoduleResult(
                        name=sub_name,
                        submodule_type=cached.submodule_type,
                        time_sec=cached.time_sec,
                        flops=cached.flops,
                        count=cached.count,
                        params_count=cached.params_count,
                        weight_memory_gb=cached.weight_memory_gb,
                        gradient_memory_gb=cached.gradient_memory_gb,
                        optimizer_memory_gb=cached.optimizer_memory_gb,
                        activation_memory_gb=cached.activation_memory_gb,
                        communication_bytes=cached.communication_bytes,
                        communication_time_sec=cached.communication_time_sec,
                        comm_ops_detail=cached.comm_ops_detail,
                        nested_submodules=cached.nested_submodules,
                    )
                )
                logger.debug(
                    f"[SUBMODULE] name={sub_name}, type={cached.submodule_type}, "
                    f"params_physical={cached.params_count / 1e9:.4f}B, "
                    f"weight_gb={cached.weight_memory_gb:.4f}GB, "
                    f"flops={cached.flops / 1e12:.4f}T"
                )

        logger.debug(
            f"[PHASE_ANALYSIS] phase_name={phase.component}, "
            f"compute_type={phase.compute_type.value}, "
            f"total_flops={flops / 1e12:.4f}T, "
            f"time_sec={time_sec:.4f}, "
            f"memory_gb={memory:.4f}, "
            f"num_submodules={len(submodules)}"
        )
        return time_sec, memory, flops, submodules

    def _evaluate_single_submodule(
        self,
        sub_name: str,
        sub_inst: ModuleInstance,
        phase: Phase,
    ) -> SubmoduleResult:
        """评估单个子模块实例."""
        delta_comm_bytes = 0
        comm_ops_detail = []
        comm_time_sec = 0.0
        parallel_degrees = self._get_parallel_degrees()
        
        comm_ops = sub_inst.total_comm_ops
        for comm_op in comm_ops:
            ptype = getattr(comm_op, "ptype", "unknown")
            comm_ops_detail.append(
                {
                    "comm_type": comm_op.comm_type,
                    "ptype": ptype,
                    "data_bytes": comm_op.data_bytes,
                }
            )
            delta_comm_bytes += comm_op.data_bytes
            degree = parallel_degrees.get(ptype, 1)
            bandwidth_gbps = self._get_bandwidth_for_ptype(ptype)
            comm_time_sec += self._estimate_single_comm_time(
                comm_op.comm_type, comm_op.data_bytes, degree, bandwidth_gbps
            )

        submodule_type = self._infer_submodule_type(sub_name, sub_inst)

        nested_submodules = []
        if submodule_type == "transformer_block":
            nested_submodules = self._analyze_nested_submodules(sub_inst, phase)

        return SubmoduleResult(
            name=sub_name,
            submodule_type=submodule_type,
            time_sec=self._estimate_submodule_time(sub_inst),
            flops=sub_inst.flops_forward_physical,
            params_count=sub_inst.params_count_physical,
            weight_memory_gb=sub_inst.weight_memory_physical / 1e9,
            gradient_memory_gb=sub_inst.gradient_memory_physical / 1e9,
            optimizer_memory_gb=sub_inst.optimizer_memory_physical / 1e9,
            activation_memory_gb=sub_inst.activation_memory_physical / 1e9,
            communication_bytes=delta_comm_bytes,
            communication_time_sec=comm_time_sec,
            comm_ops_detail=comm_ops_detail,
            nested_submodules=nested_submodules,
        )

    def _analyze_nested_submodules(
        self,
        parent_inst: ModuleInstance,
        phase: Phase,
    ) -> List["SubmoduleResult"]:
        """分析嵌套子模块（如 transformer_block 内部的 attention, ffn, moe）."""
        from .base import SubmoduleResult

        nested_items = [
            (sub_name, sub_inst)
            for sub_name, sub_inst in parent_inst._submodule_instances.items()
            if self._is_nested_submodule(sub_inst)
        ]

        if not nested_items:
            return []

        signature_groups: Dict[str, List[Tuple[str, ModuleInstance]]] = {}
        for sub_name, sub_inst in nested_items:
            sig = self._compute_structure_signature(sub_inst)
            if sig not in signature_groups:
                signature_groups[sig] = []
            signature_groups[sig].append((sub_name, sub_inst))

        logger.debug(
            f"[NESTED_GROUPS] {len(signature_groups)} unique structures, "
            f"total {len(nested_items)} nested submodules"
        )

        signature_cache: Dict[str, SubmoduleResult] = {}
        for sig, instances in signature_groups.items():
            first_name, first_inst = instances[0]
            delta_comm_bytes = sum(op.data_bytes for op in first_inst.total_comm_ops)

            nested_type = self._infer_submodule_type(first_name, first_inst)

            signature_cache[sig] = SubmoduleResult(
                name=first_name,
                submodule_type=nested_type,
                time_sec=self._estimate_submodule_time(first_inst),
                flops=first_inst.flops_forward_physical,
                count=1,
                params_count=first_inst.params_count_physical,
                weight_memory_gb=first_inst.weight_memory_physical / 1e9,
                gradient_memory_gb=first_inst.gradient_memory_physical / 1e9,
                optimizer_memory_gb=first_inst.optimizer_memory_physical / 1e9,
                activation_memory_gb=first_inst.activation_memory_physical / 1e9,
                communication_bytes=delta_comm_bytes,
                nested_submodules=[],
            )
            logger.debug(f"[NESTED_CACHE] signature={sig}, instances={len(instances)}, first={first_name}, type={nested_type}")

        nested = []
        for sig, instances in signature_groups.items():
            cached = signature_cache[sig]
            for sub_name, sub_inst in instances:
                nested.append(
                    SubmoduleResult(
                        name=sub_name,
                        submodule_type=cached.submodule_type,
                        time_sec=cached.time_sec,
                        flops=cached.flops,
                        count=cached.count,
                        params_count=cached.params_count,
                        weight_memory_gb=cached.weight_memory_gb,
                        gradient_memory_gb=cached.gradient_memory_gb,
                        optimizer_memory_gb=cached.optimizer_memory_gb,
                        activation_memory_gb=cached.activation_memory_gb,
                        communication_bytes=cached.communication_bytes,
                        nested_submodules=[],
                    )
                )

        return nested

    def _is_nested_submodule(self, sub_inst: ModuleInstance) -> bool:
        """判断是否为嵌套子模块（attention, ffn, moe, linear_attention, mla等）."""
        if hasattr(sub_inst, "module") and sub_inst.module:
            module = sub_inst.module
            
            if hasattr(module, "_submodule_name") and module._submodule_name:
                nested_types = ["attention", "linear_attention", "mla", "ffn", "moe"]
                return module._submodule_name in nested_types
            
            module_class = type(module).__name__
            inferred_name = infer_submodule_name_from_class(module_class)
            nested_types = ["attention", "linear_attention", "mla", "ffn", "moe"]
            return inferred_name in nested_types
        
        return False

    def _estimate_phase(
        self,
        component: ShardedModule,
        phase: Phase,
        compute_pattern: ComputePattern,
        params: Dict[str, Any],
        seq_len_factor: float,
        workload: WorkloadConfig,
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
                component, phase, params, workload
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

        logger.debug(
            f"[PHASE] phase={phase.name}, compute_type={phase.compute_type}, "
            f"time_sec={time_sec:.4f}s, memory_gb={memory_gb:.4f}GB, "
            f"flops={flops / 1e12:.4f}T, submodules_count={len(submodules)}"
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
        params_per_device = self._count_params(component) // tp // dp
        params_memory = params_per_device * dtype_bytes

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
        params_per_device = self._count_params(component) // tp // dp
        params_memory = params_per_device * dtype_bytes

        activations = batch_size * seq_len * hidden_size * dtype_bytes * num_layers * 4 // tp
        gradients = params_per_device * dtype_bytes

        return (params_memory + activations + gradients) / 1e9

    def _estimate_optimizer_memory(
        self,
        dtype_bytes: float,
        tp: int,
        dp: int,
        component: ShardedModule,
    ) -> float:
        """Estimate optimizer memory."""
        params_per_device = self._count_params(component) // tp // dp
        params_memory = params_per_device * dtype_bytes

        optimizer_states = params_per_device * dtype_bytes * 2

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
        """从PhaseResult.submodules的comm_ops_detail提取通信分解，按并行方式和原语双层组织.

        包含 bytes 和 time 的双重细分，便于前端展示不同原语类型的通信量和时间。
        """
        by_parallelism: Dict[str, Dict[str, Any]] = {}
        by_operation: Dict[str, Dict[str, Any]] = {}
        total_bytes = 0
        total_time_sec = 0.0

        parallel_degrees = self._get_parallel_degrees()

        for phase in phase_results:
            for sm in phase.submodules:
                for op_detail in sm.comm_ops_detail:
                    ptype = op_detail["ptype"]
                    comm_type = op_detail["comm_type"]
                    data_bytes = op_detail["data_bytes"]

                    degree = parallel_degrees.get(ptype, 1)
                    bandwidth_gbps = self._get_bandwidth_for_ptype(ptype)
                    comm_time = self._estimate_single_comm_time(comm_type, data_bytes, degree, bandwidth_gbps)

                    if ptype not in by_parallelism:
                        by_parallelism[ptype] = {"total_bytes": 0, "total_time_sec": 0.0, "operations": {}}
                    by_parallelism[ptype]["total_bytes"] += data_bytes
                    by_parallelism[ptype]["total_time_sec"] += comm_time

                    if comm_type not in by_parallelism[ptype]["operations"]:
                        by_parallelism[ptype]["operations"][comm_type] = {"total_bytes": 0, "total_time_sec": 0.0}
                    by_parallelism[ptype]["operations"][comm_type]["total_bytes"] += data_bytes
                    by_parallelism[ptype]["operations"][comm_type]["total_time_sec"] += comm_time

                    if comm_type not in by_operation:
                        by_operation[comm_type] = {"total_bytes": 0, "total_time_sec": 0.0, "by_ptype": {}}
                    by_operation[comm_type]["total_bytes"] += data_bytes
                    by_operation[comm_type]["total_time_sec"] += comm_time

                    if ptype not in by_operation[comm_type]["by_ptype"]:
                        by_operation[comm_type]["by_ptype"][ptype] = {"total_bytes": 0, "total_time_sec": 0.0}
                    by_operation[comm_type]["by_ptype"][ptype]["total_bytes"] += data_bytes
                    by_operation[comm_type]["by_ptype"][ptype]["total_time_sec"] += comm_time

                    total_bytes += data_bytes
                    total_time_sec += comm_time

        return CommunicationBreakdown(
            by_parallelism=by_parallelism,
            by_operation=by_operation,
            total_bytes=total_bytes,
            total_time_sec=total_time_sec,
        )

    def _estimate_single_comm_time(
        self, comm_type: str, data_bytes: int, num_ranks: int, bandwidth_gbps: float
    ) -> float:
        """Estimate time for a single communication operation."""
        import math

        if num_ranks <= 1:
            return 0.0

        bandwidth_bytes_per_sec = bandwidth_gbps * 1e9

        comm_type_lower = comm_type.lower()
        if comm_type_lower == "allreduce":
            steps = math.ceil(math.log2(num_ranks))
            return steps * data_bytes / bandwidth_bytes_per_sec
        elif comm_type_lower == "allgather":
            steps = math.ceil(math.log2(num_ranks))
            return steps * data_bytes / bandwidth_bytes_per_sec
        elif comm_type_lower == "alltoall":
            return data_bytes / bandwidth_bytes_per_sec
        elif comm_type_lower == "reduce_scatter":
            return num_ranks * data_bytes / bandwidth_bytes_per_sec / 2
        elif comm_type_lower == "broadcast":
            steps = math.ceil(math.log2(num_ranks))
            return steps * data_bytes / bandwidth_bytes_per_sec
        elif comm_type_lower == "p2p":
            return data_bytes / bandwidth_bytes_per_sec
        else:
            return data_bytes / bandwidth_bytes_per_sec

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

    def _get_bandwidth_for_ptype(self, ptype: str) -> float:
        """Get bandwidth for a parallel type.

        Args:
            ptype: Parallel type (tp, dp, ep, sp, pp)

        Returns:
            Bandwidth in GB/s
        """
        intra_node_bw = 400.0
        inter_node_bw = 100.0

        if hasattr(self.cluster, "topology"):
            topology = self.cluster.topology
            for level in topology.levels:
                if level.name == "node" or "intra" in level.name.lower():
                    intra_node_bw = level.bandwidth_gbps
                elif level.name == "inter_node" or "inter" in level.name.lower():
                    inter_node_bw = level.bandwidth_gbps

        elif hasattr(self.cluster, "intra_node_bandwidth_gbps"):
            intra_node_bw = self.cluster.intra_node_bandwidth_gbps
            if hasattr(self.cluster, "inter_node_bandwidth_gbps"):
                inter_node_bw = self.cluster.inter_node_bandwidth_gbps

        devices_per_node = getattr(self.cluster, "devices_per_node", 8)
        degree = self._get_parallel_degrees().get(ptype, 1)

        if degree <= devices_per_node:
            return intra_node_bw
        else:
            return inter_node_bw

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
        comm_breakdown: Optional[CommunicationBreakdown] = None,
    ) -> Dict[str, Any]:
        """Generate legacy breakdown format for frontend compatibility."""
        compute_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.FORWARD)
        backward_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.BACKWARD)
        optimizer_time = sum(p.total_time_sec for p in phases if p.compute_type == ComputeType.OPTIMIZER)
        communication_time = comm_breakdown.total_time_sec if comm_breakdown else 0.0
        effective_total_time = total_time + communication_time
        memory_time = max(0, effective_total_time - compute_time - backward_time - optimizer_time - communication_time)
        compute_percent = compute_time / effective_total_time * 100 if effective_total_time > 0 else 0
        backward_percent = backward_time / effective_total_time * 100 if effective_total_time > 0 else 0
        optimizer_percent = optimizer_time / effective_total_time * 100 if effective_total_time > 0 else 0
        communication_percent = communication_time / effective_total_time * 100 if effective_total_time > 0 else 0
        memory_percent = memory_time / effective_total_time * 100 if effective_total_time > 0 else 0

        inference_breakdown = None
        if workload.workload_type == WorkloadType.INFERENCE:
            prefill_phase = next((p for p in phases if p.name == "prefill"), None)
            decode_phase = next((p for p in phases if p.name == "decode"), None)
            prefill_time = prefill_phase.total_time_sec if prefill_phase else 0.0
            decode_time = decode_phase.total_time_sec if decode_phase else 0.0
            decode_per_token_time = decode_phase.single_time_sec if decode_phase else 0.0
            kv_cache_time_sec = 0.0
            for p in phases:
                kv_cache_time_sec += p.memory_breakdown.get("kv_cache_time_sec", 0.0)
            inference_effective_time = prefill_time + decode_time + communication_time + kv_cache_time_sec
            prefill_percent = prefill_time / inference_effective_time * 100 if inference_effective_time > 0 else 0
            decode_percent = decode_time / inference_effective_time * 100 if inference_effective_time > 0 else 0
            kv_cache_percent = kv_cache_time_sec / inference_effective_time * 100 if inference_effective_time > 0 else 0
            inf_comm_percent = communication_time / inference_effective_time * 100 if inference_effective_time > 0 else 0
            inference_breakdown = {
                "prefill_sec": prefill_time,
                "prefill_percent": prefill_percent,
                "decode_sec": decode_time,
                "decode_percent": decode_percent,
                "decode_per_token_sec": decode_per_token_time,
                "kv_cache_sec": kv_cache_time_sec,
                "kv_cache_percent": kv_cache_percent,
                "communication_sec": communication_time,
                "communication_percent": inf_comm_percent,
            }

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
                        "params_count": sm.params_count,
                        "memory": {
                            "weight_gb": sm.weight_memory_gb,
                            "gradient_gb": sm.gradient_memory_gb,
                            "optimizer_gb": sm.optimizer_memory_gb,
                            "activation_gb": sm.activation_memory_gb,
                            "total_gb": (
                                sm.weight_memory_gb
                                + sm.gradient_memory_gb
                                + sm.optimizer_memory_gb
                                + sm.activation_memory_gb
                            ),
                        },
                        "communication_gb": sm.communication_bytes / 1e9,
                    }
                )

        result = {
            "overview": {
                "total_time_sec": total_time,
                "throughput": sum(throughput.values()) if throughput else 0,
            },
            "time_breakdown": {
                "compute_sec": compute_time,
                "backward_sec": backward_time,
                "optimizer_sec": optimizer_time,
                "communication_sec": communication_time,
                "memory_sec": memory_time,
                "compute_percent": compute_percent,
                "backward_percent": backward_percent,
                "optimizer_percent": optimizer_percent,
                "communication_percent": communication_percent,
                "memory_percent": memory_percent,
            },
            "layers": layers,
            "submodules": submodule_breakdown,
        }
        if inference_breakdown:
            result["inference_breakdown"] = inference_breakdown
        return result

    def _generate_detailed_breakdown(
        self,
        phases: List[PhaseResult],
        workload: WorkloadConfig,
        comm_breakdown: Optional[CommunicationBreakdown] = None,
    ) -> Dict[str, Any]:
        """Generate detailed breakdown format for frontend compatibility.

        All memory metrics are per-device.

        Structure:
        - by_submodule_type: aggregated by module type (embedding, transformer_block, lm_head, etc.)
          each entry contains: memory (weight/gradient/optimizer/activation), compute, communication
          Note: rms_norm is merged into subsequent module (e.g., final_norm -> lm_head)
        """
        from .breakdown import _merge_norm_submodules

        # Find peak memory phase for memory display
        peak_phase = max(phases, key=lambda p: p.memory_gb) if phases else None

        # Aggregate by submodule type
        all_submodules = []
        for phase in phases:
            all_submodules.extend(phase.submodules)
        merged_submodules = _merge_norm_submodules(all_submodules)
        logger.debug(
            f"[MERGE_NORM] original_count={len(all_submodules)}, "
            f"merged_count={len(merged_submodules)}"
        )

        peak_submodules = _merge_norm_submodules(peak_phase.submodules) if peak_phase else []

        by_submodule_type: Dict[str, Dict[str, Any]] = {}
        by_nested_type: Dict[str, Dict[str, Any]] = {}

        for sm in merged_submodules:
            block_type = sm.submodule_type
            if block_type not in by_submodule_type:
                by_submodule_type[block_type] = {
                    "memory": {
                        "params_count": 0,
                        "weight_gb": 0.0,
                        "gradient_gb": 0.0,
                        "optimizer_gb": 0.0,
                        "activation_gb": 0.0,
                    },
                    "compute": {"flops": 0, "flops_gflops": 0.0, "time_sec": 0.0},
                    "communication": {"bytes": 0, "gb": 0.0, "time_sec": 0.0},
                    "nested_breakdown": {},
                }

            # Memory: use peak phase submodules
            peak_sms = [s for s in peak_submodules if s.submodule_type == block_type]
            if peak_sms and by_submodule_type[block_type]["memory"]["activation_gb"] == 0:
                by_submodule_type[block_type]["memory"]["params_count"] = sum(s.params_count for s in peak_sms)
                by_submodule_type[block_type]["memory"]["weight_gb"] = sum(s.weight_memory_gb for s in peak_sms)
                by_submodule_type[block_type]["memory"]["gradient_gb"] = sum(s.gradient_memory_gb for s in peak_sms)
                by_submodule_type[block_type]["memory"]["optimizer_gb"] = sum(s.optimizer_memory_gb for s in peak_sms)
                by_submodule_type[block_type]["memory"]["activation_gb"] = sum(s.activation_memory_gb for s in peak_sms)
                logger.debug(
                    f"[BY_TYPE] block_type={block_type}, "
                    f"count={len(peak_sms)}, "
                    f"weight_gb={by_submodule_type[block_type]['memory']['weight_gb']:.4f}GB, "
                    f"activation_gb={by_submodule_type[block_type]['memory']['activation_gb']:.4f}GB"
                )

            # Compute/communication: sum all phases
            by_submodule_type[block_type]["compute"]["flops"] += sm.flops
            by_submodule_type[block_type]["compute"]["flops_gflops"] += sm.flops / 1e9
            by_submodule_type[block_type]["compute"]["time_sec"] += sm.time_sec
            by_submodule_type[block_type]["communication"]["bytes"] += sm.communication_bytes
            by_submodule_type[block_type]["communication"]["gb"] += sm.communication_bytes / 1e9
            by_submodule_type[block_type]["communication"]["time_sec"] += sm.communication_time_sec

            # Handle nested submodules
            if sm.nested_submodules:
                peak_nested_all = []
                for peak_s in peak_sms:
                    if peak_s.nested_submodules:
                        peak_nested_all.extend(peak_s.nested_submodules)

                for nested in sm.nested_submodules:
                    nested_type = nested.submodule_type

                    if nested_type not in by_submodule_type[block_type]["nested_breakdown"]:
                        by_submodule_type[block_type]["nested_breakdown"][nested_type] = {
                            "memory": {
                                "params_count": 0,
                                "weight_gb": 0.0,
                                "gradient_gb": 0.0,
                                "optimizer_gb": 0.0,
                                "activation_gb": 0.0,
                            },
                            "compute": {"flops": 0, "time_sec": 0.0},
                            "communication": {"bytes": 0, "gb": 0.0, "time_sec": 0.0},
                        }

                    peak_nested_sms = [n for n in peak_nested_all if n.submodule_type == nested_type]
                    if (
                        peak_nested_sms
                        and by_submodule_type[block_type]["nested_breakdown"][nested_type]["memory"]["activation_gb"]
                        == 0
                    ):
                        by_submodule_type[block_type]["nested_breakdown"][nested_type]["memory"]["params_count"] = sum(
                            n.params_count for n in peak_nested_sms
                        )
                        by_submodule_type[block_type]["nested_breakdown"][nested_type]["memory"]["weight_gb"] = sum(
                            n.weight_memory_gb for n in peak_nested_sms
                        )
                        by_submodule_type[block_type]["nested_breakdown"][nested_type]["memory"]["gradient_gb"] = sum(
                            n.gradient_memory_gb for n in peak_nested_sms
                        )
                        by_submodule_type[block_type]["nested_breakdown"][nested_type]["memory"]["optimizer_gb"] = sum(
                            n.optimizer_memory_gb for n in peak_nested_sms
                        )
                        by_submodule_type[block_type]["nested_breakdown"][nested_type]["memory"]["activation_gb"] = sum(
                            n.activation_memory_gb for n in peak_nested_sms
                        )

                    by_submodule_type[block_type]["nested_breakdown"][nested_type]["compute"]["flops"] += nested.flops
                    by_submodule_type[block_type]["nested_breakdown"][nested_type]["compute"]["time_sec"] += (
                        nested.time_sec
                    )
                    by_submodule_type[block_type]["nested_breakdown"][nested_type]["communication"]["bytes"] += (
                        nested.communication_bytes
                    )
                    by_submodule_type[block_type]["nested_breakdown"][nested_type]["communication"]["gb"] += (
                        nested.communication_bytes / 1e9
                    )
                    by_submodule_type[block_type]["nested_breakdown"][nested_type]["communication"]["time_sec"] += (
                        nested.communication_time_sec
                    )

                    if nested_type not in by_nested_type:
                        by_nested_type[nested_type] = {
                            "memory": {
                                "params_count": 0,
                                "weight_gb": 0.0,
                                "gradient_gb": 0.0,
                                "optimizer_gb": 0.0,
                                "activation_gb": 0.0,
                            },
                            "compute": {"flops": 0, "flops_gflops": 0.0, "time_sec": 0.0},
                            "communication": {"bytes": 0, "gb": 0.0, "time_sec": 0.0},
                            "parent_type": block_type,
                        }

                    if peak_nested_sms and by_nested_type[nested_type]["memory"]["activation_gb"] == 0:
                        by_nested_type[nested_type]["memory"]["params_count"] = sum(
                            n.params_count for n in peak_nested_sms
                        )
                        by_nested_type[nested_type]["memory"]["weight_gb"] = sum(
                            n.weight_memory_gb for n in peak_nested_sms
                        )
                        by_nested_type[nested_type]["memory"]["gradient_gb"] = sum(
                            n.gradient_memory_gb for n in peak_nested_sms
                        )
                        by_nested_type[nested_type]["memory"]["optimizer_gb"] = sum(
                            n.optimizer_memory_gb for n in peak_nested_sms
                        )
                        by_nested_type[nested_type]["memory"]["activation_gb"] = sum(
                            n.activation_memory_gb for n in peak_nested_sms
                        )

                    by_nested_type[nested_type]["compute"]["flops"] += nested.flops
                    by_nested_type[nested_type]["compute"]["flops_gflops"] += nested.flops / 1e9
                    by_nested_type[nested_type]["compute"]["time_sec"] += nested.time_sec
                    by_nested_type[nested_type]["communication"]["bytes"] += nested.communication_bytes
                    by_nested_type[nested_type]["communication"]["gb"] += nested.communication_bytes / 1e9
                    by_nested_type[nested_type]["communication"]["time_sec"] += nested.communication_time_sec

        # Calculate total memory breakdown
        # Activation should be aggregated from phase-level (memory_breakdown["activation_gb"]),
        # NOT from submodule-level, because forward/inference mode submodules have 0 activation_memory_gb
        total_params_count = sum(data["memory"]["params_count"] for data in by_submodule_type.values())
        phase_activation_total = sum(
            phase.memory_breakdown.get("activation_gb", 0.0) for phase in phases
        )
        total_memory_breakdown = {
            "params_count": total_params_count,
            "params_count_billion": total_params_count / 1e9,
            "weight_gb": sum(data["memory"]["weight_gb"] for data in by_submodule_type.values()),
            "gradient_gb": sum(data["memory"]["gradient_gb"] for data in by_submodule_type.values()),
            "optimizer_gb": sum(data["memory"]["optimizer_gb"] for data in by_submodule_type.values()),
            "activation_gb": phase_activation_total,
            "activations_gb": phase_activation_total,
            "total_gb": 0,
        }
        total_memory_gb = (
            total_memory_breakdown["weight_gb"]
            + total_memory_breakdown["gradient_gb"]
            + total_memory_breakdown["optimizer_gb"]
            + total_memory_breakdown["activation_gb"]
        )
        total_memory_breakdown["total_gb"] = total_memory_gb
        
        # Build by_phase_activation: activation breakdown by phase
        by_phase_activation = {}
        for phase in phases:
            phase_activation = phase.memory_breakdown.get("activation_gb", 0.0)
            if phase_activation > 0:
                phase_name = phase.name
                if phase_name not in by_phase_activation:
                    by_phase_activation[phase_name] = {
                        "activation_gb": 0.0,
                        "component": phase.component,
                        "compute_type": phase.compute_type.value,
                    }
                by_phase_activation[phase_name]["activation_gb"] += phase_activation

        total_compute_flops = sum(data["compute"]["flops"] for data in by_submodule_type.values())
        total_comm_bytes = sum(data["communication"]["bytes"] for data in by_submodule_type.values())
        logger.debug(
            f"[TOTAL] weight_gb={total_memory_breakdown['weight_gb']:.2f}GB, "
            f"gradient_gb={total_memory_breakdown['gradient_gb']:.2f}GB, "
            f"optimizer_gb={total_memory_breakdown['optimizer_gb']:.2f}GB, "
            f"activation_gb={total_memory_breakdown['activation_gb']:.2f}GB, "
            f"total_gb={total_memory_breakdown['total_gb']:.2f}GB, "
            f"params_count={total_memory_breakdown['params_count_billion']:.4f}B, "
            f"compute_flops={total_compute_flops / 1e12:.4f}T, "
            f"comm_bytes={total_comm_bytes / 1e9:.4f}GB, "
            f"num_phases={len(phases)}, "
            f"num_submodule_types={len(by_submodule_type)}"
        )

        # Add backward compat to by_submodule_type and nested_breakdown
        for block_type, data in by_submodule_type.items():
            data["memory"]["activations_gb"] = data["memory"]["activation_gb"]
            for nested_type, nested_data in data.get("nested_breakdown", {}).items():
                nested_data["memory"]["activations_gb"] = nested_data["memory"].get("activation_gb", 0)

        comm_breakdown_dict = {}
        if comm_breakdown:
            comm_breakdown_dict = comm_breakdown.to_dict()

        return {
            "submodels": [
                {
                    "model_name": phase.component,
                    "model_type": phase.compute_type.value,
                    "compute_time_sec": phase.total_time_sec,
                    "memory": {
                        "summary": phase.memory_breakdown,
                        "total_gb": phase.memory_gb,
                    },
                }
                for phase in phases
            ],
            "memory": {
                "summary": total_memory_breakdown,
                "total_gb": total_memory_gb,
                "by_type": {
                    "weight": total_memory_breakdown["weight_gb"],
                    "gradient": total_memory_breakdown["gradient_gb"],
                    "optimizer": total_memory_breakdown["optimizer_gb"],
                    "activation": total_memory_breakdown["activation_gb"],
                    "total": total_memory_breakdown["total_gb"],
                },
                "by_submodel": _aggregate_submodel_memory(phases),
                "by_submodule_type": by_submodule_type,
                "by_phase_activation": by_phase_activation,
            },
            "compute": {
                "by_submodule_type": {
                    block_type: {
                        "flops": data["compute"]["flops"],
                        "flops_gflops": data["compute"]["flops_gflops"],
                        "time_sec": data["compute"]["time_sec"],
                    }
                    for block_type, data in by_submodule_type.items()
                },
                "by_nested_type": {
                    nested_type: {
                        "flops": data["compute"]["flops"],
                        "flops_gflops": data["compute"]["flops_gflops"],
                        "time_sec": data["compute"]["time_sec"],
                        "parent_type": data["parent_type"],
                    }
                    for nested_type, data in by_nested_type.items()
                },
            },
            "communication": {
                "by_submodule_type": {
                    block_type: {
                        "bytes": data["communication"]["bytes"],
                        "gb": data["communication"]["gb"],
                    }
                    for block_type, data in by_submodule_type.items()
                },
                "by_nested_type": {
                    nested_type: {
                        "bytes": data["communication"]["bytes"],
                        "gb": data["communication"]["gb"],
                        "parent_type": data["parent_type"],
                    }
                    for nested_type, data in by_nested_type.items()
                },
                "by_parallelism": comm_breakdown_dict.get("by_parallelism", {}),
                "by_operation": comm_breakdown_dict.get("by_operation", {}),
                "total_bytes": comm_breakdown_dict.get("total_bytes", 0),
            },
            "by_submodule_type": by_submodule_type,
            "by_nested_type": by_nested_type,
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

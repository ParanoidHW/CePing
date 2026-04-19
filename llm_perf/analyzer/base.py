"""Unified Analyzer base classes and data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union, Optional


class ComputeType(str, Enum):
    """Types of compute operations."""

    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    COMMUNICATION = "communication"


class WorkloadType(str, Enum):
    """Types of workloads."""

    TRAINING = "training"
    INFERENCE = "inference"
    MIXED = "mixed"


class ThroughputMetric(str, Enum):
    """Types of throughput metrics."""

    TOKENS_PER_SEC = "tokens_per_sec"
    SAMPLES_PER_SEC = "samples_per_sec"
    PIXELS_PER_SEC = "pixels_per_sec"
    IMAGES_PER_SEC = "images_per_sec"
    VIDEOS_PER_SEC = "videos_per_sec"


class ComputePattern(str, Enum):
    """Compute patterns - describes computation characteristics, not model types.

    Each pattern represents a specific computation structure:
    - TRANSFORMER_BLOCK: Attention + FFN structure (used by LLM, DiT, etc.)
    - CONV_ENCODER: Convolutional encoding (VAE Encoder, ResNet)
    - CONV_DECODER: Convolutional decoding (VAE Decoder)
    - DENSE_FORWARD: Dense/MLP forward pass
    - ATTENTION_ONLY: Pure attention without FFN (Text Encoder)
    """

    TRANSFORMER_BLOCK = "transformer_block"
    CONV_ENCODER = "conv_encoder"
    CONV_DECODER = "conv_decoder"
    DENSE_FORWARD = "dense_forward"
    ATTENTION_ONLY = "attention_only"


@dataclass
class Phase:
    """A compute phase in a workload.

    Attributes:
        name: Phase name (e.g., "prefill", "denoise", "encode")
        compute_type: Type of compute operation
        component: Generic component identifier (e.g., "main", "backbone", "encoder", "decoder")
                   NOT model-specific names like "dit", "vae"
        repeat: Repeat count - can be int or dynamic parameter name like "generation_len"
        seq_len_factor: Sequence length factor - can be float or expression like "1/seq_len"
        compute_pattern: Optional compute pattern for this phase
        extra_params: Additional phase-specific parameters
    """

    name: str
    compute_type: ComputeType
    component: str = "main"
    repeat: Union[int, str] = 1
    seq_len_factor: Union[float, str] = 1.0
    compute_pattern: Optional[ComputePattern] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "compute_type": self.compute_type.value,
            "component": self.component,
            "repeat": self.repeat,
            "seq_len_factor": self.seq_len_factor,
            "extra_params": self.extra_params,
        }
        if self.compute_pattern:
            result["compute_pattern"] = self.compute_pattern.value
        return result


@dataclass
class SubmoduleResult:
    """子模块分解结果，从ModuleInstance获取.

    Attributes:
        name: 子模块名称
        submodule_type: 子模块类型 (embedding, attention, ffn, moe, lm_head, rms_norm, conv, resblock)
        time_sec: 执行时间（秒）
        flops: FLOPs
        memory_gb: 内存占用（GB）
        communication_bytes: 通信数据量（字节）
    """

    name: str
    submodule_type: str
    time_sec: float
    flops: int
    memory_gb: float
    communication_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "submodule_type": self.submodule_type,
            "time_sec": self.time_sec,
            "time_ms": self.time_sec * 1000,
            "flops": self.flops,
            "flops_gflops": self.flops / 1e9,
            "memory_gb": self.memory_gb,
            "communication_gb": self.communication_bytes / 1e9,
        }


@dataclass
class PhaseResult:
    """Result of a single phase analysis.

    Attributes:
        name: Phase name
        component: Component name (resolved from component_mapping)
        compute_type: Type of compute operation
        single_time_sec: Time for single execution
        repeat_count: Actual repeat count (resolved from dynamic params)
        total_time_sec: Total time (single_time × repeat_count)
        memory_gb: Memory usage for this phase
        flops: FLOPs for this phase (optional)
        submodules: 子模块分解结果列表
    """

    name: str
    component: str
    compute_type: ComputeType
    single_time_sec: float = 0.0
    repeat_count: int = 1
    total_time_sec: float = 0.0
    memory_gb: float = 0.0
    flops: Optional[float] = None
    submodules: List["SubmoduleResult"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "compute_type": self.compute_type.value,
            "single_time_sec": self.single_time_sec,
            "single_time_ms": self.single_time_sec * 1000,
            "repeat_count": self.repeat_count,
            "total_time_sec": self.total_time_sec,
            "total_time_ms": self.total_time_sec * 1000,
            "memory_gb": self.memory_gb,
            "flops": self.flops,
            "submodules": [sm.to_dict() for sm in self.submodules],
        }


@dataclass
class CommunicationBreakdown:
    """通信分解结果."""

    all_reduce: Dict[str, Any] = field(default_factory=dict)
    all_gather: Dict[str, Any] = field(default_factory=dict)
    reduce_scatter: Dict[str, Any] = field(default_factory=dict)
    all_to_all: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_reduce": self.all_reduce,
            "all_gather": self.all_gather,
            "reduce_scatter": self.reduce_scatter,
            "all_to_all": self.all_to_all,
        }


@dataclass
class UnifiedResult:
    """Result of unified workload analysis.

    Attributes:
        workload_name: Workload configuration name
        workload_type: Type of workload (training/inference/mixed)
        phases: List of phase results
        total_time_sec: Total execution time
        peak_memory_gb: Peak memory usage
        throughput: Throughput metrics dictionary
        params: Resolved parameters used in analysis
        metadata: Additional metadata
        breakdown: Legacy breakdown format for frontend compatibility
        detailed_breakdown: Legacy detailed_breakdown format for frontend compatibility
        mfu: Model FLOPs Utilization (0-1), None if not applicable
        qps: Queries Per Second
        communication_breakdown: Communication breakdown by operation type
    """

    workload_name: str
    workload_type: WorkloadType
    phases: List[PhaseResult] = field(default_factory=list)
    total_time_sec: float = 0.0
    peak_memory_gb: float = 0.0
    throughput: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    breakdown: Optional[Dict[str, Any]] = None
    detailed_breakdown: Optional[Dict[str, Any]] = None
    mfu: Optional[float] = None
    qps: Optional[float] = None
    communication_breakdown: Optional["CommunicationBreakdown"] = None
    module_breakdown: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with frontend compatibility fields.

        Dimension convention:
        - Memory metrics: per-GPU (each GPU's memory usage)
        - Compute metrics: per-GPU (each GPU's FLOPs)
        - Throughput metrics: global (system-wide throughput)
        - Latency metrics: global (request-level latency)
        """
        result = {
            "workload_name": self.workload_name,
            "workload_type": self.workload_type.value,
            "phases": [p.to_dict() for p in self.phases],
            "total_time_sec": self.total_time_sec,
            "total_time_ms": self.total_time_sec * 1000,
            "peak_memory_gb": self.peak_memory_gb,
            "throughput": self.throughput,
            "params": self.params,
            "metadata": self.metadata,
            "mfu": self.mfu,
            "qps": self.qps,
            "time": self._build_time_dict(),
            "memory": self._build_memory_dict(),
            "prefill": self._build_prefill_dict(),
            "decode": self._build_decode_dict(),
            "end_to_end": self._build_end_to_end_dict(),
        }
        if self.breakdown:
            result["breakdown"] = self.breakdown
        if self.detailed_breakdown:
            result["detailed_breakdown"] = self.detailed_breakdown
        if self.communication_breakdown:
            result["communication_breakdown"] = self.communication_breakdown.to_dict()
        if self.module_breakdown:
            result["module_breakdown"] = self.module_breakdown
        return result

    def _build_time_dict(self) -> Dict[str, float]:
        """Build time dictionary - global time."""
        return {
            "time_per_step_sec": self.total_time_sec,
            "total_time_sec": self.total_time_sec,
        }

    def _build_memory_dict(self) -> Dict[str, float]:
        """Build memory dictionary - per GPU memory."""
        kv_cache_gb = 0.0
        if self.workload_type == WorkloadType.INFERENCE:
            kv_cache_gb = self.metadata.get("kv_cache_gb", 0.0)

        return {
            "memory_per_gpu_gb": self.peak_memory_gb,
            "peak_memory_gb": self.peak_memory_gb,
            "kv_cache_gb": kv_cache_gb,
        }

    def _build_prefill_dict(self) -> Optional[Dict[str, float]]:
        """Build prefill dictionary - global time, per request TTFT."""
        if self.workload_type != WorkloadType.INFERENCE:
            return None

        prefill = self.get_phase("prefill")
        if not prefill:
            return None

        return {
            "ttft_sec": prefill.total_time_sec,
            "total_time_sec": prefill.total_time_sec,
        }

    def _build_decode_dict(self) -> Optional[Dict[str, float]]:
        """Build decode dictionary - global TPS, global per-token time."""
        if self.workload_type != WorkloadType.INFERENCE:
            return None

        decode = self.get_phase("decode")
        if not decode:
            return None

        global_tps = self.throughput.get("tokens_per_sec", 0)

        return {
            "tps": global_tps,
            "tpot_sec": decode.single_time_sec,
            "single_time_sec": decode.single_time_sec,
            "repeat_count": decode.repeat_count,
        }

    def _build_end_to_end_dict(self) -> Dict[str, float]:
        """Build end_to_end dictionary - global metrics."""
        return {
            "overall_tps": self.qps or self.throughput.get("tokens_per_sec", 0),
            "total_time_sec": self.total_time_sec,
        }

    def get_phase(self, name: str) -> Optional[PhaseResult]:
        """Get a specific phase result by name."""
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    def get_component_phases(self, component: str) -> List[PhaseResult]:
        """Get all phases for a specific component."""
        return [p for p in self.phases if p.component == component]

    def get_component_time(self, component: str) -> float:
        """Get total time for a specific component."""
        return sum(p.total_time_sec for p in self.get_component_phases(component))

    def get_component_memory(self, component: str) -> float:
        """Get peak memory for a specific component."""
        phases = self.get_component_phases(component)
        return max(p.memory_gb for p in phases) if phases else 0.0


@dataclass
class WorkloadConfig:
    """Configuration for a workload.

    The workload config describes computation characteristics, NOT model types.

    Attributes:
        name: Workload name (e.g., "training", "autoregressive-inference")
        description: Human-readable description
        workload_type: Type of workload
        phases: List of compute phases
        default_params: Default parameter values
        optimizer_factor: Optimizer time factor relative to forward
        gradient_accumulation_steps: Gradient accumulation steps
        throughput_metric: Primary throughput metric type
        throughput_formula: Custom throughput formula (optional)
        component_mapping: Mapping from generic identifiers to user component names
                          e.g., {"backbone": "dit", "encoder": "text_encoder"}
        config_file: Path to source YAML file (optional)
    """

    name: str
    description: str = ""
    workload_type: WorkloadType = WorkloadType.INFERENCE
    phases: List[Phase] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    optimizer_factor: float = 1.5
    gradient_accumulation_steps: int = 1
    throughput_metric: ThroughputMetric = ThroughputMetric.TOKENS_PER_SEC
    throughput_formula: Optional[str] = None
    component_mapping: Dict[str, str] = field(default_factory=dict)
    config_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "workload_type": self.workload_type.value,
            "phases": [p.to_dict() for p in self.phases],
            "default_params": self.default_params,
            "optimizer_factor": self.optimizer_factor,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "throughput_metric": self.throughput_metric.value,
            "throughput_formula": self.throughput_formula,
            "component_mapping": self.component_mapping,
        }
        if self.config_file:
            result["config_file"] = self.config_file
        return result

    def get_required_params(self) -> List[str]:
        """Get list of required dynamic parameters."""
        required = []
        for phase in self.phases:
            if isinstance(phase.repeat, str):
                required.append(phase.repeat)
            if isinstance(phase.seq_len_factor, str):
                if "/" in phase.seq_len_factor:
                    param = phase.seq_len_factor.split("/")[-1].strip()
                    required.append(param)
                else:
                    required.append(phase.seq_len_factor)
        return list(set(required))

    def resolve_component(self, generic_name: str) -> str:
        """Resolve generic component name to user component name.

        Args:
            generic_name: Generic identifier (e.g., "backbone", "encoder")

        Returns:
            User component name from mapping, or generic_name if not mapped
        """
        return self.component_mapping.get(generic_name, generic_name)

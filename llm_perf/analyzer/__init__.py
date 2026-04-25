"""Unified performance analyzer module.

Provides unified workload analysis for:
- Training flow (forward + backward + optimizer)
- Autoregressive generation (prefill + decode)
- Iterative denoising (multi-step forward)
- Pipeline execution (multi-component)
- Mixed workloads (speculative decoding, RL PPO, etc.)

Workload describes computation characteristics, NOT model types.
"""

from .base import (
    ComputeType,
    ComputePattern,
    WorkloadType,
    ScenarioType,
    ThroughputMetric,
    Phase,
    PhaseResult,
    UnifiedResult,
    WorkloadConfig,
    SubmoduleResult,
    CommunicationBreakdown,
)
from .unified import UnifiedAnalyzer, analyze_workload
from .workload_loader import (
    get_workload,
    load_workload_from_yaml,
    register_workload,
    list_workloads,
    load_all_builtin_workloads,
    clear_cache,
    is_builtin_workload,
    BACKWARD_COMPAT_MAP,
    infer_scenario_type,
)
from .compat import infer_workload, WORKLOAD_PRESETS
from .breakdown import KernelBreakdown, LayerBreakdown

load_all_builtin_workloads()

__all__ = [
    "ComputeType",
    "ComputePattern",
    "WorkloadType",
    "ScenarioType",
    "ThroughputMetric",
    "Phase",
    "PhaseResult",
    "UnifiedResult",
    "WorkloadConfig",
    "SubmoduleResult",
    "CommunicationBreakdown",
    "UnifiedAnalyzer",
    "analyze_workload",
    "get_workload",
    "load_workload_from_yaml",
    "register_workload",
    "list_workloads",
    "load_all_builtin_workloads",
    "clear_cache",
    "is_builtin_workload",
    "BACKWARD_COMPAT_MAP",
    "infer_workload",
    "WORKLOAD_PRESETS",
    "KernelBreakdown",
    "LayerBreakdown",
    "infer_scenario_type",
]

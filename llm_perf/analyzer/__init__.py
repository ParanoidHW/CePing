"""Unified performance analyzer module.

Provides unified workload analysis for:
- LLM training/inference
- Diffusion training/inference
- MoE training/inference
- Mixed workloads (speculative decoding, RL PPO, etc.)
"""

from .base import (
    ComputeType,
    WorkloadType,
    ThroughputMetric,
    Phase,
    PhaseResult,
    UnifiedResult,
    WorkloadConfig,
)
from .unified import UnifiedAnalyzer, analyze_workload
from .presets import (
    WORKLOAD_PRESETS,
    get_workload,
    register_workload,
    list_workloads,
    infer_workload,
)
from .breakdown import KernelBreakdown, LayerBreakdown

__all__ = [
    "ComputeType",
    "WorkloadType",
    "ThroughputMetric",
    "Phase",
    "PhaseResult",
    "UnifiedResult",
    "WorkloadConfig",
    "UnifiedAnalyzer",
    "analyze_workload",
    "WORKLOAD_PRESETS",
    "get_workload",
    "register_workload",
    "list_workloads",
    "infer_workload",
    "KernelBreakdown",
    "LayerBreakdown",
]

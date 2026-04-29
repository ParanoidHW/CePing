"""Breakdown calculation for workload analysis.

Provides hierarchical breakdown:
- Stage (prefill, decode, forward, backward)
- Phase (sub-stages)
- Submodule (attention, ffn, moe, embedding, lm_head)
- Kernel (linear, flash_attention, etc.)
- Communication (allreduce, allgather, reducescatter)

This module wraps existing analyzer/breakdown.py functionality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from llm_perf.analyzer.base import UnifiedResult, PhaseResult, SubmoduleResult
from llm_perf.analyzer.breakdown import generate_module_breakdown, ModuleBreakdown


@dataclass
class KernelBreakdown:
    """Kernel-level breakdown."""

    name: str
    time_sec: float
    flops: int = 0
    bytes_accessed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "time_sec": self.time_sec,
            "time_ms": self.time_sec * 1000,
            "flops": self.flops,
            "bytes_accessed": self.bytes_accessed,
        }


@dataclass
class SubmoduleBreakdown:
    """Submodule-level breakdown."""

    name: str
    submodule_type: str
    time_sec: float
    flops: int
    params_count: int
    weight_memory_gb: float
    communication_bytes: int
    kernels: List[KernelBreakdown] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "submodule_type": self.submodule_type,
            "time_sec": self.time_sec,
            "time_ms": self.time_sec * 1000,
            "flops": self.flops,
            "params_count": self.params_count,
            "weight_memory_gb": self.weight_memory_gb,
            "communication_bytes": self.communication_bytes,
            "kernels": [k.to_dict() for k in self.kernels],
        }


@dataclass
class PhaseBreakdown:
    """Phase-level breakdown."""

    name: str
    time_sec: float
    memory_gb: float
    submodules: List[SubmoduleBreakdown] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "time_sec": self.time_sec,
            "time_ms": self.time_sec * 1000,
            "memory_gb": self.memory_gb,
            "submodules": [s.to_dict() for s in self.submodules],
        }


@dataclass
class StageBreakdown:
    """Stage-level breakdown."""

    name: str
    total_time_sec: float
    peak_memory_gb: float
    phases: List[PhaseBreakdown] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_time_sec": self.total_time_sec,
            "total_time_ms": self.total_time_sec * 1000,
            "peak_memory_gb": self.peak_memory_gb,
            "phases": [p.to_dict() for p in self.phases],
        }


@dataclass
class CommunicationBreakdown:
    """Communication breakdown."""

    by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_bytes: int = 0
    total_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "by_type": self.by_type,
            "total_bytes": self.total_bytes,
            "total_mb": self.total_bytes / 1e6,
            "total_gb": self.total_bytes / 1e9,
            "total_time_sec": self.total_time_sec,
        }


@dataclass
class WorkloadBreakdown:
    """Complete workload breakdown."""

    stages: List[StageBreakdown] = field(default_factory=list)
    by_submodule_type: Dict[str, SubmoduleBreakdown] = field(default_factory=dict)
    communication: Optional[CommunicationBreakdown] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "by_submodule_type": {k: v.to_dict() for k, v in self.by_submodule_type.items()},
            "communication": self.communication.to_dict() if self.communication else None,
            "summary": self.summary,
        }


class BreakdownCalculator:
    """Calculate breakdown from UnifiedResult."""

    def calculate(self, unified_result: UnifiedResult) -> WorkloadBreakdown:
        """Calculate breakdown from UnifiedResult.

        Args:
            unified_result: UnifiedResult from UnifiedAnalyzer

        Returns:
            WorkloadBreakdown
        """
        stages = self._aggregate_by_stage(unified_result)
        by_submodule_type = self._aggregate_by_submodule_type(unified_result)
        communication = self._aggregate_communication(unified_result)
        summary = self._calculate_summary(unified_result)

        return WorkloadBreakdown(
            stages=stages,
            by_submodule_type=by_submodule_type,
            communication=communication,
            summary=summary,
        )

    def _aggregate_by_stage(self, result: UnifiedResult) -> List[StageBreakdown]:
        """Aggregate by stage."""
        stage_dict: Dict[str, StageBreakdown] = {}

        for phase in result.phases:
            stage_name = phase.name
            if stage_name not in stage_dict:
                stage_dict[stage_name] = StageBreakdown(
                    name=stage_name,
                    total_time_sec=0.0,
                    peak_memory_gb=0.0,
                )

            stage_dict[stage_name].total_time_sec += phase.total_time_sec
            stage_dict[stage_name].peak_memory_gb = max(
                stage_dict[stage_name].peak_memory_gb, phase.memory_gb
            )

            phase_breakdown = self._phase_to_breakdown(phase)
            stage_dict[stage_name].phases.append(phase_breakdown)

        return list(stage_dict.values())

    def _aggregate_by_submodule_type(self, result: UnifiedResult) -> Dict[str, SubmoduleBreakdown]:
        """Aggregate by submodule type."""
        submodule_dict: Dict[str, SubmoduleBreakdown] = {}

        for phase in result.phases:
            for sub in phase.submodules:
                sub_type = sub.submodule_type
                if sub_type not in submodule_dict:
                    submodule_dict[sub_type] = SubmoduleBreakdown(
                        name=sub_type,
                        submodule_type=sub_type,
                        time_sec=0.0,
                        flops=0,
                        params_count=0,
                        weight_memory_gb=0.0,
                        communication_bytes=0,
                    )

                submodule_dict[sub_type].time_sec += sub.time_sec
                submodule_dict[sub_type].flops += sub.flops
                submodule_dict[sub_type].params_count += sub.params_count
                submodule_dict[sub_type].weight_memory_gb += sub.weight_memory_gb
                submodule_dict[sub_type].communication_bytes += sub.communication_bytes

        return submodule_dict

    def _aggregate_communication(self, result: UnifiedResult) -> Optional[CommunicationBreakdown]:
        """Aggregate communication."""
        if not result.communication_breakdown:
            return None

        comm_breakdown = CommunicationBreakdown(
            by_type=result.communication_breakdown.by_operation,
            total_bytes=result.communication_breakdown.total_bytes,
            total_time_sec=result.communication_breakdown.total_time_sec,
        )

        return comm_breakdown

    def _calculate_summary(self, result: UnifiedResult) -> Dict[str, Any]:
        """Calculate summary."""
        return {
            "total_time_sec": result.total_time_sec,
            "peak_memory_gb": result.peak_memory_gb,
            "throughput": result.throughput,
            "mfu": result.mfu,
            "qps": result.qps,
        }

    def _phase_to_breakdown(self, phase: PhaseResult) -> PhaseBreakdown:
        """Convert PhaseResult to PhaseBreakdown."""
        submodules = []
        for sub in phase.submodules:
            submodules.append(
                SubmoduleBreakdown(
                    name=sub.name,
                    submodule_type=sub.submodule_type,
                    time_sec=sub.time_sec,
                    flops=sub.flops,
                    params_count=sub.params_count,
                    weight_memory_gb=sub.weight_memory_gb,
                    communication_bytes=sub.communication_bytes,
                    kernels=self._extract_kernels(sub),
                )
            )

        return PhaseBreakdown(
            name=phase.name,
            time_sec=phase.total_time_sec,
            memory_gb=phase.memory_gb,
            submodules=submodules,
        )

    def _extract_kernels(self, submodule: SubmoduleResult) -> List[KernelBreakdown]:
        """Extract kernel breakdown from submodule."""
        kernels = []

        kernel_time = submodule.time_sec
        if kernel_time > 0:
            kernels.append(
                KernelBreakdown(
                    name="compute",
                    time_sec=kernel_time,
                    flops=submodule.flops,
                )
            )

        comm_time = submodule.communication_time_sec
        if comm_time > 0:
            kernels.append(
                KernelBreakdown(
                    name="communication",
                    time_sec=comm_time,
                    bytes_accessed=submodule.communication_bytes,
                )
            )

        return kernels


_calculator_instance: Optional[BreakdownCalculator] = None


def get_calculator() -> BreakdownCalculator:
    """Get singleton BreakdownCalculator instance."""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = BreakdownCalculator()
    return _calculator_instance


def calculate_breakdown(unified_result: UnifiedResult) -> WorkloadBreakdown:
    """Calculate breakdown from UnifiedResult.

    Args:
        unified_result: UnifiedResult from UnifiedAnalyzer

    Returns:
        WorkloadBreakdown
    """
    return get_calculator().calculate(unified_result)
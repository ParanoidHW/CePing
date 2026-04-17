"""Overlap feature for compute-communication overlap modeling."""

from typing import Dict, Any

from ..base import SchedulerFeature, SchedulerConfig


class OverlapFeature(SchedulerFeature):
    """Models compute-communication overlap optimization.

    Overlap allows computation and communication to execute
    concurrently, reducing effective execution time.

    The overlap efficiency depends on:
    - Hardware capabilities (overlapping compute/comm units)
    - Software implementation quality
    - Communication pattern (all-reduce, all-to-all, etc.)

    Typical overlap efficiency:
    - Ideal: 1.0 (full overlap)
    - Good: 0.7-0.9
    - Moderate: 0.5-0.7
    - Poor: 0.3-0.5
    """

    name = "overlap"
    description = "Compute-communication overlap optimization"

    def __init__(self, config: SchedulerConfig):
        """Initialize overlap feature.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.efficiency = config.overlap_efficiency

    def apply_overhead(self, base_time: float) -> float:
        """Overlap doesn't add overhead to base time.

        Args:
            base_time: Base execution time in seconds

        Returns:
            Unmodified base time
        """
        return base_time

    def apply_overlap(self, compute_time: float, comm_time: float) -> float:
        """Apply compute-communication overlap.

        Effective time = max(compute, comm) + overlap_penalty

        Where overlap_penalty accounts for imperfect overlap:
        - If compute > comm: overlap_penalty = (comm - comm * efficiency)
        - If comm > compute: overlap_penalty = (compute - compute * efficiency)

        Args:
            compute_time: Compute time in seconds
            comm_time: Communication time in seconds

        Returns:
            Effective time after overlap
        """
        if compute_time <= 0 and comm_time <= 0:
            return 0.0

        if compute_time <= 0:
            return comm_time
        if comm_time <= 0:
            return compute_time

        max_time = max(compute_time, comm_time)
        min_time = min(compute_time, comm_time)

        overlap_benefit = min_time * self.efficiency
        effective_time = max_time + (min_time - overlap_benefit)

        return effective_time

    def apply_memory_optimization(self, base_memory: int) -> int:
        """Overlap doesn't affect memory.

        Args:
            base_memory: Base memory in bytes

        Returns:
            Unmodified memory
        """
        return base_memory

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "efficiency": self.efficiency,
        }

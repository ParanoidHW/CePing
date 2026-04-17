"""Prefetch feature for KV cache prefetch modeling."""

from typing import Dict, Any

from ..base import SchedulerFeature, SchedulerConfig


class PrefetchFeature(SchedulerFeature):
    """Models KV cache prefetch for inference optimization.

    Prefetch overlaps KV cache loading with computation,
    reducing the effective latency of inference.

    Benefits:
    - Reduces attention latency by prefetching KV cache
    - Hides memory bandwidth latency
    - Improves throughput for long context

    Trade-offs:
    - Requires additional memory bandwidth
    - May not be fully effective if compute is short
    - Depends on memory hierarchy (HBM -> SRAM)
    """

    name = "prefetch"
    description = "KV cache prefetch for inference optimization"

    def __init__(self, config: SchedulerConfig):
        """Initialize prefetch feature.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.enabled = config.prefetch_enabled
        self.overlap_ratio = config.prefetch_overlap_ratio

    def apply_overhead(self, base_time: float) -> float:
        """Prefetch doesn't add overhead when compute is long enough.

        Args:
            base_time: Base execution time in seconds

        Returns:
            Unmodified base time
        """
        return base_time

    def apply_overlap(self, compute_time: float, comm_time: float) -> float:
        """Apply prefetch overlap for inference.

        Effective time = max(compute, comm) * (1 - overlap_ratio * efficiency)

        Where:
        - overlap_ratio: How much of communication can be hidden
        - efficiency: Hardware-dependent prefetch efficiency

        Args:
            compute_time: Compute time in seconds
            comm_time: Communication/memory load time in seconds

        Returns:
            Effective time after prefetch
        """
        if not self.enabled:
            return compute_time + comm_time

        if compute_time <= 0 and comm_time <= 0:
            return 0.0

        if compute_time <= 0:
            return comm_time
        if comm_time <= 0:
            return compute_time

        max_time = max(compute_time, comm_time)
        min_time = min(compute_time, comm_time)

        prefetch_benefit = min_time * self.overlap_ratio

        return max_time + (min_time - prefetch_benefit)

    def apply_memory_optimization(self, base_memory: int) -> int:
        """Prefetch may require extra memory for buffer.

        Args:
            base_memory: Base memory in bytes

        Returns:
            Memory with prefetch buffer overhead
        """
        if not self.enabled:
            return base_memory

        return int(base_memory * 1.05)

    def estimate_prefetch_effectiveness(self, compute_time: float, kv_load_time: float) -> float:
        """Estimate prefetch effectiveness.

        Prefetch is most effective when:
        - Compute time >= KV load time (can fully hide)
        - Memory bandwidth not saturated

        Args:
            compute_time: Compute time for attention
            kv_load_time: Time to load KV cache

        Returns:
            Effectiveness ratio (0.0 - 1.0)
        """
        if not self.enabled:
            return 0.0

        if compute_time <= 0 or kv_load_time <= 0:
            return 0.0

        if compute_time >= kv_load_time:
            return min(self.overlap_ratio, 0.95)

        ratio = compute_time / kv_load_time
        return ratio * self.overlap_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "overlap_ratio": self.overlap_ratio,
        }

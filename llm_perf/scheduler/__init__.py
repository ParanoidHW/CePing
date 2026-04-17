"""Scheduler module for framework scheduling feature modeling.

This module provides abstractions for modeling various framework
scheduling optimizations:
- Overlap: Compute-communication overlap
- Pipeline Bubble: PP bubble overhead
- Chunking: Sequence chunking for memory
- Prefetch: KV cache prefetch

Usage:
    from llm_perf.scheduler import SchedulerModel, SchedulerConfig

    config = SchedulerConfig(
        enabled_features=["overlap", "chunking"],
        overlap_efficiency=0.8,
        chunk_size=1024,
    )
    model = SchedulerModel(config)

    result = SchedulerResult(
        compute_time=1.0,
        comm_time=0.5,
        memory_bytes=10_000_000_000,
    )
    optimized = model.apply_all(result)
"""

from .base import (
    SchedulerFeature,
    SchedulerConfig,
    SchedulerModel,
    SchedulerResult,
)
from .features import (
    OverlapFeature,
    PipelineBubbleFeature,
    ChunkingFeature,
    PrefetchFeature,
)

__all__ = [
    "SchedulerFeature",
    "SchedulerConfig",
    "SchedulerModel",
    "SchedulerResult",
    "OverlapFeature",
    "PipelineBubbleFeature",
    "ChunkingFeature",
    "PrefetchFeature",
]

"""Scheduler base classes for framework scheduling features."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SchedulerFeature(ABC):
    """Base class for scheduler features.

    Scheduler features model various optimization techniques used by
    training/inference frameworks:
    - Overlap: Computation-communication overlap
    - Pipeline Bubble: Pipeline parallelism bubble overhead
    - Chunking: Sequence chunking for memory optimization
    - Prefetch: KV cache prefetch for inference

    Each feature provides methods to:
    - Apply overhead to base time
    - Apply overlap between compute and communication
    - Apply memory optimization effects
    """

    name: str = "base"
    description: str = "Base scheduler feature"

    @abstractmethod
    def apply_overhead(self, base_time: float) -> float:
        """Apply scheduling overhead to base time.

        Args:
            base_time: Base execution time in seconds

        Returns:
            Modified time after applying overhead
        """
        pass

    @abstractmethod
    def apply_overlap(self, compute_time: float, comm_time: float) -> float:
        """Apply compute-communication overlap optimization.

        Args:
            compute_time: Compute time in seconds
            comm_time: Communication time in seconds

        Returns:
            Effective time after overlap
        """
        pass

    @abstractmethod
    def apply_memory_optimization(self, base_memory: int) -> int:
        """Apply memory optimization effects.

        Args:
            base_memory: Base memory in bytes

        Returns:
            Modified memory after optimization
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
        }


@dataclass
class SchedulerConfig:
    """Configuration for scheduler features.

    Attributes:
        enabled_features: List of enabled feature names
        overlap_enabled: Whether compute-comm overlap is enabled
        overlap_efficiency: Overlap efficiency ratio (0.0-1.0)
        chunk_size: Chunk size for sequence chunking (None means disabled)
        prefetch_enabled: Whether KV prefetch is enabled
        prefetch_overlap_ratio: Prefetch overlap ratio (0.0-1.0)
        pipeline_bubble_ratio: Pipeline bubble ratio (0.0-1.0)
    """

    enabled_features: List[str] = field(default_factory=list)
    overlap_enabled: bool = False
    overlap_efficiency: float = 0.8
    chunk_size: Optional[int] = None
    prefetch_enabled: bool = False
    prefetch_overlap_ratio: float = 0.5
    pipeline_bubble_ratio: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled_features": self.enabled_features,
            "overlap": {
                "enabled": self.overlap_enabled,
                "efficiency": self.overlap_efficiency,
            },
            "chunking": {
                "chunk_size": self.chunk_size,
            },
            "prefetch": {
                "enabled": self.prefetch_enabled,
                "overlap_ratio": self.prefetch_overlap_ratio,
            },
            "pipeline_bubble": {
                "ratio": self.pipeline_bubble_ratio,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulerConfig":
        """Create from dictionary."""
        overlap_data = data.get("overlap", {})
        chunking_data = data.get("chunking", {})
        prefetch_data = data.get("prefetch", {})
        bubble_data = data.get("pipeline_bubble", {})

        return cls(
            enabled_features=data.get("enabled_features", []),
            overlap_enabled=overlap_data.get("enabled", False),
            overlap_efficiency=overlap_data.get("efficiency", 0.8),
            chunk_size=chunking_data.get("chunk_size"),
            prefetch_enabled=prefetch_data.get("enabled", False),
            prefetch_overlap_ratio=prefetch_data.get("overlap_ratio", 0.5),
            pipeline_bubble_ratio=bubble_data.get("ratio", 0.1),
        )


class SchedulerModel:
    """Scheduler model that combines multiple scheduler features.

    The SchedulerModel applies scheduling optimizations to performance
    analysis results. It can:
    - Combine multiple features (overlap, chunking, prefetch, etc.)
    - Apply features in sequence with configurable order
    - Provide unified interface for result modification
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """Initialize scheduler model.

        Args:
            config: Scheduler configuration. If None, uses defaults.
        """
        self.config = config or SchedulerConfig()
        self._features: List[SchedulerFeature] = []
        self._feature_map: Dict[str, SchedulerFeature] = {}

        self._initialize_features()

    def _initialize_features(self):
        """Initialize features based on configuration."""
        from .features.overlap import OverlapFeature
        from .features.pipeline_bubble import PipelineBubbleFeature
        from .features.chunking import ChunkingFeature
        from .features.prefetch import PrefetchFeature

        feature_classes = {
            "overlap": OverlapFeature,
            "pipeline_bubble": PipelineBubbleFeature,
            "chunking": ChunkingFeature,
            "prefetch": PrefetchFeature,
        }

        for feature_name in self.config.enabled_features:
            if feature_name in feature_classes:
                feature = feature_classes[feature_name](self.config)
                self._features.append(feature)
                self._feature_map[feature_name] = feature

        if self.config.overlap_enabled and "overlap" not in self._feature_map:
            feature = OverlapFeature(self.config)
            self._features.append(feature)
            self._feature_map["overlap"] = feature

    def add_feature(self, feature: SchedulerFeature):
        """Add a feature to the scheduler model.

        Args:
            feature: Feature to add
        """
        self._features.append(feature)
        self._feature_map[feature.name] = feature

    def get_feature(self, name: str) -> Optional[SchedulerFeature]:
        """Get a feature by name.

        Args:
            name: Feature name

        Returns:
            Feature instance or None if not found
        """
        return self._feature_map.get(name)

    def apply_all(self, result: "SchedulerResult") -> "SchedulerResult":
        """Apply all enabled features to a result.

        Args:
            result: Input scheduler result

        Returns:
            Modified result after applying all features
        """
        modified = SchedulerResult(
            compute_time=result.compute_time,
            comm_time=result.comm_time,
            memory_bytes=result.memory_bytes,
        )

        for feature in self._features:
            modified.compute_time = feature.apply_overhead(modified.compute_time)
            modified.overlapped_time = feature.apply_overlap(modified.compute_time, modified.comm_time)
            modified.memory_bytes = feature.apply_memory_optimization(modified.memory_bytes)

        if not self._features:
            modified.overlapped_time = modified.compute_time + modified.comm_time

        return modified

    def apply_overlap(self, compute_time: float, comm_time: float) -> float:
        """Apply overlap optimization.

        Args:
            compute_time: Compute time in seconds
            comm_time: Communication time in seconds

        Returns:
            Effective time after overlap
        """
        if "overlap" in self._feature_map:
            return self._feature_map["overlap"].apply_overlap(compute_time, comm_time)
        return compute_time + comm_time

    def apply_memory_optimization(self, memory_bytes: int) -> int:
        """Apply memory optimization.

        Args:
            memory_bytes: Base memory in bytes

        Returns:
            Optimized memory in bytes
        """
        for feature in self._features:
            memory_bytes = feature.apply_memory_optimization(memory_bytes)
        return memory_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "features": [f.to_dict() for f in self._features],
        }


@dataclass
class SchedulerResult:
    """Result after applying scheduler optimizations.

    Attributes:
        compute_time: Compute time in seconds
        comm_time: Communication time in seconds
        memory_bytes: Memory usage in bytes
        overlapped_time: Effective time after overlap optimization
        feature_metrics: Additional metrics from features
    """

    compute_time: float = 0.0
    comm_time: float = 0.0
    memory_bytes: int = 0
    overlapped_time: float = 0.0
    feature_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compute_time": self.compute_time,
            "comm_time": self.comm_time,
            "overlapped_time": self.overlapped_time,
            "memory_bytes": self.memory_bytes,
            "memory_gb": self.memory_bytes / 1e9,
            "feature_metrics": self.feature_metrics,
        }

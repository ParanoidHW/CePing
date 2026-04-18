"""Colocate evaluation scenario.

Evaluate multiple models deployed on the same cluster.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from llm_perf.analyzer import UnifiedAnalyzer, UnifiedResult
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.device import Device
from llm_perf.modeling import ShardedModule
from llm_perf.strategy.base import StrategyConfig


@dataclass
class ModelAllocation:
    """Single model's resource allocation."""

    name: str
    model: ShardedModule
    strategy: StrategyConfig
    workload: str
    allocated_ratio: float = 1.0


@dataclass
class ColocateResult:
    """Colocate evaluation result."""

    model_results: Dict[str, UnifiedResult] = field(default_factory=dict)
    total_devices: int = 0
    total_utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_results": {name: result.to_dict() for name, result in self.model_results.items()},
            "total_devices": self.total_devices,
            "total_utilization": self.total_utilization,
        }


class ColocateAnalyzer:
    """Colocate analyzer.

    Evaluate multiple models on the same cluster.
    """

    def __init__(self, device: Device, cluster: Cluster):
        """Initialize colocate analyzer.

        Args:
            device: Device configuration
            cluster: Cluster configuration
        """
        self.device = device
        self.cluster = cluster

    def analyze(self, allocations: List[ModelAllocation], **kwargs) -> ColocateResult:
        """Analyze colocate performance.

        Args:
            allocations: Each model's allocation config
            **kwargs: Parameters passed to each model

        Returns:
            ColocateResult
        """
        results = {}

        for allocation in allocations:
            analyzer = UnifiedAnalyzer(
                allocation.model,
                self.device,
                self.cluster,
                allocation.strategy,
            )

            result = analyzer.analyze(allocation.workload, **kwargs)
            results[allocation.name] = result

        total_utilization = sum(a.allocated_ratio for a in allocations)

        return ColocateResult(
            model_results=results,
            total_devices=self.cluster.num_devices,
            total_utilization=min(total_utilization, 1.0),
        )

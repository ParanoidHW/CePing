"""Base classes for performance analyzers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_perf.modeling import ShardedModule
    from llm_perf.hardware.device import Device
    from llm_perf.hardware.cluster import Cluster
    from llm_perf.strategy.base import StrategyConfig


@dataclass
class BaseResult:
    """Base result class for performance analysis."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {}


@dataclass
class PerformanceBreakdown:
    """Performance breakdown by category."""

    compute_time_sec: float = 0.0
    communication_time_sec: float = 0.0
    memory_time_sec: float = 0.0
    total_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compute_time_sec": self.compute_time_sec,
            "communication_time_sec": self.communication_time_sec,
            "memory_time_sec": self.memory_time_sec,
            "total_time_sec": self.total_time_sec,
        }


class BaseAnalyzer(ABC):
    """Base analyzer for ShardedModule models."""

    def __init__(
        self,
        model: "ShardedModule",
        device: "Device",
        cluster: "Cluster",
        strategy: "StrategyConfig",
    ):
        self.model = model
        self.device = device
        self.cluster = cluster
        self.strategy = strategy

    def _get_model_info(self) -> Dict[str, Any]:
        """Get basic model information."""
        params = self._count_params()
        return {
            "params": params,
            "params_gb": params * 2 / 1e9,
        }

    def _count_params(self) -> int:
        """Count total parameters in model."""
        total = 0
        for name, weight in self.model._weights.items():
            total += weight.numel()
        for name, submodule in self.model._submodules.items():
            total += self._count_submodule_params(submodule)
        return total

    def _count_submodule_params(self, module: "ShardedModule") -> int:
        """Count parameters in a submodule."""
        total = 0
        for name, weight in module._weights.items():
            total += weight.numel()
        for name, submodule in module._submodules.items():
            total += self._count_submodule_params(submodule)
        return total

    def _get_parallel_degrees(self) -> Dict[str, int]:
        """Get parallel degrees from strategy."""
        return {
            "tp": self.strategy.tp_degree,
            "pp": self.strategy.pp_degree,
            "dp": self.strategy.dp_degree,
            "ep": self.strategy.ep_degree or 1,
        }

    @abstractmethod
    def analyze(self, **kwargs) -> BaseResult:
        """Run performance analysis."""
        pass

"""Breakdown classes for performance analysis."""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class KernelBreakdown:
    """Breakdown by kernel type."""

    name: str
    time_sec: float
    flops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "time_sec": self.time_sec,
            "time_ms": self.time_sec * 1000,
            "flops": self.flops,
        }


@dataclass
class LayerBreakdown:
    """Breakdown by layer."""

    name: str
    kernels: List[KernelBreakdown]
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_time_sec": self.total_time,
            "kernels": [k.to_dict() for k in self.kernels],
        }

"""Base kernel classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class KernelType(Enum):
    """Types of kernels."""
    COMPUTE = "compute"
    COMMUNICATION = "communication"
    MEMORY = "memory"


@dataclass
class KernelConfig:
    """Base kernel configuration."""
    name: str
    kernel_type: KernelType
    # Performance parameters (can be overridden by benchmarks)
    measured_flops: Optional[float] = None  # Actual measured FLOPs/s
    measured_bw: Optional[float] = None     # Actual measured bytes/s
    latency_us: Optional[float] = None      # Fixed latency overhead


class Kernel(ABC):
    """Abstract base class for all kernels."""
    
    def __init__(self, config: KernelConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def kernel_type(self) -> KernelType:
        return self.config.kernel_type
    
    @abstractmethod
    def estimate_time(
        self,
        input_shape: tuple,
        output_shape: tuple,
        dtype: str,
        **kwargs
    ) -> float:
        """
        Estimate execution time for this kernel.
        
        Returns:
            Estimated time in seconds
        """
        pass
    
    @abstractmethod
    def estimate_memory(
        self,
        input_shape: tuple,
        output_shape: tuple,
        dtype: str,
        **kwargs
    ) -> int:
        """
        Estimate memory usage for this kernel.
        
        Returns:
            Estimated memory in bytes
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.config.name,
            "type": self.config.kernel_type.value,
        }


class KernelRegistry:
    """Registry for kernel implementations."""
    
    def __init__(self):
        self._kernels: Dict[str, Kernel] = {}
    
    def register(self, name: str, kernel: Kernel):
        """Register a kernel."""
        self._kernels[name] = kernel
    
    def get(self, name: str) -> Optional[Kernel]:
        """Get a kernel by name."""
        return self._kernels.get(name)
    
    def list_kernels(self) -> list:
        """List all registered kernel names."""
        return list(self._kernels.keys())

"""Kernel Backend base class.

Defines the interface for pluggable kernel evaluation strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from ..functional import KernelResult
from ..base import KernelConfig
from ...hardware.device import Device


@dataclass
class BackendConfig:
    """Backend configuration."""

    name: str
    device: Optional[Device] = None
    cluster_config: Optional[Dict[str, Any]] = None
    profiling_data_path: Optional[str] = None
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class KernelBackend(ABC):
    """Abstract base class for kernel evaluation backends.

    Each backend provides a different strategy for estimating
    kernel performance:
    - TheoryBackend: Analytical model (FLOPs, Roofline)
    - ProfilingBackend: Lookup table with interpolation
    - MicroarchBackend: Hardware microarchitecture simulation

    The backend interface is designed to be:
    1. Pluggable: Easy to switch between backends
    2. Configurable: Support device-specific tuning
    3. Extensible: Add new kernels without changing core logic
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self._initialized = False

    @property
    def name(self) -> str:
        return self.config.name

    def initialize(self) -> None:
        """Initialize backend (load data, setup models, etc.)."""
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    @abstractmethod
    def estimate_compute_time(
        self,
        kernel_name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        dtype: str,
        device: Device,
        **kwargs,
    ) -> float:
        """Estimate compute kernel execution time.

        Args:
            kernel_name: Kernel identifier (e.g., "linear", "attention", "gelu")
            input_shapes: List of input tensor shapes
            output_shape: Output tensor shape
            dtype: Data type (fp16, bf16, fp32, etc.)
            device: Target device for execution
            **kwargs: Additional kernel parameters (e.g., is_causal for attention)

        Returns:
            Estimated execution time in seconds
        """
        pass

    @abstractmethod
    def estimate_compute_time_from_result(self, result: KernelResult, device: Device, **kwargs) -> float:
        """Estimate time from KernelResult (functional API output).

        Args:
            result: KernelResult from functional API
            device: Target device
            **kwargs: Additional parameters

        Returns:
            Estimated execution time in seconds
        """
        pass

    @abstractmethod
    def estimate_comm_time(
        self, comm_type: str, data_size_bytes: int, num_ranks: int, bandwidth_gbps: float, **kwargs
    ) -> float:
        """Estimate communication kernel execution time.

        Args:
            comm_type: Communication type (allreduce, allgather, alltoall, etc.)
            data_size_bytes: Data size in bytes
            num_ranks: Number of participating ranks
            bandwidth_gbps: Network bandwidth in GB/s
            **kwargs: Additional parameters (e.g., participating_ranks list)

        Returns:
            Estimated communication time in seconds
        """
        pass

    @abstractmethod
    def estimate_memory(
        self, kernel_name: str, input_shapes: List[Tuple[int, ...]], output_shape: Tuple[int, ...], dtype: str, **kwargs
    ) -> int:
        """Estimate memory usage for kernel.

        Args:
            kernel_name: Kernel identifier
            input_shapes: List of input tensor shapes
            output_shape: Output tensor shape
            dtype: Data type
            **kwargs: Additional parameters

        Returns:
            Estimated memory usage in bytes
        """
        pass

    @abstractmethod
    def estimate_memory_from_result(self, result: KernelResult, **kwargs) -> int:
        """Estimate memory from KernelResult.

        Args:
            result: KernelResult from functional API
            **kwargs: Additional parameters

        Returns:
            Estimated memory usage in bytes
        """
        pass

    def estimate_training_time(
        self,
        kernel_name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        dtype: str,
        device: Device,
        **kwargs,
    ) -> float:
        """Estimate total training time (forward + backward).

        Default implementation: forward_time * 3 (forward + backward)

        Args:
            kernel_name: Kernel identifier
            input_shapes: List of input tensor shapes
            output_shape: Output tensor shape
            dtype: Data type
            device: Target device
            **kwargs: Additional parameters

        Returns:
            Estimated training time in seconds
        """
        forward_time = self.estimate_compute_time(kernel_name, input_shapes, output_shape, dtype, device, **kwargs)
        return forward_time * 3

    def estimate_training_time_from_result(self, result: KernelResult, device: Device, **kwargs) -> float:
        """Estimate training time from KernelResult.

        Args:
            result: KernelResult from functional API
            device: Target device
            **kwargs: Additional parameters

        Returns:
            Estimated training time in seconds
        """
        forward_time = self.estimate_compute_time_from_result(result, device, **kwargs)
        return forward_time * 3

    def get_kernel_config(self, kernel_name: str, **kwargs) -> Optional[KernelConfig]:
        """Get kernel configuration if available.

        Args:
            kernel_name: Kernel identifier
            **kwargs: Additional parameters

        Returns:
            KernelConfig if found, None otherwise
        """
        return None

    def supports_kernel(self, kernel_name: str) -> bool:
        """Check if backend supports this kernel type.

        Args:
            kernel_name: Kernel identifier

        Returns:
            True if supported
        """
        return True

    def get_backend_type(self) -> str:
        """Get backend type identifier.

        Returns:
            Backend type string (theory, profiling, microarch)
        """
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert backend info to dictionary."""
        return {
            "name": self.name,
            "type": self.get_backend_type(),
            "initialized": self._initialized,
            "config": {
                "device": self.config.device.config.name if self.config.device else None,
            },
        }

"""Kernel Backend base class.

Defines the interface for pluggable kernel evaluation strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass

from ..functional import KernelResult
from ..base import KernelConfig
from ...hardware.device import Device

if TYPE_CHECKING:
    from ...models.sharding import ShardingInfo, ShardedLayerConfig
    from ...strategy.base import StrategyConfig


@dataclass
class ShardedKernelResult:
    """Result of sharded kernel evaluation.

    Attributes:
        sharded_shape: Shape after sharding (on single device)
        sharded_flops: FLOPs after sharding
        sharded_time: Execution time after sharding
        comm_time: Communication time associated with this kernel
        sharded_memory: Memory usage after sharding
    """

    sharded_shape: Tuple[int, ...]
    sharded_flops: int
    sharded_time: float
    comm_time: float = 0.0
    sharded_memory: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharded_shape": self.sharded_shape,
            "sharded_flops": self.sharded_flops,
            "sharded_time_sec": self.sharded_time,
            "sharded_time_ms": self.sharded_time * 1000,
            "comm_time_sec": self.comm_time,
            "comm_time_ms": self.comm_time * 1000,
            "sharded_memory_bytes": self.sharded_memory,
        }


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

    def estimate_sharded_compute_time(
        self,
        original_result: KernelResult,
        sharding_info: "ShardingInfo",
        strategy: "StrategyConfig",
        device: Device,
        bandwidth_gbps: float = 400.0,
        **kwargs,
    ) -> ShardedKernelResult:
        """Estimate sharded kernel performance.

        This method computes the performance of a kernel after applying
        parallelism sharding. Each backend can implement its own strategy
        for evaluating sharded kernels.

        Args:
            original_result: Original (unsharded) KernelResult
            sharding_info: Sharding metadata from the layer
            strategy: Parallelism strategy configuration
            device: Target device
            bandwidth_gbps: Communication bandwidth (for comm time estimation)
            **kwargs: Additional parameters

        Returns:
            ShardedKernelResult with sharded shape, flops, time, and comm time
        """
        tp = strategy.tp_degree

        sharded_flops = original_result.flops // max(tp, 1)
        sharded_shape = self._compute_sharded_shape(original_result.output, sharding_info, strategy)

        sharded_time = self._estimate_time_from_flops(sharded_flops, sharded_shape, original_result.dtype, device)

        comm_time = 0.0
        if sharding_info and sharding_info.comm_patterns:
            for pattern in sharding_info.comm_patterns:
                if tp > 1:
                    comm_bytes = sharding_info.get_comm_bytes_for_tp(
                        batch_size=kwargs.get("batch_size", 1),
                        seq_len=kwargs.get("seq_len", 512),
                        hidden_size=kwargs.get("hidden_size", 4096),
                        dtype=original_result.dtype,
                    )
                    comm_time += self.estimate_comm_time(
                        pattern.comm_type,
                        comm_bytes,
                        tp,
                        bandwidth_gbps,
                    )

        sharded_memory = original_result.bytes_accessed // max(tp, 1)

        return ShardedKernelResult(
            sharded_shape=sharded_shape,
            sharded_flops=sharded_flops,
            sharded_time=sharded_time,
            comm_time=comm_time,
            sharded_memory=sharded_memory,
        )

    def estimate_sharded_layer_time(
        self,
        sharded_layer: "ShardedLayerConfig",
        device: Device,
        bandwidth_gbps: float = 400.0,
        **kwargs,
    ) -> ShardedKernelResult:
        """Estimate time for a ShardedLayerConfig directly.

        Args:
            sharded_layer: ShardedLayerConfig from build_sharded_layers()
            device: Target device
            bandwidth_gbps: Communication bandwidth
            **kwargs: Additional parameters (batch_size, seq_len, hidden_size)

        Returns:
            ShardedKernelResult with time and comm time
        """
        sharded_time = self._estimate_time_from_flops(
            sharded_layer.sharded_flops,
            sharded_layer.sharded_output_shape,
            kwargs.get("dtype", "fp16"),
            device,
        )

        comm_time = 0.0
        if sharded_layer.comm_after_bytes > 0:
            tp = kwargs.get("tp_degree", 1)
            if tp > 1:
                comm_time = self.estimate_comm_time(
                    "allreduce",
                    sharded_layer.comm_after_bytes,
                    tp,
                    bandwidth_gbps,
                )

        return ShardedKernelResult(
            sharded_shape=sharded_layer.sharded_output_shape,
            sharded_flops=sharded_layer.sharded_flops,
            sharded_time=sharded_time,
            comm_time=comm_time,
            sharded_memory=sharded_layer.sharded_activation_bytes,
        )

    def _compute_sharded_shape(
        self,
        original_shape: Tuple[int, ...],
        sharding_info: Optional["ShardingInfo"],
        strategy: "StrategyConfig",
    ) -> Tuple[int, ...]:
        """Compute sharded shape from original shape and sharding info.

        Args:
            original_shape: Original output shape
            sharding_info: Sharding metadata
            strategy: Parallelism strategy

        Returns:
            Sharded shape (on single device)
        """
        if not sharding_info or not original_shape:
            return original_shape

        tp = strategy.tp_degree
        sp = strategy.sp_degree

        sharded_shape = list(original_shape)

        if "seq_len" in sharding_info.shardable_dims and sp > 1:
            seq_dim = sharding_info.shardable_dims["seq_len"]
            sharded_seq = seq_dim.get_sharded_size(sp)
            if len(sharded_shape) >= 2:
                sharded_shape[1] = sharded_seq

        return tuple(sharded_shape)

    def _estimate_time_from_flops(
        self,
        flops: int,
        shape: Tuple[int, ...],
        dtype: str,
        device: Device,
    ) -> float:
        """Estimate time from FLOPs using Roofline model.

        Args:
            flops: FLOPs count
            shape: Output shape (for memory bound detection)
            dtype: Data type
            device: Target device

        Returns:
            Estimated time in seconds
        """
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
        dtype_size = dtype_sizes.get(dtype, 2)

        elements = 1
        for dim in shape:
            if isinstance(dim, int) and dim > 0:
                elements *= dim

        bytes_accessed = elements * dtype_size * 2

        peak_flops = device.get_compute_tflops(dtype) * 1e12
        memory_bw = device.get_memory_bw_gbps() * 1e9

        arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else float("inf")

        threshold_cube = 200.0

        if arithmetic_intensity < threshold_cube:
            achievable_flops = min(peak_flops, arithmetic_intensity * memory_bw)
        else:
            achievable_flops = peak_flops

        efficiency = 0.7
        achievable_flops *= efficiency

        time_sec = flops / achievable_flops if achievable_flops > 0 else 0

        return time_sec

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

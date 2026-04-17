"""Profiling Backend - Performance estimation from measured data.

This backend provides performance estimates based on:
1. Lookup table from profiling measurements
2. Interpolation for unmeasured configurations
3. Nearest-neighbor fallback when interpolation fails

Data format:
{
    "kernel_name": {
        "shape_key": {
            "forward_time_ms": 1.234,
            "backward_time_ms": 2.468,
            "memory_bytes": 8192,
            "dtype": "fp16",
            "device": "H100-SXM-80GB"
        }
    }
}

Shape key format: "dim1_dim2_dim3" (e.g., "4096_5120_4096" for linear)
"""

from typing import Dict, Any, Optional, Tuple, List
import json
import math
import os

from .base import KernelBackend, BackendConfig
from ..functional import KernelResult
from ...hardware.device import Device
from ...utils.constants import DTYPE_SIZES


class ProfilingBackend(KernelBackend):
    """Backend using measured profiling data.

    This backend loads profiling measurements from JSON files
    and provides lookup/interpolation capabilities.

    Features:
    - Load profiling data from JSON file
    - Exact lookup for measured configurations
    - Linear interpolation for similar shapes
    - Nearest-neighbor fallback
    - Multi-device support
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._profiling_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._device_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._interpolation_threshold = 0.2

    def initialize(self) -> None:
        data_path = self.config.profiling_data_path
        if data_path and os.path.exists(data_path):
            self._load_profiling_data(data_path)
        self._initialized = True

    def _load_profiling_data(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)

        for kernel_name, shape_data in data.items():
            if kernel_name not in self._profiling_data:
                self._profiling_data[kernel_name] = {}

            for shape_key, metrics in shape_data.items():
                device_name = metrics.get("device", "default")
                if device_name not in self._device_data:
                    self._device_data[device_name] = {}
                if kernel_name not in self._device_data[device_name]:
                    self._device_data[device_name][kernel_name] = {}

                self._profiling_data[kernel_name][shape_key] = metrics
                self._device_data[device_name][kernel_name][shape_key] = metrics

    def load_data(self, data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        for kernel_name, shape_data in data.items():
            if kernel_name not in self._profiling_data:
                self._profiling_data[kernel_name] = {}

            for shape_key, metrics in shape_data.items():
                device_name = metrics.get("device", "default")
                if device_name not in self._device_data:
                    self._device_data[device_name] = {}
                if kernel_name not in self._device_data[device_name]:
                    self._device_data[device_name][kernel_name] = {}

                self._profiling_data[kernel_name][shape_key] = metrics
                self._device_data[device_name][kernel_name][shape_key] = metrics

    def add_measurement(
        self,
        kernel_name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        dtype: str,
        device: Device,
        forward_time_ms: float,
        backward_time_ms: Optional[float] = None,
        memory_bytes: Optional[int] = None,
    ) -> None:
        shape_key = self._make_shape_key(input_shapes, output_shape)
        device_name = device.config.name

        metrics = {
            "forward_time_ms": forward_time_ms,
            "backward_time_ms": backward_time_ms or forward_time_ms * 2,
            "memory_bytes": memory_bytes or self._estimate_memory_from_shapes(input_shapes, output_shape, dtype),
            "dtype": dtype,
            "device": device_name,
            "input_shapes": [list(s) for s in input_shapes],
            "output_shape": list(output_shape),
        }

        if kernel_name not in self._profiling_data:
            self._profiling_data[kernel_name] = {}
        if device_name not in self._device_data:
            self._device_data[device_name] = {}
        if kernel_name not in self._device_data[device_name]:
            self._device_data[device_name][kernel_name] = {}

        self._profiling_data[kernel_name][shape_key] = metrics
        self._device_data[device_name][kernel_name][shape_key] = metrics

    def _make_shape_key(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
    ) -> str:
        all_dims = []
        for shape in input_shapes:
            all_dims.extend(shape)
        all_dims.extend(output_shape)
        return "_".join(str(d) for d in all_dims)

    def _estimate_memory_from_shapes(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        dtype: str,
    ) -> int:
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        total_bytes = 0
        for shape in input_shapes:
            total_bytes += math.prod(shape) * dtype_size
        total_bytes += math.prod(output_shape) * dtype_size
        return total_bytes

    def estimate_compute_time(
        self,
        kernel_name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        dtype: str,
        device: Device,
        **kwargs,
    ) -> float:
        device_name = device.config.name
        shape_key = self._make_shape_key(input_shapes, output_shape)

        exact_match = self._lookup_exact(kernel_name, shape_key, device_name, dtype)
        if exact_match is not None:
            return exact_match["forward_time_ms"] * 1e-3

        interpolated = self._interpolate(kernel_name, input_shapes, device_name, dtype)
        if interpolated is not None:
            return interpolated["forward_time_ms"] * 1e-3

        nearest = self._nearest_neighbor(kernel_name, input_shapes, device_name, dtype)
        if nearest is not None:
            return nearest["forward_time_ms"] * 1e-3

        return 0.0

    def estimate_compute_time_from_result(self, result: KernelResult, device: Device, **kwargs) -> float:
        return self.estimate_compute_time(
            kwargs.get("kernel_name", "unknown"), result.input_shapes, result.output, result.dtype, device, **kwargs
        )

    def estimate_comm_time(
        self, comm_type: str, data_size_bytes: int, num_ranks: int, bandwidth_gbps: float, **kwargs
    ) -> float:
        comm_key = f"{comm_type}_{data_size_bytes}_{num_ranks}"
        device_name = kwargs.get("device_name", "default")

        if device_name in self._device_data:
            if "communication" in self._device_data[device_name]:
                if comm_key in self._device_data[device_name]["communication"]:
                    metrics = self._device_data[device_name]["communication"][comm_key]
                    return metrics.get("time_ms", 0.0) * 1e-3

        bandwidth_bytes_per_sec = bandwidth_gbps * 1e9
        if num_ranks <= 1:
            return 0.0

        if comm_type == "allreduce":
            steps = math.ceil(math.log2(num_ranks))
            return steps * data_size_bytes / bandwidth_bytes_per_sec
        elif comm_type == "allgather":
            steps = math.ceil(math.log2(num_ranks))
            return steps * data_size_bytes / bandwidth_bytes_per_sec
        else:
            return data_size_bytes / bandwidth_bytes_per_sec

    def estimate_memory(
        self, kernel_name: str, input_shapes: List[Tuple[int, ...]], output_shape: Tuple[int, ...], dtype: str, **kwargs
    ) -> int:
        device_name = kwargs.get("device_name", "default")
        shape_key = self._make_shape_key(input_shapes, output_shape)

        exact_match = self._lookup_exact(kernel_name, shape_key, device_name, dtype)
        if exact_match is not None:
            return exact_match.get("memory_bytes", 0)

        return self._estimate_memory_from_shapes(input_shapes, output_shape, dtype)

    def estimate_memory_from_result(self, result: KernelResult, **kwargs) -> int:
        return self.estimate_memory(
            kwargs.get("kernel_name", "unknown"), result.input_shapes, result.output, result.dtype, **kwargs
        )

    def _lookup_exact(
        self,
        kernel_name: str,
        shape_key: str,
        device_name: str,
        dtype: str,
    ) -> Optional[Dict[str, Any]]:
        if device_name in self._device_data:
            if kernel_name in self._device_data[device_name]:
                if shape_key in self._device_data[device_name][kernel_name]:
                    metrics = self._device_data[device_name][kernel_name][shape_key]
                    if metrics.get("dtype") == dtype:
                        return metrics
        return None

    def _interpolate(
        self,
        kernel_name: str,
        input_shapes: List[Tuple[int, ...]],
        device_name: str,
        dtype: str,
    ) -> Optional[Dict[str, Any]]:
        if device_name not in self._device_data:
            return None
        if kernel_name not in self._device_data[device_name]:
            return None

        target_dims = []
        for shape in input_shapes:
            target_dims.extend(shape)

        if not target_dims:
            return None

        candidates = []
        for shape_key, metrics in self._device_data[device_name][kernel_name].items():
            if metrics.get("dtype") != dtype:
                continue

            stored_dims = [int(d) for d in shape_key.split("_")]
            if len(stored_dims) != len(target_dims):
                continue

            max_ratio = 0.0
            valid = True
            for t, s in zip(target_dims, stored_dims):
                if s == 0:
                    valid = False
                    break
                ratio = abs(t / s - 1.0)
                max_ratio = max(max_ratio, ratio)

            if valid and max_ratio <= self._interpolation_threshold:
                candidates.append((max_ratio, metrics, stored_dims))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])

        best_ratio, best_metrics, best_dims = candidates[0]

        if best_ratio == 0.0:
            return best_metrics

        scale_factor = math.prod(target_dims) / math.prod(best_dims)

        interpolated = {
            "forward_time_ms": best_metrics["forward_time_ms"] * scale_factor,
            "backward_time_ms": best_metrics.get("backward_time_ms", 0.0) * scale_factor,
            "memory_bytes": int(best_metrics.get("memory_bytes", 0) * scale_factor),
            "dtype": dtype,
            "device": device_name,
        }

        return interpolated

    def _nearest_neighbor(
        self,
        kernel_name: str,
        input_shapes: List[Tuple[int, ...]],
        device_name: str,
        dtype: str,
    ) -> Optional[Dict[str, Any]]:
        if device_name not in self._device_data:
            return None
        if kernel_name not in self._device_data[device_name]:
            return None

        target_dims = []
        for shape in input_shapes:
            target_dims.extend(shape)

        if not target_dims:
            return None

        best_match = None
        best_distance = float("inf")

        for shape_key, metrics in self._device_data[device_name][kernel_name].items():
            if metrics.get("dtype") != dtype:
                continue

            stored_dims = [int(d) for d in shape_key.split("_")]
            if len(stored_dims) != len(target_dims):
                continue

            distance = sum(abs(t - s) for t, s in zip(target_dims, stored_dims))
            if distance < best_distance:
                best_distance = distance
                best_match = metrics

        return best_match

    def supports_kernel(self, kernel_name: str) -> bool:
        return kernel_name in self._profiling_data

    def get_backend_type(self) -> str:
        return "profiling"

    def get_available_kernels(self) -> List[str]:
        return list(self._profiling_data.keys())

    def get_available_shapes(self, kernel_name: str) -> List[str]:
        if kernel_name not in self._profiling_data:
            return []
        return list(self._profiling_data[kernel_name].keys())

    def save_to_file(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._profiling_data, f, indent=2)

    def get_backend_type(self) -> str:
        return "profiling"

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["available_kernels"] = self.get_available_kernels()
        base_dict["profiling_data_path"] = self.config.profiling_data_path
        return base_dict

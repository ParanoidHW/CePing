"""Theory Backend - Analytical performance estimation using FLOPs/Roofline model.

This backend provides theoretical performance estimates based on:
1. FLOPs calculation for compute kernels
2. Roofline model for compute/memory bound analysis
3. Bandwidth-based communication time estimation

The TheoryBackend is the default backend, maintaining backward compatibility
with existing kernel evaluation logic.
"""

from typing import Dict, Optional, Tuple, List
import math

from .base import KernelBackend, BackendConfig
from ..functional import (
    KernelResult,
    linear,
    bmm,
    scaled_dot_product_attention,
    flash_attention,
    mla_attention,
    layer_norm,
    rms_norm,
    silu,
    gelu,
    relu,
    softmax,
    conv2d,
    conv3d,
    embedding,
)
from ..base import KernelConfig
from ...hardware.device import Device, ComputeUnitType


class TheoryBackend(KernelBackend):
    """Backend using theoretical FLOPs/Roofline model.

    This backend estimates performance analytically:
    - Compute time: FLOPs / achievable_FLOPS (roofline model)
    - Communication time: bytes / bandwidth
    - Memory: computed from tensor shapes

    The roofline model considers:
    - Peak compute throughput (TFLOPS)
    - Memory bandwidth (GB/s)
    - Arithmetic intensity (FLOPs/byte)

    When arithmetic intensity < ridge_point, kernel is memory-bound.
    When arithmetic intensity >= ridge_point, kernel is compute-bound.
    """

    KERNEL_FUNCS = {
        "linear": linear,
        "bmm": bmm,
        "matmul": bmm,
        "attention": scaled_dot_product_attention,
        "scaled_dot_product_attention": scaled_dot_product_attention,
        "flash_attention": flash_attention,
        "mla_attention": mla_attention,
        "layer_norm": layer_norm,
        "rms_norm": rms_norm,
        "silu": silu,
        "gelu": gelu,
        "relu": relu,
        "softmax": softmax,
        "conv2d": conv2d,
        "conv3d": conv3d,
        "embedding": embedding,
    }

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._kernel_configs: Dict[str, KernelConfig] = {}
        self._unit_type_map = {
            "cube": ComputeUnitType.CUBE_TENSOR_CORE,
            "vector": ComputeUnitType.VECTOR_CUDA_CORE,
        }

    def initialize(self) -> None:
        self._initialized = True

    def _get_unit_type(self, unit_type_str: str) -> ComputeUnitType:
        return self._unit_type_map.get(unit_type_str, ComputeUnitType.CUBE_TENSOR_CORE)

    def estimate_compute_time(
        self,
        kernel_name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        dtype: str,
        device: Device,
        **kwargs,
    ) -> float:
        result = self._get_kernel_result(kernel_name, input_shapes, dtype, **kwargs)
        if result is None:
            return 0.0
        return self.estimate_compute_time_from_result(result, device, **kwargs)

    def estimate_compute_time_from_result(self, result: KernelResult, device: Device, **kwargs) -> float:
        flops = result.flops
        bytes_accessed = result.bytes_accessed
        dtype = result.dtype
        unit_type = self._get_unit_type(result.unit_type)

        arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else float("inf")

        achievable_flops = device.estimate_roofline_flops(arithmetic_intensity, dtype, unit_type)

        compute_time = flops / achievable_flops

        latency_us = kwargs.get("latency_us", 0.0)
        if latency_us:
            compute_time += latency_us * 1e-6

        return compute_time

    def estimate_backward_time_from_result(self, result: KernelResult, device: Device, **kwargs) -> float:
        flops_backward = result.flops_backward
        bytes_accessed_backward = result.bytes_accessed_backward

        if flops_backward == 0:
            forward_time = self.estimate_compute_time_from_result(result, device, **kwargs)
            return forward_time * 2

        dtype = result.dtype
        unit_type = self._get_unit_type(result.unit_type)

        arithmetic_intensity_backward = (
            flops_backward / bytes_accessed_backward if bytes_accessed_backward > 0 else float("inf")
        )

        achievable_flops = device.estimate_roofline_flops(arithmetic_intensity_backward, dtype, unit_type)

        backward_time = flops_backward / achievable_flops

        latency_us = kwargs.get("latency_us", 0.0)
        if latency_us:
            backward_time += latency_us * 1e-6

        return backward_time

    def estimate_training_time_from_result(self, result: KernelResult, device: Device, **kwargs) -> float:
        forward_time = self.estimate_compute_time_from_result(result, device, **kwargs)
        backward_time = self.estimate_backward_time_from_result(result, device, **kwargs)
        return forward_time + backward_time

    def estimate_comm_time(
        self, comm_type: str, data_size_bytes: int, num_ranks: int, bandwidth_gbps: float, **kwargs
    ) -> float:
        bandwidth_bytes_per_sec = bandwidth_gbps * 1e9

        if num_ranks <= 1:
            return 0.0

        if comm_type == "allreduce":
            steps = math.ceil(math.log2(num_ranks))
            return steps * data_size_bytes / bandwidth_bytes_per_sec

        elif comm_type == "allgather":
            steps = math.ceil(math.log2(num_ranks))
            return steps * data_size_bytes / bandwidth_bytes_per_sec

        elif comm_type == "alltoall":
            return data_size_bytes / bandwidth_bytes_per_sec

        elif comm_type == "broadcast":
            steps = math.ceil(math.log2(num_ranks))
            return steps * data_size_bytes / bandwidth_bytes_per_sec

        elif comm_type == "reduce_scatter":
            return num_ranks * data_size_bytes / bandwidth_bytes_per_sec / 2

        elif comm_type == "p2p":
            return data_size_bytes / bandwidth_bytes_per_sec

        else:
            return data_size_bytes / bandwidth_bytes_per_sec

    def estimate_memory(
        self, kernel_name: str, input_shapes: List[Tuple[int, ...]], output_shape: Tuple[int, ...], dtype: str, **kwargs
    ) -> int:
        result = self._get_kernel_result(kernel_name, input_shapes, dtype, **kwargs)
        if result is None:
            return 0
        return self.estimate_memory_from_result(result, **kwargs)

    def estimate_memory_from_result(self, result: KernelResult, **kwargs) -> int:
        return result.bytes_accessed

    def estimate_backward_memory_from_result(self, result: KernelResult, **kwargs) -> int:
        return result.bytes_accessed_backward

    def _get_kernel_result(
        self, kernel_name: str, input_shapes: List[Tuple[int, ...]], dtype: str, **kwargs
    ) -> Optional[KernelResult]:
        kernel_func = self.KERNEL_FUNCS.get(kernel_name)
        if kernel_func is None:
            return None

        try:
            if kernel_name == "linear":
                if len(input_shapes) >= 2:
                    return kernel_func(input_shapes[0], input_shapes[1], dtype=dtype)
            elif kernel_name in ["bmm", "matmul"]:
                if len(input_shapes) >= 2:
                    return kernel_func(input_shapes[0], input_shapes[1], dtype=dtype)
            elif kernel_name in ["attention", "scaled_dot_product_attention"]:
                if len(input_shapes) >= 3:
                    return kernel_func(
                        input_shapes[0],
                        input_shapes[1],
                        input_shapes[2],
                        dtype=dtype,
                        is_causal=kwargs.get("is_causal", False),
                        use_gqa=kwargs.get("use_gqa", False),
                    )
            elif kernel_name == "flash_attention":
                if len(input_shapes) >= 3:
                    return kernel_func(
                        input_shapes[0],
                        input_shapes[1],
                        input_shapes[2],
                        dtype=dtype,
                        is_causal=kwargs.get("is_causal", False),
                        use_gqa=kwargs.get("use_gqa", False),
                        block_size=kwargs.get("block_size", 128),
                    )
            elif kernel_name == "mla_attention":
                if len(input_shapes) >= 2:
                    return kernel_func(
                        input_shapes[0],
                        input_shapes[1],
                        key=input_shapes[2] if len(input_shapes) > 2 else None,
                        value=input_shapes[3] if len(input_shapes) > 3 else None,
                        dtype=dtype,
                        use_absorb=kwargs.get("use_absorb", False),
                        qk_head_dim=kwargs.get("qk_head_dim", 128),
                        v_head_dim=kwargs.get("v_head_dim", 128),
                        kv_lora_rank=kwargs.get("kv_lora_rank", 512),
                    )
            elif kernel_name in ["layer_norm", "rms_norm"]:
                if len(input_shapes) >= 1:
                    normalized_shape = kwargs.get("normalized_shape", (input_shapes[0][-1],))
                    if kernel_name == "layer_norm":
                        return kernel_func(
                            input_shapes[0],
                            normalized_shape,
                            elementwise_affine=kwargs.get("elementwise_affine", True),
                            dtype=dtype,
                        )
                    else:
                        return kernel_func(input_shapes[0], dtype=dtype)
            elif kernel_name in ["silu", "gelu", "relu", "softmax"]:
                if len(input_shapes) >= 1:
                    if kernel_name == "gelu":
                        return kernel_func(input_shapes[0], approximate=kwargs.get("approximate", "none"), dtype=dtype)
                    elif kernel_name == "softmax":
                        return kernel_func(input_shapes[0], dim=kwargs.get("dim", -1), dtype=dtype)
                    else:
                        return kernel_func(input_shapes[0], dtype=dtype)
            elif kernel_name == "conv2d":
                if len(input_shapes) >= 2:
                    return kernel_func(
                        input_shapes[0],
                        input_shapes[1],
                        stride=kwargs.get("stride", (1, 1)),
                        padding=kwargs.get("padding", (0, 0)),
                        dtype=dtype,
                    )
            elif kernel_name == "conv3d":
                if len(input_shapes) >= 2:
                    return kernel_func(
                        input_shapes[0],
                        input_shapes[1],
                        stride=kwargs.get("stride", (1, 1, 1)),
                        padding=kwargs.get("padding", (0, 0, 0)),
                        dtype=dtype,
                    )
            elif kernel_name == "embedding":
                num_embeddings = kwargs.get("num_embeddings", 32000)
                embedding_dim = kwargs.get("embedding_dim", 4096)
                return kernel_func(
                    num_embeddings, embedding_dim, input_shapes[0] if input_shapes else (1,), dtype=dtype
                )
        except Exception:
            return None

        return None

    def supports_kernel(self, kernel_name: str) -> bool:
        return kernel_name in self.KERNEL_FUNCS

    def get_backend_type(self) -> str:
        return "theory"

    def compute_arithmetic_intensity(self, flops: int, bytes_accessed: int) -> float:
        if bytes_accessed == 0:
            return float("inf")
        return flops / bytes_accessed

    def is_memory_bound(
        self,
        flops: int,
        bytes_accessed: int,
        dtype: str,
        device: Device,
        unit_type: ComputeUnitType = ComputeUnitType.CUBE_TENSOR_CORE,
    ) -> bool:
        ai = self.compute_arithmetic_intensity(flops, bytes_accessed)
        peak_flops = device.get_compute_tflops(dtype, unit_type) * 1e12
        mem_bw = device.get_memory_bw_gbps() * 1e9
        ridge_point = peak_flops / mem_bw
        return ai < ridge_point

    def compute_kernel_result(
        self, kernel_name: str, input_shapes: List[Tuple[int, ...]], dtype: str = "fp16", **kwargs
    ) -> Optional[KernelResult]:
        return self._get_kernel_result(kernel_name, input_shapes, dtype, **kwargs)

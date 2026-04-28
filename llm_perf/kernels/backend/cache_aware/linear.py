"""Linear layer cache-aware memory access calculation.

Linear layer memory access pattern:
- Input: batch × seq × in_features (read once)
- Weight: in_features × out_features (read multiple times, per batch/seq position)
- Output: batch × seq × out_features (write once)

Cache behavior:
- Weight matrix is reused for each batch/seq position
  -> High cache hit rate if weight fits in L2 cache
  -> Moderate hit rate if weight fits in L2 but not L1
- Input/output tensors are stream-like (read/write once)
  -> Low cache hit rate (no reuse)

Reference:
    NVIDIA H100: 50MB L2 can hold ~25M fp16 elements (12.5M weight values)
    Large models: weight may exceed L2 capacity (e.g., 4096×5120 = 20M elements)
"""

from llm_perf.hardware.device import Device
from llm_perf.kernels.functional import KernelResult
from .base import CacheAwareCalculator, CacheAwareResult
from .registry import CacheAwareRegistry


class LinearCacheAware(CacheAwareCalculator):
    """Cache-aware calculator for Linear layer.

    Linear layer memory access analysis:
    1. Weight matrix reuse pattern:
       - Weight size = in_features × out_features × dtype_size
       - Reuse factor = batch × seq (each position needs weight)
       - If weight fits in L2: cache hit rate ~90%
       - If weight exceeds L2: cache hit rate drops

    2. Input/output streaming pattern:
       - Input: read once per element (no reuse)
       - Output: write once per element (no reuse)
       - Cache hit rate ~0% (no temporal locality)

    3. Effective HBM access:
       - If weight fits in L2: only load weight once to L2, then reuse
       - If weight exceeds L2: must load weight from HBM for each batch/seq position

    Formula:
        bytes_actual = input_bytes + output_bytes + weight_bytes_actual

        weight_bytes_actual depends on cache behavior:
        - Fits in L2: weight_bytes (load once) + cache hit bytes for reuse
        - Exceeds L2: weight_bytes × (batch × seq) (load from HBM for each position)
    """

    KERNEL_NAME = "linear"

    def __init__(self, device: Device):
        super().__init__(device)
        CacheAwareRegistry.register(self.KERNEL_NAME, LinearCacheAware)

    def supports_kernel(self, kernel_name: str) -> bool:
        return kernel_name in ["linear", "matmul", "bmm"]

    def calculate(
        self,
        kernel_result: KernelResult,
        batch_size: int,
        **kwargs,
    ) -> CacheAwareResult:
        """Calculate cache-aware memory access for Linear layer.

        Args:
            kernel_result: KernelResult from linear() functional API
            batch_size: Batch size (or batch × seq for language models)
            **kwargs: seq_len, in_features, out_features

        Returns:
            CacheAwareResult with L1/L2/HBM breakdown
        """
        bytes_theory = kernel_result.bytes_accessed

        input_shapes = kernel_result.input_shapes
        dtype_size = kernel_result.get_dtype_size()

        if len(input_shapes) < 2:
            return CacheAwareResult(
                bytes_theory=bytes_theory,
                bytes_actual=bytes_theory,
                hbm_access_bytes=bytes_theory,
                cache_efficiency=0.0,
                effective_bandwidth_gbps=self._cache_config["hbm_bw_gbps"],
            )

        input_shape = input_shapes[0]
        weight_shape = input_shapes[1]

        seq_len = kwargs.get("seq_len", 1)
        effective_batch = batch_size * seq_len if seq_len > 1 else batch_size

        in_features = input_shape[-1] if len(input_shape) > 0 else 1
        out_features = weight_shape[0] if len(weight_shape) > 0 else 1

        input_bytes = effective_batch * in_features * dtype_size
        output_bytes = effective_batch * out_features * dtype_size
        weight_bytes = in_features * out_features * dtype_size

        weight_fits_l1 = self._fits_in_l1(weight_bytes)
        weight_fits_l2 = self._fits_in_l2(weight_bytes)

        reuse_factor = effective_batch

        if weight_fits_l1:
            weight_l1_hits = weight_bytes * (reuse_factor - 1)
            weight_l2_hits = 0
            weight_hbm_access = weight_bytes
            weight_actual = weight_bytes + weight_l1_hits
        elif weight_fits_l2:
            weight_l1_hits = 0
            weight_l2_hits = weight_bytes * (reuse_factor - 1)
            weight_hbm_access = weight_bytes
        else:
            weight_l1_hits = 0
            weight_l2_hits = weight_bytes * reuse_factor * 0.1
            weight_hbm_access = weight_bytes * reuse_factor * 0.9

        input_l1_hits = 0
        input_l2_hits = 0
        input_hbm = input_bytes

        output_l1_hits = 0
        output_l2_hits = 0
        output_hbm = output_bytes

        l1_hit_bytes = weight_l1_hits + input_l1_hits + output_l1_hits
        l2_hit_bytes = weight_l2_hits + input_l2_hits + output_l2_hits
        hbm_access_bytes = weight_hbm_access + input_hbm + output_hbm

        bytes_actual = l1_hit_bytes + l2_hit_bytes + hbm_access_bytes

        cache_efficiency = (l1_hit_bytes + l2_hit_bytes) / bytes_actual if bytes_actual > 0 else 0.0

        effective_bandwidth_gbps = self._estimate_effective_bandwidth(
            l1_hit_bytes, l2_hit_bytes, hbm_access_bytes, bytes_actual
        )

        details = {
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "weight_bytes": weight_bytes,
            "reuse_factor": reuse_factor,
            "weight_fits_l1": weight_fits_l1,
            "weight_fits_l2": weight_fits_l2,
            "weight_l1_hits": weight_l1_hits,
            "weight_l2_hits": weight_l2_hits,
            "weight_hbm_access": weight_hbm_access,
            "l1_cache_kb": self._cache_config["l1_cache_kb"],
            "l2_cache_kb": self._cache_config["l2_cache_kb"],
        }

        return CacheAwareResult(
            bytes_theory=bytes_theory,
            bytes_actual=bytes_actual,
            l1_hit_bytes=l1_hit_bytes,
            l2_hit_bytes=l2_hit_bytes,
            hbm_access_bytes=hbm_access_bytes,
            cache_efficiency=cache_efficiency,
            effective_bandwidth_gbps=effective_bandwidth_gbps,
            details=details,
        )
"""Test cache-aware memory access calculation for Linear layer.

Tests cover:
- Weight reuse pattern analysis
- L1/L2 cache hit rate calculation
- Effective bandwidth estimation
- Different batch sizes and weight sizes
"""

import pytest

from llm_perf.hardware.device import Device
from llm_perf.kernels.functional import linear
from llm_perf.kernels.backend.cache_aware import LinearCacheAware, CacheAwareRegistry


class TestLinearCacheAware:
    """Test Linear cache-aware calculation."""

    @pytest.fixture
    def h100_device(self):
        """H100 device fixture."""
        return Device.from_preset("H100-SXM-80GB")

    @pytest.fixture
    def calculator(self, h100_device):
        """Linear cache-aware calculator fixture."""
        return LinearCacheAware(h100_device)

    def test_registry_registration(self, h100_device):
        """Test that LinearCacheAware is registered."""
        assert CacheAwareRegistry.has("linear")
        
        calc = CacheAwareRegistry.get("linear", h100_device)
        assert calc is not None
        assert isinstance(calc, LinearCacheAware)

    def test_supports_kernel(self, calculator):
        """Test kernel support check."""
        assert calculator.supports_kernel("linear")
        assert calculator.supports_kernel("matmul")
        assert calculator.supports_kernel("bmm")
        assert not calculator.supports_kernel("attention")

    def test_small_weight_fits_l1(self, calculator, h100_device):
        """Test small weight that fits in L1 cache.

        Small weight: 128 × 128 fp16 = 32KB < 128KB (L1 capacity)
        Expected: high L1 hit rate due to weight reuse
        """
        in_features = 128
        out_features = 128
        batch_size = 1000

        kernel_result = linear(
            (batch_size, in_features),
            (out_features, in_features),
            dtype="fp16",
        )

        cache_result = calculator.calculate(kernel_result, batch_size)

        assert cache_result.bytes_theory == kernel_result.bytes_accessed
        assert cache_result.bytes_actual > 0
        assert cache_result.l1_hit_bytes > 0
        assert cache_result.hbm_access_bytes > 0
        assert cache_result.cache_efficiency > 0.0

        assert cache_result.details["weight_fits_l1"] is True
        assert cache_result.details["reuse_factor"] == batch_size

    def test_medium_weight_fits_l2(self, calculator, h100_device):
        """Test medium weight that fits in L2 cache.

        Medium weight: 4096 × 1024 fp16 = 8MB < 50MB (L2 capacity)
        Expected: moderate L2 hit rate, low L1 hit rate
        """
        in_features = 4096
        out_features = 1024
        batch_size = 512
        seq_len = 2048

        kernel_result = linear(
            (batch_size, seq_len, in_features),
            (out_features, in_features),
            dtype="fp16",
        )

        effective_batch = batch_size * seq_len
        cache_result = calculator.calculate(kernel_result, batch_size, seq_len=seq_len)

        weight_bytes = in_features * out_features * 2
        l2_capacity = h100_device.config.l2_cache_mb * 1024 * 1024

        assert cache_result.bytes_theory == kernel_result.bytes_accessed
        assert cache_result.details["weight_fits_l1"] is False
        assert cache_result.details["reuse_factor"] == effective_batch

        if weight_bytes < l2_capacity:
            assert cache_result.details["weight_fits_l2"] is True
            assert cache_result.l2_hit_bytes > 0
            assert cache_result.cache_efficiency > 0.5

    def test_large_weight_exceeds_l2(self, calculator, h100_device):
        """Test large weight that exceeds L2 cache.

        Large weight: 12288 × 14336 fp16 = 355MB > 50MB (L2 capacity)
        Expected: low cache hit rate, high HBM access
        """
        in_features = 12288
        out_features = 14336
        batch_size = 128
        seq_len = 1024

        kernel_result = linear(
            (batch_size, seq_len, in_features),
            (out_features, in_features),
            dtype="fp16",
        )

        effective_batch = batch_size * seq_len
        cache_result = calculator.calculate(kernel_result, batch_size, seq_len=seq_len)

        weight_bytes = in_features * out_features * 2
        l2_capacity = h100_device.config.l2_cache_mb * 1024 * 1024

        assert cache_result.bytes_theory == kernel_result.bytes_accessed
        assert cache_result.details["weight_fits_l1"] is False

        if weight_bytes > l2_capacity:
            assert cache_result.details["weight_fits_l2"] is False
            assert cache_result.hbm_access_bytes > cache_result.bytes_theory * 0.5
            assert cache_result.cache_efficiency < 0.5

    def test_effective_bandwidth_calculation(self, calculator, h100_device):
        """Test effective bandwidth calculation.

        Effective bandwidth should be weighted average of:
        - L1: 1000 GB/s
        - L2: 400 GB/s
        - HBM: 3 TB/s (from device config)
        """
        kernel_result = linear((128, 128), (128, 128), dtype="fp16")
        cache_result = calculator.calculate(kernel_result, batch_size=128)

        assert cache_result.effective_bandwidth_gbps > 0
        assert cache_result.effective_bandwidth_gbps < h100_device.get_memory_bw_gbps()

        if cache_result.l1_hit_bytes > 0:
            assert cache_result.effective_bandwidth_gbps > 400

    def test_different_batch_sizes(self, calculator):
        """Test cache behavior with different batch sizes.

        Larger batch size -> more reuse -> higher cache efficiency
        """
        in_features = 1024
        out_features = 1024

        kernel_result = linear(
            (in_features,),
            (out_features, in_features),
            dtype="fp16",
        )

        small_batch_result = calculator.calculate(kernel_result, batch_size=16)
        large_batch_result = calculator.calculate(kernel_result, batch_size=1024)

        if small_batch_result.details["weight_fits_l2"]:
            assert large_batch_result.cache_efficiency > small_batch_result.cache_efficiency

    def test_mismatched_shapes(self, calculator):
        """Test handling of mismatched input shapes."""
        kernel_result = linear((1,), (1,), dtype="fp16")
        
        cache_result = calculator.calculate(kernel_result, batch_size=1)
        
        assert cache_result.bytes_theory == kernel_result.bytes_accessed
        assert cache_result.bytes_actual > 0

    def test_to_dict_output(self, calculator):
        """Test CacheAwareResult.to_dict() output."""
        kernel_result = linear((1024,), (4096, 1024), dtype="fp16")
        cache_result = calculator.calculate(kernel_result, batch_size=128)
        
        result_dict = cache_result.to_dict()
        
        assert "bytes_theory" in result_dict
        assert "bytes_actual" in result_dict
        assert "l1_hit_bytes" in result_dict
        assert "l2_hit_bytes" in result_dict
        assert "hbm_access_bytes" in result_dict
        assert "cache_efficiency" in result_dict
        assert "effective_bandwidth_gbps" in result_dict
        assert "details" in result_dict
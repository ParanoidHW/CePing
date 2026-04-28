"""Test cache-aware memory access calculation for Flash Attention.

Tests cover:
- Blocking strategy analysis
- L1/L2 cache hit rate calculation
- Causal vs non-causal attention
- Different sequence lengths and block sizes
"""

import pytest

from llm_perf.hardware.device import Device
from llm_perf.kernels.functional import flash_attention
from llm_perf.kernels.backend.cache_aware import FlashAttentionCacheAware, CacheAwareRegistry


class TestFlashAttentionCacheAware:
    """Test Flash Attention cache-aware calculation."""

    @pytest.fixture
    def h100_device(self):
        """H100 device fixture."""
        return Device.from_preset("H100-SXM-80GB")

    @pytest.fixture
    def calculator(self, h100_device):
        """Flash Attention cache-aware calculator fixture."""
        return FlashAttentionCacheAware(h100_device)

    def test_registry_registration(self, h100_device):
        """Test that FlashAttentionCacheAware is registered."""
        assert CacheAwareRegistry.has("flash_attention")
        
        calc = CacheAwareRegistry.get("flash_attention", h100_device)
        assert calc is not None
        assert isinstance(calc, FlashAttentionCacheAware)

    def test_supports_kernel(self, calculator):
        """Test kernel support check."""
        assert calculator.supports_kernel("flash_attention")
        assert calculator.supports_kernel("attention")
        assert calculator.supports_kernel("scaled_dot_product_attention")
        assert not calculator.supports_kernel("linear")

    def test_causal_attention_blocking(self, calculator):
        """Test causal attention blocking strategy.

        Causal attention: triangular pattern
        - Q loaded multiple times (for each KV block)
        - K/V loaded multiple times (triangular)
        """
        batch_size = 8
        num_heads = 32
        seq_len = 2048
        head_dim = 128
        block_size = 128

        kernel_result = flash_attention(
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, num_heads, seq_len, head_dim),
            is_causal=True,
            dtype="fp16",
            block_size=block_size,
        )

        cache_result = calculator.calculate(
            kernel_result,
            batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            is_causal=True,
            block_size=block_size,
        )

        assert cache_result.bytes_theory == kernel_result.bytes_accessed
        assert cache_result.bytes_actual > 0

        assert cache_result.details["is_causal"] is True
        assert cache_result.details["block_size"] == block_size

        num_blocks = (seq_len + block_size - 1) // block_size
        assert cache_result.details["q_loads"] == num_blocks

        assert cache_result.hbm_access_bytes > 0

    def test_non_causal_attention_blocking(self, calculator):
        """Test non-causal attention blocking strategy.

        Non-causal attention: full attention pattern
        - Q loaded once
        - K/V loaded multiple times
        """
        batch_size = 8
        num_heads = 32
        seq_len = 512
        head_dim = 128
        block_size = 128

        kernel_result = flash_attention(
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, num_heads, seq_len, head_dim),
            is_causal=False,
            dtype="fp16",
            block_size=block_size,
        )

        cache_result = calculator.calculate(
            kernel_result,
            batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            is_causal=False,
            block_size=block_size,
        )

        assert cache_result.bytes_theory == kernel_result.bytes_accessed
        assert cache_result.details["is_causal"] is False
        assert cache_result.details["q_loads"] == 1

        assert cache_result.bytes_actual < cache_result.bytes_theory * 2

    def test_block_fits_in_l1(self, calculator, h100_device):
        """Test block that fits in L1 cache.

        Block size: 128 × 128 fp16 × 4 = 128KB
        Fits in H100 L1 cache (128KB per SM)
        """
        batch_size = 1
        num_heads = 1
        seq_len = 128
        head_dim = 128
        block_size = 128

        kernel_result = flash_attention(
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, num_heads, seq_len, head_dim),
            is_causal=True,
            dtype="fp16",
            block_size=block_size,
        )

        cache_result = calculator.calculate(
            kernel_result,
            batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            is_causal=True,
            block_size=block_size,
        )

        block_bytes = block_size * head_dim * 2 * 4
        l1_capacity_kb = cache_result.details.get("l1_cache_kb", 128)
        l1_capacity = l1_capacity_kb * 1024

        if block_bytes <= l1_capacity:
            assert cache_result.details["block_fits_l1"] is True
            if cache_result.details["q_loads"] > 1:
                assert cache_result.l1_hit_bytes > 0

    def test_different_sequence_lengths(self, calculator):
        """Test cache behavior with different sequence lengths.

        Longer sequence -> more blocks -> more Q/K/V loads
        """
        batch_size = 8
        num_heads = 32
        head_dim = 128
        block_size = 128

        short_seq = 128
        kernel_result_short = flash_attention(
            (batch_size, num_heads, short_seq, head_dim),
            (batch_size, num_heads, short_seq, head_dim),
            (batch_size, num_heads, short_seq, head_dim),
            is_causal=True,
            dtype="fp16",
            block_size=block_size,
        )
        cache_result_short = calculator.calculate(
            kernel_result_short,
            batch_size,
            seq_len=short_seq,
            num_heads=num_heads,
            head_dim=head_dim,
            is_causal=True,
            block_size=block_size,
        )

        long_seq = 2048
        kernel_result_long = flash_attention(
            (batch_size, num_heads, long_seq, head_dim),
            (batch_size, num_heads, long_seq, head_dim),
            (batch_size, num_heads, long_seq, head_dim),
            is_causal=True,
            dtype="fp16",
            block_size=block_size,
        )
        cache_result_long = calculator.calculate(
            kernel_result_long,
            batch_size,
            seq_len=long_seq,
            num_heads=num_heads,
            head_dim=head_dim,
            is_causal=True,
            block_size=block_size,
        )

        assert cache_result_long.details["num_q_blocks"] > cache_result_short.details["num_q_blocks"]

        assert cache_result_long.hbm_access_bytes > cache_result_short.hbm_access_bytes

    def test_effective_bandwidth_calculation(self, calculator, h100_device):
        """Test effective bandwidth calculation.

        Flash Attention should have higher effective bandwidth due to:
        - Blocking strategy (fits in L1/L2)
        - Reduced HBM traffic (scores stay in SRAM)
        """
        batch_size = 8
        num_heads = 32
        seq_len = 512
        head_dim = 128

        kernel_result = flash_attention(
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, num_heads, seq_len, head_dim),
            is_causal=True,
            dtype="fp16",
            block_size=128,
        )

        cache_result = calculator.calculate(
            kernel_result,
            batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            is_causal=True,
            block_size=128,
        )

        assert cache_result.effective_bandwidth_gbps > 0
        assert cache_result.effective_bandwidth_gbps <= h100_device.get_memory_bw_gbps()

        if cache_result.l1_hit_bytes > 0 or cache_result.l2_hit_bytes > 0:
            assert cache_result.effective_bandwidth_gbps > 100

    def test_gqa_attention(self, calculator):
        """Test GQA (Grouped Query Attention).

        GQA: kv_num_heads < num_heads
        Memory access reduced by GQA factor
        """
        batch_size = 8
        num_heads = 32
        kv_num_heads = 8
        seq_len = 512
        head_dim = 128

        kernel_result = flash_attention(
            (batch_size, num_heads, seq_len, head_dim),
            (batch_size, kv_num_heads, seq_len, head_dim),
            (batch_size, kv_num_heads, seq_len, head_dim),
            is_causal=True,
            dtype="fp16",
            use_gqa=True,
            block_size=128,
        )

        cache_result = calculator.calculate(
            kernel_result,
            batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
            is_causal=True,
            block_size=128,
        )

        assert cache_result.bytes_theory == kernel_result.bytes_accessed
        assert cache_result.details["kv_num_heads"] == kv_num_heads

    def test_to_dict_output(self, calculator):
        """Test CacheAwareResult.to_dict() output."""
        kernel_result = flash_attention(
            (1, 1, 128, 128),
            (1, 1, 128, 128),
            (1, 1, 128, 128),
            is_causal=True,
            dtype="fp16",
        )
        cache_result = calculator.calculate(kernel_result, batch_size=1, seq_len=128)
        
        result_dict = cache_result.to_dict()
        
        assert "bytes_theory" in result_dict
        assert "bytes_actual" in result_dict
        assert "l1_hit_bytes" in result_dict
        assert "l2_hit_bytes" in result_dict
        assert "hbm_access_bytes" in result_dict
        assert "cache_efficiency" in result_dict
        assert "effective_bandwidth_gbps" in result_dict
        assert "details" in result_dict
        assert "block_size" in result_dict["details"]
        assert "is_causal" in result_dict["details"]
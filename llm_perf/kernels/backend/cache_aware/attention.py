"""Flash Attention cache-aware memory access calculation.

Flash Attention memory access pattern (tiling/blocking):
- Q, K, V loaded from HBM to SRAM in blocks
- Attention scores computed in SRAM (never materialized to HBM)
- Output written from SRAM to HBM

Key optimization:
- Standard attention: O(N^2) HBM traffic for attention scores
- Flash Attention: O(N) HBM traffic (scores stay in SRAM)

Reference:
    Flash Attention paper: https://arxiv.org/abs/2205.14135
    Block size: typically 128 or 256
    H100 L1 cache: 128KB per SM (can hold ~64K fp16 elements)
"""

from llm_perf.hardware.device import Device
from llm_perf.kernels.functional import KernelResult
from .base import CacheAwareCalculator, CacheAwareResult
from .registry import CacheAwareRegistry


class FlashAttentionCacheAware(CacheAwareCalculator):
    """Cache-aware calculator for Flash Attention.

    Flash Attention memory access analysis:
    1. Blocking strategy:
       - Block size: 128 (typical for fp16)
       - Q blocks: loaded to SRAM multiple times (for each KV block)
       - K/V blocks: loaded to SRAM once per Q block (triangular for causal)

    2. HBM traffic (for causal attention):
       - Q: seq_len / block_size × batch × heads × seq_len × head_dim
       - K/V: seq_len / block_size × batch × kv_heads × seq_len × head_dim
       - O: batch × heads × seq_len × head_dim (written once)
       - Softmax stats: negligible (batch × heads × seq_len × 4 bytes)

    3. L1 cache behavior:
       - Block fits in L1 (128KB): high hit rate for intra-block computation
       - Q block loaded multiple times: temporal reuse across KV blocks
       - K/V loaded once per Q block: limited reuse for causal

    4. Effective HBM access formula (Flash Attention):
       For causal attention with triangular pattern:
       - Q loads: ~seq_len/block_size times (each Q block processes all KV blocks up to that position)
       - K/V loads: ~seq_len/block_size times (triangular pattern)
       
       Simplified model:
       bytes_actual ≈ (Q + K + V + O) × (seq_len / block_size) for causal
       bytes_actual ≈ (Q + K + V + O) × 1 for non-causal (full attention)
    """

    KERNEL_NAME = "flash_attention"

    def __init__(self, device: Device):
        super().__init__(device)
        CacheAwareRegistry.register(self.KERNEL_NAME, FlashAttentionCacheAware)

    def supports_kernel(self, kernel_name: str) -> bool:
        return kernel_name in ["flash_attention", "attention", "scaled_dot_product_attention"]

    def calculate(
        self,
        kernel_result: KernelResult,
        batch_size: int,
        **kwargs,
    ) -> CacheAwareResult:
        """Calculate cache-aware memory access for Flash Attention.

        Args:
            kernel_result: KernelResult from flash_attention() functional API
            batch_size: Batch size
            **kwargs: seq_len, num_heads, kv_num_heads, head_dim, is_causal, block_size

        Returns:
            CacheAwareResult with L1/L2/HBM breakdown
        """
        bytes_theory = kernel_result.bytes_accessed

        input_shapes = kernel_result.input_shapes
        dtype_size = kernel_result.get_dtype_size()

        if len(input_shapes) < 3:
            return CacheAwareResult(
                bytes_theory=bytes_theory,
                bytes_actual=bytes_theory,
                hbm_access_bytes=bytes_theory,
                cache_efficiency=0.0,
                effective_bandwidth_gbps=self._cache_config["hbm_bw_gbps"],
            )

        query_shape = input_shapes[0]
        key_shape = input_shapes[1]

        seq_len = kwargs.get("seq_len", query_shape[2] if len(query_shape) > 2 else 512)
        num_heads = query_shape[1] if len(query_shape) > 1 else kwargs.get("num_heads", 32)
        kv_num_heads = key_shape[1] if len(key_shape) > 1 else kwargs.get("kv_num_heads", num_heads)
        head_dim = query_shape[3] if len(query_shape) > 3 else kwargs.get("head_dim", 128)

        is_causal = kwargs.get("is_causal", False)
        block_size = kwargs.get("block_size", 128)

        q_bytes_per_block = batch_size * num_heads * block_size * head_dim * dtype_size
        kv_bytes_per_block = batch_size * kv_num_heads * block_size * head_dim * dtype_size
        o_bytes = batch_size * num_heads * seq_len * head_dim * dtype_size

        num_q_blocks = (seq_len + block_size - 1) // block_size
        num_kv_blocks = (seq_len + block_size - 1) // block_size

        if is_causal:
            q_loads = num_q_blocks
            kv_loads = num_kv_blocks
        else:
            q_loads = 1
            kv_loads = num_kv_blocks

        block_size_bytes = block_size * head_dim * dtype_size * 4

        block_fits_l1 = self._fits_in_l1(block_size_bytes)
        block_fits_l2 = self._fits_in_l2(block_size_bytes)

        if block_fits_l1:
            block_l1_hits = q_bytes_per_block * (q_loads - 1)
            block_l2_hits = 0
        elif block_fits_l2:
            block_l1_hits = 0
            block_l2_hits = q_bytes_per_block * (q_loads - 1)
        else:
            block_l1_hits = 0
            block_l2_hits = 0

        q_hbm_access = q_bytes_per_block * q_loads
        kv_hbm_access = kv_bytes_per_block * kv_loads * 2

        l1_hit_bytes = block_l1_hits
        l2_hit_bytes = block_l2_hits
        hbm_access_bytes = q_hbm_access + kv_hbm_access + o_bytes

        bytes_actual = l1_hit_bytes + l2_hit_bytes + hbm_access_bytes

        cache_efficiency = (l1_hit_bytes + l2_hit_bytes) / bytes_actual if bytes_actual > 0 else 0.0

        effective_bandwidth_gbps = self._estimate_effective_bandwidth(
            l1_hit_bytes, l2_hit_bytes, hbm_access_bytes, bytes_actual
        )

        details = {
            "seq_len": seq_len,
            "num_heads": num_heads,
            "kv_num_heads": kv_num_heads,
            "head_dim": head_dim,
            "is_causal": is_causal,
            "block_size": block_size,
            "num_q_blocks": num_q_blocks,
            "num_kv_blocks": num_kv_blocks,
            "q_loads": q_loads,
            "kv_loads": kv_loads,
            "q_bytes_per_block": q_bytes_per_block,
            "kv_bytes_per_block": kv_bytes_per_block,
            "block_fits_l1": block_fits_l1,
            "block_fits_l2": block_fits_l2,
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
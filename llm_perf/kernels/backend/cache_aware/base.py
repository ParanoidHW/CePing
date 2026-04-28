"""Cache-aware calculation base class and result data structure.

Provides the foundation for cache-aware memory access estimation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

from llm_perf.hardware.device import Device
from llm_perf.kernels.functional import KernelResult


@dataclass
class CacheAwareResult:
    """Result of cache-aware memory access calculation.

    Attributes:
        bytes_theory: Theoretical memory access (from KernelResult.bytes_accessed)
        bytes_actual: Actual memory access considering cache effects
        l1_hit_bytes: Bytes that hit in L1 cache
        l2_hit_bytes: Bytes that hit in L2 cache
        hbm_access_bytes: Bytes that must access HBM (L1 + L2 miss)
        cache_efficiency: Ratio of cache hits (l1+l2 hits / total bytes)
        effective_bandwidth_gbps: Effective bandwidth considering cache hierarchy

    Reference:
        HBM bandwidth is the bottleneck for memory-bound kernels.
        Cache hierarchy reduces effective HBM access:
        - L1 hit: ~1000 GB/s (on-chip, ~1 cycle latency)
        - L2 hit: ~400 GB/s (on-chip, ~10 cycles latency)
        - HBM access: ~3 TB/s (off-chip, ~400 cycles latency)
    """
    bytes_theory: int
    bytes_actual: int
    l1_hit_bytes: int = 0
    l2_hit_bytes: int = 0
    hbm_access_bytes: int = 0
    cache_efficiency: float = 0.0
    effective_bandwidth_gbps: float = 0.0
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bytes_theory": self.bytes_theory,
            "bytes_actual": self.bytes_actual,
            "l1_hit_bytes": self.l1_hit_bytes,
            "l2_hit_bytes": self.l2_hit_bytes,
            "hbm_access_bytes": self.hbm_access_bytes,
            "cache_efficiency": self.cache_efficiency,
            "effective_bandwidth_gbps": self.effective_bandwidth_gbps,
            "details": self.details,
        }


class CacheAwareCalculator(ABC):
    """Base class for cache-aware memory access calculation.

    Each kernel type has specific memory access patterns that affect
    cache behavior:
    - Linear: Weight matrix reuse pattern
    - Attention: KV cache and Q@K^T@V pattern
    - Convolution: Weight reuse and sliding window pattern

    Subclasses must implement:
    - calculate(): Compute cache-aware memory access
    - supports_kernel(): Check if calculator supports this kernel

    Hardware cache parameters (from Device):
    - L1 cache: 128 KB per SM, ~1000 GB/s effective bandwidth
    - L2 cache: 40-50 MB shared, ~400 GB/s effective bandwidth
    - HBM: 3-5 TB/s bandwidth, 400+ cycles latency

    Reference:
        NVIDIA H100 architecture: 132 SMs, 128KB L1/SM, 50MB L2
        AMD MI300X: 304 CDNs, 64KB L1/CU, 256KB L2/CU
    """

    def __init__(self, device: Device):
        self.device = device
        self._cache_config = self._extract_cache_config()

    def _extract_cache_config(self) -> Dict[str, Any]:
        """Extract cache configuration from device.

        Default values based on typical GPU/NPU architecture:
        - H100: 128KB L1, 50MB L2
        - A100: 128KB L1, 40MB L2
        - MI300X: 64KB L1, 256KB L2 per CU
        """
        device_name = self.device.config.name

        cache_configs = {
            "H100-SXM-80GB": {
                "l1_cache_kb": 128,
                "l2_cache_kb": 51200,  # 50MB = 51200KB
                "num_sm": 132,
                "l1_bw_gbps": 1000,  # Approximate
                "l2_bw_gbps": 400,   # Approximate
                "hbm_bw_gbps": self.device.get_memory_bw_gbps(),
            },
            "A100-SXM-80GB": {
                "l1_cache_kb": 128,
                "l2_cache_kb": 40960,  # 40MB
                "num_sm": 108,
                "l1_bw_gbps": 1000,
                "l2_bw_gbps": 400,
                "hbm_bw_gbps": self.device.get_memory_bw_gbps(),
            },
            "MI300X": {
                "l1_cache_kb": 64,
                "l2_cache_kb": 256,  # 256KB per CU (aggregate is larger)
                "num_sm": 304,  # CUs
                "l1_bw_gbps": 800,
                "l2_bw_gbps": 300,
                "hbm_bw_gbps": self.device.get_memory_bw_gbps(),
            },
        }

        if device_name in cache_configs:
            return cache_configs[device_name]

        return {
            "l1_cache_kb": 128,
            "l2_cache_kb": 40960,
            "num_sm": 108,
            "l1_bw_gbps": 1000,
            "l2_bw_gbps": 400,
            "hbm_bw_gbps": self.device.get_memory_bw_gbps(),
        }

    @abstractmethod
    def calculate(
        self,
        kernel_result: KernelResult,
        batch_size: int,
        **kwargs,
    ) -> CacheAwareResult:
        """Calculate cache-aware memory access for a kernel.

        Args:
            kernel_result: KernelResult from functional API
            batch_size: Batch size for the computation
            **kwargs: Additional parameters (e.g., seq_len, hidden_size)

        Returns:
            CacheAwareResult with cache hierarchy breakdown
        """
        pass

    @abstractmethod
    def supports_kernel(self, kernel_name: str) -> bool:
        """Check if this calculator supports the given kernel type.

        Args:
            kernel_name: Kernel identifier (e.g., "linear", "flash_attention")

        Returns:
            True if supported
        """
        pass

    def _estimate_effective_bandwidth(
        self,
        l1_hit_bytes: int,
        l2_hit_bytes: int,
        hbm_access_bytes: int,
        total_bytes: int,
    ) -> float:
        """Estimate effective bandwidth considering cache hierarchy.

        Effective bandwidth = weighted average of cache bandwidths:
        - L1 hit: l1_bw_gbps (fastest)
        - L2 hit: l2_bw_gbps (medium)
        - HBM: hbm_bw_gbps (slowest, bottleneck)

        Formula:
        BW_eff = (l1_hit * BW_l1 + l2_hit * BW_l2 + hbm * BW_hbm) / total_bytes

        Returns:
            Effective bandwidth in GB/s
        """
        if total_bytes == 0:
            return self._cache_config["hbm_bw_gbps"]

        l1_bw = self._cache_config["l1_bw_gbps"]
        l2_bw = self._cache_config["l2_bw_gbps"]
        hbm_bw = self._cache_config["hbm_bw_gbps"]

        weighted_bw = (
            l1_hit_bytes * l1_bw
            + l2_hit_bytes * l2_bw
            + hbm_access_bytes * hbm_bw
        ) / total_bytes

        return weighted_bw

    def _fits_in_l1(self, data_size_bytes: int) -> bool:
        """Check if data fits in L1 cache per SM.

        Args:
            data_size_bytes: Data size in bytes

        Returns:
            True if fits in L1 cache
        """
        l1_capacity = self._cache_config["l1_cache_kb"] * 1024
        return data_size_bytes <= l1_capacity

    def _fits_in_l2(self, data_size_bytes: int) -> bool:
        """Check if data fits in L2 cache (shared across SMs).

        Args:
            data_size_bytes: Data size in bytes

        Returns:
            True if fits in L2 cache
        """
        l2_capacity = self._cache_config["l2_cache_kb"] * 1024
        return data_size_bytes <= l2_capacity
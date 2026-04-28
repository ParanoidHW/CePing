"""Cache-aware kernel memory access calculation.

This module provides cache-aware memory access estimation for kernels,
considering hardware cache hierarchy (L1, L2, HBM) and kernel-specific
access patterns.

Key components:
- CacheAwareCalculator: Base class for cache-aware calculation
- Registry: Central registry for cache-aware calculators
- LinearCacheAware: Linear layer cache-aware calculation
- FlashAttentionCacheAware: Flash Attention cache-aware calculation

Reference:
- Flash Attention: https://arxiv.org/abs/2205.14135
- Roofline Model: https://crd.lbl.gov/departments/computer-science/PAR/research/roofline/
"""

from .base import CacheAwareCalculator, CacheAwareResult
from .registry import CacheAwareRegistry, get_calculator
from .linear import LinearCacheAware
from .attention import FlashAttentionCacheAware

CacheAwareRegistry.register("linear", LinearCacheAware)
CacheAwareRegistry.register("flash_attention", FlashAttentionCacheAware)

__all__ = [
    "CacheAwareCalculator",
    "CacheAwareResult",
    "CacheAwareRegistry",
    "get_calculator",
    "LinearCacheAware",
    "FlashAttentionCacheAware",
]
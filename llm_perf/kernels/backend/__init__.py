"""Kernel Backend Layer - Pluggable evaluation strategies.

This module provides a backend layer for kernel evaluation,
supporting multiple evaluation strategies:
- TheoryBackend: FLOPs/Roofline theoretical estimation
- ProfilingBackend: Measured data lookup and interpolation
- MicroarchBackend: Microarchitecture-based modeling (future)

Example:
    >>> from llm_perf.kernels.backend import KernelBackendRegistry, TheoryBackend
    >>> registry = KernelBackendRegistry()
    >>> registry.register_backend("theory", TheoryBackend)
    >>> backend = registry.get_backend("theory")
    >>> time = backend.estimate_compute_time("linear", (4096,), (5120,), "fp16", device)
"""

from .base import KernelBackend
from .theory import TheoryBackend
from .profiling import ProfilingBackend
from .microarch import MicroarchBackend
from .registry import KernelBackendRegistry, get_backend_registry

__all__ = [
    "KernelBackend",
    "TheoryBackend",
    "ProfilingBackend",
    "MicroarchBackend",
    "KernelBackendRegistry",
    "get_backend_registry",
]

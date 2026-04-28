"""Registry for cache-aware calculators.

Provides a central registry for registering and retrieving
cache-aware calculators by kernel type.
"""

from typing import Dict, Type, Optional

from llm_perf.hardware.device import Device
from .base import CacheAwareCalculator


class CacheAwareRegistry:
    """Central registry for cache-aware calculators.

    Usage:
        # Register a calculator
        CacheAwareRegistry.register("linear", LinearCacheAware)

        # Get a calculator instance
        calculator = CacheAwareRegistry.get("linear", device)

        # Check if calculator exists
        if CacheAwareRegistry.has("linear"):
            ...

    Design:
        - Decoupled from specific kernel implementations
        - Easy to add new kernel types
        - Singleton pattern for global access
    """

    _calculators: Dict[str, Type[CacheAwareCalculator]] = {}

    @classmethod
    def register(cls, kernel_name: str, calculator_class: Type[CacheAwareCalculator]) -> None:
        """Register a calculator for a kernel type.

        Args:
            kernel_name: Kernel identifier (e.g., "linear", "flash_attention")
            calculator_class: Calculator class (not instance)
        """
        cls._calculators[kernel_name] = calculator_class

    @classmethod
    def get(cls, kernel_name: str, device: Device) -> Optional[CacheAwareCalculator]:
        """Get a calculator instance for a kernel type.

        Args:
            kernel_name: Kernel identifier
            device: Target device

        Returns:
            Calculator instance if registered, None otherwise
        """
        calculator_class = cls._calculators.get(kernel_name)
        if calculator_class is None:
            return None
        return calculator_class(device)

    @classmethod
    def has(cls, kernel_name: str) -> bool:
        """Check if a calculator is registered for this kernel.

        Args:
            kernel_name: Kernel identifier

        Returns:
            True if registered
        """
        return kernel_name in cls._calculators

    @classmethod
    def list_registered(cls) -> list:
        """List all registered kernel types.

        Returns:
            List of kernel names
        """
        return list(cls._calculators.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered calculators (for testing)."""
        cls._calculators.clear()


def get_calculator(kernel_name: str, device: Device) -> Optional[CacheAwareCalculator]:
    """Convenience function to get a calculator.

    Args:
        kernel_name: Kernel identifier
        device: Target device

    Returns:
        Calculator instance if registered, None otherwise
    """
    return CacheAwareRegistry.get(kernel_name, device)
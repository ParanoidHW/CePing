"""Kernel Backend Registry.

Provides a centralized registry for kernel backends,
allowing easy registration, retrieval, and switching
between different evaluation strategies.
"""

from typing import Dict, Optional, Type

from .base import KernelBackend, BackendConfig
from .theory import TheoryBackend
from .profiling import ProfilingBackend
from .microarch import MicroarchBackend
from ...hardware.device import Device


class KernelBackendRegistry:
    """Registry for kernel evaluation backends.

    Provides:
    - Backend registration: register_backend(name, backend_class)
    - Backend retrieval: get_backend(name)
    - Default backend management: set_default_backend(name)
    - Backend creation with config: create_backend(name, config)

    Example:
        >>> registry = KernelBackendRegistry()
        >>> registry.register_backend("theory", TheoryBackend)
        >>> backend = registry.get_backend("theory")
        >>> time = backend.estimate_compute_time(...)
    """

    DEFAULT_BACKEND = "theory"

    _backend_classes: Dict[str, Type[KernelBackend]] = {}
    _backend_instances: Dict[str, KernelBackend] = {}
    _default_backend_name: str = DEFAULT_BACKEND

    def __init__(self):
        self._register_default_backends()

    def _register_default_backends(self) -> None:
        self._backend_classes["theory"] = TheoryBackend
        self._backend_classes["profiling"] = ProfilingBackend
        self._backend_classes["microarch"] = MicroarchBackend

    def register_backend(
        self,
        name: str,
        backend_class: Type[KernelBackend],
    ) -> None:
        """Register a backend class.

        Args:
            name: Backend identifier
            backend_class: Backend class (not instance)
        """
        self._backend_classes[name] = backend_class

    def create_backend(
        self,
        name: str,
        device: Optional[Device] = None,
        profiling_data_path: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> KernelBackend:
        """Create and initialize a backend instance.

        Args:
            name: Backend identifier
            device: Target device
            profiling_data_path: Path to profiling data (for ProfilingBackend)
            extra: Extra configuration parameters

        Returns:
            Initialized backend instance
        """
        backend_class = self._backend_classes.get(name)
        if backend_class is None:
            raise ValueError(f"Unknown backend: {name}. Available: {list(self._backend_classes.keys())}")

        config = BackendConfig(
            name=name,
            device=device,
            profiling_data_path=profiling_data_path,
            extra=extra or {},
        )

        backend = backend_class(config)
        backend.initialize()

        self._backend_instances[name] = backend
        return backend

    def get_backend(self, name: str) -> KernelBackend:
        """Get a backend instance.

        If backend instance exists, return it.
        Otherwise, create a new instance with default config.

        Args:
            name: Backend identifier

        Returns:
            Backend instance
        """
        if name in self._backend_instances:
            return self._backend_instances[name]

        return self.create_backend(name)

    def get_backend_class(self, name: str) -> Optional[Type[KernelBackend]]:
        """Get backend class (not instance).

        Args:
            name: Backend identifier

        Returns:
            Backend class or None
        """
        return self._backend_classes.get(name)

    def set_default_backend(self, name: str) -> None:
        """Set default backend.

        Args:
            name: Backend identifier
        """
        if name not in self._backend_classes:
            raise ValueError(f"Unknown backend: {name}")
        self._default_backend_name = name

    def get_default_backend(self) -> KernelBackend:
        """Get default backend instance.

        Returns:
            Default backend instance
        """
        return self.get_backend(self._default_backend_name)

    def get_default_backend_name(self) -> str:
        """Get default backend name.

        Returns:
            Default backend identifier
        """
        return self._default_backend_name

    def list_backends(self) -> list:
        """List registered backend names.

        Returns:
            List of backend identifiers
        """
        return list(self._backend_classes.keys())

    def list_initialized_backends(self) -> list:
        """List initialized backend instances.

        Returns:
            List of initialized backend identifiers
        """
        return list(self._backend_instances.keys())

    def has_backend(self, name: str) -> bool:
        """Check if backend is registered.

        Args:
            name: Backend identifier

        Returns:
            True if registered
        """
        return name in self._backend_classes

    def has_backend_instance(self, name: str) -> bool:
        """Check if backend instance is created.

        Args:
            name: Backend identifier

        Returns:
            True if instance exists
        """
        return name in self._backend_instances

    def remove_backend_instance(self, name: str) -> bool:
        """Remove backend instance (not class registration).

        Args:
            name: Backend identifier

        Returns:
            True if removed
        """
        if name in self._backend_instances:
            del self._backend_instances[name]
            return True
        return False

    def clear_instances(self) -> None:
        """Clear all backend instances."""
        self._backend_instances.clear()

    def get_backend_info(self, name: str) -> Dict:
        """Get backend information.

        Args:
            name: Backend identifier

        Returns:
            Backend info dict
        """
        backend_class = self._backend_classes.get(name)
        if backend_class is None:
            return {"error": f"Backend {name} not registered"}

        info = {
            "name": name,
            "class": backend_class.__name__,
            "registered": True,
            "initialized": name in self._backend_instances,
        }

        if name in self._backend_instances:
            backend = self._backend_instances[name]
            info["backend_info"] = backend.to_dict()

        return info

    def to_dict(self) -> Dict:
        """Convert registry info to dictionary."""
        return {
            "registered_backends": self.list_backends(),
            "initialized_backends": self.list_initialized_backends(),
            "default_backend": self._default_backend_name,
        }


_GLOBAL_REGISTRY: Optional[KernelBackendRegistry] = None


def get_backend_registry() -> KernelBackendRegistry:
    """Get global backend registry singleton.

    Returns:
        Global KernelBackendRegistry instance
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = KernelBackendRegistry()
    return _GLOBAL_REGISTRY


def set_global_registry(registry: KernelBackendRegistry) -> None:
    """Set global registry (for testing/customization).

    Args:
        registry: KernelBackendRegistry instance
    """
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = registry


def get_backend(name: Optional[str] = None) -> KernelBackend:
    """Get backend from global registry.

    Args:
        name: Backend name (None for default)

    Returns:
        Backend instance
    """
    registry = get_backend_registry()
    if name is None:
        return registry.get_default_backend()
    return registry.get_backend(name)


def create_theory_backend(device: Optional[Device] = None) -> TheoryBackend:
    """Create TheoryBackend with optional device.

    Args:
        device: Target device

    Returns:
        TheoryBackend instance
    """
    registry = get_backend_registry()
    return registry.create_backend("theory", device=device)


def create_profiling_backend(
    device: Optional[Device] = None,
    profiling_data_path: Optional[str] = None,
) -> ProfilingBackend:
    """Create ProfilingBackend with optional data path.

    Args:
        device: Target device
        profiling_data_path: Path to profiling JSON file

    Returns:
        ProfilingBackend instance
    """
    registry = get_backend_registry()
    return registry.create_backend("profiling", device=device, profiling_data_path=profiling_data_path)


def create_microarch_backend(device: Optional[Device] = None) -> MicroarchBackend:
    """Create MicroarchBackend with optional device.

    Args:
        device: Target device

    Returns:
        MicroarchBackend instance
    """
    registry = get_backend_registry()
    return registry.create_backend("microarch", device=device)

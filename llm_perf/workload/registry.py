"""Workload and Model registry.

Provides registration and lookup for:
- Workload types
- Model presets

Registry is used by:
1. Web API to list available workloads/models
2. CLI to validate workload/model names
3. Engine to resolve workload/model configurations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .loader import get_loader, WorkloadLoader
from .schema import WorkloadSchema, ModelSchema


@dataclass
class WorkloadInfo:
    """Workload information for listing."""

    name: str
    category: str
    description: str
    workload_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "workload_type": self.workload_type,
        }


@dataclass
class ModelInfo:
    """Model information for listing."""

    name: str
    architecture: str
    sparse_type: str
    description: str
    supported_workloads: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "sparse_type": self.sparse_type,
            "description": self.description,
            "supported_workloads": self.supported_workloads,
        }


class WorkloadRegistry:
    """Registry for workload configurations.

    Provides:
    - List all workloads
    - Get workload schema
    - Get workload info
    """

    def __init__(self, loader: Optional[WorkloadLoader] = None):
        self.loader = loader or get_loader()
        self._schema_cache: Dict[str, WorkloadSchema] = {}

    def list_workloads(self) -> List[WorkloadInfo]:
        """List all available workloads.

        Returns:
            List of WorkloadInfo
        """
        result = []
        categories = self.loader.list_workload_categories()

        for category, workloads in categories.items():
            for workload_name in workloads:
                full_name = f"{category}/{workload_name}"
                try:
                    yaml_config = self.loader.load_workload_yaml(full_name)
                    result.append(
                        WorkloadInfo(
                            name=workload_name,
                            category=category,
                            description=yaml_config.get("description", ""),
                            workload_type=yaml_config.get("workload_type", "inference"),
                        )
                    )
                except FileNotFoundError:
                    pass

        return result

    def list_workload_categories(self) -> Dict[str, List[str]]:
        """List workloads by category.

        Returns:
            Dict like {"training": ["training", "denoise"], ...}
        """
        return self.loader.list_workload_categories()

    def get_workload_schema(self, workload_name: str) -> WorkloadSchema:
        """Get WorkloadSchema for frontend rendering.

        Args:
            workload_name: Like "inference/autoregressive"

        Returns:
            WorkloadSchema
        """
        if workload_name in self._schema_cache:
            return self._schema_cache[workload_name]

        schema = self.loader.get_workload_schema(workload_name)
        self._schema_cache[workload_name] = schema
        return schema

    def get_workload_info(self, workload_name: str) -> Optional[WorkloadInfo]:
        """Get WorkloadInfo for a specific workload.

        Args:
            workload_name: Like "inference/autoregressive"

        Returns:
            WorkloadInfo or None
        """
        try:
            yaml_config = self.loader.load_workload_yaml(workload_name)
            category = workload_name.split("/")[0] if "/" in workload_name else "unknown"
            workload_name_short = workload_name.split("/")[-1] if "/" in workload_name else workload_name
            return WorkloadInfo(
                name=workload_name_short,
                category=category,
                description=yaml_config.get("description", ""),
                workload_type=yaml_config.get("workload_type", "inference"),
            )
        except FileNotFoundError:
            return None

    def is_valid_workload(self, workload_name: str) -> bool:
        """Check if workload name is valid.

        Args:
            workload_name: Like "inference/autoregressive"

        Returns:
            True if valid
        """
        try:
            self.loader.load_workload_yaml(workload_name)
            return True
        except FileNotFoundError:
            return False

    def clear_cache(self) -> None:
        """Clear schema cache."""
        self._schema_cache.clear()


class ModelRegistry:
    """Registry for model configurations.

    Provides:
    - List all models
    - Get model schema
    - Get model info
    - Check model support for workload
    """

    def __init__(self, loader: Optional[WorkloadLoader] = None):
        self.loader = loader or get_loader()
        self._schema_cache: Dict[str, ModelSchema] = {}

    def list_models(self) -> List[ModelInfo]:
        """List all available models.

        Returns:
            List of ModelInfo
        """
        result = []
        models = self.loader.list_models()

        for model_name in models:
            try:
                yaml_config = self.loader.load_model_yaml(model_name)
                result.append(
                    ModelInfo(
                        name=model_name,
                        architecture=yaml_config.get("architecture", ""),
                        sparse_type=yaml_config.get("sparse_type", "dense"),
                        description=yaml_config.get("description", ""),
                        supported_workloads=yaml_config.get("supported_workloads", []),
                    )
                )
            except FileNotFoundError:
                pass

        return result

    def get_model_schema(self, model_name: str) -> ModelSchema:
        """Get ModelSchema for frontend rendering.

        Args:
            model_name: Like "llama-7b"

        Returns:
            ModelSchema
        """
        if model_name in self._schema_cache:
            return self._schema_cache[model_name]

        schema = self.loader.get_model_schema(model_name)
        self._schema_cache[model_name] = schema
        return schema

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get ModelInfo for a specific model.

        Args:
            model_name: Like "llama-7b"

        Returns:
            ModelInfo or None
        """
        try:
            yaml_config = self.loader.load_model_yaml(model_name)
            return ModelInfo(
                name=model_name,
                architecture=yaml_config.get("architecture", ""),
                sparse_type=yaml_config.get("sparse_type", "dense"),
                description=yaml_config.get("description", ""),
                supported_workloads=yaml_config.get("supported_workloads", []),
            )
        except FileNotFoundError:
            return None

    def is_valid_model(self, model_name: str) -> bool:
        """Check if model name is valid.

        Args:
            model_name: Like "llama-7b"

        Returns:
            True if valid
        """
        try:
            self.loader.load_model_yaml(model_name)
            return True
        except FileNotFoundError:
            return False

    def supports_workload(self, model_name: str, workload_category: str) -> bool:
        """Check if model supports a workload category.

        Args:
            model_name: Like "llama-7b"
            workload_category: Like "training", "inference"

        Returns:
            True if supported
        """
        supported = self.loader.get_supported_workloads_for_model(model_name)
        return workload_category in supported

    def get_models_for_workload(self, workload_category: str) -> List[ModelInfo]:
        """Get models that support a workload category.

        Args:
            workload_category: Like "training", "inference"

        Returns:
            List of ModelInfo
        """
        all_models = self.list_models()
        return [m for m in all_models if workload_category in m.supported_workloads]

    def clear_cache(self) -> None:
        """Clear schema cache."""
        self._schema_cache.clear()


_workload_registry: Optional[WorkloadRegistry] = None
_model_registry: Optional[ModelRegistry] = None


def get_workload_registry() -> WorkloadRegistry:
    """Get singleton WorkloadRegistry instance."""
    global _workload_registry
    if _workload_registry is None:
        _workload_registry = WorkloadRegistry()
    return _workload_registry


def get_model_registry() -> ModelRegistry:
    """Get singleton ModelRegistry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
"""Model and Pipeline Registry implementation.

Provides centralized registration and factory pattern for models and pipelines.
Supports dynamic model/pipeline discovery and instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

# Delayed import to avoid circular imports
# from llm_perf.legacy.models.base import BaseModel, ModelConfig

# Type variables for generic registry
# Type variables - actual bounds checked at runtime to avoid circular imports
ConfigT = TypeVar("ConfigT")  # bound=ModelConfig
ModelT = TypeVar("ModelT")  # bound=BaseModel
PipelineT = TypeVar("PipelineT", bound="Pipeline")


@dataclass
class ModelInfo:
    """Information about a registered model."""

    name: str
    config_class: Type
    model_class: Type
    description: str
    architecture: str  # Model architecture (llama, deepseek, mixtral, wan_dit, etc.)
    sparse_type: str = "dense"  # FFN type: "dense", "standard_moe", "deepseek_moe"
    attention_features: List[str] = field(default_factory=list)  # ["mla", "gqa", "standard"]
    default_config: Optional[Dict[str, Any]] = None
    _explicit_category: Optional[str] = None  # For backward compatibility tests

    @property
    def category(self) -> str:
        """Backward compatibility: category from explicit setting or architecture/sparse_type."""
        # Use explicit category if set
        if self._explicit_category is not None:
            return self._explicit_category

        # Special category for specific architectures
        architecture_to_category = {
            "wan_text_encoder": "text_encoder",
            "wan_dit": "dit",
            "wan_vae": "vae",
            "vae": "vae",
            "resnet": "vision",
        }
        if self.architecture in architecture_to_category:
            return architecture_to_category[self.architecture]

        # For LLM models, map by sparse_type
        sparse_to_category = {
            "dense": "llm",
            "standard_moe": "moe",
            "deepseek_moe": "moe",
        }
        return sparse_to_category.get(self.sparse_type, self.sparse_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "architecture": self.architecture,
            "sparse_type": self.sparse_type,
            "attention_features": self.attention_features,
            "category": self.category,  # Backward compatibility
            "config_class": self.config_class.__name__,
            "model_class": self.model_class.__name__,
            "default_config": self.default_config or {},
        }


@dataclass
class PipelineInfo:
    """Information about a registered pipeline."""

    name: str
    pipeline_class: Type["Pipeline"]
    description: str
    supported_models: List[str]  # List of model categories this pipeline supports
    default_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "supported_models": self.supported_models,
            "pipeline_class": self.pipeline_class.__name__,
            "default_config": self.default_config or {},
        }


class ModelRegistry:
    """Central registry for all models.

    Provides factory pattern for model creation and supports dynamic model discovery.
    All models should be registered through this registry for web service integration.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("llama-7b", LlamaConfig, LlamaModel, "LLaMA 7B model")
        >>> model = registry.create("llama-7b", vocab_size=32000, hidden_size=4096, ...)
    """

    _instance: Optional[ModelRegistry] = None
    _models: Dict[str, ModelInfo]

    def __new__(cls) -> ModelRegistry:
        """Singleton pattern to ensure single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def register(
        self,
        name: str,
        config_class: Type,
        model_class: Type,
        description: str = "",
        architecture: str = "",
        sparse_type: str = "dense",
        attention_features: Optional[List[str]] = None,
        default_config: Optional[Dict[str, Any]] = None,
        # Backward compatibility: category can be explicitly set or maps from sparse_type
        category: Optional[str] = None,
    ) -> None:
        """Register a model with the registry.

        Args:
            name: Unique identifier for the model
            config_class: Configuration dataclass for the model
            model_class: Model class that inherits from BaseModel
            description: Human-readable description
            architecture: Model architecture (llama, deepseek, mixtral, etc.)
            sparse_type: FFN type - "dense", "standard_moe", "deepseek_moe"
            attention_features: Attention features ["mla", "gqa", "standard"]
            default_config: Default configuration values
            category: (Optional) Explicit category override. If not set, inferred from architecture/sparse_type

        Raises:
            ValueError: If model name already registered
        """
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")

        # Backward compatibility: category -> sparse_type mapping if sparse_type not set
        if category and sparse_type == "dense":
            category_to_sparse = {
                "llm": "dense",
                "moe": "standard_moe",
            }
            sparse_type = category_to_sparse.get(category, "dense")

        # Default architecture to name if not provided
        if not architecture:
            architecture = name

        # Default attention features
        if attention_features is None:
            attention_features = []

        # Store explicit category if provided (for backward compatibility tests)
        explicit_category = category if category else None

        self._models[name] = ModelInfo(
            name=name,
            config_class=config_class,
            model_class=model_class,
            description=description,
            architecture=architecture,
            sparse_type=sparse_type,
            attention_features=attention_features,
            default_config=default_config,
            _explicit_category=explicit_category,
        )

    def unregister(self, name: str) -> None:
        """Unregister a model.

        Args:
            name: Model identifier to remove

        Raises:
            KeyError: If model not found
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        del self._models[name]

    def create(self, model_name: str, **config_kwargs: Any) -> Any:
        """Create a model instance by name.

        Args:
            model_name: Registered model identifier
            **config_kwargs: Configuration parameters passed to config class

        Returns:
            Instantiated model

        Raises:
            KeyError: If model not found
            TypeError: If config parameters are invalid
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found in registry")

        info = self._models[model_name]

        # Merge with default config
        merged_config = dict(info.default_config or {})
        merged_config.update(config_kwargs)
        # Only set name from registry if not provided in kwargs
        if "name" not in config_kwargs:
            merged_config.setdefault("name", model_name)

        # Create config and model
        config = info.config_class(**merged_config)
        return info.model_class(config)

    def get_info(self, name: str) -> ModelInfo:
        """Get model information.

        Args:
            name: Model identifier

        Returns:
            ModelInfo dataclass

        Raises:
            KeyError: If model not found
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]

    def list_models(self, sparse_type: Optional[str] = None, category: Optional[str] = None) -> List[str]:
        """List all registered model names.

        Args:
            sparse_type: Filter by sparse_type (optional) - "dense", "standard_moe", "deepseek_moe"
            category: (Deprecated) Filter by category - maps to sparse_type

        Returns:
            List of model names
        """
        # Backward compatibility: category -> sparse_type
        filter_type = sparse_type
        if category and not sparse_type:
            category_to_sparse = {"llm": "dense", "moe": "standard_moe"}
            filter_type = category_to_sparse.get(category)

        if filter_type:
            return [name for name, info in self._models.items() if info.sparse_type == filter_type]
        return list(self._models.keys())

    def list_by_sparse_type(self) -> Dict[str, List[str]]:
        """List models grouped by sparse_type.

        Returns:
            Dictionary mapping sparse_type to list of model names
        """
        result: Dict[str, List[str]] = {}
        for name, info in self._models.items():
            if info.sparse_type not in result:
                result[info.sparse_type] = []
            result[info.sparse_type].append(name)
        return result

    def list_by_architecture(self) -> Dict[str, List[str]]:
        """List models grouped by architecture.

        Returns:
            Dictionary mapping architecture to list of model names
        """
        result: Dict[str, List[str]] = {}
        for name, info in self._models.items():
            if info.architecture not in result:
                result[info.architecture] = []
            result[info.architecture].append(name)
        return result

    def list_by_category(self) -> Dict[str, List[str]]:
        """List models grouped by category (backward compatibility).

        Uses the actual category property value from each ModelInfo.

        Returns:
            Dictionary mapping category to list of model names
        """
        result: Dict[str, List[str]] = {}
        for name, info in self._models.items():
            cat = info.category
            if cat not in result:
                result[cat] = []
            result[cat].append(name)
        return result

    def get_all_infos(self) -> Dict[str, "ModelInfo"]:
        """Get all registered model informations.

        Returns:
            Dictionary mapping model names to ModelInfo
        """
        return dict(self._models)

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary representation."""
        return {
            "models": {name: info.to_dict() for name, info in self._models.items()},
            "by_sparse_type": self.list_by_sparse_type(),
            "by_architecture": self.list_by_architecture(),
            "categories": self.list_by_sparse_type(),  # Backward compatibility alias
        }

    def clear(self) -> None:
        """Clear all registered models. Use with caution."""
        self._models.clear()

    def is_registered(self, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Model identifier

        Returns:
            True if registered, False otherwise
        """
        return name in self._models

    def create_preset(
        self, name: str, preset_name: str, config: Dict[str, Any]
    ) -> None:
        """Create a preset configuration for a registered model.

        Args:
            name: Model identifier
            preset_name: Name of the preset
            config: Configuration dictionary

        Raises:
            KeyError: If model not found
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")

        info = self._models[name]
        if info.default_config is None:
            info.default_config = {}

        # Store preset under a special key
        if "presets" not in info.default_config:
            info.default_config["presets"] = {}
        info.default_config["presets"][preset_name] = config


class PipelineRegistry:
    """Central registry for all pipelines.

    Provides factory pattern for pipeline creation and supports dynamic pipeline discovery.
    All pipelines should be registered through this registry for web service integration.

    Example:
        >>> registry = PipelineRegistry()
        >>> registry.register("diffusion-video", DiffusionVideoPipeline,
        ...                   "Text-to-video diffusion pipeline")
        >>> pipeline = registry.create("diffusion-video", ...)
    """

    _instance: Optional[PipelineRegistry] = None
    _pipelines: Dict[str, PipelineInfo]

    def __new__(cls) -> PipelineRegistry:
        """Singleton pattern to ensure single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipelines = {}
        return cls._instance

    def register(
        self,
        name: str,
        pipeline_class: Type[PipelineT],
        description: str = "",
        supported_models: Optional[List[str]] = None,
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a pipeline with the registry.

        Args:
            name: Unique identifier for the pipeline
            pipeline_class: Pipeline class that inherits from Pipeline
            description: Human-readable description
            supported_models: List of model categories this pipeline supports
            default_config: Default configuration values

        Raises:
            ValueError: If pipeline name already registered
        """
        if name in self._pipelines:
            raise ValueError(f"Pipeline '{name}' is already registered")

        self._pipelines[name] = PipelineInfo(
            name=name,
            pipeline_class=pipeline_class,
            description=description,
            supported_models=supported_models or [],
            default_config=default_config,
        )

    def unregister(self, name: str) -> None:
        """Unregister a pipeline.

        Args:
            name: Pipeline identifier to remove

        Raises:
            KeyError: If pipeline not found
        """
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found in registry")
        del self._pipelines[name]

    def create(self, name: str, **kwargs: Any) -> "Pipeline":
        """Create a pipeline instance by name.

        Args:
            name: Registered pipeline identifier
            **kwargs: Parameters passed to pipeline constructor

        Returns:
            Instantiated pipeline

        Raises:
            KeyError: If pipeline not found
            TypeError: If parameters are invalid
        """
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found in registry")

        info = self._pipelines[name]

        # Merge with default config
        merged_config = dict(info.default_config or {})
        merged_config.update(kwargs)

        return info.pipeline_class(**merged_config)

    def get_info(self, name: str) -> PipelineInfo:
        """Get pipeline information.

        Args:
            name: Pipeline identifier

        Returns:
            PipelineInfo dataclass

        Raises:
            KeyError: If pipeline not found
        """
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found in registry")
        return self._pipelines[name]

    def list_pipelines(self) -> List[str]:
        """List all registered pipeline names.

        Returns:
            List of pipeline names
        """
        return list(self._pipelines.keys())

    def get_for_model_category(self, category: str) -> List[str]:
        """Get pipelines that support a specific model category.

        Args:
            category: Model category (e.g., "llm", "vae")

        Returns:
            List of pipeline names
        """
        return [
            name
            for name, info in self._pipelines.items()
            if category in info.supported_models
        ]

    def get_all_infos(self) -> Dict[str, PipelineInfo]:
        """Get all registered pipeline informations.

        Returns:
            Dictionary mapping pipeline names to PipelineInfo
        """
        return dict(self._pipelines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary representation."""
        return {
            "pipelines": {name: info.to_dict() for name, info in self._pipelines.items()}
        }

    def clear(self) -> None:
        """Clear all registered pipelines. Use with caution."""
        self._pipelines.clear()

    def is_registered(self, name: str) -> bool:
        """Check if a pipeline is registered.

        Args:
            name: Pipeline identifier

        Returns:
            True if registered, False otherwise
        """
        return name in self._pipelines


# Import Pipeline here to avoid circular imports
from .pipeline import Pipeline  # noqa: E402

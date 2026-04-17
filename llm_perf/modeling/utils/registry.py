"""Model registry and presets for modeling framework.

Provides unified interface for model creation via:
1. Preset mode: config contains 'preset' field
2. Architecture mode: config contains 'architecture' field
3. Name mode: config contains 'name' field
"""

from typing import Optional, Dict, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_perf.modeling.models import LlamaModel, DeepSeekModel
    from llm_perf.modeling.models import ShardedVAE, ShardedResNet
    from llm_perf.modeling.models import ShardedWanTextEncoder, ShardedWanDiT, ShardedWanVAE


class ModelInfo:
    """Model registration info."""

    def __init__(
        self,
        name: str,
        model_class: type,
        description: str,
        architecture: str,
        sparse_type: str,
        attention_features: list,
        default_config: dict,
    ):
        self.name = name
        self.model_class = model_class
        self.description = description
        self.architecture = architecture
        self.sparse_type = sparse_type
        self.attention_features = attention_features
        self.default_config = default_config


class ModelingRegistry:
    """Registry for sharded models."""

    _instance: Optional["ModelingRegistry"] = None
    _models: Dict[str, ModelInfo] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(
        self,
        name: str,
        model_class: type,
        description: str = "",
        architecture: str = "",
        sparse_type: str = "dense",
        attention_features: list = [],
        default_config: dict = {},
    ) -> None:
        """Register a model."""
        self._models[name] = ModelInfo(
            name=name,
            model_class=model_class,
            description=description,
            architecture=architecture,
            sparse_type=sparse_type,
            attention_features=attention_features,
            default_config=default_config,
        )

    def is_registered(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._models

    def get_model_class(self, name: str) -> Optional[type]:
        """Get model class by name."""
        info = self._models.get(name)
        return info.model_class if info else None

    def get_info(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self._models.get(name)

    def get_all_infos(self) -> Dict[str, ModelInfo]:
        """Get all registered models."""
        return self._models

    def list_models(self) -> list:
        """List all registered model names."""
        return list(self._models.keys())

    def create(self, name: str, **kwargs) -> Any:
        """Create model instance."""
        info = self._models.get(name)
        if info is None:
            raise ValueError(f"Model '{name}' not registered")

        merged_config = dict(info.default_config)
        merged_config.update(kwargs)

        return info.model_class(**merged_config)


def register_all_models() -> None:
    """Register all built-in models."""
    registry = ModelingRegistry()

    registry.register(
        name="llama",
        model_class=LlamaModel,
        description="LLaMA architecture with GQA support",
        architecture="llama",
        sparse_type="dense",
        attention_features=["gqa"],
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="deepseek",
        model_class=DeepSeekModel,
        description="DeepSeek architecture with MoE and MLA",
        architecture="deepseek",
        sparse_type="deepseek_moe",
        attention_features=["mla"],
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 36,
            "num_heads": 32,
            "num_kv_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 4096,
            "dtype": "fp16",
            "first_k_dense_layers": 1,
            "num_experts": 64,
            "num_experts_per_token": 8,
        },
    )

    registry.register(
        name="vae",
        model_class=ShardedVAE,
        description="VAE for video/image generation",
        architecture="vae",
        sparse_type="dense",
        attention_features=[],
        default_config={
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "block_out_channels": (128, 256, 512, 512),
            "use_3d": True,
            "use_attention": False,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="resnet",
        model_class=ShardedResNet,
        description="ResNet for image classification",
        architecture="resnet",
        sparse_type="dense",
        attention_features=[],
        default_config={
            "variant": "resnet50",
            "num_classes": 1000,
            "input_channels": 3,
            "input_height": 224,
            "input_width": 224,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="wan-text-encoder",
        model_class=ShardedWanTextEncoder,
        description="Wan2.1 Text Encoder (T5)",
        architecture="wan_text_encoder",
        sparse_type="dense",
        attention_features=["standard"],
        default_config={
            "vocab_size": 256384,
            "hidden_size": 4096,
            "num_layers": 24,
            "num_heads": 64,
            "intermediate_size": 10240,
            "max_seq_len": 512,
            "dtype": "bf16",
        },
    )

    registry.register(
        name="wan-dit",
        model_class=ShardedWanDiT,
        description="Wan2.1 DiT for video generation",
        architecture="wan_dit",
        sparse_type="dense",
        attention_features=["standard"],
        default_config={
            "hidden_size": 5120,
            "num_layers": 40,
            "num_heads": 40,
            "max_seq_len": 512,
            "dtype": "bf16",
        },
    )

    registry.register(
        name="wan-vae",
        model_class=ShardedWanVAE,
        description="Wan2.1 VAE for video generation",
        architecture="wan_vae",
        sparse_type="dense",
        attention_features=[],
        default_config={
            "in_channels": 3,
            "latent_channels": 16,
            "dtype": "fp32",
        },
    )


def get_model_presets() -> dict:
    """Get preset configurations."""
    return {
        "llama-7b": {
            "architecture": "llama",
            "sparse_type": "dense",
            "attention_features": ["gqa"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 4096,
            "dtype": "fp16",
            "description": "LLaMA 7B",
        },
        "llama-13b": {
            "architecture": "llama",
            "sparse_type": "dense",
            "attention_features": ["gqa"],
            "vocab_size": 32000,
            "hidden_size": 5120,
            "num_layers": 40,
            "num_heads": 40,
            "num_kv_heads": 40,
            "intermediate_size": 13824,
            "max_seq_len": 4096,
            "dtype": "fp16",
            "description": "LLaMA 13B",
        },
        "llama-70b": {
            "architecture": "llama",
            "sparse_type": "dense",
            "attention_features": ["gqa"],
            "vocab_size": 32000,
            "hidden_size": 8192,
            "num_layers": 80,
            "num_heads": 64,
            "num_kv_heads": 8,
            "intermediate_size": 28672,
            "max_seq_len": 4096,
            "dtype": "fp16",
            "description": "LLaMA 70B",
        },
        "mixtral-8x7b": {
            "architecture": "mixtral",
            "sparse_type": "standard_moe",
            "attention_features": ["gqa"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "intermediate_size": 14336,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 8,
            "num_experts_per_token": 2,
            "description": "Mixtral 8x7B",
        },
        "deepseek-v3": {
            "architecture": "deepseek",
            "sparse_type": "deepseek_moe",
            "attention_features": ["mla"],
            "vocab_size": 129280,
            "hidden_size": 7168,
            "num_layers": 61,
            "num_heads": 128,
            "num_kv_heads": 128,
            "intermediate_size": 18432,
            "max_seq_len": 163840,
            "dtype": "fp16",
            "first_k_dense_layers": 1,
            "num_experts": 256,
            "num_experts_per_token": 8,
            "description": "DeepSeek V3",
        },
        "resnet50": {
            "architecture": "resnet",
            "sparse_type": "dense",
            "attention_features": [],
            "variant": "resnet50",
            "num_classes": 1000,
            "dtype": "fp16",
            "description": "ResNet-50",
        },
        "video-vae": {
            "architecture": "vae",
            "sparse_type": "dense",
            "attention_features": [],
            "use_3d": True,
            "use_attention": False,
            "latent_channels": 4,
            "block_out_channels": (128, 256, 512, 512),
            "dtype": "fp16",
            "description": "Video VAE",
        },
    }


def get_presets_by_sparse_type() -> dict:
    """Get presets grouped by sparse_type."""
    presets = get_model_presets()
    result = {
        "dense": [],
        "sparse_standard_moe": [],
        "sparse_deepseek_moe": [],
    }

    for name, config in presets.items():
        sparse_type = config.get("sparse_type", "dense")
        preset_info = {"name": name, **config}
        if sparse_type == "dense":
            result["dense"].append(preset_info)
        elif sparse_type == "standard_moe":
            result["sparse_standard_moe"].append(preset_info)
        elif sparse_type == "deepseek_moe":
            result["sparse_deepseek_moe"].append(preset_info)

    return result


def create_model_from_config(config: dict) -> Any:
    """Create model from configuration.

    Supports:
    1. Preset mode: config contains 'preset'
    2. Architecture mode: config contains 'architecture'
    3. Name mode: config contains 'name'

    Args:
        config: Model configuration dict

    Returns:
        Model instance
    """
    registry = ModelingRegistry()
    presets = get_model_presets()

    meta_fields = {"type", "preset", "description", "architecture", "sparse_type", "attention_features"}

    preset_name = config.get("preset")
    if preset_name and preset_name in presets:
        preset_config = presets[preset_name]
        merged_config = dict(preset_config)
        merged_config.update(config)
        architecture = merged_config.get("architecture", preset_name)
        for field in meta_fields:
            merged_config.pop(field, None)
        model_name = _find_model_by_architecture(registry, architecture)
        if model_name:
            return registry.create(model_name, **merged_config)
        if registry.is_registered(architecture):
            return registry.create(architecture, **merged_config)

    architecture = config.get("architecture")
    if architecture:
        model_name = _find_model_by_architecture(registry, architecture)
        if model_name:
            model_config = dict(config)
            for field in meta_fields:
                model_config.pop(field, None)
            return registry.create(model_name, **model_config)

    model_type = config.get("type")
    if model_type and registry.is_registered(model_type):
        model_config = dict(config)
        for field in meta_fields:
            model_config.pop(field, None)
        return registry.create(model_type, **model_config)

    model_name = config.get("name")
    if model_name and registry.is_registered(model_name):
        model_config = dict(config)
        for field in meta_fields:
            model_config.pop(field, None)
        return registry.create(model_name, **model_config)

    available_models = registry.list_models()
    available_presets = list(presets.keys())
    raise ValueError(
        f"Unknown model: '{preset_name or architecture or model_type or model_name}'. "
        f"Available presets: {available_presets}. "
        f"Available models: {available_models}."
    )


def _find_model_by_architecture(registry: ModelingRegistry, architecture: str) -> Optional[str]:
    """Find model name by architecture."""
    for name, info in registry.get_all_infos().items():
        if info.architecture == architecture:
            return name
    return None


register_all_models()

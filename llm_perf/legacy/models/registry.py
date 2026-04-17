"""Model registration module.

Registers all built-in models with the ModelRegistry for dynamic discovery.
This module should be imported to register all models.
"""

from typing import Optional

from llm_perf.legacy.core.registry import ModelRegistry

# Import all model classes
from .llama import LlamaConfig, LlamaModel
from .moe import MoEConfig, MoEModel
from .deepseek import DeepSeekConfig, DeepSeekV3Config, DeepSeekModel, DeepSeekV3Model
from .resnet import ResNetConfig, ResNetModel
from .vae import VAEConfig, VAEModel
from .wan_video import (
    WanTextEncoderConfig,
    WanDiTConfig,
    WanVAEConfig,
    WanTextEncoder,
    WanDiTModel,
    WanVAEModel,
)


def register_all_models() -> None:
    """Register all built-in models with the ModelRegistry.

    This function should be called at application startup to make
    all models available through the registry.
    """
    registry = ModelRegistry()

    # Register LLaMA models (Dense architecture)
    registry.register(
        name="llama",
        config_class=LlamaConfig,
        model_class=LlamaModel,
        description="LLaMA (Large Language Model Meta AI) architecture",
        architecture="llama",
        sparse_type="dense",
        attention_features=["gqa"],  # Supports Grouped Query Attention
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
    )

    # Register Mixtral-style MoE models (Standard MoE architecture)
    registry.register(
        name="mixtral",
        config_class=MoEConfig,
        model_class=MoEModel,
        description="Mixtral Mixture of Experts (MoE) transformer architecture",
        architecture="mixtral",
        sparse_type="standard_moe",  # Periodic MoE layers
        attention_features=["gqa"],
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 8,
            "num_experts_per_token": 2,
        },
    )

    # Register DeepSeek models (DeepSeek MoE + MLA architecture)
    registry.register(
        name="deepseek",
        config_class=DeepSeekConfig,
        model_class=DeepSeekModel,
        description="DeepSeek-V2 architecture with MLA attention and MoE",
        architecture="deepseek",
        sparse_type="deepseek_moe",  # First-K Dense + MoE pattern
        attention_features=["mla"],  # Multi-Head Latent Attention
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 64,
            "num_experts_per_token": 6,
        },
    )

    registry.register(
        name="deepseek-v3",
        config_class=DeepSeekV3Config,
        model_class=DeepSeekV3Model,
        description="DeepSeek-V3 architecture with MLA and auxiliary-loss-free load balancing",
        architecture="deepseek",
        sparse_type="deepseek_moe",
        attention_features=["mla"],
        default_config={
            "vocab_size": 32000,
            "hidden_size": 7168,
            "num_layers": 61,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "intermediate_size": 18432,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 256,
            "num_experts_per_token": 8,
        },
    )

    # Register ResNet models (Vision Dense architecture)
    registry.register(
        name="resnet",
        config_class=ResNetConfig,
        model_class=ResNetModel,
        description="ResNet CNN architecture for image classification",
        architecture="resnet",
        sparse_type="dense",
        attention_features=[],  # No attention layers
        default_config={
            "vocab_size": 1000,  # ImageNet classes
            "hidden_size": 512,
            "num_layers": 50,
            "num_attention_heads": 0,
            "intermediate_size": 0,
            "max_seq_len": 1,
            "dtype": "fp16",
            "variant": "resnet50",
            "num_blocks": [3, 4, 6, 3],
        },
    )

    # Register VAE models (Dense architecture)
    registry.register(
        name="vae",
        config_class=VAEConfig,
        model_class=VAEModel,
        description="Variational Autoencoder for image generation (Stable Diffusion style)",
        architecture="vae",
        sparse_type="dense",
        attention_features=[],  # Convolutional layers, no attention
        default_config={
            "vocab_size": 1,
            "hidden_size": 512,
            "num_layers": 16,
            "num_attention_heads": 0,
            "intermediate_size": 0,
            "max_seq_len": 1,
            "dtype": "fp16",
            "in_channels": 3,
            "latent_channels": 4,
            "block_out_channels": [128, 256, 512, 512],
        },
    )

    # Register Wan2.1 Video Generation models
    registry.register(
        name="wan-text-encoder",
        config_class=WanTextEncoderConfig,
        model_class=WanTextEncoder,
        description="Wan2.1 Text Encoder (umT5-XXL) for video generation",
        architecture="wan_text_encoder",
        sparse_type="dense",
        attention_features=["standard"],
        default_config={
            "vocab_size": 256384,
            "hidden_size": 4096,
            "num_layers": 24,
            "num_attention_heads": 64,
            "intermediate_size": 10240,
            "max_seq_len": 512,
            "dtype": "bf16",
        },
    )

    registry.register(
        name="wan-dit",
        config_class=WanDiTConfig,
        model_class=WanDiTModel,
        description="Wan2.1 DiT (Diffusion Transformer) for video generation",
        architecture="wan_dit",
        sparse_type="dense",
        attention_features=["standard"],
        default_config={
            "vocab_size": 1,
            "hidden_size": 5120,
            "num_layers": 40,
            "num_attention_heads": 40,
            "intermediate_size": 0,
            "max_seq_len": 1,
            "dtype": "bf16",
            "latent_num_frames": 21,
            "latent_height": 90,
            "latent_width": 160,
            "patch_size": [1, 2, 2],
            "in_channels": 16,
        },
    )

    registry.register(
        name="wan-vae",
        config_class=WanVAEConfig,
        model_class=WanVAEModel,
        description="Wan2.1 VAE (3D Causal) for video generation",
        architecture="wan_vae",
        sparse_type="dense",
        attention_features=[],  # Convolutional layers
        default_config={
            "vocab_size": 1,
            "hidden_size": 256,
            "num_layers": 8,
            "num_attention_heads": 0,
            "intermediate_size": 0,
            "max_seq_len": 1,
            "dtype": "fp32",
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "in_channels": 3,
            "latent_channels": 16,
            "temporal_compression": 4,
            "spatial_compression": 8,
            "block_out_channels": [128, 256, 512, 512],
        },
    )

    # Backward compatibility: register "moe" alias for mixtral
    registry.register(
        name="moe",
        config_class=MoEConfig,
        model_class=MoEModel,
        description="[Alias] Mixtral MoE transformer architecture",
        architecture="mixtral",
        sparse_type="standard_moe",
        attention_features=["gqa"],
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 8,
            "num_experts_per_token": 2,
        },
    )


def get_model_presets() -> dict:
    """Get preset configurations for common model variants.

    Presets include architecture and sparse_type fields for automatic classification.
    Users only need to select a preset without setting type manually.

    Returns:
        Dictionary of preset configurations grouped by sparse_type
    """
    return {
        # ===== Dense Model Presets =====
        "llama-7b": {
            "architecture": "llama",
            "sparse_type": "dense",
            "attention_features": ["gqa"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 4096,
            "dtype": "fp16",
            "description": "LLaMA 7B - Dense transformer with GQA",
        },
        "llama-70b": {
            "architecture": "llama",
            "sparse_type": "dense",
            "attention_features": ["gqa"],
            "vocab_size": 32000,
            "hidden_size": 8192,
            "num_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 28672,
            "max_seq_len": 4096,
            "dtype": "fp16",
            "description": "LLaMA 70B - Dense transformer with GQA",
        },
        # ===== Sparse (Standard MoE) Presets =====
        "mixtral-8x7b": {
            "architecture": "mixtral",
            "sparse_type": "standard_moe",
            "attention_features": ["gqa"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 8,
            "num_experts_per_token": 2,
            "description": "Mixtral 8x7B - Standard MoE (periodic layers) with GQA",
        },
        # ===== Sparse (DeepSeek MoE) Presets =====
        "deepseek-v3": {
            "architecture": "deepseek",
            "sparse_type": "deepseek_moe",
            "attention_features": ["mla"],
            "vocab_size": 32000,
            "hidden_size": 7168,
            "num_layers": 61,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "intermediate_size": 18432,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 256,
            "num_experts_per_token": 8,
            "description": "DeepSeek V3 - DeepSeek MoE + MLA attention",
        },
        # ===== Video Generation Pipeline Preset =====
        "wan-t2v-14b": {
            "architecture": "wan_pipeline",
            "sparse_type": "dense",
            "attention_features": [],
            "text_encoder": "wan-text-encoder",
            "dit": "wan-dit",
            "vae": "wan-vae",
            "description": "Wan2.1 Text-to-Video 14B pipeline",
        },
    }


def get_presets_by_sparse_type() -> dict:
    """Get presets grouped by sparse_type for UI display.

    Returns:
        Dictionary with 'dense' and 'sparse' keys containing preset lists
    """
    presets = get_model_presets()
    result = {
        "dense": [],
        "sparse_standard_moe": [],
        "sparse_deepseek_moe": [],
    }

    for name, config in presets.items():
        sparse_type = config.get("sparse_type", "dense")
        # Add preset info with name
        preset_info = {"name": name, **config}
        if sparse_type == "dense":
            result["dense"].append(preset_info)
        elif sparse_type == "standard_moe":
            result["sparse_standard_moe"].append(preset_info)
        elif sparse_type == "deepseek_moe":
            result["sparse_deepseek_moe"].append(preset_info)

    return result


def create_model_from_config(config: dict):
    """Create model from configuration using ModelRegistry.

    This is the unified factory function used by CLI and Web interface.
    Supports three modes:
    1. Preset mode: config contains 'preset' field - architecture/sparse_type auto-filled
    2. Architecture mode: config contains 'architecture' field
    3. Name mode: config contains 'name' field matching a registered model

    Args:
        config: Model configuration dictionary

    Returns:
        Instantiated model from registry

    Raises:
        ValueError: If model architecture/name/preset is unknown or not registered
    """
    registry = ModelRegistry()
    presets = get_model_presets()

    # Meta fields that should not be passed to model config
    meta_fields = {"type", "preset", "description", "architecture", "sparse_type", "attention_features"}

    # Check for preset first (e.g., "llama-7b", "mixtral-8x7b")
    preset_name = config.get("preset")
    if preset_name and preset_name in presets:
        preset_config = presets[preset_name]
        # Merge preset with user overrides
        merged_config = dict(preset_config)
        merged_config.update(config)
        # Get architecture from merged config (preset provides it)
        architecture = merged_config.get("architecture", preset_name)
        # Remove meta fields before passing to model
        for field in meta_fields:
            merged_config.pop(field, None)
        # Find registered model by architecture
        model_name = _find_model_by_architecture(registry, architecture)
        if model_name:
            return registry.create(model_name, **merged_config)
        # Fallback: try preset name as registered name
        if registry.is_registered(architecture):
            return registry.create(architecture, **merged_config)

    # Use 'architecture' field to find registered model
    architecture = config.get("architecture")
    if architecture:
        model_name = _find_model_by_architecture(registry, architecture)
        if model_name:
            model_config = dict(config)
            for field in meta_fields:
                model_config.pop(field, None)
            return registry.create(model_name, **model_config)

    # Backward compatibility: Use 'type' field
    model_type = config.get("type")
    if model_type and registry.is_registered(model_type):
        model_config = dict(config)
        for field in meta_fields:
            model_config.pop(field, None)
        return registry.create(model_type, **model_config)

    # Use 'name' field as model identifier
    model_name = config.get("name")
    if model_name and registry.is_registered(model_name):
        model_config = dict(config)
        for field in meta_fields:
            model_config.pop(field, None)
        return registry.create(model_name, **model_config)

    # Fallback: raise error with available options
    available_models = registry.list_models()
    available_presets = list(presets.keys())
    raise ValueError(
        f"Unknown model: '{preset_name or architecture or model_type or model_name}'. "
        f"Available presets: {available_presets}. "
        f"Available models: {available_models}."
    )


def _find_model_by_architecture(registry: ModelRegistry, architecture: str) -> Optional[str]:
    """Find registered model name by architecture.

    Args:
        registry: ModelRegistry instance
        architecture: Architecture name (e.g., "llama", "mixtral", "deepseek")

    Returns:
        Registered model name or None if not found
    """
    for name, info in registry.get_all_infos().items():
        if info.architecture == architecture:
            return name
    return None


# Auto-register on import
register_all_models()

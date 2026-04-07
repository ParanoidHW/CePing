"""Model registration module.

Registers all built-in models with the ModelRegistry for dynamic discovery.
This module should be imported to register all models.
"""

from ..core.registry import ModelRegistry

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

    # Register LLaMA models
    registry.register(
        name="llama",
        config_class=LlamaConfig,
        model_class=LlamaModel,
        description="LLaMA (Large Language Model Meta AI) architecture",
        category="llm",
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

    # Register MoE models
    registry.register(
        name="moe",
        config_class=MoEConfig,
        model_class=MoEModel,
        description="Mixture of Experts (MoE) transformer architecture",
        category="moe",
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

    # Register DeepSeek models
    registry.register(
        name="deepseek",
        config_class=DeepSeekConfig,
        model_class=DeepSeekModel,
        description="DeepSeek-V2 architecture with MLA attention",
        category="llm",
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
        description="DeepSeek-V3 architecture with auxiliary-loss-free load balancing",
        category="moe",
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

    # Register ResNet models
    registry.register(
        name="resnet",
        config_class=ResNetConfig,
        model_class=ResNetModel,
        description="ResNet CNN architecture for image classification",
        category="vision",
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

    # Register VAE models
    registry.register(
        name="vae",
        config_class=VAEConfig,
        model_class=VAEModel,
        description="Variational Autoencoder for image generation (Stable Diffusion style)",
        category="vae",
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
        category="text_encoder",
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
        category="dit",
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
        category="vae",
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


def get_model_presets() -> dict:
    """Get preset configurations for common model variants.

    Returns:
        Dictionary of preset configurations
    """
    return {
        # LLaMA presets
        "llama-7b": {
            "type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
        "llama-70b": {
            "type": "llama",
            "vocab_size": 32000,
            "hidden_size": 8192,
            "num_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 28672,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
        # MoE presets
        "mixtral-8x7b": {
            "type": "moe",
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
        "deepseek-v3": {
            "type": "deepseek-v3",
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
        # Video generation presets
        "wan-t2v-14b": {
            "type": "wan-pipeline",
            "text_encoder": "wan-text-encoder",
            "dit": "wan-dit",
            "vae": "wan-vae",
            "description": "Wan2.1 Text-to-Video 14B model",
        },
    }


# Auto-register on import
register_all_models()

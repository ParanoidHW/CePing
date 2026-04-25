"""Model registry and presets for modeling framework.

Provides unified interface for model creation via:
1. Preset mode: config contains 'preset' field
2. Architecture mode: config contains 'architecture' field
3. Name mode: config contains 'name' field
"""

import inspect
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

import yaml

from llm_perf.modeling.models import LlamaModel, DeepSeekModel
from llm_perf.modeling.vision import ShardedResNet
from llm_perf.modeling.encoder import ShardedVAE
from llm_perf.modeling.wan import ShardedWanTextEncoder, ShardedWanDiT, ShardedWanVAE
from llm_perf.modeling.qwen3_5 import Qwen3_5MoEModel, Qwen3_5Model
from llm_perf.modeling.hunyuan_image import (
    HunyuanImage3TextModel,
    HunyuanImage3DiffusionModel,
    HunyuanT5Encoder,
    HunyuanVAEEncoder,
    HunyuanVAEDecoder,
)

if TYPE_CHECKING:
    pass

_PRESETS_CACHE: Optional[Dict[str, Any]] = None


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
        supported_workloads: list = None,
    ):
        self.name = name
        self.model_class = model_class
        self.description = description
        self.architecture = architecture
        self.sparse_type = sparse_type
        self.attention_features = attention_features
        self.default_config = default_config
        self.supported_workloads = supported_workloads or ["training", "inference"]


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
        supported_workloads: list = None,
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
            supported_workloads=supported_workloads,
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
        filtered_config = _filter_params_for_model(info.model_class, merged_config)

        return info.model_class(**filtered_config)


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

    registry.register(
        name="qwen3_5_moe",
        model_class=Qwen3_5MoEModel,
        description="Qwen3.5 MoE with hybrid linear/full attention",
        architecture="qwen3_5_moe",
        sparse_type="qwen3_5_moe",
        attention_features=["linear_attention", "gqa"],
        default_config={
            "vocab_size": 248320,
            "hidden_size": 2048,
            "num_layers": 40,
            "num_heads": 16,
            "num_kv_heads": 2,
            "head_dim": 256,
            "linear_num_heads": 16,
            "linear_num_kv_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_kernel_dim": 4,
            "intermediate_size": 512,
            "num_experts": 256,
            "num_experts_per_token": 8,
            "shared_expert_intermediate": 512,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="qwen3_5",
        model_class=Qwen3_5Model,
        description="Qwen3.5 Dense with hybrid linear/full attention and SwiGLU FFN",
        architecture="qwen3_5",
        sparse_type="dense",
        attention_features=["linear_attention", "gqa"],
        default_config={
            "vocab_size": 248320,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 16,
            "num_kv_heads": 4,
            "intermediate_size": 12288,
            "linear_num_heads": 16,
            "linear_num_kv_heads": 32,
            "linear_key_head_dim": 256,
            "linear_value_head_dim": 256,
            "linear_kernel_dim": 4,
            "tie_word_embeddings": False,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="hunyuan_image_3_text",
        model_class=HunyuanImage3TextModel,
        description="HunyuanImage 3.0 Text Model with MoE and QK Norm",
        architecture="hunyuan_image_3_text",
        sparse_type="hunyuan_moe",
        attention_features=["qk_norm", "gqa"],
        default_config={
            "vocab_size": 133120,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "moe_intermediate_size": 3072,
            "num_experts": 64,
            "num_experts_per_token": 8,
            "num_shared_experts": 1,
            "use_qk_norm": True,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="hunyuan_image_3_diffusion",
        model_class=HunyuanImage3DiffusionModel,
        description="HunyuanImage 3.0 Diffusion Model for image generation",
        architecture="hunyuan_image_3_diffusion",
        sparse_type="hunyuan_moe",
        attention_features=["qk_norm", "gqa"],
        default_config={
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "moe_intermediate_size": 3072,
            "num_experts": 64,
            "num_experts_per_token": 8,
            "num_shared_experts": 1,
            "use_qk_norm": True,
            "image_height": 64,
            "image_width": 64,
            "latent_channels": 16,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="hunyuan-t5-encoder",
        model_class=HunyuanT5Encoder,
        description="HunyuanImage T5 Text Encoder (t5-v1_1-xxl)",
        architecture="t5_encoder",
        sparse_type="dense",
        attention_features=["standard"],
        default_config={
            "vocab_size": 32128,
            "hidden_size": 4096,
            "num_layers": 24,
            "num_heads": 64,
            "intermediate_size": 10240,
            "max_text_len": 512,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="hunyuan-vae-encoder",
        model_class=HunyuanVAEEncoder,
        description="HunyuanImage VAE Encoder for image generation",
        architecture="vae_encoder",
        sparse_type="dense",
        attention_features=[],
        default_config={
            "in_channels": 3,
            "latent_channels": 16,
            "block_out_channels": (128, 256, 512, 512),
            "use_3d": True,
            "use_attention": False,
            "dtype": "fp16",
        },
    )

    registry.register(
        name="hunyuan-vae-decoder",
        model_class=HunyuanVAEDecoder,
        description="HunyuanImage VAE Decoder for image generation",
        architecture="vae_decoder",
        sparse_type="dense",
        attention_features=[],
        default_config={
            "out_channels": 3,
            "latent_channels": 16,
            "block_out_channels": (128, 256, 512, 512),
            "use_3d": True,
            "use_attention": False,
            "dtype": "fp16",
        },
    )


def _get_llm_param_schema() -> dict:
    """Get parameter schema for LLM models (llama, mixtral, deepseek)."""
    return {
        "training": [
            {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 32},
            {"name": "seq_len", "label": "Sequence Length", "type": "number", "default": 4096},
        ],
        "inference": [
            {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 8},
            {"name": "prompt_len", "label": "Prompt Length", "type": "number", "default": 1024},
            {"name": "generation_len", "label": "Generation Length", "type": "number", "default": 128},
        ],
    }


def _get_video_param_schema() -> dict:
    """Get parameter schema for video generation models (wan-t2v-14b, wan-dit)."""
    return {
        "training": [
            {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 1},
            {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 81},
            {"name": "height", "label": "Height", "type": "number", "default": 720},
            {"name": "width", "label": "Width", "type": "number", "default": 1280},
        ],
        "inference": [
            {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 1},
            {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 81},
            {"name": "height", "label": "Height", "type": "number", "default": 720},
            {"name": "width", "label": "Width", "type": "number", "default": 1280},
            {"name": "num_steps", "label": "Inference Steps", "type": "number", "default": 50},
            {"name": "use_cfg", "label": "Use CFG", "type": "select", "default": "true", "options": ["true", "false"]},
        ],
    }


def _get_vae_param_schema() -> dict:
    """Get parameter schema for VAE models."""
    return {
        "training": [
            {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 8},
            {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 17},
            {"name": "height", "label": "Height", "type": "number", "default": 256},
            {"name": "width", "label": "Width", "type": "number", "default": 256},
        ],
        "inference": [
            {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 8},
            {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 17},
            {"name": "height", "label": "Height", "type": "number", "default": 256},
            {"name": "width", "label": "Width", "type": "number", "default": 256},
        ],
    }


def _get_resnet_param_schema() -> dict:
    """Get parameter schema for ResNet models."""
    return {
        "training": [
            {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 32},
            {"name": "height", "label": "Image Height", "type": "number", "default": 224},
            {"name": "width", "label": "Image Width", "type": "number", "default": 224},
        ],
        "inference": [
            {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 32},
            {"name": "height", "label": "Image Height", "type": "number", "default": 224},
            {"name": "width", "label": "Image Width", "type": "number", "default": 224},
        ],
    }


def _load_presets_from_yaml() -> dict:
    """Load preset configurations from YAML files.

    Scans configs/models/*.yaml and returns a dict of presets.
    Falls back to hardcoded presets if YAML files not found.
    """
    config_dir = Path(__file__).parent.parent.parent / "configs" / "models"

    if not config_dir.exists():
        return _get_hardcoded_presets()

    presets = {}
    for yaml_file in config_dir.glob("*.yaml"):
        with open(yaml_file, encoding="utf-8") as f:
            preset_data = yaml.safe_load(f)

        preset_name = yaml_file.stem

        preset = {
            "description": preset_data.get("description", ""),
            "preset_type": preset_data.get("preset_type", "model"),
            "architecture": preset_data.get("architecture", preset_name),
            "sparse_type": preset_data.get("sparse_type", "dense"),
            "attention_features": preset_data.get("attention_features", []),
            "supported_workloads": preset_data.get("supported_workloads", ["training", "inference"]),
        }

        if "config" in preset_data:
            preset.update(preset_data["config"])

        if "param_schema" in preset_data:
            preset["param_schema"] = preset_data["param_schema"]

        if "model_class_map" in preset_data:
            preset["model_class_map"] = preset_data["model_class_map"]

        presets[preset_name] = preset

    return presets if presets else _get_hardcoded_presets()


def _get_hardcoded_presets() -> dict:
    """Get preset configurations."""
    llm_schema = _get_llm_param_schema()
    video_schema = _get_video_param_schema()
    vae_schema = _get_vae_param_schema()
    resnet_schema = _get_resnet_param_schema()

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
            "param_schema": llm_schema,
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
            "param_schema": llm_schema,
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
            "param_schema": llm_schema,
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
            "param_schema": llm_schema,
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
            "param_schema": llm_schema,
        },
        "resnet50": {
            "architecture": "resnet",
            "sparse_type": "dense",
            "attention_features": [],
            "variant": "resnet50",
            "num_classes": 1000,
            "dtype": "fp16",
            "description": "ResNet-50",
            "param_schema": resnet_schema,
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
            "param_schema": vae_schema,
        },
        "wan-t2v-14b": {
            "architecture": "wan_pipeline",
            "sparse_type": "dense",
            "attention_features": [],
            "text_encoder": "wan-text-encoder",
            "dit": "wan-dit",
            "vae": "wan-vae",
            "description": "Wan2.1 Text-to-Video 14B pipeline",
            "param_schema": video_schema,
        },
        "wan-dit": {
            "architecture": "wan_dit",
            "sparse_type": "dense",
            "attention_features": ["standard"],
            "hidden_size": 5120,
            "num_layers": 40,
            "num_heads": 40,
            "dtype": "bf16",
            "description": "Wan2.1 DiT 14B",
            "param_schema": video_schema,
        },
    }


def get_model_presets() -> dict:
    """Get preset configurations."""
    global _PRESETS_CACHE
    if _PRESETS_CACHE is None:
        _PRESETS_CACHE = _load_presets_from_yaml()
    return _PRESETS_CACHE


def get_presets_by_sparse_type() -> dict:
    """Get presets grouped by sparse_type.
    
    Automatically groups presets into dense/sparse based on sparse_type field.
    All non-dense types (standard_moe, deepseek_moe, qwen3_5_moe, hunyuan_moe, etc.)
    are unified as 'sparse'.
    """
    presets = get_model_presets()
    result = {
        "dense": [],
        "sparse": [],
    }

    for name, config in presets.items():
        sparse_type = config.get("sparse_type", "dense")
        preset_info = {"name": name, **config}
        if sparse_type == "dense":
            result["dense"].append(preset_info)
        else:
            result["sparse"].append(preset_info)

    return result


def get_presets_by_workload(workload_type: str) -> dict:
    """Get presets filtered by workload type.
    
    Args:
        workload_type: Workload type string (training, inference, diffusion, etc.)
        
    Returns:
        Dict of presets that support the given workload type
    """
    presets = get_model_presets()
    filtered = {}
    
    for name, config in presets.items():
        supported_workloads = config.get("supported_workloads", ["training", "inference"])
        if workload_type in supported_workloads:
            filtered[name] = config
    
    return filtered


def get_presets_by_workload_grouped(workload_type: str) -> dict:
    """Get presets filtered by workload type, grouped by sparse_type.
    
    Args:
        workload_type: Workload type string
        
    Returns:
        Dict grouped by sparse_type (dense/sparse) with only workload-supporting presets.
        All non-dense types are unified as 'sparse'.
    """
    filtered_presets = get_presets_by_workload(workload_type)
    result = {
        "dense": [],
        "sparse": [],
    }
    
    for name, config in filtered_presets.items():
        sparse_type = config.get("sparse_type", "dense")
        preset_info = {"name": name, **config}
        if sparse_type == "dense":
            result["dense"].append(preset_info)
        else:
            result["sparse"].append(preset_info)
    
    return result


def create_model_from_config(config: dict, workload_type: Optional[str] = None) -> Any:
    """Create model from configuration.

    Supports:
    1. Preset mode: config contains 'preset'
    2. Architecture mode: config contains 'architecture'
    3. Name mode: config contains 'name'

    Args:
        config: Model configuration dict
        workload_type: Workload type (training, inference, diffusion, etc.)
                       Used to select model class from preset's model_class_map

    Returns:
        Model instance
    """
    registry = ModelingRegistry()
    presets = get_model_presets()

    meta_fields = {"type", "preset", "description", "architecture", "sparse_type", "attention_features", "model_class_map"}

    preset_name = config.get("preset")
    if preset_name and preset_name in presets:
        preset_config = presets[preset_name]
        merged_config = dict(preset_config)
        merged_config.update(config)
        architecture = merged_config.get("architecture", preset_name)
        
        model_class_map = merged_config.get("model_class_map", {})
        if workload_type and workload_type in model_class_map:
            model_name = model_class_map[workload_type]
            if registry.is_registered(model_name):
                for field in meta_fields:
                    merged_config.pop(field, None)
                model_class = registry.get_all_infos()[model_name].model_class
                filtered_config = _filter_params_for_model(model_class, merged_config)
                return registry.create(model_name, **filtered_config)
        
        for field in meta_fields:
            merged_config.pop(field, None)
        model_name = _find_model_by_architecture(registry, architecture)
        if model_name:
            model_class = registry.get_all_infos()[model_name].model_class
            filtered_config = _filter_params_for_model(model_class, merged_config)
            return registry.create(model_name, **filtered_config)
        if registry.is_registered(architecture):
            model_class = registry.get_all_infos()[architecture].model_class
            filtered_config = _filter_params_for_model(model_class, merged_config)
            return registry.create(architecture, **filtered_config)

    architecture = config.get("architecture")
    if architecture:
        model_name = _find_model_by_architecture(registry, architecture)
        if model_name:
            model_config = dict(config)
            for field in meta_fields:
                model_config.pop(field, None)
            model_class = registry.get_all_infos()[model_name].model_class
            filtered_config = _filter_params_for_model(model_class, model_config)
            return registry.create(model_name, **filtered_config)

    model_type = config.get("type")
    if model_type:
        if registry.is_registered(model_type):
            model_config = dict(config)
            for field in meta_fields:
                model_config.pop(field, None)
            model_class = registry.get_all_infos()[model_type].model_class
            filtered_config = _filter_params_for_model(model_class, model_config)
            return registry.create(model_type, **filtered_config)
        if model_type in presets:
            preset_config = presets[model_type]
            merged_config = dict(preset_config)
            merged_config.update(config)
            architecture = merged_config.get("architecture", model_type)
            for field in meta_fields:
                merged_config.pop(field, None)
            model_name = _find_model_by_architecture(registry, architecture)
            if model_name:
                model_class = registry.get_all_infos()[model_name].model_class
                filtered_config = _filter_params_for_model(model_class, merged_config)
                return registry.create(model_name, **filtered_config)

    model_name = config.get("name")
    if model_name and registry.is_registered(model_name):
        model_config = dict(config)
        for field in meta_fields:
            model_config.pop(field, None)
        model_class = registry.get_all_infos()[model_name].model_class
        filtered_config = _filter_params_for_model(model_class, model_config)
        return registry.create(model_name, **filtered_config)

    available_presets = list(presets.keys())
    raise ValueError(
        f"Unknown preset: '{preset_name or architecture or model_type or model_name}'. "
        f"Available presets: {available_presets}."
    )


def _find_model_by_architecture(registry: ModelingRegistry, architecture: str) -> Optional[str]:
    """Find model name by architecture."""
    for name, info in registry.get_all_infos().items():
        if info.architecture == architecture:
            return name
    return None


def _filter_params_for_model(model_class: Any, config: dict) -> dict:
    """Filter config params based on model class __init__ signature.

    Args:
        model_class: Model class to check signature
        config: Configuration dict

    Returns:
        Filtered config with only valid parameters
    """
    try:
        sig = inspect.signature(model_class.__init__)
        valid_params = {k for k in sig.parameters if k != "self"}
        return {k: v for k, v in config.items() if k in valid_params}
    except (ValueError, TypeError):
        return config


register_all_models()

"""Pipeline registration module.

Registers all built-in pipelines with the PipelineRegistry for dynamic discovery.
This module should be imported to register all pipelines.
"""

from ..core.registry import PipelineRegistry

# Import all pipeline classes
from .inference import InferencePipeline
from .diffusion_video import DiffusionVideoPipeline


def register_all_pipelines() -> None:
    """Register all built-in pipelines with the PipelineRegistry.

    This function should be called at application startup to make
    all pipelines available through the registry.
    """
    registry = PipelineRegistry()

    # Register inference pipeline for LLMs
    if not registry.is_registered("inference"):
        registry.register(
            name="inference",
            pipeline_class=InferencePipeline,
            description="Standard LLM inference pipeline (prefill + decode)",
            supported_models=["llm", "moe"],
            default_config={
                "batch_size": 1,
                "prompt_len": 512,
                "generation_len": 128,
            },
        )

    # Register diffusion video generation pipeline
    if not registry.is_registered("diffusion-video"):
        registry.register(
            name="diffusion-video",
            pipeline_class=DiffusionVideoPipeline,
            description="Text-to-video generation pipeline with iterative denoising",
            supported_models=["text_encoder", "dit", "vae"],
            default_config={
                "num_inference_steps": 50,
                "use_cfg": True,
                "batch_size": 1,
                "num_frames": 81,
                "height": 720,
                "width": 1280,
            },
        )


def get_pipeline_presets() -> dict:
    """Get preset configurations for common pipeline setups.

    Returns:
        Dictionary of preset configurations
    """
    return {
        "llm-inference": {
            "type": "inference",
            "description": "Standard LLM inference",
            "models": ["llama", "moe", "deepseek", "deepseek-v3"],
            "default_strategy": {
                "tp": 1,
                "pp": 1,
                "dp": 1,
                "ep": 1,
            },
        },
        "video-generation": {
            "type": "diffusion-video",
            "description": "Text-to-video generation (Wan2.1 style)",
            "models": ["wan-text-encoder", "wan-dit", "wan-vae"],
            "default_strategy": {
                "tp": 8,
                "pp": 1,
                "dp": 1,
                "sp": 1,
            },
        },
    }


# Auto-register on import
register_all_pipelines()

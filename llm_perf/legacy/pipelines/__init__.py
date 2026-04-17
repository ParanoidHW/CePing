"""Pipeline implementations for model composition.

Provides ready-to-use pipelines for common ML workflows:
- LLM inference pipeline
- Diffusion image generation pipeline
- Diffusion video generation pipeline

Architecture:
- core/pipeline.py: Abstract base classes (Pipeline, PipelineStep, etc.)
- core/registry.py: Registry classes (ModelRegistry, PipelineRegistry)
- pipelines/: Concrete implementations (InferencePipeline, DiffusionVideoPipeline)
- pipelines/register.py: Registration functions
"""

from .diffusion_video import DiffusionVideoPipeline, create_wan_t2v_pipeline
from .inference import InferencePipeline

__all__ = [
    "InferencePipeline",
    "DiffusionVideoPipeline",
    "create_wan_t2v_pipeline",
]

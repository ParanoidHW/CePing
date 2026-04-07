"""Pipeline implementations for model composition.

Provides ready-to-use pipelines for common ML workflows:
- LLM inference pipeline
- Diffusion image generation pipeline  
- Diffusion video generation pipeline
"""

from .base import InferencePipeline
from .diffusion_video import DiffusionVideoPipeline, create_wan_t2v_pipeline

__all__ = [
    "InferencePipeline",
    "DiffusionVideoPipeline", 
    "create_wan_t2v_pipeline",
]

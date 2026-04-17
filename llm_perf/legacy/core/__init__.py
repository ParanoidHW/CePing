"""Legacy core module - Core utilities for legacy models."""

from .registry import ModelRegistry
from .pipeline import Pipeline, PipelineResult, PipelineStage

__all__ = [
    "ModelRegistry",
    "Pipeline",
    "PipelineResult",
    "PipelineStage",
]

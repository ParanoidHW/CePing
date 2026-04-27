"""Handler registry for model-type-specific dispatch.

Provides explicit type-based handler selection, following architecture principle:
explicit dispatch over implicit hasattr checks.
"""

from llm_perf.analyzer.base import WorkloadType

from .base_handler import BaseModelHandler
from .diffusion_handler import DiffusionHandler
from .llm_handler import LLMHandler

HANDLER_REGISTRY = {
    WorkloadType.TRAINING: LLMHandler(),
    WorkloadType.INFERENCE: LLMHandler(),
    WorkloadType.DIFFUSION: DiffusionHandler(),
}

DEFAULT_HANDLER = LLMHandler()


def get_handler(workload_type: WorkloadType) -> BaseModelHandler:
    """Get handler for workload type.
    
    Args:
        workload_type: Workload type (TRAINING, INFERENCE, DIFFUSION, MIXED)
    
    Returns:
        Handler instance for this workload type
    
    Note:
        Vision models use compute_pattern-based dispatch (CONV_ENCODER, CONV_DECODER),
        handled by VisionHandler in _analyze_phase_with_submodules.
    """
    return HANDLER_REGISTRY.get(workload_type, DEFAULT_HANDLER)


__all__ = [
    "BaseModelHandler",
    "DiffusionHandler",
    "LLMHandler",
    "get_handler",
]
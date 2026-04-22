"""Validation module - Unified entry point for all validators.

This module provides configuration validation decoupled from evaluation logic:
1. Validation only checks parameters, does not change evaluation results
2. Validation failures return error messages, do not affect evaluation code
3. Validation success allows evaluation flow to proceed unchanged
"""

import logging
from typing import TYPE_CHECKING, Optional, Dict, Any

from .errors import ValidationErrors, ValidationError, ValidationLevel, ValidationCategory
from .strategy_validator import validate_strategy
from .model_validator import validate_model
from .sequence_validator import validate_sequence
from .memory_validator import validate_memory
from .special_validator import validate_special, validate_vpp

if TYPE_CHECKING:
    from llm_perf.strategy.parallel_context import ParallelContext

logger = logging.getLogger(__name__)


def validate_all(
    ctx: "ParallelContext",
    num_gpus: int,
    vocab_size: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    seq_len: int,
    num_kv_heads: Optional[int] = None,
    weight_memory_gb: Optional[float] = None,
    activation_memory_gb: Optional[float] = None,
    device_memory_gb: Optional[float] = None,
    gradient_memory_gb: Optional[float] = None,
    optimizer_memory_gb: Optional[float] = None,
    mode: str = "training",
    model_type: Optional[str] = None,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    patch_size: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_layers: Optional[int] = None,
    vpp_degree: int = 1,
    num_micro_batches: Optional[int] = None,
    pipeline_schedule: Optional[str] = None,
    num_experts: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
) -> ValidationErrors:
    """Run all validators and return combined results.
    
    Args:
        ctx: ParallelContext with parallel strategy configuration
        num_gpus: Total number of GPUs in the cluster
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        seq_len: Sequence length
        num_kv_heads: Number of KV heads for GQA
        weight_memory_gb: Weight memory per GPU in GB
        activation_memory_gb: Activation memory per GPU in GB
        device_memory_gb: Device memory capacity in GB
        gradient_memory_gb: Gradient memory per GPU in GB
        optimizer_memory_gb: Optimizer state memory per GPU in GB
        mode: Mode ("training" or "inference")
        model_type: Model type (e.g., "dit", "wan")
        image_height: Image height (for DiT)
        image_width: Image width (for DiT)
        patch_size: Patch size (for DiT)
        num_frames: Number of frames (for video models)
        num_layers: Number of layers (for VPP validation)
        vpp_degree: VPP degree
        num_micro_batches: Number of micro batches
        pipeline_schedule: Pipeline schedule type
        num_experts: Number of routed experts (for EP divisibility)
        global_batch_size: Global batch size (for batch divisibility)
        micro_batch_size: Micro batch size (for mini/micro divisibility)
    
    Returns:
        ValidationErrors containing all validation results
    """
    errors = ValidationErrors()
    
    logger.info(f"[Validation] Starting validation for mode={mode}, num_gpus={num_gpus}")
    
    errors.merge(validate_strategy(
        ctx, num_gpus, num_experts, global_batch_size, micro_batch_size
    ))
    
    errors.merge(validate_model(
        ctx,
        vocab_size,
        hidden_size,
        num_heads,
        intermediate_size,
        num_kv_heads,
    ))
    
    errors.merge(validate_sequence(ctx, seq_len, num_heads))
    
    if weight_memory_gb is not None and device_memory_gb is not None:
        errors.merge(validate_memory(
            ctx,
            weight_memory_gb,
            activation_memory_gb or 0.0,
            device_memory_gb,
            gradient_memory_gb,
            optimizer_memory_gb,
            mode,
        ))
    
    if model_type:
        errors.merge(validate_special(
            ctx,
            model_type,
            num_heads,
            image_height,
            image_width,
            patch_size,
            num_frames,
        ))
    
    if num_layers is not None and vpp_degree > 1:
        errors.merge(validate_vpp(
            ctx,
            num_layers,
            vpp_degree,
            num_micro_batches,
            pipeline_schedule,
        ))
    
    logger.info(f"[Validation] Completed: {errors}")
    
    return errors


def validate_to_dict(
    ctx: "ParallelContext",
    **kwargs,
) -> Dict[str, Any]:
    """Run validation and return dictionary result for JSON response.
    
    Args:
        ctx: ParallelContext
        **kwargs: Arguments for validate_all
    
    Returns:
        Dictionary with validation results suitable for JSON response
    """
    errors = validate_all(ctx, **kwargs)
    
    return {
        "success": not errors.has_errors(),
        "validation": errors.to_dict(),
    }


__all__ = [
    "ValidationErrors",
    "ValidationError",
    "ValidationLevel",
    "ValidationCategory",
    "validate_all",
    "validate_to_dict",
    "validate_strategy",
    "validate_model",
    "validate_sequence",
    "validate_memory",
    "validate_special",
    "validate_vpp",
]
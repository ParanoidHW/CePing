"""MemoryValidator - Memory capacity validation."""

import logging
from typing import TYPE_CHECKING, Optional

from .errors import ValidationError, ValidationErrors, ValidationLevel, ValidationCategory

if TYPE_CHECKING:
    from llm_perf.strategy.parallel_context import ParallelContext

logger = logging.getLogger(__name__)


def validate_memory(
    ctx: "ParallelContext",
    weight_memory_gb: float,
    activation_memory_gb: float,
    device_memory_gb: float,
    gradient_memory_gb: Optional[float] = None,
    optimizer_memory_gb: Optional[float] = None,
    mode: str = "training",
) -> ValidationErrors:
    """Validate memory capacity against device limits.
    
    Args:
        ctx: ParallelContext with parallel strategy configuration
        weight_memory_gb: Weight memory per GPU in GB
        activation_memory_gb: Activation memory per GPU in GB
        device_memory_gb: Device memory capacity in GB
        gradient_memory_gb: Gradient memory per GPU in GB (training only)
        optimizer_memory_gb: Optimizer state memory per GPU in GB (training only)
        mode: Mode ("training" or "inference")
    
    Returns:
        ValidationErrors containing any validation errors
    """
    errors = ValidationErrors()
    
    errors.merge(_validate_weight_memory(ctx, weight_memory_gb, device_memory_gb))
    
    if mode == "training":
        errors.merge(_validate_training_memory(
            ctx,
            weight_memory_gb,
            activation_memory_gb,
            device_memory_gb,
            gradient_memory_gb,
            optimizer_memory_gb,
        ))
    else:
        errors.merge(_validate_inference_memory(
            weight_memory_gb,
            activation_memory_gb,
            device_memory_gb,
        ))
    
    errors.merge(_validate_activation_threshold(activation_memory_gb, device_memory_gb))
    
    return errors


def _validate_weight_memory(
    ctx: "ParallelContext",
    weight_memory_gb: float,
    device_memory_gb: float,
) -> ValidationErrors:
    """Validate weight memory fits in device memory.
    
    Rule: weight_memory_physical <= device_memory_gb
    
    Note: Without ZeRO/offload, weights cannot be split
    """
    errors = ValidationErrors()
    
    if weight_memory_gb > device_memory_gb:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.MEMORY,
            code="WEIGHT_MEMORY_EXCEEDED",
            message=f"Weight memory ({weight_memory_gb:.2f} GB) exceeds device memory ({device_memory_gb:.2f} GB)",
            suggestion=f"Increase TP/PP degree to reduce weight memory per GPU, or enable ZeRO-3/offload",
            details={
                "weight_memory_gb": weight_memory_gb,
                "device_memory_gb": device_memory_gb,
                "tp_degree": ctx.tp_degree,
                "pp_degree": ctx.pp_degree,
                "zero_stage": ctx.zero_stage,
            },
        ))
    
    logger.info(f"[MemoryValidator] Weight memory check: {weight_memory_gb:.2f} GB vs {device_memory_gb:.2f} GB")
    return errors


def _validate_training_memory(
    ctx: "ParallelContext",
    weight_memory_gb: float,
    activation_memory_gb: float,
    device_memory_gb: float,
    gradient_memory_gb: Optional[float],
    optimizer_memory_gb: Optional[float],
) -> ValidationErrors:
    """Validate total training memory fits in device memory.
    
    Components:
    - Weight (static)
    - Activation (dynamic, depends on batch_size)
    - Gradient (static, = weight size without ZeRO)
    - Optimizer (static, Adam = weight × 2)
    
    ZeRO stages:
    - ZeRO-1: Optimizer sharded across DP
    - ZeRO-2: Gradient + Optimizer sharded
    - ZeRO-3: Weight + Gradient + Optimizer sharded
    """
    errors = ValidationErrors()
    
    gradient_memory = gradient_memory_gb if gradient_memory_gb is not None else weight_memory_gb
    optimizer_memory = optimizer_memory_gb if optimizer_memory_gb is not None else weight_memory_gb * 2
    
    total_memory = weight_memory_gb + activation_memory_gb + gradient_memory + optimizer_memory
    
    if ctx.zero_stage >= 1:
        optimizer_memory = optimizer_memory / ctx.dp_degree
        total_memory = weight_memory_gb + activation_memory_gb + gradient_memory + optimizer_memory
    
    if ctx.zero_stage >= 2:
        gradient_memory = gradient_memory / ctx.dp_degree
        total_memory = weight_memory_gb + activation_memory_gb + gradient_memory + optimizer_memory
    
    if ctx.zero_stage >= 3:
        weight_memory_gb = weight_memory_gb / ctx.dp_degree
        total_memory = weight_memory_gb + activation_memory_gb + gradient_memory + optimizer_memory
    
    if total_memory > device_memory_gb:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.MEMORY,
            code="TRAINING_MEMORY_EXCEEDED",
            message=f"Total training memory ({total_memory:.2f} GB) exceeds device memory ({device_memory_gb:.2f} GB)",
            suggestion=f"Enable activation checkpointing, reduce batch_size, or use higher ZeRO stage",
            details={
                "weight_memory_gb": weight_memory_gb,
                "activation_memory_gb": activation_memory_gb,
                "gradient_memory_gb": gradient_memory,
                "optimizer_memory_gb": optimizer_memory,
                "total_memory_gb": total_memory,
                "device_memory_gb": device_memory_gb,
                "zero_stage": ctx.zero_stage,
            },
        ))
    
    logger.info(f"[MemoryValidator] Training memory check: total={total_memory:.2f} GB vs device={device_memory_gb:.2f} GB")
    return errors


def _validate_inference_memory(
    weight_memory_gb: float,
    activation_memory_gb: float,
    device_memory_gb: float,
) -> ValidationErrors:
    """Validate total inference memory fits in device memory.
    
    Components:
    - Weight (static)
    - Activation (dynamic)
    """
    errors = ValidationErrors()
    
    total_memory = weight_memory_gb + activation_memory_gb
    
    if total_memory > device_memory_gb:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.MEMORY,
            code="INFERENCE_MEMORY_EXCEEDED",
            message=f"Total inference memory ({total_memory:.2f} GB) exceeds device memory ({device_memory_gb:.2f} GB)",
            suggestion=f"Increase TP degree or reduce batch_size",
            details={
                "weight_memory_gb": weight_memory_gb,
                "activation_memory_gb": activation_memory_gb,
                "total_memory_gb": total_memory,
                "device_memory_gb": device_memory_gb,
            },
        ))
    
    logger.info(f"[MemoryValidator] Inference memory check: total={total_memory:.2f} GB vs device={device_memory_gb:.2f} GB")
    return errors


def _validate_activation_threshold(
    activation_memory_gb: float,
    device_memory_gb: float,
) -> ValidationErrors:
    """Validate activation memory threshold (warning).
    
    Rule: activation_memory < 50% device_memory (recommended)
    
    Reference: Large activation memory may affect training stability
    """
    errors = ValidationErrors()
    
    threshold = device_memory_gb * 0.5
    
    if activation_memory_gb > threshold:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.MEMORY,
            code="ACTIVATION_MEMORY_HIGH",
            message=f"Activation memory ({activation_memory_gb:.2f} GB) exceeds 50% of device memory ({threshold:.2f} GB)",
            suggestion="Consider enabling activation checkpointing to reduce activation memory",
            details={
                "activation_memory_gb": activation_memory_gb,
                "device_memory_gb": device_memory_gb,
                "threshold_gb": threshold,
                "ratio": activation_memory_gb / device_memory_gb,
            },
        ))
    
    logger.info(f"[MemoryValidator] Activation threshold check: {activation_memory_gb:.2f} GB vs 50%={threshold:.2f} GB")
    return errors
"""ModelValidator - Model specification validation."""

import logging
from typing import TYPE_CHECKING, Optional

from .errors import ValidationError, ValidationErrors, ValidationLevel, ValidationCategory

if TYPE_CHECKING:
    from llm_perf.strategy.parallel_context import ParallelContext

logger = logging.getLogger(__name__)


def validate_model(
    ctx: "ParallelContext",
    vocab_size: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    num_kv_heads: Optional[int] = None,
) -> ValidationErrors:
    """Validate model specification against parallel strategy.
    
    Args:
        ctx: ParallelContext with parallel strategy configuration
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        num_kv_heads: Number of KV heads for GQA (default: same as num_heads)
    
    Returns:
        ValidationErrors containing any validation errors
    """
    errors = ValidationErrors()
    
    errors.merge(_validate_vocab_size(ctx, vocab_size))
    errors.merge(_validate_hidden_size(ctx, hidden_size))
    errors.merge(_validate_num_heads(ctx, num_heads))
    errors.merge(_validate_intermediate_size(ctx, intermediate_size))
    
    if num_kv_heads is not None:
        errors.merge(_validate_num_kv_heads(ctx, num_kv_heads, num_heads))
    
    return errors


def _validate_vocab_size(
    ctx: "ParallelContext",
    vocab_size: int,
) -> ValidationErrors:
    """Validate vocab_size is divisible by TP degree.
    
    Rule: vocab_size % tp_degree == 0
    
    Reference: Megatron-LM embedding parallelism
    """
    errors = ValidationErrors()
    
    if vocab_size % ctx.tp_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.MODEL,
            code="VOCAB_SIZE_NOT_DIVISIBLE",
            message=f"vocab_size ({vocab_size}) is not divisible by TP degree ({ctx.tp_degree})",
            suggestion=f"Adjust vocab_size or TP degree so vocab_size % TP == 0",
            details={
                "vocab_size": vocab_size,
                "tp_degree": ctx.tp_degree,
                "remainder": vocab_size % ctx.tp_degree,
            },
        ))
    
    logger.info(f"[ModelValidator] vocab_size check: {vocab_size} % {ctx.tp_degree} = {vocab_size % ctx.tp_degree}")
    return errors


def _validate_hidden_size(
    ctx: "ParallelContext",
    hidden_size: int,
) -> ValidationErrors:
    """Validate hidden_size is divisible by TP degree.
    
    Rule: hidden_size % tp_degree == 0
    
    Reference: Megatron-LM column parallel linear
    """
    errors = ValidationErrors()
    
    if hidden_size % ctx.tp_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.MODEL,
            code="HIDDEN_SIZE_NOT_DIVISIBLE",
            message=f"hidden_size ({hidden_size}) is not divisible by TP degree ({ctx.tp_degree})",
            suggestion=f"Adjust hidden_size or TP degree so hidden_size % TP == 0",
            details={
                "hidden_size": hidden_size,
                "tp_degree": ctx.tp_degree,
                "remainder": hidden_size % ctx.tp_degree,
            },
        ))
    
    logger.info(f"[ModelValidator] hidden_size check: {hidden_size} % {ctx.tp_degree} = {hidden_size % ctx.tp_degree}")
    return errors


def _validate_num_heads(
    ctx: "ParallelContext",
    num_heads: int,
) -> ValidationErrors:
    """Validate num_heads is divisible by TP degree.
    
    Rule: num_heads % tp_degree == 0
    
    Reference: Attention heads must be evenly distributed across TP ranks
    """
    errors = ValidationErrors()
    
    if num_heads % ctx.tp_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.MODEL,
            code="NUM_HEADS_NOT_DIVISIBLE",
            message=f"num_heads ({num_heads}) is not divisible by TP degree ({ctx.tp_degree})",
            suggestion=f"Adjust num_heads or TP degree so num_heads % TP == 0",
            details={
                "num_heads": num_heads,
                "tp_degree": ctx.tp_degree,
                "remainder": num_heads % ctx.tp_degree,
            },
        ))
    
    logger.info(f"[ModelValidator] num_heads check: {num_heads} % {ctx.tp_degree} = {num_heads % ctx.tp_degree}")
    return errors


def _validate_intermediate_size(
    ctx: "ParallelContext",
    intermediate_size: int,
) -> ValidationErrors:
    """Validate intermediate_size is divisible by TP degree.
    
    Rule: intermediate_size % tp_degree == 0
    
    Reference: FFN intermediate dimension must be evenly distributed
    """
    errors = ValidationErrors()
    
    if intermediate_size % ctx.tp_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.MODEL,
            code="INTERMEDIATE_SIZE_NOT_DIVISIBLE",
            message=f"intermediate_size ({intermediate_size}) is not divisible by TP degree ({ctx.tp_degree})",
            suggestion=f"Adjust intermediate_size or TP degree so intermediate_size % TP == 0",
            details={
                "intermediate_size": intermediate_size,
                "tp_degree": ctx.tp_degree,
                "remainder": intermediate_size % ctx.tp_degree,
            },
        ))
    
    logger.info(f"[ModelValidator] intermediate_size check: {intermediate_size} % {ctx.tp_degree} = {intermediate_size % ctx.tp_degree}")
    return errors


def _validate_num_kv_heads(
    ctx: "ParallelContext",
    num_kv_heads: int,
    num_heads: int,
) -> ValidationErrors:
    """Validate num_kv_heads for GQA/MQA.
    
    Rule: num_kv_heads % tp_degree == 0 (for efficient KV cache sharding)
    
    Reference: Llama-2/3 GQA design
    """
    errors = ValidationErrors()
    
    if num_kv_heads >= ctx.tp_degree:
        if num_kv_heads % ctx.tp_degree != 0:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.MODEL,
                code="NUM_KV_HEADS_NOT_DIVISIBLE",
                message=f"num_kv_heads ({num_kv_heads}) is not divisible by TP degree ({ctx.tp_degree})",
                suggestion=f"Adjust num_kv_heads or TP degree so num_kv_heads % TP == 0",
                details={
                    "num_kv_heads": num_kv_heads,
                    "tp_degree": ctx.tp_degree,
                    "remainder": num_kv_heads % ctx.tp_degree,
                },
            ))
    elif num_kv_heads == 1:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.MODEL,
            code="MQA_DETECTED",
            message=f"MQA detected: num_kv_heads=1 requires KV cache replication across TP ranks",
            suggestion="Consider using GQA (num_kv_heads > 1) for better memory efficiency",
            details={
                "num_kv_heads": num_kv_heads,
                "num_heads": num_heads,
                "tp_degree": ctx.tp_degree,
            },
        ))
    
    logger.info(f"[ModelValidator] num_kv_heads check: {num_kv_heads}, num_heads={num_heads}, TP={ctx.tp_degree}")
    return errors
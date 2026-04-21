"""StrategyValidator - Parallel strategy validation."""

import logging
from typing import TYPE_CHECKING

from .errors import ValidationError, ValidationErrors, ValidationLevel, ValidationCategory

if TYPE_CHECKING:
    from llm_perf.strategy.parallel_context import ParallelContext

logger = logging.getLogger(__name__)


def validate_strategy(
    ctx: "ParallelContext",
    num_gpus: int,
) -> ValidationErrors:
    """Validate parallel strategy configuration.
    
    Args:
        ctx: ParallelContext with parallel strategy configuration
        num_gpus: Total number of GPUs in the cluster
    
    Returns:
        ValidationErrors containing any validation errors
    """
    errors = ValidationErrors()
    
    errors.merge(_validate_parallel_product(ctx, num_gpus))
    errors.merge(_validate_ep_constraint(ctx))
    errors.merge(_validate_parallel_degrees(ctx))
    
    return errors


def _validate_parallel_product(
    ctx: "ParallelContext",
    num_gpus: int,
) -> ValidationErrors:
    """Validate parallel degree product equals GPU count.
    
    Rule: TP×PP×DP×EP×SP = total_gpus
    
    Reference: Megatron-LM, DeepSpeed official documentation
    """
    errors = ValidationErrors()
    
    product = (
        ctx.tp_degree
        * ctx.pp_degree
        * ctx.dp_degree
        * ctx.ep_degree
        * ctx.sp_degree
    )
    
    if product != num_gpus:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="PARALLEL_PRODUCT_MISMATCH",
            message=f"Parallel degree product ({product}) does not equal GPU count ({num_gpus})",
            suggestion=f"Adjust parallel degrees so TP×PP×DP×EP×SP = {num_gpus}",
            details={
                "tp_degree": ctx.tp_degree,
                "pp_degree": ctx.pp_degree,
                "dp_degree": ctx.dp_degree,
                "ep_degree": ctx.ep_degree,
                "sp_degree": ctx.sp_degree,
                "product": product,
                "num_gpus": num_gpus,
            },
        ))
    
    logger.info(f"[StrategyValidator] Parallel product check: {product} vs {num_gpus}")
    return errors


def _validate_ep_constraint(ctx: "ParallelContext") -> ValidationErrors:
    """Validate EP degree constraint.
    
    Rule: EP <= TP (Expert Parallelism depends on Tensor Parallelism)
    
    Reference: DeepSpeed-MoE, MegaBlocks
    """
    errors = ValidationErrors()
    
    if ctx.ep_degree > ctx.tp_degree:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="EP_EXCEEDS_TP",
            message=f"EP degree ({ctx.ep_degree}) exceeds TP degree ({ctx.tp_degree})",
            suggestion=f"Set EP degree <= TP degree (max {ctx.tp_degree})",
            details={
                "ep_degree": ctx.ep_degree,
                "tp_degree": ctx.tp_degree,
            },
        ))
    
    logger.info(f"[StrategyValidator] EP constraint check: EP={ctx.ep_degree} vs TP={ctx.tp_degree}")
    return errors


def _validate_parallel_degrees(ctx: "ParallelContext") -> ValidationErrors:
    """Validate all parallel degrees are >= 1.
    
    Rule: All parallel degrees must be positive integers >= 1
    """
    errors = ValidationErrors()
    
    degrees = {
        "tp_degree": ctx.tp_degree,
        "pp_degree": ctx.pp_degree,
        "dp_degree": ctx.dp_degree,
        "ep_degree": ctx.ep_degree,
        "sp_degree": ctx.sp_degree,
    }
    
    for name, value in degrees.items():
        if value < 1:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.STRATEGY,
                code="INVALID_PARALLEL_DEGREE",
                message=f"{name} ({value}) must be >= 1",
                suggestion=f"Set {name} to a positive integer >= 1",
                details={
                    "degree_name": name,
                    "value": value,
                },
            ))
    
    logger.info(f"[StrategyValidator] Parallel degrees check: {degrees}")
    return errors
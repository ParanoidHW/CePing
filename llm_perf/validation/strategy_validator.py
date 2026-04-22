"""StrategyValidator - Parallel strategy validation.

Supports layered parallelism:
- Attention layers: tp_degree × dp_degree × pp_degree × sp_degree = world_size
- MoE layers: expert_tp_degree × ep_degree × dp_degree × pp_degree × sp_degree = world_size
"""

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
    
    Supports layered parallelism where Attention and MoE can have different TP strategies.
    
    Args:
        ctx: ParallelContext with parallel strategy configuration
        num_gpus: Total number of GPUs in the cluster
    
    Returns:
        ValidationErrors containing any validation errors
    """
    errors = ValidationErrors()
    
    errors.merge(_validate_layered_parallelism(ctx, num_gpus))
    errors.merge(_validate_ep_performance_suggestion(ctx))
    errors.merge(_validate_parallel_degrees(ctx))
    
    return errors


def _validate_layered_parallelism(
    ctx: "ParallelContext",
    num_gpus: int,
) -> ValidationErrors:
    """Validate layered parallelism strategy.
    
    Layered Parallelism Rules:
    - Attention part: tp_degree × dp_degree × pp_degree × sp_degree = num_gpus
    - MoE part: expert_tp_degree × ep_degree × dp_degree × pp_degree × sp_degree = num_gpus
    
    If expert_tp_degree == tp_degree (uniform TP), both rules are equivalent.
    If expert_tp_degree != tp_degree (layered TP), both must satisfy independently.
    
    Reference: DeepSeek-V3, Mixtral MoE sharding strategies
    """
    errors = ValidationErrors()
    
    attn_product = (
        ctx.tp_degree
        * ctx.dp_degree
        * ctx.pp_degree
        * ctx.sp_degree
    )
    
    effective_expert_tp = ctx.expert_tp_degree or ctx.tp_degree
    moe_product = (
        effective_expert_tp
        * ctx.ep_degree
        * ctx.dp_degree
        * ctx.pp_degree
        * ctx.sp_degree
    )
    
    is_layered = (effective_expert_tp != ctx.tp_degree) or (ctx.ep_degree > 1)
    
    if attn_product != num_gpus:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="ATTN_PARALLEL_PRODUCT_MISMATCH",
            message=f"Attention parallel product ({attn_product}) does not equal GPU count ({num_gpus})",
            suggestion=f"Adjust parallel degrees so tp × dp × pp × sp = {num_gpus}",
            details={
                "tp_degree": ctx.tp_degree,
                "dp_degree": ctx.dp_degree,
                "pp_degree": ctx.pp_degree,
                "sp_degree": ctx.sp_degree,
                "attn_product": attn_product,
                "num_gpus": num_gpus,
                "layered_parallelism": is_layered,
            },
        ))
    
    if moe_product != num_gpus:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="MOE_PARALLEL_PRODUCT_MISMATCH",
            message=f"MoE parallel product ({moe_product}) does not equal GPU count ({num_gpus})",
            suggestion=f"Adjust parallel degrees so expert_tp × ep × dp × pp × sp = {num_gpus}",
            details={
                "expert_tp_degree": effective_expert_tp,
                "ep_degree": ctx.ep_degree,
                "dp_degree": ctx.dp_degree,
                "pp_degree": ctx.pp_degree,
                "sp_degree": ctx.sp_degree,
                "moe_product": moe_product,
                "num_gpus": num_gpus,
                "layered_parallelism": is_layered,
            },
        ))
    
    logger.info(
        f"[StrategyValidator] Layered parallelism check: "
        f"attn_product={attn_product}, moe_product={moe_product}, num_gpus={num_gpus}, "
        f"layered={is_layered}"
    )
    return errors


def _validate_ep_performance_suggestion(ctx: "ParallelContext") -> ValidationErrors:
    """Validate EP degree for performance optimization (WARNING level).
    
    Performance Suggestion: EP should not significantly exceed expert_tp_degree for
    optimal expert weight distribution.
    
    Reason:
    - Expert weights are sharded: {0: "ep", 2: "tp"}
    - When EP > expert_tp, expert distribution may be uneven
    - Performance may degrade but NOT a hard error
    
    This was previously a hard validation rule (EP <= TP), now relaxed to WARNING.
    
    Reference: DeepSpeed-MoE, MegaBlocks expert sharding strategies
    """
    errors = ValidationErrors()
    
    effective_expert_tp = ctx.expert_tp_degree or ctx.tp_degree
    
    if ctx.ep_degree > effective_expert_tp * 2:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.STRATEGY,
            code="EP_EXCEEDS_EXPERT_TP_WARNING",
            message=f"EP degree ({ctx.ep_degree}) significantly exceeds expert_tp ({effective_expert_tp}), may impact performance",
            suggestion=f"Consider setting EP <= expert_tp × 2 for optimal expert weight distribution",
            details={
                "ep_degree": ctx.ep_degree,
                "expert_tp_degree": effective_expert_tp,
                "tp_degree": ctx.tp_degree,
                "ratio": ctx.ep_degree / effective_expert_tp,
            },
        ))
    
    logger.info(
        f"[StrategyValidator] EP performance check: EP={ctx.ep_degree}, "
        f"expert_tp={effective_expert_tp}, tp={ctx.tp_degree}"
    )
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
        "expert_tp_degree": ctx.expert_tp_degree or ctx.tp_degree,
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
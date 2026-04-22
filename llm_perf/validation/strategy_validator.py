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
    num_devices: int,
    num_experts: int = None,
    global_batch_size: int = None,
    micro_batch_size: int = None,
) -> ValidationErrors:
    """Validate parallel strategy configuration.
    
    Supports layered parallelism where Attention and MoE can have different TP strategies.
    
    Args:
        ctx: ParallelContext with parallel strategy configuration
        num_devices: Total number of devices in the cluster
        num_experts: Number of routed experts (for EP divisibility check)
        global_batch_size: Global batch size (for batch divisibility check)
        micro_batch_size: Micro batch size (for mini/micro divisibility check)
    
    Returns:
        ValidationErrors containing any validation errors
    """
    errors = ValidationErrors()
    
    errors.merge(_validate_layered_parallelism(ctx, num_devices))
    errors.merge(_validate_ep_performance_suggestion(ctx))
    errors.merge(_validate_parallel_degrees(ctx))
    errors.merge(_validate_ep_expert_divisibility(ctx, num_experts))
    errors.merge(_validate_batch_size_divisibility(ctx, global_batch_size, micro_batch_size))
    
    return errors


def _validate_layered_parallelism(
    ctx: "ParallelContext",
    num_devices: int,
) -> ValidationErrors:
    """Validate layered parallelism strategy.
    
    Layered Parallelism Rules:
    - Attention part: tp_degree × dp_degree × pp_degree × sp_degree = num_devices
    - MoE part: expert_tp_degree × ep_degree × dp_degree × pp_degree × sp_degree = num_devices
    
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
    
    if attn_product != num_devices:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="ATTN_PARALLEL_PRODUCT_MISMATCH",
            message=f"Attention parallel product ({attn_product}) does not equal device count ({num_devices})",
            suggestion=f"Adjust parallel degrees so tp × dp × pp × sp = {num_devices}",
            details={
                "tp_degree": ctx.tp_degree,
                "dp_degree": ctx.dp_degree,
                "pp_degree": ctx.pp_degree,
                "sp_degree": ctx.sp_degree,
                "attn_product": attn_product,
                "num_devices": num_devices,
                "layered_parallelism": is_layered,
            },
        ))
    
    if moe_product != num_devices:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="MOE_PARALLEL_PRODUCT_MISMATCH",
            message=f"MoE parallel product ({moe_product}) does not equal device count ({num_devices})",
            suggestion=f"Adjust parallel degrees so expert_tp × ep × dp × pp × sp = {num_devices}",
            details={
                "expert_tp_degree": effective_expert_tp,
                "ep_degree": ctx.ep_degree,
                "dp_degree": ctx.dp_degree,
                "pp_degree": ctx.pp_degree,
                "sp_degree": ctx.sp_degree,
                "moe_product": moe_product,
                "num_devices": num_devices,
                "layered_parallelism": is_layered,
            },
        ))
    
    logger.info(
        f"[StrategyValidator] Layered parallelism check: "
        f"attn_product={attn_product}, moe_product={moe_product}, num_devices={num_devices}, "
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


def _validate_ep_expert_divisibility(
    ctx: "ParallelContext",
    num_experts: int,
) -> ValidationErrors:
    """Validate EP divides num_experts evenly.
    
    Rule: num_experts % ep_degree == 0 (ERROR)
    
    Reason:
    - MoE models require experts to be evenly distributed across EP ranks
    - Partial expert assignment is not possible
    
    Reference: DeepSeek-V3, Mixtral expert parallelism
    """
    errors = ValidationErrors()
    
    if num_experts is None or ctx.ep_degree <= 1:
        return errors
    
    if num_experts % ctx.ep_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="EP_EXPERT_DIVISIBILITY",
            message=f"num_experts ({num_experts}) is not divisible by EP ({ctx.ep_degree})",
            suggestion=f"EP should be a divisor of {num_experts} (e.g., {[d for d in range(1, num_experts+1) if num_experts % d == 0]})",
            details={
                "num_experts": num_experts,
                "ep_degree": ctx.ep_degree,
                "remainder": num_experts % ctx.ep_degree,
            },
        ))
        logger.warning(
            f"[StrategyValidator] EP divisibility check failed: "
            f"{num_experts} % {ctx.ep_degree} = {num_experts % ctx.ep_degree}"
        )
    else:
        logger.info(
            f"[StrategyValidator] EP divisibility check passed: "
            f"{num_experts} / {ctx.ep_degree} = {num_experts // ctx.ep_degree} experts/device"
        )
    
    return errors


def _validate_batch_size_divisibility(
    ctx: "ParallelContext",
    global_batch_size: int,
    micro_batch_size: int,
) -> ValidationErrors:
    """Validate batch size divisibility constraints.
    
    Rules:
    1. global_batch_size % dp_degree == 0 (ERROR)
    2. mini_batch_size = global_batch_size / dp_degree
    3. mini_batch_size % micro_batch_size == 0 (ERROR)
    
    Reason:
    - Global batch is split across DP ranks
    - Mini batch is further split into micro batches for gradient accumulation
    
    Reference: DeepSpeed, Megatron-LM batch size handling
    """
    errors = ValidationErrors()
    
    if global_batch_size is None:
        return errors
    
    if ctx.dp_degree <= 1:
        return errors
    
    if global_batch_size % ctx.dp_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="GLOBAL_BATCH_DP_DIVISIBILITY",
            message=f"global_batch_size ({global_batch_size}) is not divisible by DP ({ctx.dp_degree})",
            suggestion=f"global_batch_size should be a multiple of {ctx.dp_degree}",
            details={
                "global_batch_size": global_batch_size,
                "dp_degree": ctx.dp_degree,
                "remainder": global_batch_size % ctx.dp_degree,
            },
        ))
        logger.warning(
            f"[StrategyValidator] Global batch divisibility check failed: "
            f"{global_batch_size} % {ctx.dp_degree} = {global_batch_size % ctx.dp_degree}"
        )
        return errors
    
    mini_batch_size = global_batch_size // ctx.dp_degree
    
    if micro_batch_size is not None and mini_batch_size % micro_batch_size != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="MINI_BATCH_MICRO_DIVISIBILITY",
            message=f"mini_batch_size ({mini_batch_size}) is not divisible by micro_batch_size ({micro_batch_size})",
            suggestion=f"micro_batch_size should be a divisor of {mini_batch_size}",
            details={
                "global_batch_size": global_batch_size,
                "dp_degree": ctx.dp_degree,
                "mini_batch_size": mini_batch_size,
                "micro_batch_size": micro_batch_size,
                "remainder": mini_batch_size % micro_batch_size,
            },
        ))
        logger.warning(
            f"[StrategyValidator] Mini/micro batch divisibility check failed: "
            f"{mini_batch_size} % {micro_batch_size} = {mini_batch_size % micro_batch_size}"
        )
    else:
        logger.info(
            f"[StrategyValidator] Batch size divisibility check passed: "
            f"global_batch={global_batch_size}, mini_batch={mini_batch_size}, "
            f"micro_batch={micro_batch_size}"
        )
    
    return errors
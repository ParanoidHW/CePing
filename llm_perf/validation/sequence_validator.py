"""SequenceValidator - Sequence parallelism validation."""

import logging
from typing import Optional

from .errors import ValidationError, ValidationErrors, ValidationLevel, ValidationCategory

logger = logging.getLogger(__name__)


def validate_sequence(
    ctx: "ParallelContext",
    seq_len: int,
    num_heads: Optional[int] = None,
) -> ValidationErrors:
    """Validate sequence parallelism configuration.
    
    Args:
        ctx: ParallelContext with parallel strategy configuration
        seq_len: Sequence length
        num_heads: Number of attention heads (required for Ulysses SP)
    
    Returns:
        ValidationErrors containing any validation errors
    
    Note:
        seq_len divisible checks are soft constraints (WARNING level).
        They can be handled by padding in the algorithm.
    """
    errors = ValidationErrors()
    
    errors.merge(_validate_sp_degree(ctx, seq_len))
    
    if ctx.sp_type.value == "ulysses" or ctx.ulysses_degree > 1:
        errors.merge(_validate_ulysses_sp(ctx, seq_len, num_heads))
    
    if ctx.sp_type.value == "ring_p2p" or ctx.sp_type.value == "ring_allgather" or ctx.ring_degree > 1:
        errors.merge(_validate_ring_attention(ctx, seq_len))
    
    if ctx.sp_type.value == "megatron":
        errors.merge(_validate_megatron_sp(ctx, seq_len))
    
    if ctx.sp_type.value == "unified_2d" or (ctx.ulysses_degree > 1 and ctx.ring_degree > 1):
        errors.merge(_validate_unified_2d_sp(ctx, seq_len, num_heads))
    
    errors.merge(_validate_ulysses_tp_compatibility(ctx, num_heads))
    
    return errors


def _validate_sp_degree(
    ctx: "ParallelContext",
    seq_len: int,
) -> ValidationErrors:
    """Validate SP degree constraints.
    
    Rule: sp_degree <= tp_degree (SP is extension of TP) - HARD constraint
    Rule: seq_len % sp_degree == 0 - SOFT constraint (WARNING)
    """
    errors = ValidationErrors()
    
    if ctx.sp_degree > ctx.tp_degree:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.SEQUENCE,
            code="SP_EXCEEDS_TP",
            message=f"SP degree ({ctx.sp_degree}) exceeds TP degree ({ctx.tp_degree})",
            suggestion=f"Set SP degree <= TP degree (max {ctx.tp_degree})",
            details={
                "sp_degree": ctx.sp_degree,
                "tp_degree": ctx.tp_degree,
            },
        ))
    
    if ctx.sp_degree > 1 and seq_len % ctx.sp_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.SEQUENCE,
            code="SEQ_LEN_NOT_DIVISIBLE_BY_SP",
            message=f"seq_len ({seq_len}) is not divisible by SP degree ({ctx.sp_degree}), can be padded",
            suggestion=f"Padding seq_len to {((seq_len // ctx.sp_degree) + 1) * ctx.sp_degree} for even split",
            details={
                "seq_len": seq_len,
                "sp_degree": ctx.sp_degree,
                "remainder": seq_len % ctx.sp_degree,
                "padded_seq_len": ((seq_len // ctx.sp_degree) + 1) * ctx.sp_degree,
            },
        ))
    
    logger.info(f"[SequenceValidator] SP degree check: SP={ctx.sp_degree}, TP={ctx.tp_degree}, seq_len={seq_len}")
    return errors


def _validate_ulysses_sp(
    ctx: "ParallelContext",
    seq_len: int,
    num_heads: Optional[int],
) -> ValidationErrors:
    """Validate Ulysses Sequence Parallelism.
    
    Rules:
    - ulysses_degree >= 1 (HARD)
    - seq_len % ulysses_degree == 0 (SOFT - WARNING)
    - ulysses_degree * tp_degree <= num_heads (HARD)
    - num_heads % (ulysses_degree * tp_degree) == 0 (HARD)
    
    Reference: DeepSpeed Ulysses (arXiv:2309.14509)
    """
    errors = ValidationErrors()
    
    if ctx.ulysses_degree < 1:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.SEQUENCE,
            code="INVALID_ULYSSES_DEGREE",
            message=f"ulysses_degree ({ctx.ulysses_degree}) must be >= 1",
            suggestion="Set ulysses_degree to a positive integer >= 1",
            details={"ulysses_degree": ctx.ulysses_degree},
        ))
        return errors
    
    if seq_len % ctx.ulysses_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.SEQUENCE,
            code="SEQ_LEN_NOT_DIVISIBLE_BY_ULYSSES",
            message=f"seq_len ({seq_len}) is not divisible by ulysses_degree ({ctx.ulysses_degree}), can be padded",
            suggestion=f"Padding seq_len to {((seq_len // ctx.ulysses_degree) + 1) * ctx.ulysses_degree}",
            details={
                "seq_len": seq_len,
                "ulysses_degree": ctx.ulysses_degree,
                "remainder": seq_len % ctx.ulysses_degree,
                "padded_seq_len": ((seq_len // ctx.ulysses_degree) + 1) * ctx.ulysses_degree,
            },
        ))
    
    if num_heads is not None:
        total_degree = ctx.ulysses_degree * ctx.tp_degree
        if total_degree > num_heads:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SEQUENCE,
                code="ULYSSES_TP_EXCEEDS_HEADS",
                message=f"ulysses_degree * tp_degree ({total_degree}) exceeds num_heads ({num_heads})",
                suggestion=f"Reduce ulysses_degree or tp_degree so ulysses_degree * tp_degree <= {num_heads}",
                details={
                    "ulysses_degree": ctx.ulysses_degree,
                    "tp_degree": ctx.tp_degree,
                    "total_degree": total_degree,
                    "num_heads": num_heads,
                },
            ))
        elif num_heads % total_degree != 0:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SEQUENCE,
                code="HEADS_NOT_DIVISIBLE_BY_ULYSSES_TP",
                message=f"num_heads ({num_heads}) is not divisible by ulysses_degree * tp_degree ({total_degree})",
                suggestion=f"Adjust num_heads or parallel degrees so num_heads % (ulysses_degree * tp_degree) == 0",
                details={
                    "num_heads": num_heads,
                    "ulysses_degree": ctx.ulysses_degree,
                    "tp_degree": ctx.tp_degree,
                    "total_degree": total_degree,
                    "remainder": num_heads % total_degree,
                },
            ))
    
    logger.info(f"[SequenceValidator] Ulysses SP check: ulysses={ctx.ulysses_degree}, seq_len={seq_len}, heads={num_heads}")
    return errors


def _validate_ring_attention(
    ctx: "ParallelContext",
    seq_len: int,
) -> ValidationErrors:
    """Validate Ring Attention configuration.
    
    Rules:
    - ring_degree >= 1 (HARD)
    - seq_len % ring_degree == 0 (SOFT - WARNING)
    
    Reference: Ring Attention (arXiv:2309.14509)
    """
    errors = ValidationErrors()
    
    if ctx.ring_degree < 1:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.SEQUENCE,
            code="INVALID_RING_DEGREE",
            message=f"ring_degree ({ctx.ring_degree}) must be >= 1",
            suggestion="Set ring_degree to a positive integer >= 1",
            details={"ring_degree": ctx.ring_degree},
        ))
        return errors
    
    if ctx.ring_degree > 1 and seq_len % ctx.ring_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.SEQUENCE,
            code="SEQ_LEN_NOT_DIVISIBLE_BY_RING",
            message=f"seq_len ({seq_len}) is not divisible by ring_degree ({ctx.ring_degree}), can be padded",
            suggestion=f"Padding seq_len to {((seq_len // ctx.ring_degree) + 1) * ctx.ring_degree}",
            details={
                "seq_len": seq_len,
                "ring_degree": ctx.ring_degree,
                "remainder": seq_len % ctx.ring_degree,
                "padded_seq_len": ((seq_len // ctx.ring_degree) + 1) * ctx.ring_degree,
            },
        ))
    
    logger.info(f"[SequenceValidator] Ring Attention check: ring={ctx.ring_degree}, seq_len={seq_len}")
    return errors


def _validate_megatron_sp(
    ctx: "ParallelContext",
    seq_len: int,
) -> ValidationErrors:
    """Validate Megatron-SP configuration.
    
    Megatron-SP is a boolean switch that changes TP communication pattern:
    - Enabled: TP uses allgather + reducescatter (seq_len dimension split)
    - Disabled: TP uses allreduce (standard TP)
    
    Rules:
    - sp_degree must equal tp_degree (HARD - when megatron_sp enabled)
    - seq_len % tp_degree == 0 (SOFT - WARNING, can be padded)
    
    Reference: Megatron-LM SP (2022)
    """
    errors = ValidationErrors()
    
    if ctx.sp_degree != ctx.tp_degree:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.SEQUENCE,
            code="MEGATRON_SP_MISMATCH",
            message=f"Megatron-SP requires sp_degree ({ctx.sp_degree}) to equal tp_degree ({ctx.tp_degree})",
            suggestion=f"Set sp_degree = tp_degree = {ctx.tp_degree}",
            details={
                "sp_degree": ctx.sp_degree,
                "tp_degree": ctx.tp_degree,
            },
        ))
    
    if seq_len % ctx.tp_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.SEQUENCE,
            code="SEQ_LEN_NOT_DIVISIBLE_BY_TP",
            message=f"seq_len ({seq_len}) is not divisible by tp_degree ({ctx.tp_degree}) for Megatron-SP, can be padded",
            suggestion=f"Padding seq_len to {((seq_len // ctx.tp_degree) + 1) * ctx.tp_degree}",
            details={
                "seq_len": seq_len,
                "tp_degree": ctx.tp_degree,
                "remainder": seq_len % ctx.tp_degree,
                "padded_seq_len": ((seq_len // ctx.tp_degree) + 1) * ctx.tp_degree,
            },
        ))
    
    logger.info(f"[SequenceValidator] Megatron-SP check: SP={ctx.sp_degree}, TP={ctx.tp_degree}, seq_len={seq_len}")
    return errors


def _validate_unified_2d_sp(
    ctx: "ParallelContext",
    seq_len: int,
    num_heads: Optional[int],
) -> ValidationErrors:
    """Validate Unified 2D-SP (Ulysses + Ring) configuration.
    
    Rules:
    - sp_degree = ulysses_degree * ring_degree (HARD)
    - seq_len % sp_degree == 0 (SOFT - WARNING)
    - ulysses_degree * tp_degree <= num_heads (HARD)
    - num_heads % (ulysses_degree * tp_degree) == 0 (HARD)
    
    Reference: USP (arXiv:2405.07719)
    """
    errors = ValidationErrors()
    
    computed_sp = ctx.ulysses_degree * ctx.ring_degree
    if computed_sp != ctx.sp_degree:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.SEQUENCE,
            code="UNIFIED_2D_SP_MISMATCH",
            message=f"sp_degree ({ctx.sp_degree}) must equal ulysses_degree * ring_degree ({computed_sp})",
            suggestion=f"Set sp_degree = {computed_sp}",
            details={
                "sp_degree": ctx.sp_degree,
                "ulysses_degree": ctx.ulysses_degree,
                "ring_degree": ctx.ring_degree,
                "computed_sp": computed_sp,
            },
        ))
    
    if seq_len % ctx.sp_degree != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.SEQUENCE,
            code="SEQ_LEN_NOT_DIVISIBLE_BY_UNIFIED_SP",
            message=f"seq_len ({seq_len}) is not divisible by unified SP degree ({ctx.sp_degree}), can be padded",
            suggestion=f"Padding seq_len to {((seq_len // ctx.sp_degree) + 1) * ctx.sp_degree}",
            details={
                "seq_len": seq_len,
                "sp_degree": ctx.sp_degree,
                "remainder": seq_len % ctx.sp_degree,
                "padded_seq_len": ((seq_len // ctx.sp_degree) + 1) * ctx.sp_degree,
            },
        ))
    
    if num_heads is not None:
        total_degree = ctx.ulysses_degree * ctx.tp_degree
        if total_degree > num_heads:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SEQUENCE,
                code="UNIFIED_2D_HEADS_EXCEEDED",
                message=f"ulysses_degree * tp_degree ({total_degree}) exceeds num_heads ({num_heads})",
                suggestion=f"Reduce ulysses_degree or tp_degree so ulysses_degree * tp_degree <= {num_heads}",
                details={
                    "ulysses_degree": ctx.ulysses_degree,
                    "tp_degree": ctx.tp_degree,
                    "total_degree": total_degree,
                    "num_heads": num_heads,
                },
            ))
        elif num_heads % total_degree != 0:
            errors.add_error(ValidationError(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SEQUENCE,
                code="HEADS_NOT_DIVISIBLE_BY_ULYSSES_TP",
                message=f"num_heads ({num_heads}) is not divisible by ulysses_degree * tp_degree ({total_degree})",
                suggestion=f"Consider adjusting degrees for optimal head distribution",
                details={
                    "num_heads": num_heads,
                    "ulysses_degree": ctx.ulysses_degree,
                    "tp_degree": ctx.tp_degree,
                    "total_degree": total_degree,
                },
            ))
    
    logger.info(f"[SequenceValidator] Unified 2D-SP check: ulysses={ctx.ulysses_degree}, ring={ctx.ring_degree}, SP={ctx.sp_degree}")
    return errors


def _validate_ulysses_tp_compatibility(
    ctx: "ParallelContext",
    num_heads: Optional[int],
) -> ValidationErrors:
    """Validate Ulysses + TP compatibility with Megatron-SP.
    
    When Ulysses + TP is combined:
    - Megatron-SP is disabled (mutual exclusion)
    - TP splits on head dimension (not within head)
    - Validation: num_heads % (ulysses_degree * tp_degree) == 0
    
    Reference: DeepSpeed Ulysses + 360-LLaMA-Factory (arXiv:2505.22296)
    """
    errors = ValidationErrors()
    
    has_ulysses_tp = ctx.ulysses_degree > 1 and ctx.tp_degree > 1
    
    if has_ulysses_tp and ctx.sp_type.value == "megatron":
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.SEQUENCE,
            code="MEGATRON_SP_DISABLED_BY_ULYSSES_TP",
            message="Megatron-SP is disabled when Ulysses + TP combination is used",
            suggestion="Use Ulysses SP type instead of Megatron-SP for Ulysses + TP combination",
            details={
                "ulysses_degree": ctx.ulysses_degree,
                "tp_degree": ctx.tp_degree,
                "sp_type": ctx.sp_type.value,
            },
        ))
    
    if has_ulysses_tp and num_heads is not None:
        total_degree = ctx.ulysses_degree * ctx.tp_degree
        if total_degree > num_heads:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SEQUENCE,
                code="ULYSSES_TP_HEADS_EXCEEDED",
                message=f"ulysses_degree * tp_degree ({total_degree}) exceeds num_heads ({num_heads})",
                suggestion=f"Reduce ulysses_degree or tp_degree, or use Dummy-Head technique (arXiv:2505.22296)",
                details={
                    "ulysses_degree": ctx.ulysses_degree,
                    "tp_degree": ctx.tp_degree,
                    "total_degree": total_degree,
                    "num_heads": num_heads,
                    "tp_mode": "head_dimension_split",
                },
            ))
    
    logger.info(f"[SequenceValidator] Ulysses+TP compatibility: ulysses={ctx.ulysses_degree}, TP={ctx.tp_degree}, Megatron-SP disabled={has_ulysses_tp}")
    return errors
"""SpecialValidator - Special scenario validation."""

import logging
from typing import Optional

from .errors import ValidationError, ValidationErrors, ValidationLevel, ValidationCategory

logger = logging.getLogger(__name__)


def validate_special(
    ctx: "ParallelContext",
    model_type: Optional[str] = None,
    num_heads: Optional[int] = None,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    patch_size: Optional[int] = None,
    num_frames: Optional[int] = None,
) -> ValidationErrors:
    """Validate special scenarios (multimodal, DiT, etc).
    
    Args:
        ctx: ParallelContext with parallel strategy configuration
        model_type: Model type (e.g., "dit", "wan", "vae")
        num_heads: Number of attention heads (for DiT/Ulysses)
        image_height: Image height (for DiT)
        image_width: Image width (for DiT)
        patch_size: Patch size (for DiT)
        num_frames: Number of frames (for video models)
    
    Returns:
        ValidationErrors containing any validation errors
    """
    errors = ValidationErrors()
    
    if model_type in ["dit", "flux", "sd3", "wan"]:
        errors.merge(_validate_dit(ctx, num_heads, image_height, image_width, patch_size))
    
    if model_type == "wan" and num_frames is not None:
        errors.merge(_validate_video(ctx, num_frames, image_height, image_width, patch_size))
    
    return errors


def _validate_dit(
    ctx: "ParallelContext",
    num_heads: Optional[int],
    image_height: Optional[int],
    image_width: Optional[int],
    patch_size: Optional[int],
) -> ValidationErrors:
    """Validate DiT/Vision Transformer configuration.
    
    Rules:
    - DiT num_heads >= ulysses_degree * tp_degree
    - image_height % patch_size == 0
    - image_width % patch_size == 0
    - num_patches % sp_degree == 0
    
    Reference: DiT (Peebles & Xie, 2023)
    """
    errors = ValidationErrors()
    
    if num_heads is not None and ctx.ulysses_degree > 1:
        total_degree = ctx.ulysses_degree * ctx.tp_degree
        if total_degree > num_heads:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SPECIAL,
                code="DIT_HEADS_EXCEEDED",
                message=f"DiT num_heads ({num_heads}) is less than ulysses_degree * tp_degree ({total_degree})",
                suggestion=f"Reduce ulysses_degree or tp_degree so ulysses_degree * tp_degree <= {num_heads}",
                details={
                    "num_heads": num_heads,
                    "ulysses_degree": ctx.ulysses_degree,
                    "tp_degree": ctx.tp_degree,
                    "total_degree": total_degree,
                },
            ))
    
    if image_height is not None and patch_size is not None:
        if image_height % patch_size != 0:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SPECIAL,
                code="IMAGE_HEIGHT_NOT_DIVISIBLE",
                message=f"Image height ({image_height}) is not divisible by patch_size ({patch_size})",
                suggestion=f"Adjust image height or patch_size so height % patch_size == 0",
                details={
                    "image_height": image_height,
                    "patch_size": patch_size,
                    "remainder": image_height % patch_size,
                },
            ))
    
    if image_width is not None and patch_size is not None:
        if image_width % patch_size != 0:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SPECIAL,
                code="IMAGE_WIDTH_NOT_DIVISIBLE",
                message=f"Image width ({image_width}) is not divisible by patch_size ({patch_size})",
                suggestion=f"Adjust image width or patch_size so width % patch_size == 0",
                details={
                    "image_width": image_width,
                    "patch_size": patch_size,
                    "remainder": image_width % patch_size,
                },
            ))
    
    if image_height and image_width and patch_size and ctx.sp_degree > 1:
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        if num_patches % ctx.sp_degree != 0:
            errors.add_error(ValidationError(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SPECIAL,
                code="PATCHES_NOT_DIVISIBLE_BY_SP",
                message=f"Num patches ({num_patches}) is not divisible by SP degree ({ctx.sp_degree})",
                suggestion=f"Adjust image size or SP degree so num_patches % sp_degree == 0",
                details={
                    "image_height": image_height,
                    "image_width": image_width,
                    "patch_size": patch_size,
                    "num_patches": num_patches,
                    "sp_degree": ctx.sp_degree,
                    "remainder": num_patches % ctx.sp_degree,
                },
            ))
    
    logger.info(f"[SpecialValidator] DiT check: heads={num_heads}, image={image_height}x{image_width}, patch={patch_size}")
    return errors


def _validate_video(
    ctx: "ParallelContext",
    num_frames: int,
    image_height: Optional[int],
    image_width: Optional[int],
    patch_size: Optional[int],
) -> ValidationErrors:
    """Validate video/3D content configuration.
    
    Rules:
    - num_frames % temporal_sp_degree == 0 (if temporal SP)
    - num_spatial_patches % spatial_sp_degree == 0
    
    Reference: Wan2.1 (arXiv:2503.20314)
    """
    errors = ValidationErrors()
    
    temporal_compress = 4
    if ctx.sp_degree > 1 and num_frames % (ctx.sp_degree * temporal_compress) != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.SPECIAL,
            code="VIDEO_FRAMES_NOT_DIVISIBLE",
            message=f"Num frames ({num_frames}) may not be divisible by temporal compression ({temporal_compress})",
            suggestion=f"Consider adjusting num_frames for optimal temporal compression",
            details={
                "num_frames": num_frames,
                "temporal_compress": temporal_compress,
                "sp_degree": ctx.sp_degree,
            },
        ))
    
    if image_height and image_width and patch_size:
        spatial_compress = 8
        spatial_tokens = (image_height // spatial_compress) * (image_width // spatial_compress)
        if ctx.sp_degree > 1 and spatial_tokens % ctx.sp_degree != 0:
            errors.add_error(ValidationError(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SPECIAL,
                code="VIDEO_SPATIAL_NOT_DIVISIBLE",
                message=f"Spatial tokens ({spatial_tokens}) may not be divisible by SP degree ({ctx.sp_degree})",
                suggestion=f"Consider adjusting image size or SP degree",
                details={
                    "image_height": image_height,
                    "image_width": image_width,
                    "spatial_compress": spatial_compress,
                    "spatial_tokens": spatial_tokens,
                    "sp_degree": ctx.sp_degree,
                },
            ))
    
    logger.info(f"[SpecialValidator] Video check: frames={num_frames}, image={image_height}x{image_width}")
    return errors


def validate_vpp(
    ctx: "ParallelContext",
    num_layers: int,
    vpp_degree: int,
    num_micro_batches: Optional[int] = None,
    pipeline_schedule: Optional[str] = None,
    uniform_pp_stages: bool = True,
) -> ValidationErrors:
    """Validate Virtual Pipeline Parallelism (VPP) configuration.
    
    Rules:
    - vpp_degree >= 1 (HARD)
    - num_layers % (pp_degree * vpp_degree) == 0 (OPTIONAL, controlled by uniform_pp_stages)
    - num_micro_batches >= pp_degree * vpp_degree * 2 (for interleaved) (WARNING)
    
    Reference: Megatron-LM Interleaved Pipeline Schedule
    
    Args:
        ctx: ParallelContext
        num_layers: Number of transformer layers
        vpp_degree: Virtual pipeline parallelism degree
        num_micro_batches: Number of micro batches for pipeline schedule
        pipeline_schedule: Pipeline schedule type ("1f1b", "gpipe", "interleaved")
        uniform_pp_stages: If True, validate num_layers divisibility (default True)
            When False, allows non-uniform stage distribution (different layers per stage)
    """
    errors = ValidationErrors()
    
    if vpp_degree < 1:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.SPECIAL,
            code="INVALID_VPP_DEGREE",
            message=f"vpp_degree ({vpp_degree}) must be >= 1",
            suggestion="Set vpp_degree to a positive integer >= 1",
            details={"vpp_degree": vpp_degree},
        ))
        return errors
    
    total_stages = ctx.pp_degree * vpp_degree
    if uniform_pp_stages and num_layers % total_stages != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.SPECIAL,
            code="LAYERS_NOT_DIVISIBLE_BY_VPP",
            message=f"num_layers ({num_layers}) is not divisible by pp_degree * vpp_degree ({total_stages})",
            suggestion=f"Adjust num_layers or VPP degrees, or set uniform_pp_stages=False for non-uniform distribution",
            details={
                "num_layers": num_layers,
                "pp_degree": ctx.pp_degree,
                "vpp_degree": vpp_degree,
                "total_stages": total_stages,
                "remainder": num_layers % total_stages,
                "uniform_pp_stages": uniform_pp_stages,
            },
        ))
    
    if not uniform_pp_stages and num_layers % total_stages != 0:
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.SPECIAL,
            code="VPP_NON_UNIFORM_DISTRIBUTION",
            message=f"num_layers ({num_layers}) will have non-uniform distribution across VPP stages ({total_stages})",
            suggestion=f"Stage distribution: {num_layers // total_stages} base layers + {num_layers % total_stages} extra layers",
            details={
                "num_layers": num_layers,
                "pp_degree": ctx.pp_degree,
                "vpp_degree": vpp_degree,
                "total_stages": total_stages,
                "base_layers_per_stage": num_layers // total_stages,
                "extra_layers": num_layers % total_stages,
            },
        ))
    
    if pipeline_schedule == "interleaved" and num_micro_batches is not None:
        min_micro_batches = ctx.pp_degree * vpp_degree * 2
        if num_micro_batches < min_micro_batches:
            errors.add_error(ValidationError(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SPECIAL,
                code="INSUFFICIENT_MICRO_BATCHES",
                message=f"num_micro_batches ({num_micro_batches}) is less than recommended ({min_micro_batches}) for interleaved schedule",
                suggestion=f"Set num_micro_batches >= {min_micro_batches} for optimal pipeline efficiency",
                details={
                    "num_micro_batches": num_micro_batches,
                    "min_micro_batches": min_micro_batches,
                    "pp_degree": ctx.pp_degree,
                    "vpp_degree": vpp_degree,
                },
            ))
    
    logger.info(f"[SpecialValidator] VPP check: layers={num_layers}, PP={ctx.pp_degree}, VPP={vpp_degree}, uniform={uniform_pp_stages}")
    return errors
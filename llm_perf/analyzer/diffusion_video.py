"""Diffusion Video Generation Performance Analyzer.

Analyzes the complete text-to-video generation pipeline including:
1. Text Encoder (umT5-XXL)
2. DiT Backbone (Diffusion Transformer)
3. VAE (Encoder and Decoder)
4. Full denoising pipeline

Based on Wan2.1 architecture with Flow Matching framework.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from llm_perf.legacy.models.wan_video import WanTextEncoder, WanDiTModel, WanVAEModel
from ..hardware.device import Device, ComputeUnitType
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig
from ..kernels.compute import ComputeKernelRegistry
from ..utils.constants import DTYPE_SIZES

from .detailed_breakdown import DetailedPerformanceResult
from .breakdown_generator import BreakdownGenerator, PipelineBreakdownGenerator
from .result_base import BaseResult


@dataclass
class DiffusionVideoResult(BaseResult):
    """Result of diffusion video generation performance analysis."""

    # Component-specific times (seconds)
    text_encoder_time_sec: float
    dit_single_step_time_sec: float
    vae_encoder_time_sec: float
    vae_decoder_time_sec: float

    # Full pipeline time
    total_generation_time_sec: float

    # Memory usage (GB per GPU)
    peak_memory_text_encoder_gb: float
    peak_memory_dit_gb: float
    peak_memory_vae_gb: float
    peak_memory_total_gb: float

    # Throughput metrics
    video_pixels_per_sec: float  # Total pixels generated per second

    # Breakdown
    component_breakdown: Dict[str, float]

    # Detailed breakdown (optional)
    detailed_breakdown: Optional[DetailedPerformanceResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "time": {
                "text_encoder_sec": self.text_encoder_time_sec,
                "dit_single_step_sec": self.dit_single_step_time_sec,
                "vae_encoder_sec": self.vae_encoder_time_sec,
                "vae_decoder_sec": self.vae_decoder_time_sec,
                "total_generation_sec": self.total_generation_time_sec,
            },
            "memory": {
                "text_encoder_gb": self.peak_memory_text_encoder_gb,
                "dit_gb": self.peak_memory_dit_gb,
                "vae_gb": self.peak_memory_vae_gb,
                "total_gb": self.peak_memory_total_gb,
            },
            "throughput": {
                "video_pixels_per_sec": self.video_pixels_per_sec,
            },
            "component_breakdown": self.component_breakdown,
        }
        if self.detailed_breakdown is not None:
            result["detailed_breakdown"] = self.detailed_breakdown.to_dict()
        return result


class DiffusionVideoAnalyzer:
    """Analyzer for diffusion-based video generation.
    
    Evaluates performance of Wan2.1-style text-to-video generation
    with separate assessment of Text Encoder, DiT, and VAE components.
    """
    
    def __init__(
        self,
        text_encoder: WanTextEncoder,
        dit: WanDiTModel,
        vae: WanVAEModel,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae = vae
        self.device = device
        self.cluster = cluster
        self.strategy = strategy
        
        self.compute_registry = ComputeKernelRegistry(device)
    
    def analyze(
        self,
        num_frames: int = 81,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 50,
        use_cfg: bool = True,
        include_detailed_breakdown: bool = True,
    ) -> DiffusionVideoResult:
        """
        Analyze video generation performance.

        Args:
            num_frames: Number of video frames to generate
            height: Video height in pixels
            width: Video width in pixels
            num_inference_steps: Number of denoising steps
            use_cfg: Whether to use Classifier-Free Guidance (doubles batch)
            include_detailed_breakdown: Whether to include detailed breakdown

        Returns:
            DiffusionVideoResult with performance metrics
        """
        # Text Encoder evaluation
        text_encoder_time = self._estimate_text_encoder_time()
        text_encoder_memory = self._estimate_text_encoder_memory(inference_mode=True)

        # DiT single step evaluation
        dit_single_step_time = self._estimate_dit_single_step(
            num_frames, height, width, use_cfg
        )
        dit_memory = self._estimate_dit_memory(
            num_frames, height, width, use_cfg, inference_mode=True
        )

        # VAE evaluation
        vae_encoder_time, vae_decoder_time = self._estimate_vae_time(
            num_frames, height, width
        )
        vae_memory = self._estimate_vae_memory(num_frames, height, width)

        # Full pipeline time
        # Text encoder runs once
        # DiT runs for num_inference_steps
        # VAE encoder runs once (for reference video if provided)
        # VAE decoder runs once at the end
        total_time = (
            text_encoder_time +
            dit_single_step_time * num_inference_steps +
            vae_decoder_time  # Decoder only for generation
        )

        # Peak memory is the max of all components
        peak_memory = max(text_encoder_memory, dit_memory, vae_memory)

        # Throughput: total pixels generated per second
        total_pixels = num_frames * height * width * num_inference_steps
        pixels_per_sec = total_pixels / total_time if total_time > 0 else 0

        # Ensure total_time is not zero
        if total_time <= 0:
            total_time = 1e-6  # Minimum time to avoid division by zero

        # Component breakdown
        breakdown = {
            "text_encoder_pct": text_encoder_time / total_time * 100,
            "dit_pct": (dit_single_step_time * num_inference_steps) / total_time * 100,
            "vae_decoder_pct": vae_decoder_time / total_time * 100,
            "text_encoder_time": text_encoder_time,
            "dit_total_time": dit_single_step_time * num_inference_steps,
            "dit_single_step_time": dit_single_step_time,
            "vae_decoder_time": vae_decoder_time,
        }

        # Generate detailed breakdown
        detailed_breakdown = None
        if include_detailed_breakdown:
            detailed_breakdown = self._generate_detailed_breakdown(
                num_frames, height, width, num_inference_steps, use_cfg,
                text_encoder_time, dit_single_step_time, vae_encoder_time,
                vae_decoder_time, total_time, pixels_per_sec
            )

        return DiffusionVideoResult(
            text_encoder_time_sec=text_encoder_time,
            dit_single_step_time_sec=dit_single_step_time,
            vae_encoder_time_sec=vae_encoder_time,
            vae_decoder_time_sec=vae_decoder_time,
            total_generation_time_sec=total_time,
            peak_memory_text_encoder_gb=text_encoder_memory / 1024**3,
            peak_memory_dit_gb=dit_memory / 1024**3,
            peak_memory_vae_gb=vae_memory / 1024**3,
            peak_memory_total_gb=peak_memory / 1024**3,
            video_pixels_per_sec=pixels_per_sec,
            component_breakdown=breakdown,
            detailed_breakdown=detailed_breakdown,
        )

    def _generate_detailed_breakdown(
        self,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int,
        use_cfg: bool,
        text_encoder_time: float,
        dit_single_step_time: float,
        vae_encoder_time: float,
        vae_decoder_time: float,
        total_time: float,
        pixels_per_sec: float,
    ) -> DetailedPerformanceResult:
        """Generate detailed breakdown for the pipeline (inference mode)."""
        # Create generators for each sub-model (is_training=False for inference)
        text_encoder_gen = BreakdownGenerator(
            self.text_encoder, self.device, self.cluster, self.strategy, is_training=False
        )
        dit_gen = BreakdownGenerator(
            self.dit, self.device, self.cluster, self.strategy, is_training=False
        )
        vae_gen = BreakdownGenerator(
            self.vae, self.device, self.cluster, self.strategy, is_training=False
        )

        # Create pipeline generator
        pipeline_gen = PipelineBreakdownGenerator(
            [
                ("Text Encoder", "text_encoder", text_encoder_gen),
                ("DiT", "dit", dit_gen),
                ("VAE", "vae", vae_gen),
            ],
            self.device,
            self.cluster,
            self.strategy,
        )

        # Generate detailed result
        return pipeline_gen.generate(
            total_time_sec=total_time,
            throughput=pixels_per_sec,
            metadata={
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "use_cfg": use_cfg,
            },
        )
    
    def _estimate_text_encoder_time(self) -> float:
        """Estimate text encoder forward pass time."""
        dtype = self.text_encoder.config.dtype
        total_flops = self.text_encoder.total_flops_forward
        
        # Use roofline model for time estimation
        achievable_flops = self.device.estimate_roofline_flops(
            arithmetic_intensity=total_flops / (self.text_encoder.activation_memory + 1),
            dtype=dtype,
            unit_type=ComputeUnitType.CUBE_TENSOR_CORE
        )
        
        if achievable_flops <= 0:
            # Fallback: use theoretical peak FLOPS
            achievable_flops = self.device.get_compute_tflops(dtype) * 1e12
        
        return total_flops / achievable_flops if achievable_flops > 0 else 0.0
    
    def _estimate_text_encoder_memory(
        self, inference_mode: bool = True
    ) -> int:
        """Estimate text encoder peak memory in bytes.

        Uses the generic memory estimation from BaseModel with video-specific
        configurations.

        Args:
            inference_mode: If True, use inference memory estimation (layer-wise
                activation reuse). If False, use training memory (all layers saved).

        Returns:
            Estimated peak memory in bytes
        """
        # Use BaseModel's generic estimate_memory method
        # Text encoder uses batch_size=1 (single text prompt)
        return self.text_encoder.estimate_memory(
            inference_mode=inference_mode,
            batch_size=1,
            is_distributed=False,  # Text encoder typically runs on single GPU
            apply_calibration=True,
        )
    
    def _estimate_dit_single_step(
        self,
        num_frames: int,
        height: int,
        width: int,
        use_cfg: bool,
    ) -> float:
        """Estimate DiT single denoising step time."""
        cfg = self.dit.config
        dtype = cfg.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        # Calculate sequence length after patchify
        pt, ph, pw = cfg.patch_size
        latent_t = (num_frames - 1) // 4 + 1  # VAE temporal compression
        latent_h = height // 8  # VAE spatial compression
        latent_w = width // 8
        
        seq_len = (latent_t // pt) * (latent_h // ph) * (latent_w // pw)
        batch_size = 2 if use_cfg else 1  # CFG doubles batch
        
        # Total FLOPs for single step
        total_flops = self.dit.total_flops_forward
        
        # Adjust for actual sequence length and batch
        # FLOPs scale linearly with sequence length and batch
        base_seq_len = (
            (cfg.latent_num_frames // pt) *
            (cfg.latent_height // ph) *
            (cfg.latent_width // pw)
        )
        scale_factor = (seq_len / base_seq_len) * batch_size
        adjusted_flops = total_flops * scale_factor
        
        # Estimate using roofline model
        activation_memory = (
            seq_len * batch_size * cfg.hidden_size * dtype_size *
            cfg.num_layers * 4  # Rough estimate for all activations
        )
        
        achievable_flops = self.device.estimate_roofline_flops(
            arithmetic_intensity=adjusted_flops / (activation_memory + 1),
            dtype=dtype,
            unit_type=ComputeUnitType.CUBE_TENSOR_CORE
        )
        
        if achievable_flops <= 0:
            achievable_flops = self.device.get_compute_tflops(dtype) * 1e12
        
        return adjusted_flops / achievable_flops if achievable_flops > 0 else 0.0
    
    def _estimate_dit_memory(
        self,
        num_frames: int,
        height: int,
        width: int,
        use_cfg: bool,
        inference_mode: bool = True,
    ) -> int:
        """Estimate DiT peak memory for single step.

        Uses BaseModel's generic memory estimation with video-specific
        scaling for sequence length and CFG batch size.

        Args:
            num_frames: Number of frames
            height: Video height
            width: Video width
            use_cfg: Whether to use classifier-free guidance
            inference_mode: If True, use inference memory estimation (layer-wise
                activation reuse). If False, use training memory (all layers saved).

        Returns:
            Estimated peak memory in bytes
        """
        cfg = self.dit.config

        # Calculate effective batch size (CFG doubles batch)
        batch_size = 2 if use_cfg else 1

        # Use BaseModel's generic estimate_memory method
        # This properly handles layer-wise activation lifecycle
        base_memory = self.dit.estimate_memory(
            inference_mode=inference_mode,
            batch_size=batch_size,
            is_distributed=self.strategy.tp_degree > 1 or self.strategy.dp_degree > 1,
            apply_calibration=True,
        )

        # Scale memory based on actual sequence length vs base config
        # Base config uses latent_num_frames/height/width from config
        pt, ph, pw = cfg.patch_size
        latent_t = (num_frames - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8

        actual_seq_len = (latent_t // pt) * (latent_h // ph) * (latent_w // pw)
        base_seq_len = (
            (cfg.latent_num_frames // pt)
            * (cfg.latent_height // ph)
            * (cfg.latent_width // pw)
        )

        # Memory scales roughly linearly with sequence length for activations
        # Parameters don't scale, so we use a conservative scaling factor
        seq_scale = actual_seq_len / max(base_seq_len, 1)

        # Activation memory scales with seq_len, but parameter memory doesn't
        # So we apply partial scaling (rough estimate: 60% activation, 40% params)
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        param_memory = self.dit.total_params * dtype_size
        activation_memory = base_memory - param_memory

        scaled_activation = int(activation_memory * seq_scale)

        return param_memory + scaled_activation
    
    def _estimate_vae_time(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tuple[float, float]:
        """Estimate VAE encoder and decoder time."""
        dtype = self.vae.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        # Encoder FLOPs
        encoder_flops = sum(
            layer.flops for layer in self.vae.layers
            if "encoder" in layer.name
        )
        
        # Decoder FLOPs
        decoder_flops = sum(
            layer.flops for layer in self.vae.layers
            if "decoder" in layer.name
        )
        
        # Scale by actual resolution
        base_pixels = self.vae.config.num_frames * self.vae.config.height * self.vae.config.width
        actual_pixels = num_frames * height * width
        scale_factor = actual_pixels / base_pixels
        
        encoder_flops *= scale_factor
        decoder_flops *= scale_factor
        
        # Estimate using roofline
        encoder_memory = self.vae.config.latent_channels * num_frames * height * width * dtype_size
        decoder_memory = self.vae.config.in_channels * num_frames * height * width * dtype_size
        
        encoder_flops_rate = self.device.estimate_roofline_flops(
            arithmetic_intensity=encoder_flops / (encoder_memory + 1),
            dtype=dtype,
            unit_type=ComputeUnitType.CUBE_TENSOR_CORE
        )
        
        decoder_flops_rate = self.device.estimate_roofline_flops(
            arithmetic_intensity=decoder_flops / (decoder_memory + 1),
            dtype=dtype,
            unit_type=ComputeUnitType.CUBE_TENSOR_CORE
        )
        
        # Fallback if roofline returns 0
        if encoder_flops_rate <= 0:
            encoder_flops_rate = self.device.get_compute_tflops(dtype) * 1e12
        if decoder_flops_rate <= 0:
            decoder_flops_rate = self.device.get_compute_tflops(dtype) * 1e12
        
        encoder_time = encoder_flops / encoder_flops_rate if encoder_flops_rate > 0 else 0.0
        decoder_time = decoder_flops / decoder_flops_rate if decoder_flops_rate > 0 else 0.0
        
        return encoder_time, decoder_time
    
    def _estimate_vae_memory(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> int:
        """Estimate VAE peak memory."""
        cfg = self.vae.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        
        # Parameters
        param_memory = self.vae.total_params * dtype_size
        
        # Peak activation: input video + latent + intermediate
        input_memory = num_frames * height * width * cfg.in_channels * dtype_size
        latent_t = (num_frames - 1) // cfg.temporal_compression + 1
        latent_h = height // cfg.spatial_compression
        latent_w = width // cfg.spatial_compression
        latent_memory = latent_t * latent_h * latent_w * cfg.latent_channels * dtype_size
        
        # Intermediate (rough estimate: ~2x largest feature map)
        max_channels = max(cfg.block_out_channels)
        intermediate_memory = latent_t * latent_h * latent_w * max_channels * dtype_size * 2
        
        return param_memory + max(input_memory, latent_memory + intermediate_memory)
    
    def analyze_components_separately(
        self,
        num_frames: int = 81,
        height: int = 720,
        width: int = 1280,
    ) -> Dict[str, Any]:
        """
        Analyze each component separately for detailed breakdown.
        
        Returns detailed analysis of Text Encoder, DiT, and VAE.
        """
        results = {}
        
        # Text Encoder analysis
        results["text_encoder"] = {
            "total_params": self.text_encoder.total_params,
            "total_flops_forward": self.text_encoder.total_flops_forward,
            "activation_memory_bytes": self.text_encoder.activation_memory,
            "estimated_time_sec": self._estimate_text_encoder_time(),
            "estimated_memory_gb": self._estimate_text_encoder_memory(inference_mode=True) / 1024**3,
        }
        
        # DiT analysis
        results["dit"] = {
            "total_params": self.dit.total_params,
            "total_flops_forward": self.dit.total_flops_forward,
            "activation_memory_bytes": self.dit.activation_memory,
            "estimated_single_step_time_sec": self._estimate_dit_single_step(
                num_frames, height, width, use_cfg=False
            ),
            "estimated_memory_gb": self._estimate_dit_memory(
                num_frames, height, width, use_cfg=False, inference_mode=True
            ) / 1024**3,
            "num_layers": self.dit.config.num_layers,
            "hidden_size": self.dit.config.hidden_size,
            "num_attention_heads": self.dit.config.num_attention_heads,
        }
        
        # VAE analysis
        encoder_flops = sum(layer.flops for layer in self.vae.layers if "encoder" in layer.name)
        decoder_flops = sum(layer.flops for layer in self.vae.layers if "decoder" in layer.name)
        encoder_time, decoder_time = self._estimate_vae_time(num_frames, height, width)
        
        results["vae"] = {
            "total_params": self.vae.total_params,
            "encoder_params": sum(layer.params_count for layer in self.vae.layers if "encoder" in layer.name),
            "decoder_params": sum(layer.params_count for layer in self.vae.layers if "decoder" in layer.name),
            "encoder_flops": encoder_flops,
            "decoder_flops": decoder_flops,
            "encoder_time_sec": encoder_time,
            "decoder_time_sec": decoder_time,
            "estimated_memory_gb": self._estimate_vae_memory(num_frames, height, width) / 1024**3,
        }
        
        return results


def create_wan_analyzer(
    device: Device,
    cluster: Cluster,
    strategy: StrategyConfig,
    num_frames: int = 81,
    height: int = 720,
    width: int = 1280,
    dtype: str = "bf16",
) -> DiffusionVideoAnalyzer:
    """Convenience function to create a Wan2.1 analyzer."""
    from llm_perf.legacy.models.wan_video import (
        create_wan_t2v_14b_text_encoder,
        create_wan_t2v_14b_dit,
        create_wan_t2v_vae,
    )
    
    text_encoder = create_wan_t2v_14b_text_encoder(dtype=dtype)
    dit = create_wan_t2v_14b_dit(num_frames=num_frames, height=height, width=width, dtype=dtype)
    vae = create_wan_t2v_vae(num_frames=num_frames, height=height, width=width, dtype=dtype)
    
    return DiffusionVideoAnalyzer(
        text_encoder=text_encoder,
        dit=dit,
        vae=vae,
        device=device,
        cluster=cluster,
        strategy=strategy,
    )

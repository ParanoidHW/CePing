"""Diffusion Video Generation Pipeline.

Implements the complete text-to-video generation pipeline including:
1. Text Encoder (once)
2. DiT Denoising (iterative, e.g., 50 steps)
3. VAE Decoder (once)

Based on Wan2.1 architecture with Flow Matching framework.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..analyzer.diffusion_video import DiffusionVideoAnalyzer
from ..core.pipeline import (
    IterationConfig,
    Pipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStep,
)
from ..core.registry import ModelRegistry
from ..hardware.cluster import Cluster
from ..hardware.device import Device
from llm_perf.legacy.models.base import BaseModel
from ..strategy.base import StrategyConfig


class DiffusionVideoPipeline(Pipeline):
    """Pipeline for diffusion-based video generation.

    Chains together Text Encoder -> DiT (iterative) -> VAE Decoder
    for end-to-end text-to-video generation performance analysis.

    Example:
        >>> pipeline = DiffusionVideoPipeline(
        ...     text_encoder=text_encoder,
        ...     dit=dit_model,
        ...     vae=vae_model,
        ...     device=device,
        ...     cluster=cluster,
        ...     strategy=strategy,
        ...     num_inference_steps=50,
        ... )
        >>> result = pipeline.run({
        ...     "num_frames": 81,
        ...     "height": 720,
        ...     "width": 1280,
        ...     "use_cfg": True,
        ... })
    """

    def __init__(
        self,
        text_encoder: BaseModel,
        dit: BaseModel,
        vae: BaseModel,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
        num_inference_steps: int = 50,
        use_cfg: bool = True,
        batch_size: int = 1,
    ):
        """Initialize diffusion video pipeline.

        Args:
            text_encoder: Text encoder model (e.g., umT5-XXL)
            dit: Diffusion Transformer model
            vae: VAE model for latent encoding/decoding
            device: Device configuration
            cluster: Cluster configuration
            strategy: Parallelism strategy
            num_inference_steps: Number of denoising steps
            use_cfg: Whether to use Classifier-Free Guidance
            batch_size: Batch size (usually 1 for video generation)
        """
        config = PipelineConfig(
            name="diffusion_video",
            device=device,
            cluster=cluster,
            strategy=strategy,
            batch_size=batch_size,
        )
        super().__init__(config)

        self.text_encoder = text_encoder
        self.dit = dit
        self.vae = vae
        self.device = device
        self.cluster = cluster
        self.strategy = strategy
        self.num_inference_steps = num_inference_steps
        self.use_cfg = use_cfg
        self.batch_size = batch_size

        # Create analyzer for performance estimation
        self.analyzer = DiffusionVideoAnalyzer(
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

    def build_steps(self) -> List[PipelineStep]:
        """Build pipeline steps.

        Returns:
            List of pipeline steps in execution order:
            1. text_encoder (single)
            2. dit_denoising (iterative)
            3. vae_decoder (single)
        """
        return [
            PipelineStep(
                name="text_encoder",
                model=self.text_encoder,
                is_iterative=False,
            ),
            PipelineStep(
                name="dit_denoising",
                model=self.dit,
                is_iterative=True,
                iteration_config=IterationConfig(
                    num_iterations=self.num_inference_steps,
                    early_stopping=False,
                ),
                depends_on=["text_encoder"],
            ),
            PipelineStep(
                name="vae_decoder",
                model=self.vae,
                is_iterative=False,
                depends_on=["dit_denoising"],
            ),
        ]

    def execute_step(self, step: PipelineStep, inputs: Dict[str, Any]) -> Tuple[Any, float]:
        """Execute a single pipeline step.

        Args:
            step: Pipeline step to execute
            inputs: Input data containing generation parameters

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        # Extract generation parameters
        num_frames = inputs.get("num_frames", 81)
        height = inputs.get("height", 720)
        width = inputs.get("width", 1280)
        use_cfg = inputs.get("use_cfg", self.use_cfg)

        # Get component times from analyzer based on step name
        if step.name == "text_encoder":
            time_sec = self.analyzer._estimate_text_encoder_time()
            result = {
                "time_sec": time_sec,
                "memory_bytes": self.analyzer._estimate_text_encoder_memory(),
            }
        elif step.name == "dit_denoising":
            time_sec = self.analyzer._estimate_dit_single_step(num_frames, height, width, use_cfg)
            result = {
                "time_sec": time_sec,
                "memory_bytes": self.analyzer._estimate_dit_memory(num_frames, height, width, use_cfg),
            }
        elif step.name == "vae_decoder":
            _, time_sec = self.analyzer._estimate_vae_time(num_frames, height, width)
            result = {
                "time_sec": time_sec,
                "memory_bytes": self.analyzer._estimate_vae_memory(num_frames, height, width),
            }
        else:
            raise ValueError(f"Unknown step: {step.name}")

        return result, time_sec

    def execute_iterative_step(
        self,
        step: PipelineStep,
        inputs: Dict[str, Any],
        iteration_config: IterationConfig,
    ) -> Tuple[Any, float, int]:
        """Execute an iterative pipeline step.

        For DiT denoising, we estimate the time for all iterations
        rather than simulating each step individually.

        Args:
            step: Pipeline step to execute
            inputs: Initial input data
            iteration_config: Iteration configuration

        Returns:
            Tuple of (final_result, total_time_seconds, num_iterations_executed)
        """
        if step.name == "dit_denoising":
            # Get single step time
            num_frames = inputs.get("num_frames", 81)
            height = inputs.get("height", 720)
            width = inputs.get("width", 1280)
            use_cfg = inputs.get("use_cfg", self.use_cfg)

            single_step_time = self.analyzer._estimate_dit_single_step(num_frames, height, width, use_cfg)

            # Total time is single step * num iterations
            num_iters = iteration_config.num_iterations
            total_time = single_step_time * num_iters

            result = {
                "time_sec": total_time,
                "single_step_time_sec": single_step_time,
                "num_iterations": num_iters,
                "memory_bytes": self.analyzer._estimate_dit_memory(num_frames, height, width, use_cfg),
            }

            return result, total_time, num_iters
        else:
            # Fall back to base implementation for other steps
            return super().execute_iterative_step(step, inputs, iteration_config)

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Run the complete video generation pipeline.

        Args:
            inputs: Generation parameters including:
                - num_frames: Number of frames (default: 81)
                - height: Video height (default: 720)
                - width: Video width (default: 1280)
                - use_cfg: Use classifier-free guidance (default: True)

        Returns:
            PipelineResult with execution metrics
        """
        inputs = inputs or {}

        num_frames = inputs.get("num_frames", 81)
        height = inputs.get("height", 720)
        width = inputs.get("width", 1280)
        use_cfg = inputs.get("use_cfg", self.use_cfg)

        # Use analyzer for complete analysis
        analysis_result = self.analyzer.analyze(
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=self.num_inference_steps,
            use_cfg=use_cfg,
        )

        # Build step times from analysis result
        step_times = {
            "text_encoder": analysis_result.text_encoder_time_sec,
            "dit_denoising": analysis_result.dit_single_step_time_sec * self.num_inference_steps,
            "vae_decoder": analysis_result.vae_decoder_time_sec,
        }

        step_results = {
            "text_encoder": {
                "time_sec": analysis_result.text_encoder_time_sec,
                "memory_gb": analysis_result.peak_memory_text_encoder_gb,
            },
            "dit_denoising": {
                "single_step_time_sec": analysis_result.dit_single_step_time_sec,
                "total_time_sec": analysis_result.dit_single_step_time_sec * self.num_inference_steps,
                "memory_gb": analysis_result.peak_memory_dit_gb,
            },
            "vae_decoder": {
                "time_sec": analysis_result.vae_decoder_time_sec,
                "memory_gb": analysis_result.peak_memory_vae_gb,
            },
        }

        # Calculate throughput
        total_pixels = num_frames * height * width
        pixels_per_sec = (
            total_pixels / analysis_result.total_generation_time_sec
            if analysis_result.total_generation_time_sec > 0
            else 0
        )

        return PipelineResult(
            total_time_sec=analysis_result.total_generation_time_sec,
            step_times=step_times,
            step_results=step_results,
            memory_peak_gb=analysis_result.peak_memory_total_gb,
            throughput=pixels_per_sec,
            metadata={
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": self.num_inference_steps,
                "use_cfg": use_cfg,
                "component_breakdown": analysis_result.component_breakdown,
            },
            detailed_breakdown=(
                analysis_result.detailed_breakdown.to_dict() if analysis_result.detailed_breakdown else None
            ),
        )

    def analyze_components(self) -> Dict[str, Any]:
        """Analyze each pipeline component separately.

        Returns:
            Dictionary with detailed analysis of each component
        """
        return self.analyzer.analyze_components_separately()


def create_wan_t2v_pipeline(
    device: Device,
    cluster: Cluster,
    strategy: StrategyConfig,
    num_frames: int = 81,
    height: int = 720,
    width: int = 1280,
    num_inference_steps: int = 50,
    dtype: str = "bf16",
) -> DiffusionVideoPipeline:
    """Create a Wan2.1 T2V-14B pipeline.

    Convenience function that creates all required models and the pipeline.

    Args:
        device: Device configuration
        cluster: Cluster configuration
        strategy: Parallelism strategy
        num_frames: Number of frames to generate
        height: Video height in pixels
        width: Video width in pixels
        num_inference_steps: Number of denoising steps
        dtype: Data type for models

    Returns:
        Configured DiffusionVideoPipeline
    """
    registry = ModelRegistry()

    # Create models through registry
    text_encoder = registry.create(
        "wan-text-encoder",
        name="WanTextEncoder",
        vocab_size=256384,
        hidden_size=4096,
        num_layers=24,
        num_attention_heads=64,
        intermediate_size=10240,
        max_seq_len=512,
        dtype=dtype,
    )

    dit = registry.create(
        "wan-dit",
        name="WanDiT",
        vocab_size=1,
        hidden_size=5120,
        num_layers=40,
        num_attention_heads=40,
        intermediate_size=20480,
        max_seq_len=1,
        dtype=dtype,
        latent_num_frames=(num_frames - 1) // 4 + 1,
        latent_height=height // 8,
        latent_width=width // 8,
        patch_size=[1, 2, 2],
        in_channels=16,
    )

    vae = registry.create(
        "wan-vae",
        name="WanVAE",
        vocab_size=1,
        hidden_size=256,
        num_layers=8,
        num_attention_heads=0,
        intermediate_size=1024,
        max_seq_len=1,
        dtype="fp32",  # VAE typically uses fp32
        num_frames=num_frames,
        height=height,
        width=width,
        in_channels=3,
        latent_channels=16,
        temporal_compression=4,
        spatial_compression=8,
        block_out_channels=[128, 256, 512, 512],
    )

    return DiffusionVideoPipeline(
        text_encoder=text_encoder,
        dit=dit,
        vae=vae,
        device=device,
        cluster=cluster,
        strategy=strategy,
        num_inference_steps=num_inference_steps,
    )

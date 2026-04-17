"""Diffusion Generation Scenario implementation.

Diffusion-based generation scenario for text-to-video/image workflows:
- Text Encoder: Process text prompts (single pass)
- DiT/Diffusion Model: Iterative denoising (multiple passes)
- VAE Decoder: Decode latent to output (single pass)

Uses DiffusionVideoPipeline and DiffusionVideoAnalyzer for estimation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Scenario, ScenarioConfig, ScenarioResult, ScenarioType, ParallelismType
from llm_perf.legacy.models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig
from ..pipelines.diffusion_video import DiffusionVideoPipeline
from ..analyzer.diffusion_video import DiffusionVideoAnalyzer


@dataclass
class DiffusionComponentConfig:
    """Configuration for a diffusion pipeline component."""

    role: str
    tp_degree: int = 1
    sp_degree: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "tp_degree": self.tp_degree,
            "sp_degree": self.sp_degree,
        }


@dataclass
class DiffusionConfig(ScenarioConfig):
    """Configuration for diffusion generation scenario.

    Attributes:
        text_encoder_config: Text encoder configuration
        dit_config: DiT model configuration
        vae_config: VAE model configuration
        num_inference_steps: Number of denoising steps
        use_cfg: Use classifier-free guidance
        num_frames: Number of frames (for video)
        height: Output height
        width: Output width
        batch_size: Batch size
    """

    text_encoder_config: DiffusionComponentConfig = None
    dit_config: DiffusionComponentConfig = None
    vae_config: DiffusionComponentConfig = None
    num_inference_steps: int = 50
    use_cfg: bool = True
    num_frames: int = 81
    height: int = 720
    width: int = 1280
    batch_size: int = 1

    def __post_init__(self):
        """Set scenario type and required models."""
        self.scenario_type = ScenarioType.DIFFUSION
        if not self.required_models:
            self.required_models = ["text_encoder", "dit", "vae"]
        if not self.supported_parallelisms:
            self.supported_parallelisms = [
                ParallelismType.TP,
                ParallelismType.SP,
            ]
        if self.text_encoder_config is None:
            self.text_encoder_config = DiffusionComponentConfig(role="text_encoder", tp_degree=1)
        if self.dit_config is None:
            self.dit_config = DiffusionComponentConfig(role="dit", tp_degree=8)
        if self.vae_config is None:
            self.vae_config = DiffusionComponentConfig(role="vae", tp_degree=1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "text_encoder_config": self.text_encoder_config.to_dict(),
                "dit_config": self.dit_config.to_dict(),
                "vae_config": self.vae_config.to_dict(),
                "num_inference_steps": self.num_inference_steps,
                "use_cfg": self.use_cfg,
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
                "batch_size": self.batch_size,
            }
        )
        return base


@dataclass
class ComponentResult:
    """Result for a single diffusion component."""

    role: str
    time_sec: float = 0.0
    memory_gb: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "time_sec": self.time_sec,
            "time_ms": self.time_sec * 1000,
            "memory_gb": self.memory_gb,
            "throughput": self.throughput,
        }


@dataclass
class DiffusionResult(ScenarioResult):
    """Result of diffusion generation scenario evaluation.

    Attributes:
        text_encoder_result: Text encoder performance
        dit_result: DiT denoising performance
        vae_result: VAE decoder performance
        total_generation_time_sec: Total generation time
        dit_single_step_time_sec: Time for single denoising step
        pixels_per_sec: Throughput in pixels/second
        frames_per_sec: Throughput in frames/second
    """

    text_encoder_result: ComponentResult = None
    dit_result: ComponentResult = None
    vae_result: ComponentResult = None
    total_generation_time_sec: float = 0.0
    dit_single_step_time_sec: float = 0.0
    pixels_per_sec: float = 0.0
    frames_per_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "text_encoder": self.text_encoder_result.to_dict() if self.text_encoder_result else {},
                "dit": self.dit_result.to_dict() if self.dit_result else {},
                "vae": self.vae_result.to_dict() if self.vae_result else {},
                "total_generation_time_sec": self.total_generation_time_sec,
                "dit_single_step_time_sec": self.dit_single_step_time_sec,
                "pixels_per_sec": self.pixels_per_sec,
                "frames_per_sec": self.frames_per_sec,
            }
        )
        return base


class DiffusionScenario(Scenario):
    """Diffusion generation performance scenario.

    Evaluates performance of diffusion-based generation pipelines:
    - Text-to-video generation
    - Text-to-image generation

    Uses DiffusionVideoPipeline for end-to-end estimation.

    Example:
        >>> from llm_perf.scenarios.registry import ScenarioRegistry
        >>> from llm_perf.modeling import WanTextEncoder, WanDiTModel, WanVAEModel
        >>> from llm_perf.hardware.device import Device
        >>> from llm_perf.hardware.cluster import Cluster
        >>> from llm_perf.strategy.base import StrategyConfig
        >>>
        >>> registry = ScenarioRegistry()
        >>> text_encoder = WanTextEncoder(config)
        >>> dit = WanDiTModel(config)
        >>> vae = WanVAEModel(config)
        >>>
        >>> scenario = registry.create_scenario(
        ...     "diffusion",
        ...     models={"text_encoder": text_encoder, "dit": dit, "vae": vae},
        ...     device=device,
        ...     cluster=cluster,
        ...     strategy=strategy,
        ...     num_inference_steps=50,
        ... )
        >>> result = scenario.analyze(num_frames=81, height=720, width=1280)
    """

    def __init__(
        self,
        config: DiffusionConfig,
        models: Dict[str, BaseModel],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize diffusion scenario.

        Args:
            config: Diffusion configuration
            models: Dictionary with "text_encoder", "dit", "vae" models
            device: Device configuration
            cluster: Cluster configuration
            strategy: Base strategy (used as template)
        """
        super().__init__(config, models, device, cluster, strategy)
        self._pipeline: DiffusionVideoPipeline = None
        self._analyzer: DiffusionVideoAnalyzer = None

    def _create_pipeline(self) -> DiffusionVideoPipeline:
        """Create the diffusion pipeline.

        Returns:
            DiffusionVideoPipeline instance
        """
        text_encoder = self.models.get("text_encoder")
        dit = self.models.get("dit")
        vae = self.models.get("vae")

        return DiffusionVideoPipeline(
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
            device=self.device,
            cluster=self.cluster,
            strategy=self.strategy,
            num_inference_steps=self.config.num_inference_steps,
            use_cfg=self.config.use_cfg,
            batch_size=self.config.batch_size,
        )

    def _create_analyzer(self) -> DiffusionVideoAnalyzer:
        """Create the diffusion analyzer.

        Returns:
            DiffusionVideoAnalyzer instance
        """
        text_encoder = self.models.get("text_encoder")
        dit = self.models.get("dit")
        vae = self.models.get("vae")

        return DiffusionVideoAnalyzer(
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
            device=self.device,
            cluster=self.cluster,
            strategy=self.strategy,
        )

    def get_analyzer(self) -> DiffusionVideoAnalyzer:
        """Get or create the DiffusionVideoAnalyzer instance.

        Returns:
            DiffusionVideoAnalyzer for this scenario
        """
        if self._analyzer is None:
            self._analyzer = self._create_analyzer()
        return self._analyzer

    def get_pipeline(self) -> DiffusionVideoPipeline:
        """Get or create the DiffusionVideoPipeline instance.

        Returns:
            DiffusionVideoPipeline for this scenario
        """
        if self._pipeline is None:
            self._pipeline = self._create_pipeline()
        return self._pipeline

    def analyze(
        self,
        num_frames: int = None,
        height: int = None,
        width: int = None,
        num_inference_steps: int = None,
        use_cfg: bool = None,
        **kwargs,
    ) -> DiffusionResult:
        """Run diffusion generation analysis.

        Args:
            num_frames: Number of frames (for video)
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            use_cfg: Use classifier-free guidance
            **kwargs: Additional parameters

        Returns:
            DiffusionResult with performance metrics
        """
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        use_cfg = use_cfg if use_cfg is not None else self.config.use_cfg

        pipeline = self.get_pipeline()

        pipeline_result = pipeline.run(
            {
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "use_cfg": use_cfg,
            }
        )

        analyzer = self.get_analyzer()
        analysis_result = analyzer.analyze(
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            use_cfg=use_cfg,
        )

        text_encoder_result = ComponentResult(
            role="text_encoder",
            time_sec=analysis_result.text_encoder_time_sec,
            memory_gb=analysis_result.peak_memory_text_encoder_gb,
            throughput=1.0 / analysis_result.text_encoder_time_sec if analysis_result.text_encoder_time_sec > 0 else 0,
        )

        dit_total_time = analysis_result.dit_single_step_time_sec * num_inference_steps
        dit_result = ComponentResult(
            role="dit",
            time_sec=dit_total_time,
            memory_gb=analysis_result.peak_memory_dit_gb,
            throughput=num_inference_steps / dit_total_time if dit_total_time > 0 else 0,
        )

        vae_result = ComponentResult(
            role="vae",
            time_sec=analysis_result.vae_decoder_time_sec,
            memory_gb=analysis_result.peak_memory_vae_gb,
            throughput=1.0 / analysis_result.vae_decoder_time_sec if analysis_result.vae_decoder_time_sec > 0 else 0,
        )

        total_pixels = num_frames * height * width
        pixels_per_sec = (
            total_pixels / analysis_result.total_generation_time_sec
            if analysis_result.total_generation_time_sec > 0
            else 0
        )
        frames_per_sec = (
            num_frames / analysis_result.total_generation_time_sec
            if analysis_result.total_generation_time_sec > 0
            else 0
        )

        return DiffusionResult(
            scenario_name=self.config.name,
            total_time_sec=analysis_result.total_generation_time_sec,
            throughput=pixels_per_sec,
            memory_peak_gb=analysis_result.peak_memory_total_gb,
            text_encoder_result=text_encoder_result,
            dit_result=dit_result,
            vae_result=vae_result,
            total_generation_time_sec=analysis_result.total_generation_time_sec,
            dit_single_step_time_sec=analysis_result.dit_single_step_time_sec,
            pixels_per_sec=pixels_per_sec,
            frames_per_sec=frames_per_sec,
            breakdown={
                "text_encoder_time_sec": analysis_result.text_encoder_time_sec,
                "dit_time_sec": dit_total_time,
                "vae_time_sec": analysis_result.vae_decoder_time_sec,
                "text_encoder_percent": analysis_result.text_encoder_time_sec
                / analysis_result.total_generation_time_sec
                * 100
                if analysis_result.total_generation_time_sec > 0
                else 0,
                "dit_percent": dit_total_time / analysis_result.total_generation_time_sec * 100
                if analysis_result.total_generation_time_sec > 0
                else 0,
                "vae_percent": analysis_result.vae_decoder_time_sec / analysis_result.total_generation_time_sec * 100
                if analysis_result.total_generation_time_sec > 0
                else 0,
                "component_breakdown": analysis_result.component_breakdown,
            },
            metadata={
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "use_cfg": use_cfg,
                "total_pixels": total_pixels,
                "strategy": self.strategy.to_dict(),
            },
        )

    def estimate_single_step_time(
        self,
        num_frames: int = None,
        height: int = None,
        width: int = None,
        use_cfg: bool = None,
    ) -> float:
        """Estimate time for a single denoising step.

        Args:
            num_frames: Number of frames
            height: Output height
            width: Output width
            use_cfg: Use classifier-free guidance

        Returns:
            Single step time in seconds
        """
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width
        use_cfg = use_cfg if use_cfg is not None else self.config.use_cfg

        analyzer = self.get_analyzer()
        return analyzer._estimate_dit_single_step(num_frames, height, width, use_cfg)

    def estimate_memory(
        self,
        num_frames: int = None,
        height: int = None,
        width: int = None,
        use_cfg: bool = None,
    ) -> Dict[str, float]:
        """Estimate memory requirements.

        Args:
            num_frames: Number of frames
            height: Output height
            width: Output width
            use_cfg: Use classifier-free guidance

        Returns:
            Dictionary with memory estimates
        """
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width
        use_cfg = use_cfg if use_cfg is not None else self.config.use_cfg

        analyzer = self.get_analyzer()

        return {
            "text_encoder_gb": analyzer._estimate_text_encoder_memory() / 1024**3,
            "dit_gb": analyzer._estimate_dit_memory(num_frames, height, width, use_cfg) / 1024**3,
            "vae_gb": analyzer._estimate_vae_memory(num_frames, height, width) / 1024**3,
            "total_gb": analyzer._estimate_total_memory(num_frames, height, width, use_cfg) / 1024**3,
        }

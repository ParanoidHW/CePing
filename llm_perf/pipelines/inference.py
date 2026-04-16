"""Base pipeline implementations."""

from typing import Any, Dict, List, Tuple

from ..analyzer.inference import InferenceAnalyzer
from ..core.pipeline import Pipeline, PipelineConfig, PipelineResult, PipelineStep
from ..hardware.cluster import Cluster
from ..hardware.device import Device
from ..models.base import BaseModel
from ..strategy.base import StrategyConfig


class InferencePipeline(Pipeline):
    """Standard inference pipeline for single models.

    Simple wrapper that uses InferenceAnalyzer for performance estimation.

    Example:
        >>> pipeline = InferencePipeline(model, device, cluster, strategy)
        >>> result = pipeline.run({"batch_size": 1, "prompt_len": 512})
    """

    def __init__(
        self,
        model: BaseModel,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
        batch_size: int = 1,
        prompt_len: int = 512,
        generation_len: int = 128,
    ):
        """Initialize inference pipeline.

        Args:
            model: Model to evaluate
            device: Device configuration
            cluster: Cluster configuration
            strategy: Parallelism strategy
            batch_size: Batch size for inference
            prompt_len: Prompt sequence length
            generation_len: Generation sequence length
        """
        config = PipelineConfig(
            name=f"inference_{model.config.name}",
            device=device,
            cluster=cluster,
            strategy=strategy,
            batch_size=batch_size,
        )
        super().__init__(config)
        self.model = model
        self.device = device
        self.cluster = cluster
        self.strategy = strategy
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.generation_len = generation_len
        self.analyzer = InferenceAnalyzer(model, device, cluster, strategy)

    def build_steps(self) -> List[PipelineStep]:
        """Build single inference step.

        Returns:
            List with single pipeline step
        """
        return [
            PipelineStep(
                name="inference",
                model=self.model,
            )
        ]

    def execute_step(
        self, step: PipelineStep, inputs: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Execute inference using analyzer.

        Args:
            step: Pipeline step (unused, we use analyzer directly)
            inputs: Input parameters

        Returns:
            Tuple of (analyzer result, estimated time)
        """
        # Get parameters from inputs or use defaults
        batch_size = inputs.get("batch_size", self.batch_size)
        prompt_len = inputs.get("prompt_len", self.prompt_len)
        generation_len = inputs.get("generation_len", self.generation_len)

        # Run analyzer
        result = self.analyzer.analyze(
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=generation_len,
        )

        return result, result.total_time_sec

    def run(self, inputs: Dict[str, Any] | None = None) -> PipelineResult:
        """Run inference pipeline.

        Args:
            inputs: Optional input parameters

        Returns:
            PipelineResult with execution metrics
        """
        inputs = inputs or {}

        batch_size = inputs.get("batch_size", self.batch_size)
        prompt_len = inputs.get("prompt_len", self.prompt_len)
        generation_len = inputs.get("generation_len", self.generation_len)

        # Run analyzer
        result = self.analyzer.analyze(
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=generation_len,
        )

        return PipelineResult(
            total_time_sec=result.total_time_sec,
            step_times={
                "prefill": result.prefill_time_sec,
                "decode": result.decode_time_per_step_sec * generation_len,
            },
            step_results={
                "prefill_tokens_per_sec": result.prefill_tokens_per_sec,
                "decode_tokens_per_sec": result.decode_tokens_per_sec,
                "memory_per_gpu_gb": result.memory_per_gpu_gb,
            },
            memory_peak_gb=result.memory_per_gpu_gb,
            throughput=result.decode_tokens_per_sec,
        )

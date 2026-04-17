"""LLM Training Scenario implementation.

Standard training scenario for single model with forward and backward passes.
Uses TrainingAnalyzer for performance estimation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from .base import Scenario, ScenarioConfig, ScenarioResult, ScenarioType, ParallelismType
from ..analyzer.training import TrainingAnalyzer, TrainingResult
from llm_perf.legacy.models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig


@dataclass
class LLMTrainingConfig(ScenarioConfig):
    """Configuration for LLM training scenario.

    Attributes:
        batch_size: Training batch size
        seq_len: Sequence length
        num_steps: Number of training steps
        gradient_accumulation_steps: Gradient accumulation steps
    """

    batch_size: int = 1
    seq_len: int = 2048
    num_steps: int = 1000
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        """Set scenario type and required models."""
        self.scenario_type = ScenarioType.LLM_TRAINING
        if not self.required_models:
            self.required_models = ["main"]
        if not self.supported_parallelisms:
            self.supported_parallelisms = [
                ParallelismType.TP,
                ParallelismType.PP,
                ParallelismType.DP,
                ParallelismType.EP,
                ParallelismType.SP,
                ParallelismType.CP,
            ]


@dataclass
class LLMTrainingResult(ScenarioResult):
    """Result of LLM training scenario evaluation.

    Attributes:
        samples_per_sec: Training throughput (samples/second)
        tokens_per_sec: Training throughput (tokens/second)
        time_per_step_sec: Time per training step
        time_to_solution_sec: Time to complete all steps
        memory_per_gpu_gb: Memory usage per GPU
        training_result: Detailed TrainingResult from analyzer
    """

    samples_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    time_per_step_sec: float = 0.0
    time_to_solution_sec: float = 0.0
    memory_per_gpu_gb: float = 0.0
    training_result: TrainingResult = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "samples_per_sec": self.samples_per_sec,
                "tokens_per_sec": self.tokens_per_sec,
                "time_per_step_sec": self.time_per_step_sec,
                "time_to_solution_sec": self.time_to_solution_sec,
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
                "training_breakdown": self.training_result.to_dict() if self.training_result else None,
            }
        )
        return base


class LLMTrainingScenario(Scenario):
    """LLM training performance scenario.

    Evaluates training performance for a single LLM model.
    Uses TrainingAnalyzer for compute, communication, and memory estimation.

    Example:
        >>> from llm_perf.modeling import create_model_from_config
        >>> from llm_perf.hardware.device import Device
        >>> from llm_perf.hardware.cluster import Cluster
        >>> from llm_perf.strategy.base import StrategyConfig
        >>> from llm_perf.scenarios.registry import ScenarioRegistry
        >>>
        >>> model = create_model_from_config({"preset": "llama-7b"})
        >>> device = Device(name="H100", memory_gb=80)
        >>> cluster = Cluster(num_nodes=4, devices_per_node=8)
        >>> strategy = StrategyConfig(tp_degree=8, pp_degree=2, dp_degree=2)
        >>>
        >>> registry = ScenarioRegistry()
        >>> scenario = registry.create_scenario(
        ...     "llm-training",
        ...     models={"main": model},
        ...     device=device,
        ...     cluster=cluster,
        ...     strategy=strategy,
        ... )
        >>> result = scenario.analyze(batch_size=4, seq_len=2048)
    """

    def __init__(
        self,
        config: LLMTrainingConfig,
        models: Dict[str, BaseModel],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize LLM training scenario.

        Args:
            config: Training scenario configuration
            models: Dictionary with "main" model
            device: Device configuration
            cluster: Cluster configuration
            strategy: Parallelism strategy
        """
        super().__init__(config, models, device, cluster, strategy)
        self._analyzer: TrainingAnalyzer = None

    def _get_main_model(self) -> BaseModel:
        """Get the main training model."""
        return self.models.get("main")

    def get_analyzer(self) -> TrainingAnalyzer:
        """Get or create the TrainingAnalyzer instance.

        Returns:
            TrainingAnalyzer for this scenario
        """
        if self._analyzer is None:
            model = self._get_main_model()
            self._analyzer = TrainingAnalyzer(
                model=model,
                device=self.device,
                cluster=self.cluster,
                strategy=self.strategy,
            )
        return self._analyzer

    def analyze(
        self,
        batch_size: int = None,
        seq_len: int = None,
        num_steps: int = None,
        **kwargs,
    ) -> LLMTrainingResult:
        """Run training performance analysis.

        Args:
            batch_size: Training batch size (uses config default if not provided)
            seq_len: Sequence length (uses config default if not provided)
            num_steps: Number of training steps (uses config default if not provided)
            **kwargs: Additional parameters

        Returns:
            LLMTrainingResult with performance metrics
        """
        batch_size = batch_size or self.config.batch_size
        seq_len = seq_len or self.config.seq_len
        num_steps = num_steps or self.config.num_steps

        analyzer = self.get_analyzer()
        training_result = analyzer.analyze(
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
        )

        total_gpus = self.strategy.world_size

        return LLMTrainingResult(
            scenario_name=self.config.name,
            total_time_sec=training_result.time_per_step_sec * num_steps,
            throughput=training_result.tokens_per_sec,
            memory_peak_gb=training_result.memory_per_gpu_gb,
            samples_per_sec=training_result.samples_per_sec,
            tokens_per_sec=training_result.tokens_per_sec,
            time_per_step_sec=training_result.time_per_step_sec,
            time_to_solution_sec=training_result.time_to_solution_sec,
            memory_per_gpu_gb=training_result.memory_per_gpu_gb,
            training_result=training_result,
            breakdown=training_result.breakdown.to_dict() if training_result.breakdown else {},
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_steps": num_steps,
                "total_gpus": total_gpus,
                "strategy": self.strategy.to_dict(),
            },
        )

    def estimate_memory(self, batch_size: int = None, seq_len: int = None) -> float:
        """Estimate training memory requirements.

        Args:
            batch_size: Training batch size
            seq_len: Sequence length

        Returns:
            Memory estimate in GB
        """
        batch_size = batch_size or self.config.batch_size
        seq_len = seq_len or self.config.seq_len

        analyzer = self.get_analyzer()
        result = analyzer.analyze(batch_size=batch_size, seq_len=seq_len)
        return result.memory_per_gpu_gb

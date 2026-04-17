"""LLM Training Scenario implementation.

Standard training scenario for single model with forward and backward passes.
Uses TrainingAnalyzer for performance estimation.
"""

from dataclasses import dataclass
from typing import Any, Dict

from .base import Scenario, ScenarioConfig, ScenarioResult, ScenarioType, ParallelismType
from llm_perf.analyzer import TrainingAnalyzer, TrainingResult
from llm_perf.modeling import ShardedModule
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig


@dataclass
class LLMTrainingConfig(ScenarioConfig):
    """Configuration for LLM training scenario."""

    batch_size: int = 1
    seq_len: int = 2048

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
    """Result of LLM training scenario evaluation."""

    samples_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    time_per_step_sec: float = 0.0
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
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
                "training_breakdown": self.training_result.to_dict() if self.training_result else None,
            }
        )
        return base


class LLMTrainingScenario(Scenario):
    """LLM training performance scenario.

    Evaluates training performance for a single LLM model.
    Uses TrainingAnalyzer for compute, communication, and memory estimation.
    """

    def __init__(
        self,
        config: LLMTrainingConfig,
        models: Dict[str, ShardedModule],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize LLM training scenario."""
        super().__init__(config, models, device, cluster, strategy)
        self._analyzer: TrainingAnalyzer = None

    def _get_main_model(self) -> ShardedModule:
        """Get the main training model."""
        return self.models.get("main")

    def get_analyzer(self) -> TrainingAnalyzer:
        """Get or create the TrainingAnalyzer instance."""
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
        **kwargs,
    ) -> LLMTrainingResult:
        """Run training performance analysis."""
        batch_size = batch_size or self.config.batch_size
        seq_len = seq_len or self.config.seq_len

        analyzer = self.get_analyzer()
        training_result = analyzer.analyze(
            batch_size=batch_size,
            seq_len=seq_len,
        )

        total_gpus = self.strategy.world_size

        return LLMTrainingResult(
            scenario_name=self.config.name,
            total_time_sec=training_result.time_per_step_sec,
            throughput=training_result.tokens_per_sec,
            memory_peak_gb=training_result.memory_per_gpu_gb,
            samples_per_sec=training_result.samples_per_sec,
            tokens_per_sec=training_result.tokens_per_sec,
            time_per_step_sec=training_result.time_per_step_sec,
            memory_per_gpu_gb=training_result.memory_per_gpu_gb,
            training_result=training_result,
            breakdown=training_result.breakdown.to_dict() if training_result.breakdown else {},
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "total_gpus": total_gpus,
                "strategy": self.strategy.to_dict(),
            },
        )

    def estimate_memory(self, batch_size: int = None, seq_len: int = None) -> float:
        """Estimate training memory requirements."""
        batch_size = batch_size or self.config.batch_size
        seq_len = seq_len or self.config.seq_len

        analyzer = self.get_analyzer()
        result = analyzer.analyze(batch_size=batch_size, seq_len=seq_len)
        return result.memory_per_gpu_gb

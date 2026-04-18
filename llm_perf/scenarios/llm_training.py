"""LLM Training Scenario implementation.

Standard training scenario for single model with forward and backward passes.
Uses UnifiedAnalyzer for performance estimation.
"""

from dataclasses import dataclass
from typing import Any, Dict

from .base import Scenario, ScenarioConfig, ScenarioResult, ScenarioType, ParallelismType
from llm_perf.analyzer import UnifiedAnalyzer, UnifiedResult
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
    unified_result: UnifiedResult = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "samples_per_sec": self.samples_per_sec,
                "tokens_per_sec": self.tokens_per_sec,
                "time_per_step_sec": self.time_per_step_sec,
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
                "unified_result": self.unified_result.to_dict() if self.unified_result else None,
            }
        )
        return base


class LLMTrainingScenario(Scenario):
    """LLM training performance scenario.

    Evaluates training performance for a single LLM model.
    Uses UnifiedAnalyzer with training workload.
    """

    def __init__(
        self,
        config: LLMTrainingConfig,
        models: Dict[str, ShardedModule],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        super().__init__(config, models, device, cluster, strategy)
        self._analyzer: UnifiedAnalyzer = None

    def _get_main_model(self) -> ShardedModule:
        return self.models.get("main")

    def get_analyzer(self) -> UnifiedAnalyzer:
        if self._analyzer is None:
            model = self._get_main_model()
            self._analyzer = UnifiedAnalyzer(model, self.device, self.cluster, self.strategy)
        return self._analyzer

    def analyze(
        self,
        batch_size: int = None,
        seq_len: int = None,
        **kwargs,
    ) -> LLMTrainingResult:
        batch_size = batch_size or self.config.batch_size
        seq_len = seq_len or self.config.seq_len

        analyzer = self.get_analyzer()
        unified_result = analyzer.analyze(
            "training",
            batch_size=batch_size,
            seq_len=seq_len,
        )

        throughput = unified_result.throughput

        return LLMTrainingResult(
            scenario_name=self.config.name,
            total_time_sec=unified_result.total_time_sec,
            throughput=throughput.get("tokens_per_sec", throughput.get("samples_per_sec", 0)),
            memory_peak_gb=unified_result.peak_memory_gb,
            samples_per_sec=throughput.get("samples_per_sec", 0),
            tokens_per_sec=throughput.get("tokens_per_sec", 0),
            time_per_step_sec=unified_result.total_time_sec,
            memory_per_gpu_gb=unified_result.peak_memory_gb,
            unified_result=unified_result,
            breakdown=unified_result.to_dict(),
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "strategy": self.strategy.to_dict(),
            },
        )

    def estimate_memory(self, batch_size: int = None, seq_len: int = None) -> float:
        batch_size = batch_size or self.config.batch_size
        seq_len = seq_len or self.config.seq_len

        analyzer = self.get_analyzer()
        result = analyzer.analyze("training", batch_size=batch_size, seq_len=seq_len)
        return result.peak_memory_gb

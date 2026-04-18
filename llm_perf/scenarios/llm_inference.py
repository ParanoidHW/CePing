"""LLM Inference Scenario implementation.

Standard inference scenario for single model with prefill and decode phases.
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
class LLMInferenceConfig(ScenarioConfig):
    """Configuration for LLM inference scenario."""

    batch_size: int = 1
    prompt_len: int = 512
    generation_len: int = 128

    def __post_init__(self):
        self.scenario_type = ScenarioType.LLM_INFERENCE
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
class LLMInferenceResult(ScenarioResult):
    """Result of LLM inference scenario evaluation."""

    prefill_time_sec: float = 0.0
    decode_time_per_step_sec: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    memory_per_gpu_gb: float = 0.0
    unified_result: UnifiedResult = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "prefill": {
                    "time_sec": self.prefill_time_sec,
                    "time_ms": self.prefill_time_sec * 1000,
                    "tokens_per_sec": self.prefill_tokens_per_sec,
                },
                "decode": {
                    "time_per_step_sec": self.decode_time_per_step_sec,
                    "time_per_step_ms": self.decode_time_per_step_sec * 1000,
                    "tokens_per_sec": self.decode_tokens_per_sec,
                },
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
                "unified_result": self.unified_result.to_dict() if self.unified_result else None,
            }
        )
        return base


class LLMInferenceScenario(Scenario):
    """LLM inference performance scenario.

    Evaluates inference performance for a single LLM model.
    Uses UnifiedAnalyzer with llm-inference workload.
    """

    def __init__(
        self,
        config: LLMInferenceConfig,
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
        prompt_len: int = None,
        generation_len: int = None,
        **kwargs,
    ) -> LLMInferenceResult:
        batch_size = batch_size or self.config.batch_size
        prompt_len = prompt_len or self.config.prompt_len
        generation_len = generation_len or self.config.generation_len

        analyzer = self.get_analyzer()
        unified_result = analyzer.analyze(
            "llm-inference",
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=generation_len,
        )

        prefill_phase = unified_result.get_phase("prefill")
        decode_phase = unified_result.get_phase("decode")

        prefill_time = prefill_phase.total_time_sec if prefill_phase else 0
        decode_time_per_step = decode_phase.single_time_sec if decode_phase else 0

        throughput = unified_result.throughput
        total_time = prefill_time + decode_time_per_step * generation_len

        prefill_tps = (batch_size * prompt_len) / prefill_time if prefill_time > 0 else 0
        decode_tps = batch_size / decode_time_per_step if decode_time_per_step > 0 else 0

        return LLMInferenceResult(
            scenario_name=self.config.name,
            total_time_sec=total_time,
            throughput=decode_tps,
            memory_peak_gb=unified_result.peak_memory_gb,
            prefill_time_sec=prefill_time,
            decode_time_per_step_sec=decode_time_per_step,
            prefill_tokens_per_sec=prefill_tps,
            decode_tokens_per_sec=decode_tps,
            memory_per_gpu_gb=unified_result.peak_memory_gb,
            unified_result=unified_result,
            breakdown=unified_result.to_dict(),
            metadata={
                "batch_size": batch_size,
                "prompt_len": prompt_len,
                "generation_len": generation_len,
                "strategy": self.strategy.to_dict(),
            },
        )

    def estimate_memory(
        self,
        batch_size: int = None,
        max_seq_len: int = None,
    ) -> Dict[str, float]:
        batch_size = batch_size or self.config.batch_size
        max_seq_len = max_seq_len or (self.config.prompt_len + self.config.generation_len)

        analyzer = self.get_analyzer()
        result = analyzer.analyze(
            "llm-inference",
            batch_size=batch_size,
            prompt_len=max_seq_len // 2,
            generation_len=max_seq_len // 2,
        )

        return {
            "total_memory_gb": result.peak_memory_gb,
        }

    def estimate_latency(
        self,
        batch_size: int = None,
        prompt_len: int = None,
        generation_len: int = None,
    ) -> Dict[str, float]:
        batch_size = batch_size or self.config.batch_size
        prompt_len = prompt_len or self.config.prompt_len
        generation_len = generation_len or self.config.generation_len

        result = self.analyze(batch_size, prompt_len, generation_len)

        total_time = result.prefill_time_sec + result.decode_time_per_step_sec * generation_len

        return {
            "ttft_sec": result.prefill_time_sec,
            "ttft_ms": result.prefill_time_sec * 1000,
            "tpot_sec": result.decode_time_per_step_sec,
            "tpot_ms": result.decode_time_per_step_sec * 1000,
            "total_time_sec": total_time,
        }

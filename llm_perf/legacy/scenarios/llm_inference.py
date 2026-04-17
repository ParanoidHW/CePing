"""LLM Inference Scenario implementation.

Standard inference scenario for single model with prefill and decode phases.
Uses InferenceAnalyzer for performance estimation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from .base import Scenario, ScenarioConfig, ScenarioResult, ScenarioType, ParallelismType
from ..analyzer.inference import InferenceAnalyzer, InferenceResult
from .models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig


@dataclass
class LLMInferenceConfig(ScenarioConfig):
    """Configuration for LLM inference scenario.

    Attributes:
        batch_size: Inference batch size
        prompt_len: Prompt sequence length
        generation_len: Generation sequence length
    """

    batch_size: int = 1
    prompt_len: int = 512
    generation_len: int = 128

    def __post_init__(self):
        """Set scenario type and required models."""
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
    """Result of LLM inference scenario evaluation.

    Attributes:
        prefill_time_sec: Prefill phase time
        decode_time_per_step_sec: Decode time per token
        prefill_tokens_per_sec: Prefill throughput
        decode_tokens_per_sec: Decode throughput
        total_tokens: Total tokens processed
        kv_cache_memory_gb: KV cache memory usage
        inference_result: Detailed InferenceResult from analyzer
    """

    prefill_time_sec: float = 0.0
    decode_time_per_step_sec: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    total_tokens: int = 0
    kv_cache_memory_gb: float = 0.0
    inference_result: InferenceResult = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
                "total_tokens": self.total_tokens,
                "kv_cache_memory_gb": self.kv_cache_memory_gb,
                "inference_breakdown": self.inference_result.to_dict() if self.inference_result else None,
            }
        )
        return base


class LLMInferenceScenario(Scenario):
    """LLM inference performance scenario.

    Evaluates inference performance for a single LLM model.
    Uses InferenceAnalyzer for prefill and decode estimation.

    Example:
        >>> from llm_perf.modeling import create_model_from_config
        >>> from llm_perf.hardware.device import Device
        >>> from llm_perf.hardware.cluster import Cluster
        >>> from llm_perf.strategy.base import StrategyConfig
        >>> from llm_perf.legacy.scenarios.registry import ScenarioRegistry
        >>>
        >>> model = create_model_from_config({"preset": "llama-7b"})
        >>> device = Device(name="H100", memory_gb=80)
        >>> cluster = Cluster(num_nodes=1, devices_per_node=8)
        >>> strategy = StrategyConfig(tp_degree=8)
        >>>
        >>> registry = ScenarioRegistry()
        >>> scenario = registry.create_scenario(
        ...     "llm-inference",
        ...     models={"main": model},
        ...     device=device,
        ...     cluster=cluster,
        ...     strategy=strategy,
        ... )
        >>> result = scenario.analyze(batch_size=1, prompt_len=512, generation_len=128)
    """

    def __init__(
        self,
        config: LLMInferenceConfig,
        models: Dict[str, BaseModel],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize LLM inference scenario.

        Args:
            config: Inference scenario configuration
            models: Dictionary with "main" model
            device: Device configuration
            cluster: Cluster configuration
            strategy: Parallelism strategy
        """
        super().__init__(config, models, device, cluster, strategy)
        self._analyzer: InferenceAnalyzer = None

    def _get_main_model(self) -> BaseModel:
        """Get the main inference model."""
        return self.models.get("main")

    def get_analyzer(self) -> InferenceAnalyzer:
        """Get or create the InferenceAnalyzer instance.

        Returns:
            InferenceAnalyzer for this scenario
        """
        if self._analyzer is None:
            model = self._get_main_model()
            self._analyzer = InferenceAnalyzer(
                model=model,
                device=self.device,
                cluster=self.cluster,
                strategy=self.strategy,
            )
        return self._analyzer

    def analyze(
        self,
        batch_size: int = None,
        prompt_len: int = None,
        generation_len: int = None,
        **kwargs,
    ) -> LLMInferenceResult:
        """Run inference performance analysis.

        Args:
            batch_size: Inference batch size (uses config default if not provided)
            prompt_len: Prompt sequence length (uses config default if not provided)
            generation_len: Generation sequence length (uses config default if not provided)
            **kwargs: Additional parameters

        Returns:
            LLMInferenceResult with performance metrics
        """
        batch_size = batch_size or self.config.batch_size
        prompt_len = prompt_len or self.config.prompt_len
        generation_len = generation_len or self.config.generation_len

        analyzer = self.get_analyzer()
        inference_result = analyzer.analyze(
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=generation_len,
        )

        total_gpus = self.strategy.world_size

        return LLMInferenceResult(
            scenario_name=self.config.name,
            total_time_sec=inference_result.total_time_sec,
            throughput=inference_result.decode_tokens_per_sec,
            memory_peak_gb=inference_result.memory_per_gpu_gb,
            prefill_time_sec=inference_result.prefill_time_sec,
            decode_time_per_step_sec=inference_result.decode_time_per_step_sec,
            prefill_tokens_per_sec=inference_result.prefill_tokens_per_sec,
            decode_tokens_per_sec=inference_result.decode_tokens_per_sec,
            total_tokens=inference_result.total_tokens,
            kv_cache_memory_gb=inference_result.kv_cache_memory_gb,
            inference_result=inference_result,
            breakdown={
                "prefill": inference_result.prefill_breakdown.to_dict() if inference_result.prefill_breakdown else {},
                "decode": inference_result.decode_breakdown.to_dict() if inference_result.decode_breakdown else {},
            },
            metadata={
                "batch_size": batch_size,
                "prompt_len": prompt_len,
                "generation_len": generation_len,
                "total_gpus": total_gpus,
                "strategy": self.strategy.to_dict(),
            },
        )

    def estimate_memory(
        self,
        batch_size: int = None,
        max_seq_len: int = None,
    ) -> Dict[str, float]:
        """Estimate inference memory requirements.

        Args:
            batch_size: Inference batch size
            max_seq_len: Maximum sequence length

        Returns:
            Dictionary with memory estimates
        """
        batch_size = batch_size or self.config.batch_size
        max_seq_len = max_seq_len or (self.config.prompt_len + self.config.generation_len)

        analyzer = self.get_analyzer()
        result = analyzer.analyze(
            batch_size=batch_size,
            prompt_len=max_seq_len // 2,
            generation_len=max_seq_len // 2,
        )

        return {
            "total_memory_gb": result.memory_per_gpu_gb,
            "kv_cache_gb": result.kv_cache_memory_gb,
        }

    def estimate_latency(
        self,
        batch_size: int = None,
        prompt_len: int = None,
        generation_len: int = None,
    ) -> Dict[str, float]:
        """Estimate inference latency.

        Args:
            batch_size: Inference batch size
            prompt_len: Prompt sequence length
            generation_len: Generation sequence length

        Returns:
            Dictionary with latency estimates
        """
        batch_size = batch_size or self.config.batch_size
        prompt_len = prompt_len or self.config.prompt_len
        generation_len = generation_len or self.config.generation_len

        analyzer = self.get_analyzer()
        result = analyzer.analyze(
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=generation_len,
        )

        return {
            "ttft_sec": result.prefill_time_sec,
            "ttft_ms": result.prefill_time_sec * 1000,
            "tpot_sec": result.decode_time_per_step_sec,
            "tpot_ms": result.decode_time_per_step_sec * 1000,
            "total_time_sec": result.total_time_sec,
        }

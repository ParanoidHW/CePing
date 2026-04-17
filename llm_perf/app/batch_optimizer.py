"""BatchOptimizer: Find optimal batch size.

Searches for maximum batch size given memory or latency constraints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import time

from llm_perf.modeling import ShardedModule
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from .evaluator import Evaluator


@dataclass
class LatencyBudget:
    """Latency budget for inference with prefill/decode separation."""

    ttft_budget_ms: Optional[float] = None
    tpot_budget_ms: Optional[float] = None
    total_latency_budget_ms: Optional[float] = None
    generation_len: Optional[int] = None


@dataclass
class BatchSearchResult:
    """Result of batch size search."""

    best_batch_size: int
    best_metric: float
    all_results: List[Dict[str, Any]]
    search_time_sec: float
    reason: str
    constraint_type: Optional[str] = None


class BatchOptimizer:
    """Find optimal batch size for training or inference."""

    def __init__(self):
        self.evaluator = Evaluator()

    def find_max_batch(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategy: Union[str, Dict, StrategyConfig],
        mode: str = "training",
        latency_budget: Optional[Union[float, LatencyBudget]] = None,
        memory_budget_gb: Optional[float] = None,
        target_tps: Optional[float] = None,
        batch_step: int = 1,
        max_batch: int = 256,
        **kwargs,
    ) -> BatchSearchResult:
        """Find maximum batch size under constraints.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategy: Strategy specification
            mode: "training" or "inference"
            latency_budget: Latency budget (float or LatencyBudget)
            memory_budget_gb: Maximum memory per GPU
            target_tps: Minimum throughput requirement
            batch_step: Step size for batch iteration
            max_batch: Maximum batch size to search

        Returns:
            BatchSearchResult with optimal batch size
        """
        start_time = time.perf_counter()

        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)
        strategy_obj = self.evaluator._resolve_strategy(strategy, **kwargs)

        if isinstance(latency_budget, float):
            latency_budget = LatencyBudget(ttft_budget_ms=latency_budget)

        all_results = []
        best_batch_size = 1
        best_metric = 0.0
        stop_reason = "max_batch_limit"
        constraint_type = None

        batch_size = kwargs.get("batch_size", 1)

        while batch_size <= max_batch:
            try:
                if mode == "training":
                    result = self.evaluator.evaluate_training(
                        model_obj,
                        cluster_obj,
                        strategy_obj,
                        batch_size=batch_size,
                        seq_len=kwargs.get("seq_len", 2048),
                    )
                    metric = result.samples_per_sec
                    memory_gb = result.memory_per_gpu_gb

                    result_data = {
                        "batch_size": batch_size,
                        "samples_per_sec": result.samples_per_sec,
                        "tokens_per_sec": result.tokens_per_sec,
                        "memory_gb": memory_gb,
                        "valid": True,
                    }

                else:
                    result = self.evaluator.evaluate_inference(
                        model_obj,
                        cluster_obj,
                        strategy_obj,
                        batch_size=batch_size,
                        prompt_len=kwargs.get("prompt_len", 512),
                        generation_len=kwargs.get("generation_len", 128),
                    )
                    metric = result.decode_tokens_per_sec
                    memory_gb = result.memory_per_gpu_gb

                    ttft_ms = result.prefill_time_sec * 1000
                    tpot_ms = result.decode_time_per_step_sec * 1000

                    result_data = {
                        "batch_size": batch_size,
                        "prefill_tps": result.prefill_tokens_per_sec,
                        "decode_tps": result.decode_tokens_per_sec,
                        "ttft_ms": ttft_ms,
                        "tpot_ms": tpot_ms,
                        "memory_gb": memory_gb,
                        "valid": True,
                    }

                should_stop = False

                if memory_budget_gb is not None and memory_gb > memory_budget_gb:
                    stop_reason = "memory_budget_exceeded"
                    constraint_type = "memory"
                    should_stop = True

                if mode == "inference" and latency_budget is not None:
                    if latency_budget.ttft_budget_ms and result_data.get("ttft_ms", 0) > latency_budget.ttft_budget_ms:
                        stop_reason = "latency_budget_exceeded"
                        constraint_type = "ttft"
                        should_stop = True

                    if latency_budget.tpot_budget_ms and result_data.get("tpot_ms", 0) > latency_budget.tpot_budget_ms:
                        stop_reason = "latency_budget_exceeded"
                        constraint_type = "tpot"
                        should_stop = True

                if target_tps is not None and metric < target_tps:
                    stop_reason = "target_tps_not_met"
                    constraint_type = "tps"
                    should_stop = True

                all_results.append(result_data)

                if not should_stop:
                    if metric > best_metric:
                        best_metric = metric
                        best_batch_size = batch_size
                else:
                    break

                batch_size += batch_step

            except Exception as e:
                all_results.append(
                    {
                        "batch_size": batch_size,
                        "valid": False,
                        "error": str(e),
                    }
                )
                stop_reason = "evaluation_error"
                constraint_type = "error"
                break

        search_time = time.perf_counter() - start_time

        return BatchSearchResult(
            best_batch_size=best_batch_size,
            best_metric=best_metric,
            all_results=all_results,
            search_time_sec=search_time,
            reason=stop_reason,
            constraint_type=constraint_type,
        )

    def compare_batch_sizes(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategy: Union[str, Dict, StrategyConfig],
        batch_sizes: List[int],
        mode: str = "training",
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare performance across different batch sizes.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategy: Strategy specification
            batch_sizes: List of batch sizes to compare
            mode: "training" or "inference"

        Returns:
            Comparison results
        """
        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)
        strategy_obj = self.evaluator._resolve_strategy(strategy, **kwargs)

        all_results = []

        for batch_size in batch_sizes:
            try:
                if mode == "training":
                    result = self.evaluator.evaluate_training(
                        model_obj,
                        cluster_obj,
                        strategy_obj,
                        batch_size=batch_size,
                        seq_len=kwargs.get("seq_len", 2048),
                    )
                    all_results.append(
                        {
                            "batch_size": batch_size,
                            "samples_per_sec": result.samples_per_sec,
                            "tokens_per_sec": result.tokens_per_sec,
                            "memory_gb": result.memory_per_gpu_gb,
                            "valid": True,
                        }
                    )
                else:
                    result = self.evaluator.evaluate_inference(
                        model_obj,
                        cluster_obj,
                        strategy_obj,
                        batch_size=batch_size,
                        prompt_len=kwargs.get("prompt_len", 512),
                        generation_len=kwargs.get("generation_len", 128),
                    )
                    all_results.append(
                        {
                            "batch_size": batch_size,
                            "prefill_tps": result.prefill_tokens_per_sec,
                            "decode_tps": result.decode_tokens_per_sec,
                            "ttft_ms": result.prefill_time_sec * 1000,
                            "memory_gb": result.memory_per_gpu_gb,
                            "valid": True,
                        }
                    )

            except Exception as e:
                all_results.append(
                    {
                        "batch_size": batch_size,
                        "valid": False,
                        "error": str(e),
                    }
                )

        return {
            "mode": mode,
            "results": all_results,
        }

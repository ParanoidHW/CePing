"""BatchOptimizer: Find optimal batch size.

Searches for maximum batch size given memory or latency constraints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import time

from llm_perf.modeling import ShardedModule
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer import infer_workload
from .evaluator import Evaluator


@dataclass
class LatencyBudget:
    """Latency budget for inference."""

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
        workload: Optional[str] = None,
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
            workload: Workload preset name (auto-inferred if None)
            latency_budget: Latency budget
            memory_budget_gb: Maximum memory per device
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

        if workload is None:
            mode = kwargs.get("mode", "training")
            model_type = self.evaluator._get_model_type(model)
            workload = infer_workload(model_type, mode)

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
                result = self.evaluator.evaluate(
                    model_obj,
                    cluster_obj,
                    workload,
                    strategy_obj,
                    batch_size=batch_size,
                    **kwargs,
                )

                throughput = result.throughput.get(
                    "tokens_per_sec",
                    result.throughput.get("samples_per_sec", result.throughput.get("pixels_per_sec", 0)),
                )
                memory_gb = result.peak_memory_gb

                prefill_phase = result.get_phase("prefill")
                decode_phase = result.get_phase("decode")

                ttft_ms = prefill_phase.total_time_sec * 1000 if prefill_phase else 0
                tpot_ms = decode_phase.single_time_sec * 1000 if decode_phase else 0

                result_data = {
                    "batch_size": batch_size,
                    "throughput": throughput,
                    "total_throughput": result.throughput,
                    "memory_gb": memory_gb,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "total_time_sec": result.total_time_sec,
                    "valid": True,
                }

                should_stop = False

                if memory_budget_gb is not None and memory_gb > memory_budget_gb:
                    stop_reason = "memory_budget_exceeded"
                    constraint_type = "memory"
                    should_stop = True

                if latency_budget is not None:
                    if latency_budget.ttft_budget_ms and ttft_ms > latency_budget.ttft_budget_ms:
                        stop_reason = "latency_budget_exceeded"
                        constraint_type = "ttft"
                        should_stop = True

                    if latency_budget.tpot_budget_ms and tpot_ms > latency_budget.tpot_budget_ms:
                        stop_reason = "latency_budget_exceeded"
                        constraint_type = "tpot"
                        should_stop = True

                if target_tps is not None and throughput < target_tps:
                    stop_reason = "target_tps_not_met"
                    constraint_type = "throughput"
                    should_stop = True

                all_results.append(result_data)

                if should_stop:
                    break

                if throughput > best_metric:
                    best_metric = throughput
                    best_batch_size = batch_size

            except Exception:
                stop_reason = "error"
                break

            batch_size += batch_step

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
        workload: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare different batch sizes.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategy: Strategy specification
            batch_sizes: List of batch sizes to compare
            workload: Workload preset name

        Returns:
            Dict with comparison results
        """
        if workload is None:
            mode = kwargs.get("mode", "training")
            model_type = self.evaluator._get_model_type(model)
            workload = infer_workload(model_type, mode)

        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)
        strategy_obj = self.evaluator._resolve_strategy(strategy, **kwargs)

        results = []
        for batch_size in batch_sizes:
            result = self.evaluator.evaluate(
                model_obj,
                cluster_obj,
                workload,
                strategy_obj,
                batch_size=batch_size,
                **kwargs,
            )

            results.append(
                {
                    "batch_size": batch_size,
                    "throughput": result.throughput,
                    "total_time_sec": result.total_time_sec,
                    "peak_memory_gb": result.peak_memory_gb,
                    "phases": [p.to_dict() for p in result.phases],
                }
            )

        throughput_key = "tokens_per_sec"
        best_idx = 0
        best_metric = 0.0
        for i, r in enumerate(results):
            metric = r["throughput"].get(throughput_key, r["throughput"].get("samples_per_sec", 0))
            if metric > best_metric:
                best_metric = metric
                best_idx = i

        return {
            "workload": workload,
            "best_batch_size": batch_sizes[best_idx],
            "best_metric": best_metric,
            "results": results,
        }

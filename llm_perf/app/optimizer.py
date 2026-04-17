"""StrategyOptimizer: Search optimal parallelism strategy.

Provides methods to find best strategy based on constraints and objectives.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time

from llm_perf.modeling import ShardedModule
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from .evaluator import Evaluator


class SearchMethod(Enum):
    """Strategy search methods."""

    GRID = "grid"
    GREEDY = "greedy"


class OptimizeObjective(Enum):
    """Optimization objectives."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"


@dataclass
class StrategyConstraints:
    """Constraints for strategy search."""

    max_memory_gb: Optional[float] = None
    max_gpus: Optional[int] = None
    min_gpus: int = 1
    max_tp: Optional[int] = None
    max_pp: Optional[int] = None
    max_dp: Optional[int] = None
    require_tp: bool = False
    require_pp: bool = False
    latency_budget_ms: Optional[float] = None
    throughput_min: Optional[float] = None


@dataclass
class StrategySearchResult:
    """Result of strategy search."""

    best_strategy: StrategyConfig
    best_metric: float
    all_results: List[Dict[str, Any]]
    search_time_sec: float
    method: str


class StrategyOptimizer:
    """Search optimal parallelism strategy."""

    def __init__(self):
        self.evaluator = Evaluator()

    def search_best_strategy(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        mode: str = "training",
        constraints: Optional[StrategyConstraints] = None,
        objective: OptimizeObjective = OptimizeObjective.THROUGHPUT,
        method: SearchMethod = SearchMethod.GRID,
        **kwargs,
    ) -> StrategySearchResult:
        """Search for best strategy.

        Args:
            model: Model specification
            hardware: Hardware specification
            mode: "training" or "inference"
            constraints: Search constraints
            objective: Optimization objective
            method: Search method

        Returns:
            StrategySearchResult with best strategy
        """
        constraints = constraints or StrategyConstraints()

        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)

        start_time = time.perf_counter()

        candidates = self._generate_candidates(cluster_obj, constraints)

        all_results = []
        best_strategy = None
        best_metric = 0.0 if objective == OptimizeObjective.THROUGHPUT else float("inf")

        for candidate in candidates:
            try:
                if mode == "training":
                    result = self.evaluator.evaluate_training(model_obj, cluster_obj, candidate, **kwargs)
                    metric = result.samples_per_sec
                    memory_gb = result.memory_per_gpu_gb
                else:
                    result = self.evaluator.evaluate_inference(model_obj, cluster_obj, candidate, **kwargs)
                    metric = result.prefill_tokens_per_sec
                    memory_gb = result.memory_per_gpu_gb

                if constraints.max_memory_gb and memory_gb > constraints.max_memory_gb:
                    continue

                if constraints.latency_budget_ms and mode == "inference":
                    latency = result.prefill_time_sec * 1000
                    if latency > constraints.latency_budget_ms:
                        continue

                if objective == OptimizeObjective.THROUGHPUT:
                    if metric > best_metric:
                        best_metric = metric
                        best_strategy = candidate
                elif objective == OptimizeObjective.LATENCY:
                    if metric < best_metric:
                        best_metric = metric
                        best_strategy = candidate
                elif objective == OptimizeObjective.MEMORY:
                    if memory_gb < best_metric:
                        best_metric = memory_gb
                        best_strategy = candidate

                all_results.append(
                    {
                        "strategy": {
                            "tp": candidate.tp_degree,
                            "pp": candidate.pp_degree,
                            "dp": candidate.dp_degree,
                        },
                        "metric": metric,
                        "memory_gb": memory_gb,
                    }
                )

            except Exception:
                continue

        search_time = time.perf_counter() - start_time

        if best_strategy is None:
            best_strategy = StrategyConfig(tp_degree=1)
            best_metric = 0.0

        return StrategySearchResult(
            best_strategy=best_strategy,
            best_metric=best_metric,
            all_results=all_results,
            search_time_sec=search_time,
            method=method.value,
        )

    def compare_specific_strategies(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategies: list,
        mode: str = "training",
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare specific strategies.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategies: List of strategy specs to compare
            mode: "training" or "inference"

        Returns:
            Comparison results
        """
        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)

        results = []
        for strategy_spec in strategies:
            strategy = self.evaluator._resolve_strategy(strategy_spec, **kwargs)

            if mode == "training":
                result = self.evaluator.evaluate_training(model_obj, cluster_obj, strategy, **kwargs)
                results.append(
                    {
                        "strategy": strategy,
                        "samples_per_sec": result.samples_per_sec,
                        "tokens_per_sec": result.tokens_per_sec,
                        "memory_gb": result.memory_per_gpu_gb,
                    }
                )
            else:
                result = self.evaluator.evaluate_inference(model_obj, cluster_obj, strategy, **kwargs)
                results.append(
                    {
                        "strategy": strategy,
                        "prefill_tps": result.prefill_tokens_per_sec,
                        "decode_tps": result.decode_tokens_per_sec,
                        "memory_gb": result.memory_per_gpu_gb,
                    }
                )

        if results:
            best = max(results, key=lambda x: x.get("samples_per_sec") or x.get("prefill_tps", 0))
        else:
            best = None

        return {
            "best_strategy": best,
            "all_results": results,
            "mode": mode,
        }

    def _generate_candidates(
        self,
        cluster: Cluster,
        constraints: StrategyConstraints,
    ) -> List[StrategyConfig]:
        """Generate strategy candidates."""
        num_devices = cluster.num_devices

        candidates = []

        max_tp = constraints.max_tp or num_devices
        max_pp = constraints.max_pp or num_devices
        max_dp = constraints.max_dp or num_devices

        if constraints.require_tp:
            tp_values = [t for t in range(2, max_tp + 1) if num_devices >= t]
        else:
            tp_values = [t for t in range(1, max_tp + 1) if num_devices >= t]

        if constraints.require_pp:
            pp_values = [p for p in range(2, max_pp + 1) if num_devices >= p]
        else:
            pp_values = [p for p in range(1, max_pp + 1) if num_devices >= p]

        dp_values = [d for d in range(1, max_dp + 1)]

        for tp in tp_values:
            for pp in pp_values:
                for dp in dp_values:
                    if tp * pp * dp <= num_devices:
                        if tp * pp * dp >= constraints.min_gpus:
                            candidates.append(
                                StrategyConfig(
                                    tp_degree=tp,
                                    pp_degree=pp,
                                    dp_degree=dp,
                                )
                            )

        return candidates

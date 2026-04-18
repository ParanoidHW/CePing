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
from llm_perf.analyzer import UnifiedResult, infer_workload
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
        workload: Optional[str] = None,
        constraints: Optional[StrategyConstraints] = None,
        objective: OptimizeObjective = OptimizeObjective.THROUGHPUT,
        method: SearchMethod = SearchMethod.GRID,
        **kwargs,
    ) -> StrategySearchResult:
        """Search for best strategy.

        Args:
            model: Model specification
            hardware: Hardware specification
            workload: Workload preset name (auto-inferred if None)
            constraints: Search constraints
            objective: Optimization objective
            method: Search method

        Returns:
            StrategySearchResult with best strategy
        """
        constraints = constraints or StrategyConstraints()

        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)

        if workload is None:
            mode = kwargs.get("mode", "training")
            model_type = self.evaluator._get_model_type(model)
            workload = infer_workload(model_type, mode)

        start_time = time.perf_counter()

        candidates = self._generate_candidates(cluster_obj, constraints)

        all_results = []
        best_strategy = None
        best_metric = 0.0 if objective == OptimizeObjective.THROUGHPUT else float("inf")

        for candidate in candidates:
            try:
                result = self.evaluator.evaluate(model_obj, cluster_obj, workload, candidate, **kwargs)

                throughput = result.throughput.get("tokens_per_sec", result.throughput.get("samples_per_sec", 0))
                memory_gb = result.peak_memory_gb

                prefill_phase = result.get_phase("prefill")
                latency = prefill_phase.total_time_sec * 1000 if prefill_phase else 0

                if constraints.max_memory_gb and memory_gb > constraints.max_memory_gb:
                    continue

                if constraints.latency_budget_ms and latency > constraints.latency_budget_ms:
                    continue

                if objective == OptimizeObjective.THROUGHPUT:
                    metric = throughput
                    if metric > best_metric:
                        best_metric = metric
                        best_strategy = candidate
                elif objective == OptimizeObjective.LATENCY:
                    metric = latency
                    if metric < best_metric:
                        best_metric = metric
                        best_strategy = candidate
                elif objective == OptimizeObjective.MEMORY:
                    metric = memory_gb
                    if metric < best_metric:
                        best_metric = metric
                        best_strategy = candidate

                all_results.append(
                    {
                        "strategy": {
                            "tp": candidate.tp_degree,
                            "pp": candidate.pp_degree,
                            "dp": candidate.dp_degree,
                        },
                        "metric": metric,
                        "throughput": throughput,
                        "memory_gb": memory_gb,
                        "latency_ms": latency,
                    }
                )

            except Exception:
                continue

        search_time = time.perf_counter() - start_time

        if best_strategy is None:
            best_strategy = StrategyConfig()
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
        strategies: List[Union[str, Dict, StrategyConfig]],
        workload: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare specific strategies.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategies: List of strategy configurations
            workload: Workload preset name

        Returns:
            Dict with comparison results
        """
        if workload is None:
            mode = kwargs.get("mode", "training")
            model_type = self.evaluator._get_model_type(model)
            workload = infer_workload(model_type, mode)

        results = self.evaluator.compare_strategies(model, hardware, strategies, workload, **kwargs)

        best_name = None
        best_metric = 0.0

        throughput_key = "tokens_per_sec"
        for name, result in results.items():
            throughput = result.throughput.get(throughput_key, result.throughput.get("samples_per_sec", 0))
            if throughput > best_metric:
                best_metric = throughput
                best_name = name

        return {
            "best_strategy": best_name,
            "best_metric": best_metric,
            "all_results": {
                name: {
                    "total_time_sec": result.total_time_sec,
                    "throughput": result.throughput,
                    "peak_memory_gb": result.peak_memory_gb,
                    "phases": [p.to_dict() for p in result.phases],
                }
                for name, result in results.items()
            },
        }

    def _generate_candidates(
        self,
        cluster: Cluster,
        constraints: StrategyConstraints,
    ) -> List[StrategyConfig]:
        """Generate strategy candidates based on constraints."""
        candidates = []

        num_devices = cluster.num_devices

        max_tp = constraints.max_tp or num_devices
        max_pp = constraints.max_pp or num_devices
        max_dp = constraints.max_dp or num_devices

        tp_range = range(1, min(max_tp + 1, num_devices + 1))
        pp_range = range(1, min(max_pp + 1, num_devices + 1))
        dp_range = range(1, min(max_dp + 1, num_devices + 1))

        for tp in tp_range:
            for pp in pp_range:
                for dp in dp_range:
                    if tp * pp * dp > num_devices:
                        continue
                    if tp * pp * dp < constraints.min_gpus:
                        continue
                    if constraints.require_tp and tp < 2:
                        continue
                    if constraints.require_pp and pp < 2:
                        continue

                    candidates.append(StrategyConfig(tp_degree=tp, pp_degree=pp, dp_degree=dp))

        if not candidates:
            candidates.append(StrategyConfig())

        return candidates

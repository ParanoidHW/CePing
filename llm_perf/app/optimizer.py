"""StrategyOptimizer: Search optimal parallelism strategy.

Provides methods to find best strategy based on constraints and objectives.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from ..models.base import BaseModel, ModelConfig
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig
from .evaluator import Evaluator


class SearchMethod(Enum):
    """Strategy search methods."""

    GRID = "grid"
    GREEDY = "greedy"
    GENETIC = "genetic"


class OptimizeObjective(Enum):
    """Optimization objectives."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"


@dataclass
class StrategyConstraints:
    """Constraints for strategy search.

    Attributes:
        max_memory_gb: Maximum memory per GPU
        max_gpus: Maximum number of GPUs
        min_gpus: Minimum number of GPUs
        max_tp: Maximum TP degree
        max_pp: Maximum PP degree
        max_dp: Maximum DP degree
        require_tp: Must use TP (TP > 1)
        require_pp: Must use PP (PP > 1)
        latency_budget_ms: Maximum latency budget (for inference)
        throughput_min: Minimum throughput requirement
    """

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
    """Result of strategy search.

    Attributes:
        best_strategy: Best strategy configuration
        best_metric: Best metric value
        all_results: All evaluated strategies
        search_time_sec: Time spent searching
        method: Search method used
    """

    best_strategy: StrategyConfig
    best_metric: float
    all_results: List[Dict[str, Any]]
    search_time_sec: float
    method: str


class StrategyOptimizer:
    """Search optimal parallelism strategy.

    Finds best strategy configuration based on:
    - Constraints (memory, GPU count, latency)
    - Optimization objective (throughput, latency, memory)
    - Search method (grid, greedy, genetic)

    Example:
        >>> optimizer = StrategyOptimizer()
        >>> constraints = StrategyConstraints(max_gpus=8, max_memory_gb=80)
        >>> result = optimizer.search_best_strategy(
        ...     "llama-7b", "h100_8gpu", "training",
        ...     constraints, OptimizeObjective.THROUGHPUT
        ... )
        >>> print(f"Best strategy: TP={result.best_strategy.tp_degree}")
    """

    def __init__(self):
        self.evaluator = Evaluator()

    def search_best_strategy(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
        hardware: Union[str, Dict, Cluster],
        mode: str,
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
            constraints: Strategy constraints
            objective: Optimization objective
            method: Search method
            **kwargs: Additional evaluation parameters

        Returns:
            StrategySearchResult with best strategy and metrics

        Example:
            >>> constraints = StrategyConstraints(max_gpus=8)
            >>> result = optimizer.search_best_strategy(
            ...     "llama-7b", "h100_8gpu", "training",
            ...     constraints, OptimizeObjective.THROUGHPUT, SearchMethod.GRID
            ... )
        """
        import time

        start_time = time.perf_counter()

        constraints = constraints or StrategyConstraints()

        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)

        max_gpus = constraints.max_gpus or cluster_obj.num_devices

        if method == SearchMethod.GRID:
            strategies = self._grid_search_strategies(model_obj, cluster_obj, max_gpus, constraints)
        elif method == SearchMethod.GREEDY:
            strategies = self._greedy_search_strategies(model_obj, cluster_obj, max_gpus, constraints, objective)
        elif method == SearchMethod.GENETIC:
            strategies = self._genetic_search_strategies(model_obj, cluster_obj, max_gpus, constraints, objective)
        else:
            raise ValueError(f"Unknown search method: {method}")

        all_results = []
        for strategy_config in strategies:
            try:
                if mode == "training":
                    eval_kwargs = {
                        "batch_size": kwargs.get("batch_size", 32),
                        "seq_len": kwargs.get("seq_len", 2048),
                    }
                    result = self.evaluator.evaluate_training(model_obj, cluster_obj, strategy_config, **eval_kwargs)

                    metric = self._get_training_metric(result, objective)
                    result_data = {
                        "strategy": self._strategy_to_str(strategy_config),
                        "tp": strategy_config.tp_degree,
                        "pp": strategy_config.pp_degree,
                        "dp": strategy_config.dp_degree,
                        "world_size": strategy_config.world_size,
                        "metric": metric,
                        "samples_per_sec": result.samples_per_sec,
                        "tokens_per_sec": result.tokens_per_sec,
                        "memory_gb": result.memory_per_gpu_gb,
                        "valid": True,
                    }

                else:
                    eval_kwargs = {
                        "batch_size": kwargs.get("batch_size", 1),
                        "prompt_len": kwargs.get("prompt_len", 512),
                        "generation_len": kwargs.get("generation_len", 128),
                    }
                    result = self.evaluator.evaluate_inference(model_obj, cluster_obj, strategy_config, **eval_kwargs)

                    metric = self._get_inference_metric(result, objective)
                    result_data = {
                        "strategy": self._strategy_to_str(strategy_config),
                        "tp": strategy_config.tp_degree,
                        "pp": strategy_config.pp_degree,
                        "dp": strategy_config.dp_degree,
                        "world_size": strategy_config.world_size,
                        "metric": metric,
                        "prefill_tps": result.prefill_tokens_per_sec,
                        "decode_tps": result.decode_tokens_per_sec,
                        "ttft_ms": result.prefill_time_sec * 1000,
                        "memory_gb": result.memory_per_gpu_gb,
                        "valid": True,
                    }

                if self._check_constraints(result_data, constraints, mode):
                    all_results.append(result_data)

            except Exception as e:
                all_results.append(
                    {
                        "strategy": self._strategy_to_str(strategy_config),
                        "tp": strategy_config.tp_degree,
                        "pp": strategy_config.pp_degree,
                        "dp": strategy_config.dp_degree,
                        "world_size": strategy_config.world_size,
                        "valid": False,
                        "error": str(e),
                    }
                )

        best_result = self._find_best_result(all_results, objective)

        if best_result is None:
            raise ValueError("No valid strategy found under constraints")

        best_strategy = StrategyConfig(
            model_name=model_obj.config.name,
            tp_degree=best_result["tp"],
            pp_degree=best_result["pp"],
            dp_degree=best_result["dp"],
        )

        search_time = time.perf_counter() - start_time

        return StrategySearchResult(
            best_strategy=best_strategy,
            best_metric=best_result["metric"],
            all_results=all_results,
            search_time_sec=search_time,
            method=method.value,
        )

    def compare_strategies(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
        hardware: Union[str, Dict, Cluster],
        strategies: List[Union[str, Dict, StrategyConfig]],
        mode: str = "training",
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare specific strategies.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategies: List of strategies to compare
            mode: "training" or "inference"
            **kwargs: Additional parameters

        Returns:
            Comparison results

        Example:
            >>> result = optimizer.compare_strategies(
            ...     "llama-7b", "h100_8gpu",
            ...     ["tp1", "tp2", "tp4", "tp8"],
            ...     mode="training"
            ... )
        """
        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)

        all_results = []
        for strategy_spec in strategies:
            strategy_config = self.evaluator._resolve_strategy(strategy_spec, model_obj.config.name)

            try:
                if mode == "training":
                    eval_kwargs = {
                        "batch_size": kwargs.get("batch_size", 32),
                        "seq_len": kwargs.get("seq_len", 2048),
                    }
                    result = self.evaluator.evaluate_training(model_obj, cluster_obj, strategy_config, **eval_kwargs)

                    result_data = {
                        "strategy": self._strategy_to_str(strategy_config),
                        "tp": strategy_config.tp_degree,
                        "pp": strategy_config.pp_degree,
                        "dp": strategy_config.dp_degree,
                        "samples_per_sec": result.samples_per_sec,
                        "tokens_per_sec": result.tokens_per_sec,
                        "memory_gb": result.memory_per_gpu_gb,
                        "valid": True,
                    }
                else:
                    eval_kwargs = {
                        "batch_size": kwargs.get("batch_size", 1),
                        "prompt_len": kwargs.get("prompt_len", 512),
                        "generation_len": kwargs.get("generation_len", 128),
                    }
                    result = self.evaluator.evaluate_inference(model_obj, cluster_obj, strategy_config, **eval_kwargs)

                    result_data = {
                        "strategy": self._strategy_to_str(strategy_config),
                        "tp": strategy_config.tp_degree,
                        "pp": strategy_config.pp_degree,
                        "dp": strategy_config.dp_degree,
                        "prefill_tps": result.prefill_tokens_per_sec,
                        "decode_tps": result.decode_tokens_per_sec,
                        "ttft_ms": result.prefill_time_sec * 1000,
                        "memory_gb": result.memory_per_gpu_gb,
                        "valid": True,
                    }

                all_results.append(result_data)

            except Exception as e:
                all_results.append(
                    {
                        "strategy": self._strategy_to_str(strategy_config),
                        "valid": False,
                        "error": str(e),
                    }
                )

        return {
            "model": model_obj.config.name,
            "hardware": getattr(cluster_obj, "name", f"{cluster_obj.num_devices}-gpu cluster"),
            "mode": mode,
            "results": all_results,
        }

    def _grid_search_strategies(
        self,
        model: BaseModel,
        cluster: Cluster,
        max_gpus: int,
        constraints: StrategyConstraints,
    ) -> List[StrategyConfig]:
        """Generate all feasible strategy combinations.

        Args:
            model: Model to evaluate
            cluster: Cluster configuration
            max_gpus: Maximum GPUs to use
            constraints: Strategy constraints

        Returns:
            List of StrategyConfig to evaluate
        """
        strategies = []

        hidden_size = model.config.hidden_size
        num_layers = model.config.num_layers

        max_tp = constraints.max_tp or max(1, min(max_gpus, hidden_size // 64))
        max_pp = constraints.max_pp or max(1, min(max_gpus, num_layers // 4))
        max_dp = constraints.max_dp or max_gpus

        tp_candidates = [1]
        for tp in [2, 4, 8, 16, 32, 64]:
            if tp <= max_tp and tp <= max_gpus:
                if hidden_size % tp == 0:
                    tp_candidates.append(tp)

        pp_candidates = [1]
        for pp in [2, 4, 8, 16, 32]:
            if pp <= max_pp and pp <= max_gpus:
                if num_layers % pp == 0:
                    pp_candidates.append(pp)

        dp_candidates = [1]
        for dp in [2, 4, 8, 16, 32, 64]:
            if dp <= max_dp:
                dp_candidates.append(dp)

        for tp in tp_candidates:
            for pp in pp_candidates:
                for dp in dp_candidates:
                    world_size = tp * pp * dp

                    if world_size < constraints.min_gpus:
                        continue
                    if world_size > max_gpus:
                        continue
                    if world_size > cluster.num_devices:
                        continue

                    if constraints.require_tp and tp <= 1:
                        continue
                    if constraints.require_pp and pp <= 1:
                        continue

                    strategy = StrategyConfig(
                        model_name=model.config.name,
                        tp_degree=tp,
                        pp_degree=pp,
                        dp_degree=dp,
                    )
                    strategies.append(strategy)

        return strategies

    def _greedy_search_strategies(
        self,
        model: BaseModel,
        cluster: Cluster,
        max_gpus: int,
        constraints: StrategyConstraints,
        objective: OptimizeObjective,
    ) -> List[StrategyConfig]:
        """Greedy search for best strategy.

        Args:
            model: Model to evaluate
            cluster: Cluster configuration
            max_gpus: Maximum GPUs
            constraints: Strategy constraints
            objective: Optimization objective

        Returns:
            List of strategies to evaluate
        """
        strategies = []

        strategies.append(
            StrategyConfig(
                model_name=model.config.name,
                tp_degree=1,
                pp_degree=1,
                dp_degree=1,
            )
        )

        hidden_size = model.config.hidden_size
        num_layers = model.config.num_layers

        if objective == OptimizeObjective.THROUGHPUT:
            if hidden_size >= 64:
                for tp in [2, 4, 8, 16, 32, 64]:
                    if tp <= max_gpus and hidden_size % tp == 0:
                        strategies.append(
                            StrategyConfig(
                                model_name=model.config.name,
                                tp_degree=tp,
                                pp_degree=1,
                                dp_degree=1,
                            )
                        )

            for pp in [2, 4, 8, 16]:
                if pp <= max_gpus and num_layers % pp == 0:
                    strategies.append(
                        StrategyConfig(
                            model_name=model.config.name,
                            tp_degree=1,
                            pp_degree=pp,
                            dp_degree=1,
                        )
                    )

            if max_gpus >= 8:
                strategies.append(
                    StrategyConfig(
                        model_name=model.config.name,
                        tp_degree=8,
                        pp_degree=1,
                        dp_degree=1,
                    )
                )

        elif objective == OptimizeObjective.LATENCY:
            if max_gpus >= 4 and hidden_size % 4 == 0:
                strategies.append(
                    StrategyConfig(
                        model_name=model.config.name,
                        tp_degree=4,
                        pp_degree=1,
                        dp_degree=1,
                    )
                )

            if max_gpus >= 2:
                strategies.append(
                    StrategyConfig(
                        model_name=model.config.name,
                        tp_degree=2,
                        pp_degree=1,
                        dp_degree=1,
                    )
                )

        elif objective == OptimizeObjective.MEMORY:
            max_tp = min(max_gpus, hidden_size // 64)
            strategies.append(
                StrategyConfig(
                    model_name=model.config.name,
                    tp_degree=max_tp,
                    pp_degree=1,
                    dp_degree=1,
                )
            )

            if max_gpus >= 4:
                strategies.append(
                    StrategyConfig(
                        model_name=model.config.name,
                        tp_degree=max_gpus // 2,
                        pp_degree=2,
                        dp_degree=1,
                    )
                )

        return strategies

    def _genetic_search_strategies(
        self,
        model: BaseModel,
        cluster: Cluster,
        max_gpus: int,
        constraints: StrategyConstraints,
        objective: OptimizeObjective,
    ) -> List[StrategyConfig]:
        """Genetic algorithm search (placeholder).

        Args:
            model: Model to evaluate
            cluster: Cluster configuration
            max_gpus: Maximum GPUs
            constraints: Strategy constraints
            objective: Optimization objective

        Returns:
            List of strategies to evaluate

        Note:
            Currently uses grid search as fallback.
        """
        return self._grid_search_strategies(model, cluster, max_gpus, constraints)

    def _get_training_metric(
        self,
        result,
        objective: OptimizeObjective,
    ) -> float:
        """Extract metric for training.

        Args:
            result: TrainingResult
            objective: Optimization objective

        Returns:
            Metric value
        """
        if objective == OptimizeObjective.THROUGHPUT:
            return result.samples_per_sec
        elif objective == OptimizeObjective.LATENCY:
            return -result.time_per_step_sec
        elif objective == OptimizeObjective.MEMORY:
            return -result.memory_per_gpu_gb
        else:
            return result.samples_per_sec

    def _get_inference_metric(
        self,
        result,
        objective: OptimizeObjective,
    ) -> float:
        """Extract metric for inference.

        Args:
            result: InferenceResult
            objective: Optimization objective

        Returns:
            Metric value
        """
        if objective == OptimizeObjective.THROUGHPUT:
            return result.decode_tokens_per_sec
        elif objective == OptimizeObjective.LATENCY:
            return -result.prefill_time_sec
        elif objective == OptimizeObjective.MEMORY:
            return -result.memory_per_gpu_gb
        else:
            return result.decode_tokens_per_sec

    def _check_constraints(
        self,
        result_data: Dict[str, Any],
        constraints: StrategyConstraints,
        mode: str,
    ) -> bool:
        """Check if result satisfies constraints.

        Args:
            result_data: Evaluation result
            constraints: Strategy constraints
            mode: "training" or "inference"

        Returns:
            True if satisfies constraints
        """
        if not result_data.get("valid", False):
            return False

        if constraints.max_memory_gb is not None:
            if result_data.get("memory_gb", 0) > constraints.max_memory_gb:
                return False

        if mode == "inference" and constraints.latency_budget_ms is not None:
            if result_data.get("ttft_ms", 0) > constraints.latency_budget_ms:
                return False

        if constraints.throughput_min is not None:
            if mode == "training":
                if result_data.get("samples_per_sec", 0) < constraints.throughput_min:
                    return False
            else:
                if result_data.get("decode_tps", 0) < constraints.throughput_min:
                    return False

        return True

    def _find_best_result(
        self,
        results: List[Dict[str, Any]],
        objective: OptimizeObjective,
    ) -> Optional[Dict[str, Any]]:
        """Find best result among all.

        Args:
            results: List of results
            objective: Optimization objective

        Returns:
            Best result or None
        """
        valid_results = [r for r in results if r.get("valid", False)]

        if not valid_results:
            return None

        return max(valid_results, key=lambda x: x.get("metric", 0))

    def _strategy_to_str(self, strategy: StrategyConfig) -> str:
        """Convert strategy to string representation.

        Args:
            strategy: StrategyConfig

        Returns:
            String like "tp4_pp2_dp1"
        """
        parts = []
        if strategy.tp_degree > 1:
            parts.append(f"tp{strategy.tp_degree}")
        if strategy.pp_degree > 1:
            parts.append(f"pp{strategy.pp_degree}")
        if strategy.dp_degree > 1:
            parts.append(f"dp{strategy.dp_degree}")

        if not parts:
            return "tp1_pp1_dp1"

        return "_".join(parts)

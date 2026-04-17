"""BatchOptimizer: Find optimal batch size.

Searches for maximum batch size given memory or latency constraints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ..models.base import BaseModel, ModelConfig
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig
from .evaluator import Evaluator


@dataclass
class LatencyBudget:
    """Latency budget for inference with prefill/decode separation.

    Attributes:
        ttft_budget_ms: Time To First Token budget (prefill phase latency)
        tpot_budget_ms: Time Per Output Token budget (decode phase per-token latency)
        total_latency_budget_ms: Total generation latency budget (optional)
        generation_len: Generation length for total latency calculation (optional)

    Example:
        >>> # Strict TTFT constraint
        >>> budget = LatencyBudget(ttft_budget_ms=50.0)

        >>> # Strict TPOT constraint
        >>> budget = LatencyBudget(tpot_budget_ms=2.0)

        >>> # Combined constraints
        >>> budget = LatencyBudget(ttft_budget_ms=100.0, tpot_budget_ms=1.0)

        >>> # Total latency constraint
        >>> budget = LatencyBudget(total_latency_budget_ms=500.0, generation_len=128)
    """

    ttft_budget_ms: Optional[float] = None
    tpot_budget_ms: Optional[float] = None
    total_latency_budget_ms: Optional[float] = None
    generation_len: Optional[int] = None


@dataclass
class BatchSearchResult:
    """Result of batch size search.

    Attributes:
        best_batch_size: Optimal batch size found
        best_metric: Metric value at optimal batch size
        all_results: All evaluated batch sizes
        search_time_sec: Time spent searching
        reason: Reason for stopping search (memory/latency/etc)
        constraint_type: Which constraint triggered stop (ttft/tpot/memory/etc)
    """

    best_batch_size: int
    best_metric: float
    all_results: List[Dict[str, Any]]
    search_time_sec: float
    reason: str
    constraint_type: Optional[str] = None


@dataclass
class StrategyConstraintsForBatch:
    """Constraints for strategy when searching batch size.

    Attributes:
        max_tp: Maximum TP degree
        max_pp: Maximum PP degree
        min_tp: Minimum TP degree
        min_pp: Minimum PP degree
    """

    max_tp: Optional[int] = None
    max_pp: Optional[int] = None
    min_tp: int = 1
    min_pp: int = 1


class BatchOptimizer:
    """Find optimal batch size for training or inference.

    Supports constraints:
    - Memory budget: maximum memory per GPU
    - Latency budget: separate TTFT/TPOT/Total constraints for inference
    - Target TPS: minimum throughput requirement

    Example:
        >>> optimizer = BatchOptimizer()
        >>> result = optimizer.find_max_batch(
        ...     "llama-7b", "h100_8gpu", "tp4",
        ...     memory_budget_gb=80
        ... )
        >>> print(f"Max batch: {result.best_batch_size}")
    """

    def __init__(self):
        self.evaluator = Evaluator()

    def _parse_latency_budget(
        self,
        latency_budget: Optional[Union[float, LatencyBudget]],
        kwargs: Dict[str, Any],
    ) -> Optional[LatencyBudget]:
        """Parse latency budget from various input formats.

        Args:
            latency_budget: Input budget (float or LatencyBudget)
            kwargs: Additional kwargs for generation_len

        Returns:
            LatencyBudget object or None
        """
        if latency_budget is None:
            return None

        if isinstance(latency_budget, float):
            return LatencyBudget(ttft_budget_ms=latency_budget)

        if isinstance(latency_budget, LatencyBudget):
            if latency_budget.generation_len is None:
                latency_budget.generation_len = kwargs.get("generation_len", 128)
            return latency_budget

        return None

    def _check_latency_budget(
        self,
        result_data: Dict[str, Any],
        latency_budget: LatencyBudget,
    ) -> tuple[bool, str]:
        """Check if latency budget constraints are exceeded.

        Args:
            result_data: Evaluation result data
            latency_budget: Latency budget constraints

        Returns:
            Tuple of (should_stop, constraint_type)
        """
        ttft_ms = result_data.get("ttft_ms", 0.0)
        tpot_ms = result_data.get("tpot_ms", 0.0)
        total_latency_ms = result_data.get("total_latency_ms", 0.0)

        if latency_budget.ttft_budget_ms is not None:
            if ttft_ms > latency_budget.ttft_budget_ms:
                return True, "ttft"

        if latency_budget.tpot_budget_ms is not None:
            if tpot_ms > latency_budget.tpot_budget_ms:
                return True, "tpot"

        if latency_budget.total_latency_budget_ms is not None:
            if total_latency_ms > latency_budget.total_latency_budget_ms:
                return True, "total_latency"

        return False, ""

    def find_max_batch(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
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
            latency_budget: Latency budget for inference:
                - float: legacy single value (treated as ttft_budget)
                - LatencyBudget: separate ttft/tpot/total budgets
            memory_budget_gb: Maximum memory per GPU
            target_tps: Minimum throughput requirement
            batch_step: Step size for batch iteration
            max_batch: Maximum batch size to search
            **kwargs: Additional parameters (seq_len, prompt_len, generation_len, etc.)

        Returns:
            BatchSearchResult with optimal batch size

        Example:
            >>> # Memory constraint
            >>> result = optimizer.find_max_batch(
            ...     "llama-7b", "h100_8gpu", "tp4",
            ...     memory_budget_gb=80, mode="training"
            ... )

            >>> # TTFT constraint only
            >>> result = optimizer.find_max_batch(
            ...     "llama-7b", "h100_8gpu", "tp4", mode="inference",
            ...     latency_budget=LatencyBudget(ttft_budget_ms=50.0)
            ... )

            >>> # TPOT constraint only
            >>> result = optimizer.find_max_batch(
            ...     "llama-7b", "h100_8gpu", "tp4", mode="inference",
            ...     latency_budget=LatencyBudget(tpot_budget_ms=2.0)
            ... )

            >>> # Combined TTFT + TPOT constraints
            >>> result = optimizer.find_max_batch(
            ...     "llama-7b", "h100_8gpu", "tp4", mode="inference",
            ...     latency_budget=LatencyBudget(ttft_budget_ms=100.0, tpot_budget_ms=1.0)
            ... )

            >>> # Total latency constraint
            >>> result = optimizer.find_max_batch(
            ...     "llama-7b", "h100_8gpu", "tp4", mode="inference",
            ...     latency_budget=LatencyBudget(total_latency_budget_ms=500.0, generation_len=128)
            ... )
        """
        import time

        start_time = time.perf_counter()

        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)
        strategy_obj = self.evaluator._resolve_strategy(strategy, model_obj.config.name, **kwargs)

        latency_budget_obj = self._parse_latency_budget(latency_budget, kwargs)

        all_results = []
        best_batch_size = 1
        best_metric = 0.0
        stop_reason = "max_batch_limit"
        constraint_type = None

        batch_size = kwargs.get("batch_size", 1)

        while batch_size <= max_batch:
            try:
                if mode == "training":
                    eval_kwargs = {
                        "batch_size": batch_size,
                        "seq_len": kwargs.get("seq_len", 2048),
                    }
                    result = self.evaluator.evaluate_training(model_obj, cluster_obj, strategy_obj, **eval_kwargs)

                    metric = result.samples_per_sec
                    memory_gb = result.memory_per_gpu_gb

                    result_data = {
                        "batch_size": batch_size,
                        "samples_per_sec": result.samples_per_sec,
                        "tokens_per_sec": result.tokens_per_sec,
                        "memory_gb": memory_gb,
                        "time_per_step_ms": result.time_per_step_sec * 1000,
                        "valid": True,
                    }

                else:
                    eval_kwargs = {
                        "batch_size": batch_size,
                        "prompt_len": kwargs.get("prompt_len", 512),
                        "generation_len": kwargs.get("generation_len", 128),
                    }
                    result = self.evaluator.evaluate_inference(model_obj, cluster_obj, strategy_obj, **eval_kwargs)

                    metric = result.decode_tokens_per_sec
                    memory_gb = result.memory_per_gpu_gb

                    ttft_ms = result.prefill_time_sec * 1000
                    tpot_ms = result.decode_time_per_step_sec * 1000
                    generation_len = latency_budget_obj.generation_len or kwargs.get("generation_len", 128)
                    total_latency_ms = ttft_ms + tpot_ms * generation_len

                    result_data = {
                        "batch_size": batch_size,
                        "prefill_tps": result.prefill_tokens_per_sec,
                        "decode_tps": result.decode_tokens_per_sec,
                        "ttft_ms": ttft_ms,
                        "tpot_ms": tpot_ms,
                        "total_latency_ms": total_latency_ms,
                        "memory_gb": memory_gb,
                        "valid": True,
                    }

                should_stop = False

                if memory_budget_gb is not None and memory_gb > memory_budget_gb:
                    stop_reason = "memory_budget_exceeded"
                    constraint_type = "memory"
                    should_stop = True

                if mode == "inference" and latency_budget_obj is not None:
                    latency_stop, latency_constraint = self._check_latency_budget(result_data, latency_budget_obj)
                    if latency_stop:
                        stop_reason = "latency_budget_exceeded"
                        constraint_type = latency_constraint
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

                if should_stop:
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

    def find_max_tps(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
        hardware: Union[str, Dict, Cluster],
        mode: str = "training",
        strategy_constraints: Optional[StrategyConstraintsForBatch] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Find maximum throughput by optimizing strategy and batch size.

        Args:
            model: Model specification
            hardware: Hardware specification
            mode: "training" or "inference"
            strategy_constraints: Strategy search constraints
            batch_size: Starting batch size
            **kwargs: Additional parameters

        Returns:
            Dictionary with best configuration and metrics

        Example:
            >>> result = optimizer.find_max_tps(
            ...     "llama-7b", "h100_8gpu",
            ...     strategy_constraints=StrategyConstraintsForBatch(max_tp=8)
            ... )
        """
        from .optimizer import StrategyOptimizer, StrategyConstraints, OptimizeObjective, SearchMethod

        strategy_constraints = strategy_constraints or StrategyConstraintsForBatch()

        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)

        max_gpus = cluster_obj.num_devices

        optimizer = StrategyOptimizer()

        search_constraints = StrategyConstraints(
            max_gpus=max_gpus,
            max_tp=strategy_constraints.max_tp,
            max_pp=strategy_constraints.max_pp,
            min_gpus=strategy_constraints.min_tp * strategy_constraints.min_pp,
        )

        objective = OptimizeObjective.THROUGHPUT

        strategy_result = optimizer.search_best_strategy(
            model_obj,
            cluster_obj,
            mode,
            search_constraints,
            objective,
            SearchMethod.GRID,
            batch_size=batch_size,
            **kwargs,
        )

        best_strategy = strategy_result.best_strategy

        max_batch_result = self.find_max_batch(
            model_obj,
            cluster_obj,
            best_strategy,
            mode,
            batch_step=kwargs.get("batch_step", 1),
            max_batch=kwargs.get("max_batch", 256),
            **kwargs,
        )

        if mode == "training":
            final_eval_kwargs = {
                "batch_size": max_batch_result.best_batch_size,
                "seq_len": kwargs.get("seq_len", 2048),
            }
            final_result = self.evaluator.evaluate_training(model_obj, cluster_obj, best_strategy, **final_eval_kwargs)

            return {
                "best_strategy": {
                    "tp": best_strategy.tp_degree,
                    "pp": best_strategy.pp_degree,
                    "dp": best_strategy.dp_degree,
                    "world_size": best_strategy.world_size,
                },
                "best_batch_size": max_batch_result.best_batch_size,
                "samples_per_sec": final_result.samples_per_sec,
                "tokens_per_sec": final_result.tokens_per_sec,
                "memory_gb": final_result.memory_per_gpu_gb,
                "strategy_search": strategy_result.all_results,
                "batch_search": max_batch_result.all_results,
            }
        else:
            final_eval_kwargs = {
                "batch_size": max_batch_result.best_batch_size,
                "prompt_len": kwargs.get("prompt_len", 512),
                "generation_len": kwargs.get("generation_len", 128),
            }
            final_result = self.evaluator.evaluate_inference(model_obj, cluster_obj, best_strategy, **final_eval_kwargs)

            return {
                "best_strategy": {
                    "tp": best_strategy.tp_degree,
                    "pp": best_strategy.pp_degree,
                    "dp": best_strategy.dp_degree,
                    "world_size": best_strategy.world_size,
                },
                "best_batch_size": max_batch_result.best_batch_size,
                "prefill_tps": final_result.prefill_tokens_per_sec,
                "decode_tps": final_result.decode_tokens_per_sec,
                "ttft_ms": final_result.prefill_time_sec * 1000,
                "memory_gb": final_result.memory_per_gpu_gb,
                "strategy_search": strategy_result.all_results,
                "batch_search": max_batch_result.all_results,
            }

    def estimate_memory_for_batch(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
        hardware: Union[str, Dict, Cluster],
        strategy: Union[str, Dict, StrategyConfig],
        batch_size: int,
        mode: str = "training",
        **kwargs,
    ) -> float:
        """Estimate memory usage for a specific batch size.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategy: Strategy specification
            batch_size: Batch size to estimate
            mode: "training" or "inference"
            **kwargs: Additional parameters

        Returns:
            Estimated memory in GB

        Example:
            >>> memory_gb = optimizer.estimate_memory_for_batch(
            ...     "llama-7b", "h100_8gpu", "tp4", batch_size=32
            ... )
        """
        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)
        strategy_obj = self.evaluator._resolve_strategy(strategy, model_obj.config.name, **kwargs)

        if mode == "training":
            eval_kwargs = {
                "batch_size": batch_size,
                "seq_len": kwargs.get("seq_len", 2048),
            }
            result = self.evaluator.evaluate_training(model_obj, cluster_obj, strategy_obj, **eval_kwargs)
        else:
            eval_kwargs = {
                "batch_size": batch_size,
                "prompt_len": kwargs.get("prompt_len", 512),
                "generation_len": kwargs.get("generation_len", 128),
            }
            result = self.evaluator.evaluate_inference(model_obj, cluster_obj, strategy_obj, **eval_kwargs)

        return result.memory_per_gpu_gb

    def compare_batch_sizes(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
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
            **kwargs: Additional parameters

        Returns:
            Comparison results

        Example:
            >>> result = optimizer.compare_batch_sizes(
            ...     "llama-7b", "h100_8gpu", "tp4",
            ...     [1, 8, 16, 32, 64]
            ... )
        """
        model_obj = self.evaluator._resolve_model(model, **kwargs)
        cluster_obj = self.evaluator._resolve_hardware(hardware, **kwargs)
        strategy_obj = self.evaluator._resolve_strategy(strategy, model_obj.config.name, **kwargs)

        all_results = []
        for batch_size in batch_sizes:
            try:
                if mode == "training":
                    eval_kwargs = {
                        "batch_size": batch_size,
                        "seq_len": kwargs.get("seq_len", 2048),
                    }
                    result = self.evaluator.evaluate_training(model_obj, cluster_obj, strategy_obj, **eval_kwargs)

                    result_data = {
                        "batch_size": batch_size,
                        "samples_per_sec": result.samples_per_sec,
                        "tokens_per_sec": result.tokens_per_sec,
                        "memory_gb": result.memory_per_gpu_gb,
                        "time_per_step_ms": result.time_per_step_sec * 1000,
                        "valid": True,
                    }
                else:
                    eval_kwargs = {
                        "batch_size": batch_size,
                        "prompt_len": kwargs.get("prompt_len", 512),
                        "generation_len": kwargs.get("generation_len", 128),
                    }
                    result = self.evaluator.evaluate_inference(model_obj, cluster_obj, strategy_obj, **eval_kwargs)

                    result_data = {
                        "batch_size": batch_size,
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
                        "batch_size": batch_size,
                        "valid": False,
                        "error": str(e),
                    }
                )

        return {
            "model": model_obj.config.name,
            "hardware": getattr(cluster_obj, "name", f"{cluster_obj.num_devices}-gpu cluster"),
            "strategy": strategy_obj.model_name,
            "mode": mode,
            "results": all_results,
        }

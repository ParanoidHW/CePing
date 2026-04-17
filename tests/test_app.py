"""Tests for Application Layer (Evaluator, StrategyOptimizer, BatchOptimizer)."""

import pytest

from llm_perf.app import Evaluator, StrategyOptimizer, BatchOptimizer
from llm_perf.app.optimizer import StrategyConstraints, OptimizeObjective, SearchMethod
from llm_perf.app.batch_optimizer import StrategyConstraintsForBatch


class TestEvaluator:
    """Test Evaluator convenience API."""

    def test_evaluator_init(self):
        """Test Evaluator initialization."""
        evaluator = Evaluator()
        assert evaluator is not None
        assert evaluator._model_cache == {}
        assert evaluator._cluster_cache == {}

    def test_list_available_presets(self):
        """Test listing available presets."""
        evaluator = Evaluator()
        presets = evaluator.list_available_presets()

        assert "models" in presets
        assert "hardware" in presets
        assert "strategies" in presets
        assert "devices" in presets

        assert "llama-7b" in presets["models"]
        assert "h100_8gpu" in presets["hardware"]
        assert "tp1" in presets["strategies"]

    def test_resolve_model_from_preset(self):
        """Test resolving model from preset name."""
        evaluator = Evaluator()
        model = evaluator._resolve_model("llama-7b")

        assert model is not None
        assert model.config.name == "llama-7b"
        assert model.config.hidden_size == 4096
        assert model.config.num_layers == 32

    def test_resolve_model_from_dict(self):
        """Test resolving model from config dict."""
        evaluator = Evaluator()
        model_config = {
            "type": "llama",
            "name": "test-model",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
        }

        model = evaluator._resolve_model(model_config)
        assert model is not None
        assert model.config.name == "test-model"

    def test_resolve_hardware_from_preset(self):
        """Test resolving hardware from preset name."""
        evaluator = Evaluator()
        cluster = evaluator._resolve_hardware("h100_8gpu")

        assert cluster is not None
        assert cluster.num_devices == 8

    def test_resolve_strategy_from_preset(self):
        """Test resolving strategy from preset name."""
        evaluator = Evaluator()
        strategy = evaluator._resolve_strategy("tp4", "llama-7b")

        assert strategy is not None
        assert strategy.tp_degree == 4
        assert strategy.pp_degree == 1
        assert strategy.dp_degree == 1
        assert strategy.model_name == "llama-7b"

    def test_resolve_strategy_from_dict(self):
        """Test resolving strategy from config dict."""
        evaluator = Evaluator()
        strategy_config = {"tp": 8, "pp": 2, "dp": 1}

        strategy = evaluator._resolve_strategy(strategy_config, "test-model")
        assert strategy.tp_degree == 8
        assert strategy.pp_degree == 2
        assert strategy.dp_degree == 1

    def test_model_caching(self):
        """Test that models are cached."""
        evaluator = Evaluator()

        model1 = evaluator._resolve_model("llama-7b")
        model2 = evaluator._resolve_model("llama-7b")

        assert model1 is model2
        assert "llama-7b" in evaluator._model_cache

    def test_hardware_caching(self):
        """Test that hardware configs are cached."""
        evaluator = Evaluator()

        cluster1 = evaluator._resolve_hardware("h100_8gpu")
        cluster2 = evaluator._resolve_hardware("h100_8gpu")

        assert cluster1 is cluster2
        assert "h100_8gpu" in evaluator._cluster_cache

    def test_evaluate_training_basic(self):
        """Test basic training evaluation."""
        evaluator = Evaluator()

        result = evaluator.evaluate_training(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            batch_size=32,
            seq_len=2048,
        )

        assert result is not None
        assert result.samples_per_sec > 0
        assert result.tokens_per_sec > 0
        assert result.memory_per_gpu_gb > 0

    def test_evaluate_training_with_tp(self):
        """Test training evaluation with tensor parallelism."""
        evaluator = Evaluator()

        result = evaluator.evaluate_training(
            "llama-7b",
            "h100_8gpu",
            "tp4",
            batch_size=32,
            seq_len=2048,
        )

        assert result is not None
        assert result.samples_per_sec > 0

    def test_evaluate_inference_basic(self):
        """Test basic inference evaluation."""
        evaluator = Evaluator()

        result = evaluator.evaluate_inference(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            batch_size=1,
            prompt_len=512,
            generation_len=128,
        )

        assert result is not None
        assert result.prefill_time_sec > 0
        assert result.decode_time_per_step_sec > 0
        assert result.prefill_tokens_per_sec > 0
        assert result.decode_tokens_per_sec > 0

    def test_evaluate_inference_with_tp(self):
        """Test inference evaluation with tensor parallelism."""
        evaluator = Evaluator()

        result = evaluator.evaluate_inference(
            "llama-7b",
            "h100_8gpu",
            "tp4",
            batch_size=1,
        )

        assert result is not None
        assert result.prefill_tokens_per_sec > 0
        assert result.decode_tokens_per_sec > 0

    def test_compare_strategies(self):
        """Test comparing multiple strategies."""
        evaluator = Evaluator()

        result = evaluator.compare_strategies(
            "llama-7b",
            "h100_8gpu",
            ["tp1", "tp2", "tp4"],
            mode="training",
            batch_size=32,
        )

        assert result is not None
        assert "model" in result
        assert "hardware" in result
        assert "results" in result
        assert len(result["results"]) == 3

    def test_strategy_fits_hardware_validation(self):
        """Test that strategy world_size must fit hardware."""
        evaluator = Evaluator()

        with pytest.raises(ValueError, match="requires.*devices"):
            evaluator.evaluate_training(
                "llama-7b",
                "h100_8gpu",
                {"tp": 16, "pp": 2, "dp": 4},
                batch_size=32,
            )

    def test_moe_model_evaluation(self):
        """Test evaluation with MoE model."""
        evaluator = Evaluator()

        result = evaluator.evaluate_training(
            "mixtral-8x7b",
            "h100_8gpu",
            {"tp": 1, "ep": 4},
            batch_size=16,
        )

        assert result is not None
        assert result.samples_per_sec > 0


class TestStrategyOptimizer:
    """Test StrategyOptimizer strategy search."""

    def test_optimizer_init(self):
        """Test StrategyOptimizer initialization."""
        optimizer = StrategyOptimizer()
        assert optimizer is not None
        assert optimizer.evaluator is not None

    def test_grid_search_generates_strategies(self):
        """Test that grid search generates feasible strategies."""
        optimizer = StrategyOptimizer()
        evaluator = Evaluator()

        model = evaluator._resolve_model("llama-7b")
        cluster = evaluator._resolve_hardware("h100_8gpu")

        strategies = optimizer._grid_search_strategies(model, cluster, max_gpus=8, constraints=StrategyConstraints())

        assert len(strategies) > 0

        for strategy in strategies:
            assert strategy.world_size <= 8
            assert strategy.world_size <= cluster.num_devices

    def test_grid_search_respects_max_gpus(self):
        """Test that grid search respects max_gpus constraint."""
        optimizer = StrategyOptimizer()
        evaluator = Evaluator()

        model = evaluator._resolve_model("llama-7b")
        cluster = evaluator._resolve_hardware("h100_8gpu")

        strategies = optimizer._grid_search_strategies(
            model, cluster, max_gpus=4, constraints=StrategyConstraints(max_gpus=4)
        )

        for strategy in strategies:
            assert strategy.world_size <= 4

    def test_grid_search_respects_model_divisibility(self):
        """Test that grid search respects model divisibility."""
        optimizer = StrategyOptimizer()
        evaluator = Evaluator()

        model = evaluator._resolve_model("llama-7b")
        cluster = evaluator._resolve_hardware("h100_8gpu")

        strategies = optimizer._grid_search_strategies(model, cluster, max_gpus=8, constraints=StrategyConstraints())

        hidden_size = model.config.hidden_size
        num_layers = model.config.num_layers

        for strategy in strategies:
            if strategy.tp_degree > 1:
                assert hidden_size % strategy.tp_degree == 0
            if strategy.pp_degree > 1:
                assert num_layers % strategy.pp_degree == 0

    def test_search_best_strategy_throughput(self):
        """Test searching for best strategy by throughput."""
        optimizer = StrategyOptimizer()

        result = optimizer.search_best_strategy(
            "llama-7b",
            "h100_8gpu",
            mode="training",
            constraints=StrategyConstraints(max_gpus=8),
            objective=OptimizeObjective.THROUGHPUT,
            method=SearchMethod.GRID,
            batch_size=32,
        )

        assert result is not None
        assert result.best_strategy is not None
        assert result.best_metric > 0
        assert result.method == "grid"
        assert len(result.all_results) > 0

    def test_search_best_strategy_memory(self):
        """Test searching for best strategy by memory."""
        optimizer = StrategyOptimizer()

        result = optimizer.search_best_strategy(
            "llama-7b",
            "h100_8gpu",
            mode="training",
            constraints=StrategyConstraints(max_memory_gb=80, max_gpus=8),
            objective=OptimizeObjective.MEMORY,
            method=SearchMethod.GRID,
            batch_size=8,
        )

        assert result is not None
        assert result.best_strategy is not None

    def test_search_best_strategy_inference(self):
        """Test searching for best strategy for inference."""
        optimizer = StrategyOptimizer()

        result = optimizer.search_best_strategy(
            "llama-7b",
            "h100_8gpu",
            mode="inference",
            constraints=StrategyConstraints(max_gpus=8),
            objective=OptimizeObjective.THROUGHPUT,
            method=SearchMethod.GRID,
            batch_size=1,
        )

        assert result is not None
        assert result.best_strategy is not None

    def test_compare_specific_strategies(self):
        """Test comparing specific strategies."""
        optimizer = StrategyOptimizer()

        result = optimizer.compare_strategies(
            "llama-7b",
            "h100_8gpu",
            ["tp1", "tp2", "tp4", "tp8"],
            mode="training",
            batch_size=32,
        )

        assert result is not None
        assert len(result["results"]) == 4

        valid_results = [r for r in result["results"] if r.get("valid")]
        assert len(valid_results) > 0

    def test_strategy_to_str(self):
        """Test strategy to string conversion."""
        optimizer = StrategyOptimizer()

        from llm_perf.strategy.base import StrategyConfig

        strategy = StrategyConfig(tp_degree=4, pp_degree=2, dp_degree=1)
        str_repr = optimizer._strategy_to_str(strategy)

        assert "tp4" in str_repr
        assert "pp2" in str_repr

    def test_constraints_require_tp(self):
        """Test constraint requiring TP."""
        optimizer = StrategyOptimizer()
        evaluator = Evaluator()

        model = evaluator._resolve_model("llama-7b")
        cluster = evaluator._resolve_hardware("h100_8gpu")

        strategies = optimizer._grid_search_strategies(
            model, cluster, max_gpus=8, constraints=StrategyConstraints(require_tp=True)
        )

        for strategy in strategies:
            assert strategy.tp_degree > 1

    def test_greedy_search_returns_strategies(self):
        """Test that greedy search returns strategies."""
        optimizer = StrategyOptimizer()
        evaluator = Evaluator()

        model = evaluator._resolve_model("llama-7b")
        cluster = evaluator._resolve_hardware("h100_8gpu")

        strategies = optimizer._greedy_search_strategies(
            model, cluster, max_gpus=8, constraints=StrategyConstraints(), objective=OptimizeObjective.THROUGHPUT
        )

        assert len(strategies) > 0

    def test_latency_budget_constraint(self):
        """Test latency budget constraint for inference."""
        optimizer = StrategyOptimizer()

        result = optimizer.search_best_strategy(
            "llama-7b",
            "h100_8gpu",
            mode="inference",
            constraints=StrategyConstraints(latency_budget_ms=100),
            objective=OptimizeObjective.THROUGHPUT,
            method=SearchMethod.GRID,
            batch_size=1,
        )

        assert result is not None


class TestBatchOptimizer:
    """Test BatchOptimizer batch size search."""

    def test_batch_optimizer_init(self):
        """Test BatchOptimizer initialization."""
        optimizer = BatchOptimizer()
        assert optimizer is not None
        assert optimizer.evaluator is not None

    def test_find_max_batch_basic(self):
        """Test basic batch size search."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="training",
            batch_step=8,
            max_batch=64,
        )

        assert result is not None
        assert result.best_batch_size > 0
        assert result.best_batch_size <= 64
        assert len(result.all_results) > 0

    def test_find_max_batch_memory_constraint(self):
        """Test batch search with memory constraint."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp4",
            mode="training",
            memory_budget_gb=80,
            batch_step=4,
            max_batch=128,
        )

        assert result is not None
        assert result.best_batch_size > 0

        for eval_result in result.all_results:
            if eval_result.get("valid") and eval_result["batch_size"] <= result.best_batch_size:
                assert eval_result["memory_gb"] <= 80

    def test_find_max_batch_latency_constraint(self):
        """Test batch search with latency constraint."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="inference",
            latency_budget_ms=500,
            batch_step=1,
            max_batch=32,
        )

        assert result is not None
        assert result.best_batch_size > 0

    def test_find_max_batch_stop_reason(self):
        """Test that stop reason is provided."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="training",
            memory_budget_gb=40,
            batch_step=8,
            max_batch=256,
        )

        assert result.reason in [
            "memory_budget_exceeded",
            "latency_budget_exceeded",
            "max_batch_limit",
            "target_tps_not_met",
            "evaluation_error",
        ]

    def test_estimate_memory_for_batch(self):
        """Test memory estimation for specific batch."""
        optimizer = BatchOptimizer()

        memory_gb = optimizer.estimate_memory_for_batch(
            "llama-7b",
            "h100_8gpu",
            "tp4",
            batch_size=32,
            mode="training",
        )

        assert memory_gb > 0

    def test_compare_batch_sizes(self):
        """Test comparing different batch sizes."""
        optimizer = BatchOptimizer()

        result = optimizer.compare_batch_sizes(
            "llama-7b",
            "h100_8gpu",
            "tp4",
            batch_sizes=[1, 8, 16, 32],
            mode="training",
        )

        assert result is not None
        assert "results" in result
        assert len(result["results"]) == 4

    def test_find_max_tps(self):
        """Test finding maximum TPS by optimizing strategy and batch."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_tps(
            "llama-7b",
            "h100_8gpu",
            mode="training",
            strategy_constraints=StrategyConstraintsForBatch(max_tp=8),
            batch_size=32,
        )

        assert result is not None
        assert "best_strategy" in result
        assert "best_batch_size" in result
        assert "samples_per_sec" in result

    def test_batch_increases_throughput(self):
        """Test that larger batch increases throughput."""
        optimizer = BatchOptimizer()

        result = optimizer.compare_batch_sizes(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            batch_sizes=[8, 16, 32],
            mode="training",
        )

        valid_results = [r for r in result["results"] if r.get("valid")]
        if len(valid_results) >= 2:
            samples_per_sec_values = [r["samples_per_sec"] for r in valid_results]
            assert samples_per_sec_values[-1] >= samples_per_sec_values[0]

    def test_batch_memory_increases(self):
        """Test that larger batch increases memory."""
        optimizer = BatchOptimizer()

        result = optimizer.compare_batch_sizes(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            batch_sizes=[8, 16, 32],
            mode="training",
        )

        valid_results = [r for r in result["results"] if r.get("valid")]
        if len(valid_results) >= 2:
            memory_values = [r["memory_gb"] for r in valid_results]
            assert memory_values[-1] >= memory_values[0]

    def test_latency_budget_ttft_only(self):
        """Test TTFT budget constraint only."""
        from llm_perf.app import LatencyBudget

        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="inference",
            latency_budget=LatencyBudget(ttft_budget_ms=100.0),
            batch_step=1,
            max_batch=64,
            prompt_len=512,
            generation_len=128,
        )

        assert result.best_batch_size > 0
        assert result.constraint_type in ["ttft", None]

    def test_latency_budget_tpot_only(self):
        """Test TPOT budget constraint only."""
        from llm_perf.app import LatencyBudget

        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="inference",
            latency_budget=LatencyBudget(tpot_budget_ms=5.0),
            batch_step=1,
            max_batch=64,
            prompt_len=512,
            generation_len=128,
        )

        assert result.best_batch_size > 0
        assert result.constraint_type in ["tpot", None]

    def test_latency_budget_combined_ttft_tpot(self):
        """Test combined TTFT + TPOT budget constraints."""
        from llm_perf.app import LatencyBudget

        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="inference",
            latency_budget=LatencyBudget(ttft_budget_ms=100.0, tpot_budget_ms=3.0),
            batch_step=1,
            max_batch=64,
            prompt_len=512,
            generation_len=128,
        )

        assert result.best_batch_size > 0
        assert result.constraint_type in ["ttft", "tpot", None]

    def test_latency_budget_total_latency(self):
        """Test total latency budget constraint."""
        from llm_perf.app import LatencyBudget

        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="inference",
            latency_budget=LatencyBudget(
                total_latency_budget_ms=500.0,
                generation_len=128,
            ),
            batch_step=1,
            max_batch=64,
            prompt_len=512,
            generation_len=128,
        )

        assert result.best_batch_size > 0
        assert result.constraint_type in ["total_latency", None]

    def test_latency_budget_legacy_float(self):
        """Test legacy float latency budget (backward compatibility)."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="inference",
            latency_budget=100.0,
            batch_step=1,
            max_batch=64,
            prompt_len=512,
            generation_len=128,
        )

        assert result.best_batch_size > 0
        assert result.constraint_type in ["ttft", None]

    def test_latency_budget_result_data_structure(self):
        """Test that result data includes TTFT/TPOT/Total latency."""
        from llm_perf.app import LatencyBudget

        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="inference",
            memory_budget_gb=80,
            batch_step=8,
            max_batch=64,
            prompt_len=512,
            generation_len=128,
        )

        assert len(result.all_results) > 0
        valid_result = result.all_results[0]
        if valid_result.get("valid"):
            assert "ttft_ms" in valid_result
            assert "tpot_ms" in valid_result
            assert "total_latency_ms" in valid_result


class TestIntegration:
    """Integration tests combining Evaluator, StrategyOptimizer, BatchOptimizer."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        evaluator = Evaluator()

        result = evaluator.evaluate_training(
            "llama-7b",
            "h100_8gpu",
            "tp4",
            batch_size=32,
        )

        assert result.samples_per_sec > 0

        optimizer = StrategyOptimizer()
        strategy_result = optimizer.search_best_strategy(
            "llama-7b",
            "h100_8gpu",
            mode="training",
            constraints=StrategyConstraints(max_gpus=8),
            objective=OptimizeObjective.THROUGHPUT,
            batch_size=32,
        )

        assert strategy_result.best_strategy is not None

        batch_optimizer = BatchOptimizer()
        batch_result = batch_optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            strategy_result.best_strategy,
            mode="training",
            memory_budget_gb=80,
        )

        assert batch_result.best_batch_size > 0

    def test_different_models(self):
        """Test evaluation with different models."""
        evaluator = Evaluator()

        models = ["llama-7b", "llama-13b"]
        results = []

        for model in models:
            result = evaluator.evaluate_training(
                model,
                "h100_8gpu",
                "tp1",
                batch_size=32,
            )
            results.append(result)

        assert all(r.samples_per_sec > 0 for r in results)

    def test_different_hardware(self):
        """Test evaluation with different hardware."""
        evaluator = Evaluator()

        hardware_presets = ["a100_8gpu", "h100_8gpu"]
        results = []

        for hw in hardware_presets:
            result = evaluator.evaluate_training(
                "llama-7b",
                hw,
                "tp1",
                batch_size=32,
            )
            results.append(result)

        assert all(r.samples_per_sec > 0 for r in results)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_model_preset(self):
        """Test handling of invalid model preset."""
        evaluator = Evaluator()

        with pytest.raises(ValueError):
            evaluator._resolve_model("invalid-model-name")

    def test_invalid_hardware_preset(self):
        """Test handling of invalid hardware preset."""
        evaluator = Evaluator()

        with pytest.raises(ValueError):
            evaluator._resolve_hardware("invalid-hardware")

    def test_invalid_strategy_preset(self):
        """Test handling of invalid strategy preset."""
        evaluator = Evaluator()

        with pytest.raises(ValueError):
            evaluator._resolve_strategy("invalid-strategy", "test")

    def test_empty_strategy_comparison(self):
        """Test comparing empty strategy list."""
        optimizer = StrategyOptimizer()

        result = optimizer.compare_strategies(
            "llama-7b",
            "h100_8gpu",
            [],
            mode="training",
        )

        assert len(result["results"]) == 0

    def test_batch_size_zero(self):
        """Test handling of zero batch size."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "h100_8gpu",
            "tp1",
            mode="training",
            batch_step=1,
            max_batch=0,
        )

        assert result.best_batch_size >= 1

    def test_constraints_extreme_values(self):
        """Test handling of extreme constraint values."""
        optimizer = StrategyOptimizer()

        result = optimizer.search_best_strategy(
            "llama-7b",
            "h100_8gpu",
            mode="training",
            constraints=StrategyConstraints(
                max_memory_gb=200,
                max_gpus=8,
            ),
            objective=OptimizeObjective.MEMORY,
            batch_size=1,
        )

        assert result is not None

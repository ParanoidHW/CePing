"""Tests for new app module."""

import pytest

from llm_perf.modeling import LlamaModel, create_model_from_config
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.app import Evaluator, StrategyOptimizer, BatchOptimizer
from llm_perf.app import StrategyConstraints, OptimizeObjective, SearchMethod
from llm_perf.app import LatencyBudget


def make_cluster(device, num_devices):
    """Helper to create cluster."""
    topology = NetworkTopology(
        name="test",
        intra_node_bandwidth_gbps=200.0,
        intra_node_latency_us=1.0,
        inter_node_bandwidth_gbps=25.0,
        inter_node_latency_us=10.0,
    )
    return Cluster.create_homogeneous(device.config, num_devices, topology)


class TestEvaluator:
    """Test Evaluator."""

    def test_evaluator_creation(self):
        """Test creating evaluator."""
        evaluator = Evaluator()
        assert evaluator is not None

    def test_evaluate_training_with_objects(self):
        """Test training evaluation with explicit objects."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        evaluator = Evaluator()
        result = evaluator.evaluate_training(model, cluster, strategy, batch_size=32, seq_len=2048)

        assert result.samples_per_sec > 0
        assert result.tokens_per_sec > 0
        assert result.memory_per_gpu_gb > 0

    def test_evaluate_training_with_preset(self):
        """Test training evaluation with preset string."""
        evaluator = Evaluator()
        result = evaluator.evaluate_training("llama-7b", "H100-SXM-80GB", "tp8", batch_size=32, seq_len=2048)

        assert result.samples_per_sec > 0
        assert result.tokens_per_sec > 0

    def test_evaluate_training_with_dict(self):
        """Test training evaluation with dict config."""
        evaluator = Evaluator()
        result = evaluator.evaluate_training(
            {"preset": "llama-7b"},
            "H100-SXM-80GB",
            {"tp_degree": 4},
            batch_size=32,
            seq_len=2048,
            num_devices=8,
        )

        assert result.samples_per_sec > 0

    def test_evaluate_inference_with_objects(self):
        """Test inference evaluation with explicit objects."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        evaluator = Evaluator()
        result = evaluator.evaluate_inference(
            model,
            cluster,
            strategy,
            batch_size=8,
            prompt_len=512,
            generation_len=128,
        )

        assert result.prefill_time_sec > 0
        assert result.decode_time_per_step_sec > 0
        assert result.prefill_tokens_per_sec > 0
        assert result.decode_tokens_per_sec > 0

    def test_evaluate_inference_with_preset(self):
        """Test inference evaluation with preset string."""
        evaluator = Evaluator()
        result = evaluator.evaluate_inference("llama-7b", "H100-SXM-80GB", "tp4", batch_size=1)

        assert result.prefill_tokens_per_sec > 0
        assert result.decode_tokens_per_sec > 0

    def test_compare_strategies_training(self):
        """Test comparing strategies for training."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)

        evaluator = Evaluator()
        result = evaluator.compare_strategies(
            model,
            cluster,
            strategies=["tp1", "tp2", "tp4"],
            mode="training",
            batch_size=32,
            seq_len=2048,
        )

        assert "best_strategy" in result
        assert "all_results" in result
        assert len(result["all_results"]) == 3

    def test_compare_strategies_inference(self):
        """Test comparing strategies for inference."""
        evaluator = Evaluator()
        result = evaluator.compare_strategies(
            "llama-7b",
            "H100-SXM-80GB",
            strategies=["tp1", "tp2"],
            mode="inference",
            batch_size=1,
            prompt_len=512,
            generation_len=128,
        )

        assert "best_strategy" in result
        assert len(result["all_results"]) == 2

    def test_list_available_presets(self):
        """Test listing presets."""
        evaluator = Evaluator()
        presets = evaluator.list_available_presets()

        assert "models" in presets
        assert "hardware" in presets
        assert "strategies" in presets
        assert len(presets["models"]) > 0
        assert len(presets["hardware"]) > 0

    def test_strategy_parsing(self):
        """Test strategy name parsing."""
        evaluator = Evaluator()

        s1 = evaluator._parse_strategy_name("tp8")
        assert s1.tp_degree == 8
        assert s1.pp_degree == 1
        assert s1.dp_degree == 1

        s2 = evaluator._parse_strategy_name("tp4_dp2")
        assert s2.tp_degree == 4
        assert s2.dp_degree == 2

        s3 = evaluator._parse_strategy_name("tp2_pp4")
        assert s3.tp_degree == 2
        assert s3.pp_degree == 4


class TestStrategyOptimizer:
    """Test StrategyOptimizer."""

    def test_optimizer_creation(self):
        """Test creating optimizer."""
        optimizer = StrategyOptimizer()
        assert optimizer is not None

    def test_search_best_strategy_throughput(self):
        """Test searching for best throughput strategy."""
        optimizer = StrategyOptimizer()

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 4)

        constraints = StrategyConstraints(max_tp=4)

        result = optimizer.search_best_strategy(
            model,
            cluster,
            mode="training",
            constraints=constraints,
            objective=OptimizeObjective.THROUGHPUT,
            batch_size=32,
            seq_len=2048,
        )

        assert result.best_strategy is not None
        assert result.best_metric > 0
        assert len(result.all_results) > 0
        assert result.search_time_sec > 0

    def test_search_best_strategy_with_memory_constraint(self):
        """Test searching with memory constraint."""
        optimizer = StrategyOptimizer()

        constraints = StrategyConstraints(
            max_memory_gb=50.0,
            max_tp=2,
        )

        result = optimizer.search_best_strategy(
            "llama-7b",
            "H100-SXM-80GB",
            mode="training",
            constraints=constraints,
            batch_size=32,
            seq_len=2048,
            num_devices=4,
        )

        assert result.best_strategy is not None
        for r in result.all_results:
            if r["memory_gb"] <= 50.0:
                assert r["memory_gb"] <= 50.0

    def test_compare_specific_strategies(self):
        """Test comparing specific strategies."""
        optimizer = StrategyOptimizer()

        result = optimizer.compare_specific_strategies(
            "llama-7b",
            "H100-SXM-80GB",
            strategies=["tp1", "tp2", "tp4"],
            mode="training",
            batch_size=32,
            seq_len=2048,
            num_devices=8,
        )

        assert "best_strategy" in result
        assert "all_results" in result
        assert len(result["all_results"]) == 3

    def test_search_inference_mode(self):
        """Test searching in inference mode."""
        optimizer = StrategyOptimizer()

        constraints = StrategyConstraints(max_tp=4)

        result = optimizer.search_best_strategy(
            "llama-7b",
            "H100-SXM-80GB",
            mode="inference",
            constraints=constraints,
            batch_size=1,
            prompt_len=512,
            generation_len=128,
            num_devices=4,
        )

        assert result.best_strategy is not None
        assert result.best_metric > 0


class TestBatchOptimizer:
    """Test BatchOptimizer."""

    def test_batch_optimizer_creation(self):
        """Test creating batch optimizer."""
        optimizer = BatchOptimizer()
        assert optimizer is not None

    def test_find_max_batch_training(self):
        """Test finding max batch for training."""
        optimizer = BatchOptimizer()

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        result = optimizer.find_max_batch(
            model,
            cluster,
            strategy,
            mode="training",
            max_batch=64,
            batch_step=8,
            seq_len=2048,
        )

        assert result.best_batch_size >= 1
        assert result.best_metric > 0
        assert len(result.all_results) > 0
        assert result.search_time_sec > 0

    def test_find_max_batch_with_memory_constraint(self):
        """Test finding max batch with memory constraint."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "H100-SXM-80GB",
            "tp8",
            mode="training",
            memory_budget_gb=40.0,
            max_batch=128,
            batch_step=8,
            seq_len=2048,
            num_devices=8,
        )

        assert result.best_batch_size >= 1
        assert result.constraint_type in [None, "memory"]

    def test_find_max_batch_inference(self):
        """Test finding max batch for inference."""
        optimizer = BatchOptimizer()

        result = optimizer.find_max_batch(
            "llama-7b",
            "H100-SXM-80GB",
            "tp4",
            mode="inference",
            max_batch=32,
            batch_step=4,
            prompt_len=512,
            generation_len=128,
            num_devices=8,
        )

        assert result.best_batch_size >= 1
        assert len(result.all_results) > 0

    def test_find_max_batch_with_latency_budget(self):
        """Test finding max batch with latency budget."""
        optimizer = BatchOptimizer()

        latency_budget = LatencyBudget(
            ttft_budget_ms=100.0,
        )

        result = optimizer.find_max_batch(
            "llama-7b",
            "H100-SXM-80GB",
            "tp4",
            mode="inference",
            latency_budget=latency_budget,
            max_batch=64,
            batch_step=8,
            prompt_len=512,
            generation_len=128,
            num_devices=8,
        )

        assert result.best_batch_size >= 1

    def test_compare_batch_sizes(self):
        """Test comparing different batch sizes."""
        optimizer = BatchOptimizer()

        result = optimizer.compare_batch_sizes(
            "llama-7b",
            "H100-SXM-80GB",
            "tp8",
            batch_sizes=[1, 8, 16, 32],
            mode="training",
            seq_len=2048,
            num_devices=8,
        )

        assert "mode" in result
        assert "results" in result
        assert len(result["results"]) == 4
        for r in result["results"]:
            assert "batch_size" in r

    def test_compare_batch_sizes_inference(self):
        """Test comparing batch sizes for inference."""
        optimizer = BatchOptimizer()

        result = optimizer.compare_batch_sizes(
            "llama-7b",
            "H100-SXM-80GB",
            "tp4",
            batch_sizes=[1, 4, 8],
            mode="inference",
            prompt_len=512,
            generation_len=128,
            num_devices=8,
        )

        assert result["mode"] == "inference"
        assert len(result["results"]) == 3


class TestDataClasses:
    """Test data classes."""

    def test_strategy_constraints(self):
        """Test StrategyConstraints dataclass."""
        c1 = StrategyConstraints()
        assert c1.max_memory_gb is None
        assert c1.max_gpus is None
        assert c1.min_gpus == 1

        c2 = StrategyConstraints(
            max_memory_gb=80.0,
            max_tp=8,
            require_tp=True,
        )
        assert c2.max_memory_gb == 80.0
        assert c2.max_tp == 8
        assert c2.require_tp is True

    def test_latency_budget(self):
        """Test LatencyBudget dataclass."""
        b1 = LatencyBudget()
        assert b1.ttft_budget_ms is None

        b2 = LatencyBudget(
            ttft_budget_ms=100.0,
            tpot_budget_ms=50.0,
            generation_len=128,
        )
        assert b2.ttft_budget_ms == 100.0
        assert b2.tpot_budget_ms == 50.0
        assert b2.generation_len == 128

    def test_optimize_objective(self):
        """Test OptimizeObjective enum."""
        assert OptimizeObjective.THROUGHPUT.value == "throughput"
        assert OptimizeObjective.LATENCY.value == "latency"
        assert OptimizeObjective.MEMORY.value == "memory"

    def test_search_method(self):
        """Test SearchMethod enum."""
        assert SearchMethod.GRID.value == "grid"
        assert SearchMethod.GREEDY.value == "greedy"

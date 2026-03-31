"""Tests for strategy classes."""

import unittest
from llm_perf.strategy.base import (
    ParallelType,
    ParallelConfig,
    StrategyConfig,
    ParallelStrategy,
)
from llm_perf.strategy.planner import StrategyConstraints, StrategyPlanner


class TestParallelType(unittest.TestCase):
    """Test ParallelType enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        self.assertEqual(ParallelType.TENSOR.value, "tp")
        self.assertEqual(ParallelType.PIPELINE.value, "pp")
        self.assertEqual(ParallelType.DATA.value, "dp")
        self.assertEqual(ParallelType.EXPERT.value, "ep")


class TestParallelConfig(unittest.TestCase):
    """Test ParallelConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = ParallelConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.degree, 1)
        self.assertEqual(config.options, {})


class TestStrategyConfig(unittest.TestCase):
    """Test StrategyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StrategyConfig()
        self.assertEqual(config.tp_degree, 1)
        self.assertEqual(config.pp_degree, 1)
        self.assertEqual(config.dp_degree, 1)
        self.assertEqual(config.ep_degree, 1)
        self.assertEqual(config.pipeline_schedule, "1f1b")
        self.assertFalse(config.activation_checkpointing)

    def test_world_size(self):
        """Test world size calculation."""
        config = StrategyConfig(tp_degree=4, pp_degree=2, dp_degree=2)
        self.assertEqual(config.world_size, 16)

    def test_world_size_with_ep(self):
        """Test world size with expert parallelism."""
        config = StrategyConfig(tp_degree=2, pp_degree=2, dp_degree=2, ep_degree=4)
        self.assertEqual(config.world_size, 32)

    def test_to_dict(self):
        """Test serialization to dict."""
        config = StrategyConfig(
            model_name="llama-7b",
            tp_degree=4,
            dp_degree=2,
            zero_stage=1,
        )
        data = config.to_dict()
        self.assertEqual(data["model_name"], "llama-7b")
        self.assertEqual(data["parallelism"]["tp"], 4)
        self.assertEqual(data["parallelism"]["dp"], 2)
        self.assertEqual(data["world_size"], 8)
        self.assertEqual(data["optimization"]["zero_stage"], 1)

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "model_name": "test-model",
            "parallelism": {"tp": 8, "pp": 2, "dp": 4},
            "scheduling": {"pipeline_schedule": "gpipe", "micro_batch_size": 2},
            "optimization": {"activation_checkpointing": True, "zero_stage": 2},
        }
        config = StrategyConfig.from_dict(data)
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.tp_degree, 8)
        self.assertEqual(config.pp_degree, 2)
        self.assertEqual(config.dp_degree, 4)
        self.assertEqual(config.pipeline_schedule, "gpipe")
        self.assertEqual(config.micro_batch_size, 2)
        self.assertTrue(config.activation_checkpointing)
        self.assertEqual(config.zero_stage, 2)

    def test_from_dict_defaults(self):
        """Test that missing keys use defaults."""
        config = StrategyConfig.from_dict({})
        self.assertEqual(config.tp_degree, 1)
        self.assertEqual(config.pp_degree, 1)
        self.assertEqual(config.dp_degree, 1)
        self.assertEqual(config.pipeline_schedule, "1f1b")


class TestParallelStrategy(unittest.TestCase):
    """Test ParallelStrategy class."""

    def setUp(self):
        self.config = StrategyConfig(tp_degree=4, pp_degree=2, dp_degree=2)
        self.strategy = ParallelStrategy(self.config)

    def test_is_tp_enabled(self):
        """Test TP enabled check."""
        self.assertTrue(self.strategy.is_tp_enabled())
        strategy = ParallelStrategy(StrategyConfig())
        self.assertFalse(strategy.is_tp_enabled())

    def test_is_pp_enabled(self):
        """Test PP enabled check."""
        self.assertTrue(self.strategy.is_pp_enabled())
        strategy = ParallelStrategy(StrategyConfig())
        self.assertFalse(strategy.is_pp_enabled())

    def test_is_dp_enabled(self):
        """Test DP enabled check."""
        self.assertTrue(self.strategy.is_dp_enabled())
        strategy = ParallelStrategy(StrategyConfig())
        self.assertFalse(strategy.is_dp_enabled())

    def test_is_ep_enabled(self):
        """Test EP enabled check."""
        self.assertFalse(self.strategy.is_ep_enabled())
        strategy = ParallelStrategy(StrategyConfig(ep_degree=4))
        self.assertTrue(strategy.is_ep_enabled())

    def test_get_tp_group(self):
        """Test TP group retrieval."""
        # With TP=4, DP=2: ranks 0-3 are group, 4-7 are group
        self.assertEqual(self.strategy.get_tp_group(0), [0, 1, 2, 3])
        self.assertEqual(self.strategy.get_tp_group(5), [4, 5, 6, 7])

    def test_get_tp_group_disabled(self):
        """Test TP group when disabled."""
        strategy = ParallelStrategy(StrategyConfig())
        self.assertEqual(strategy.get_tp_group(3), [3])

    def test_get_dp_group(self):
        """Test DP group retrieval."""
        # With TP=4, world_size=16: DP groups by TP position step by 4
        self.assertEqual(self.strategy.get_dp_group(0), [0, 4, 8, 12])
        self.assertEqual(self.strategy.get_dp_group(1), [1, 5, 9, 13])

    def test_get_dp_group_disabled(self):
        """Test DP group when disabled."""
        strategy = ParallelStrategy(StrategyConfig())
        self.assertEqual(strategy.get_dp_group(3), [3])

    def test_get_pp_group(self):
        """Test PP group retrieval."""
        # With TP=4, PP=2, DP=2: world_size=16, ranks_per_stage=8
        # PP group for rank 0 includes all ranks 0-15
        group = self.strategy.get_pp_group(0)
        self.assertEqual(len(group), 16)
        self.assertIn(0, group)
        self.assertIn(15, group)

    def test_get_pp_group_disabled(self):
        """Test PP group when disabled."""
        strategy = ParallelStrategy(StrategyConfig())
        self.assertEqual(strategy.get_pp_group(3), [3])

    def test_get_ep_group(self):
        """Test EP group retrieval."""
        config = StrategyConfig(tp_degree=2, ep_degree=4)
        strategy = ParallelStrategy(config)
        self.assertEqual(strategy.get_ep_group(0), [0, 1, 2, 3])
        self.assertEqual(strategy.get_ep_group(3), [4, 5, 6, 7])

    def test_layer_assignment(self):
        """Test layer to pipeline stage assignment."""
        self.strategy.assign_layer_to_stage("layer_0", 0)
        self.strategy.assign_layer_to_stage("layer_1", 1)
        self.assertEqual(self.strategy.get_layer_stage("layer_0"), 0)
        self.assertEqual(self.strategy.get_layer_stage("layer_1"), 1)
        self.assertEqual(self.strategy.get_layer_stage("unknown"), 0)

    def test_to_dict(self):
        """Test serialization."""
        self.strategy.assign_layer_to_stage("layer_0", 0)
        data = self.strategy.to_dict()
        self.assertIn("config", data)
        self.assertIn("layer_assignment", data)
        self.assertEqual(data["layer_assignment"]["layer_0"], 0)


class TestStrategyPlanner(unittest.TestCase):
    """Test StrategyPlanner class."""

    def setUp(self):
        self.planner = StrategyPlanner()

    def test_plan_strategy_small_model(self):
        """Test strategy for small model that fits on one device."""
        config = self.planner.plan_strategy(
            model_name="tiny",
            model_params_b=1.0,
            world_size=8,
            batch_size=16,
        )
        self.assertEqual(config.tp_degree, 1)
        self.assertGreaterEqual(config.dp_degree, 1)
        self.assertEqual(config.world_size, config.dp_degree * config.pp_degree)

    def test_plan_strategy_large_model_needs_tp(self):
        """Test strategy for large model requiring tensor parallelism."""
        config = self.planner.plan_strategy(
            model_name="huge",
            model_params_b=200.0,
            world_size=64,
            batch_size=32,
            memory_per_device_gb=80.0,
        )
        self.assertGreater(config.tp_degree, 1)
        self.assertLessEqual(config.tp_degree, 8)

    def test_plan_strategy_moe(self):
        """Test strategy for MoE model."""
        config = self.planner.plan_strategy(
            model_name="moe",
            model_params_b=50.0,
            world_size=32,
            batch_size=16,
            is_moe=True,
            num_experts=8,
        )
        self.assertGreaterEqual(config.ep_degree, 1)

    def test_recommend_strategy(self):
        """Test strategy recommendation with explanations."""
        result = self.planner.recommend_strategy(
            model_params_b=70.0,
            world_size=64,
            target_batch_size=32,
        )
        self.assertIn("config", result)
        self.assertIn("explanations", result)
        self.assertIn("memory_estimate_gb", result)
        self.assertGreater(len(result["explanations"]), 0)

    def test_next_power_of_2(self):
        """Test next power of 2 calculation."""
        self.assertEqual(self.planner._next_power_of_2(1), 1)
        self.assertEqual(self.planner._next_power_of_2(2), 2)
        self.assertEqual(self.planner._next_power_of_2(3), 4)
        self.assertEqual(self.planner._next_power_of_2(5), 8)
        self.assertEqual(self.planner._next_power_of_2(9), 16)

    def test_constraints(self):
        """Test custom constraints."""
        constraints = StrategyConstraints(max_tp_degree=4, max_dp_degree=16)
        planner = StrategyPlanner(constraints)
        config = planner.plan_strategy(
            model_name="large",
            model_params_b=100.0,
            world_size=32,
            batch_size=16,
            memory_per_device_gb=80.0,
        )
        self.assertLessEqual(config.tp_degree, 4)
        self.assertLessEqual(config.dp_degree, 16)


if __name__ == "__main__":
    unittest.main()

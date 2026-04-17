"""Tests for Scenario Layer.

Tests the Scenario abstraction including:
- Base classes (ScenarioConfig, Scenario, ScenarioResult)
- Registry (ScenarioRegistry)
- Concrete scenarios (LLMTraining, LLMInference, PDDisagg, RLTraining, Diffusion)
"""

import unittest
from unittest.mock import Mock, MagicMock

from llm_perf.scenarios.base import (
    Scenario,
    ScenarioConfig,
    ScenarioResult,
    ScenarioType,
    ParallelismType,
)
from llm_perf.scenarios.registry import (
    ScenarioRegistry,
    ScenarioInfo,
    register_all_scenarios,
)
from llm_perf.scenarios.llm_training import (
    LLMTrainingConfig,
    LLMTrainingResult,
)
from llm_perf.scenarios.llm_inference import (
    LLMInferenceConfig,
    LLMInferenceResult,
)
from llm_perf.scenarios.pd_disagg import (
    PDDisaggConfig,
    PDDisaggResult,
    PDNodeConfig,
)
from llm_perf.scenarios.rl_training import (
    RLTrainingConfig,
    RLTrainingResult,
    RLModelConfig,
)
from llm_perf.scenarios.diffusion import (
    DiffusionConfig,
    DiffusionResult,
)


class TestScenarioConfig(unittest.TestCase):
    """Test ScenarioConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a basic scenario config."""
        config = ScenarioConfig(
            name="test-scenario",
            description="Test scenario",
        )
        self.assertEqual(config.name, "test-scenario")
        self.assertEqual(config.description, "Test scenario")
        self.assertEqual(config.scenario_type, ScenarioType.LLM_INFERENCE)
        self.assertEqual(config.required_models, [])

    def test_with_custom_type(self):
        """Test creating config with custom scenario type."""
        config = ScenarioConfig(
            name="training-scenario",
            scenario_type=ScenarioType.LLM_TRAINING,
            required_models=["main"],
        )
        self.assertEqual(config.scenario_type, ScenarioType.LLM_TRAINING)
        self.assertEqual(config.required_models, ["main"])

    def test_to_dict(self):
        """Test dictionary serialization."""
        config = ScenarioConfig(
            name="test",
            description="desc",
            scenario_type=ScenarioType.PD_DISAGG,
            required_models=["prefill", "decode"],
        )
        data = config.to_dict()
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["scenario_type"], "pd_disagg")
        self.assertEqual(data["required_models"], ["prefill", "decode"])

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "test",
            "description": "desc",
            "scenario_type": "rl_training",
            "required_models": ["policy", "reward"],
        }
        config = ScenarioConfig.from_dict(data)
        self.assertEqual(config.name, "test")
        self.assertEqual(config.scenario_type, ScenarioType.RL_TRAINING)


class TestScenarioResult(unittest.TestCase):
    """Test ScenarioResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic result."""
        result = ScenarioResult(
            scenario_name="test",
            total_time_sec=1.0,
            throughput=100.0,
        )
        self.assertEqual(result.scenario_name, "test")
        self.assertEqual(result.total_time_sec, 1.0)
        self.assertEqual(result.throughput, 100.0)

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = ScenarioResult(
            scenario_name="test",
            total_time_sec=1.0,
            throughput=100.0,
            memory_peak_gb=50.0,
            breakdown={"compute": 0.5, "comm": 0.3},
        )
        data = result.to_dict()
        self.assertEqual(data["scenario_name"], "test")
        self.assertEqual(data["breakdown"]["compute"], 0.5)


class TestScenarioRegistry(unittest.TestCase):
    """Test ScenarioRegistry."""

    def setUp(self):
        """Clear registry before each test."""
        self.registry = ScenarioRegistry()
        self.registry.clear()

    def test_register(self):
        """Test registering a scenario."""
        from llm_perf.scenarios.llm_training import LLMTrainingScenario

        self.registry.register(
            name="test-training",
            scenario_class=LLMTrainingScenario,
            description="Test training scenario",
            scenario_type=ScenarioType.LLM_TRAINING,
        )
        self.assertTrue(self.registry.is_registered("test-training"))

    def test_register_duplicate(self):
        """Test registering duplicate scenario raises error."""
        from llm_perf.scenarios.llm_training import LLMTrainingScenario

        self.registry.register(
            name="test",
            scenario_class=LLMTrainingScenario,
        )
        with self.assertRaises(ValueError):
            self.registry.register(
                name="test",
                scenario_class=LLMTrainingScenario,
            )

    def test_unregister(self):
        """Test unregistering a scenario."""
        from llm_perf.scenarios.llm_training import LLMTrainingScenario

        self.registry.register(
            name="test",
            scenario_class=LLMTrainingScenario,
        )
        self.registry.unregister("test")
        self.assertFalse(self.registry.is_registered("test"))

    def test_list_scenarios(self):
        """Test listing scenarios."""
        from llm_perf.scenarios.llm_training import LLMTrainingScenario
        from llm_perf.scenarios.llm_inference import LLMInferenceScenario

        self.registry.register(
            name="training",
            scenario_class=LLMTrainingScenario,
            scenario_type=ScenarioType.LLM_TRAINING,
        )
        self.registry.register(
            name="inference",
            scenario_class=LLMInferenceScenario,
            scenario_type=ScenarioType.LLM_INFERENCE,
        )

        all_scenarios = self.registry.list_scenarios()
        self.assertIn("training", all_scenarios)
        self.assertIn("inference", all_scenarios)

        training_only = self.registry.list_scenarios(ScenarioType.LLM_TRAINING)
        self.assertEqual(training_only, ["training"])

    def test_get_info(self):
        """Test getting scenario info."""
        from llm_perf.scenarios.llm_training import LLMTrainingScenario

        self.registry.register(
            name="test",
            scenario_class=LLMTrainingScenario,
            description="Test scenario",
        )
        info = self.registry.get("test")
        self.assertEqual(info.name, "test")
        self.assertEqual(info.description, "Test scenario")

    def test_get_nonexistent(self):
        """Test getting nonexistent scenario raises error."""
        with self.assertRaises(KeyError):
            self.registry.get("nonexistent")

    def test_list_by_type(self):
        """Test listing scenarios by type."""
        from llm_perf.scenarios.llm_training import LLMTrainingScenario
        from llm_perf.scenarios.llm_inference import LLMInferenceScenario
        from llm_perf.scenarios.rl_training import RLTrainingScenario

        self.registry.register(
            name="train",
            scenario_class=LLMTrainingScenario,
            scenario_type=ScenarioType.LLM_TRAINING,
        )
        self.registry.register(
            name="infer",
            scenario_class=LLMInferenceScenario,
            scenario_type=ScenarioType.LLM_INFERENCE,
        )
        self.registry.register(
            name="rl",
            scenario_class=RLTrainingScenario,
            scenario_type=ScenarioType.RL_TRAINING,
        )

        by_type = self.registry.list_by_type()
        self.assertIn("llm_training", by_type)
        self.assertIn("llm_inference", by_type)
        self.assertIn("rl_training", by_type)


class TestLLMTrainingConfig(unittest.TestCase):
    """Test LLMTrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMTrainingConfig(name="test")
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.seq_len, 2048)
        self.assertEqual(config.num_steps, 1000)
        self.assertEqual(config.scenario_type, ScenarioType.LLM_TRAINING)
        self.assertEqual(config.required_models, ["main"])

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMTrainingConfig(
            name="test",
            batch_size=4,
            seq_len=4096,
            num_steps=500,
        )
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.seq_len, 4096)
        self.assertEqual(config.num_steps, 500)


class TestLLMInferenceConfig(unittest.TestCase):
    """Test LLMInferenceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMInferenceConfig(name="test")
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.prompt_len, 512)
        self.assertEqual(config.generation_len, 128)
        self.assertEqual(config.scenario_type, ScenarioType.LLM_INFERENCE)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMInferenceConfig(
            name="test",
            batch_size=8,
            prompt_len=1024,
            generation_len=256,
        )
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.prompt_len, 1024)
        self.assertEqual(config.generation_len, 256)


class TestPDDisaggConfig(unittest.TestCase):
    """Test PDDisaggConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PDDisaggConfig(name="test")
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.prompt_len, 512)
        self.assertEqual(config.generation_len, 128)
        self.assertEqual(config.scenario_type, ScenarioType.PD_DISAGG)
        self.assertEqual(config.required_models, ["prefill", "decode"])

    def test_node_configs(self):
        """Test PD node configurations."""
        prefill_config = PDNodeConfig(role="prefill", tp_degree=8, num_nodes=2)
        decode_config = PDNodeConfig(role="decode", tp_degree=2, num_nodes=4)

        config = PDDisaggConfig(
            name="test",
            prefill_config=prefill_config,
            decode_config=decode_config,
        )

        self.assertEqual(config.prefill_config.tp_degree, 8)
        self.assertEqual(config.prefill_config.total_devices, 16)
        self.assertEqual(config.decode_config.tp_degree, 2)
        self.assertEqual(config.decode_config.total_devices, 32)


class TestRLTrainingConfig(unittest.TestCase):
    """Test RLTrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RLTrainingConfig(name="test")
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.seq_len, 512)
        self.assertEqual(config.num_ppo_steps, 100)
        self.assertEqual(config.scenario_type, ScenarioType.RL_TRAINING)
        self.assertEqual(config.required_models, ["policy", "reward", "reference"])

    def test_model_configs(self):
        """Test RL model configurations."""
        policy_config = RLModelConfig(role="policy", tp_degree=4)
        reward_config = RLModelConfig(role="reward", tp_degree=1)

        config = RLTrainingConfig(
            name="test",
            policy_config=policy_config,
            reward_config=reward_config,
        )

        self.assertEqual(config.policy_config.tp_degree, 4)
        self.assertEqual(config.reward_config.tp_degree, 1)


class TestDiffusionConfig(unittest.TestCase):
    """Test DiffusionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DiffusionConfig(name="test")
        self.assertEqual(config.num_inference_steps, 50)
        self.assertTrue(config.use_cfg)
        self.assertEqual(config.num_frames, 81)
        self.assertEqual(config.height, 720)
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.scenario_type, ScenarioType.DIFFUSION)
        self.assertEqual(config.required_models, ["text_encoder", "dit", "vae"])

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DiffusionConfig(
            name="test",
            num_inference_steps=25,
            use_cfg=False,
            num_frames=41,
            height=480,
            width=640,
        )
        self.assertEqual(config.num_inference_steps, 25)
        self.assertFalse(config.use_cfg)
        self.assertEqual(config.num_frames, 41)


class TestScenarioRegistryWithBuiltins(unittest.TestCase):
    """Test ScenarioRegistry with built-in scenarios."""

    def setUp(self):
        """Clear and re-register built-in scenarios."""
        self.registry = ScenarioRegistry()
        self.registry.clear()
        register_all_scenarios()

    def test_builtin_scenarios_registered(self):
        """Test that built-in scenarios are registered."""
        self.assertTrue(self.registry.is_registered("llm-training"))
        self.assertTrue(self.registry.is_registered("llm-inference"))
        self.assertTrue(self.registry.is_registered("pd-disagg"))
        self.assertTrue(self.registry.is_registered("rl-training"))
        self.assertTrue(self.registry.is_registered("diffusion"))

    def test_scenario_info(self):
        """Test getting built-in scenario info."""
        info = self.registry.get("llm-training")
        self.assertEqual(info.scenario_type, ScenarioType.LLM_TRAINING)
        self.assertEqual(info.required_models, ["main"])

        info = self.registry.get("pd-disagg")
        self.assertEqual(info.scenario_type, ScenarioType.PD_DISAGG)
        self.assertEqual(info.required_models, ["prefill", "decode"])

    def test_list_by_type(self):
        """Test listing built-in scenarios by type."""
        by_type = self.registry.list_by_type()
        self.assertEqual(len(by_type["llm_training"]), 1)
        self.assertEqual(len(by_type["llm_inference"]), 1)
        self.assertEqual(len(by_type["pd_disagg"]), 1)
        self.assertEqual(len(by_type["rl_training"]), 1)
        self.assertEqual(len(by_type["diffusion"]), 1)


class TestScenarioType(unittest.TestCase):
    """Test ScenarioType enum."""

    def test_all_types(self):
        """Test all scenario types exist."""
        self.assertEqual(ScenarioType.LLM_TRAINING.value, "llm_training")
        self.assertEqual(ScenarioType.LLM_INFERENCE.value, "llm_inference")
        self.assertEqual(ScenarioType.PD_DISAGG.value, "pd_disagg")
        self.assertEqual(ScenarioType.RL_TRAINING.value, "rl_training")
        self.assertEqual(ScenarioType.DIFFUSION.value, "diffusion")


class TestParallelismType(unittest.TestCase):
    """Test ParallelismType enum."""

    def test_all_types(self):
        """Test all parallelism types exist."""
        self.assertEqual(ParallelismType.TP.value, "tp")
        self.assertEqual(ParallelismType.PP.value, "pp")
        self.assertEqual(ParallelismType.DP.value, "dp")
        self.assertEqual(ParallelismType.EP.value, "ep")
        self.assertEqual(ParallelismType.SP.value, "sp")
        self.assertEqual(ParallelismType.CP.value, "cp")


if __name__ == "__main__":
    unittest.main()

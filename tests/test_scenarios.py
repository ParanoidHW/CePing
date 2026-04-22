"""Tests for new scenarios module."""

import pytest

from llm_perf.modeling import LlamaModel, create_model_from_config
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.scenarios import (
    Scenario,
    ScenarioConfig,
    ScenarioResult,
    ScenarioType,
    ParallelismType,
    ScenarioRegistry,
    ScenarioInfo,
    LLMTrainingScenario,
    LLMTrainingConfig,
    LLMTrainingResult,
    LLMInferenceScenario,
    LLMInferenceConfig,
    LLMInferenceResult,
)


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


class TestScenarioTypes:
    """Test scenario type enums."""

    def test_scenario_type_values(self):
        """Test ScenarioType enum values."""
        assert ScenarioType.LLM_TRAINING.value == "llm_training"
        assert ScenarioType.LLM_INFERENCE.value == "llm_inference"
        assert ScenarioType.PD_DISAGG.value == "pd_disagg"
        assert ScenarioType.RL_TRAINING.value == "rl_training"
        assert ScenarioType.DIFFUSION.value == "diffusion"

    def test_parallelism_type_values(self):
        """Test ParallelismType enum values."""
        assert ParallelismType.TP.value == "tp"
        assert ParallelismType.PP.value == "pp"
        assert ParallelismType.DP.value == "dp"
        assert ParallelismType.EP.value == "ep"
        assert ParallelismType.SP.value == "sp"
        assert ParallelismType.CP.value == "cp"


class TestScenarioConfig:
    """Test ScenarioConfig."""

    def test_config_creation(self):
        """Test creating scenario config."""
        config = ScenarioConfig(
            name="test-scenario",
            description="Test scenario",
        )
        assert config.name == "test-scenario"
        assert config.description == "Test scenario"
        assert config.scenario_type == ScenarioType.LLM_INFERENCE

    def test_config_to_dict(self):
        """Test config serialization."""
        config = ScenarioConfig(
            name="test",
            scenario_type=ScenarioType.LLM_TRAINING,
            required_models=["main"],
        )
        result = config.to_dict()
        assert result["name"] == "test"
        assert result["scenario_type"] == "llm_training"
        assert result["required_models"] == ["main"]

    def test_config_from_dict(self):
        """Test config deserialization."""
        data = {
            "name": "test",
            "scenario_type": "llm_training",
            "required_models": ["policy", "reward"],
        }
        config = ScenarioConfig.from_dict(data)
        assert config.name == "test"
        assert config.scenario_type == ScenarioType.LLM_TRAINING
        assert config.required_models == ["policy", "reward"]


class TestScenarioResult:
    """Test ScenarioResult."""

    def test_result_creation(self):
        """Test creating scenario result."""
        result = ScenarioResult(
            scenario_name="test",
            total_time_sec=10.0,
            throughput=1000.0,
            memory_peak_gb=50.0,
        )
        assert result.scenario_name == "test"
        assert result.total_time_sec == 10.0
        assert result.throughput == 1000.0

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ScenarioResult(
            scenario_name="test",
            total_time_sec=10.0,
            breakdown={"compute": 5.0, "comm": 5.0},
        )
        data = result.to_dict()
        assert data["scenario_name"] == "test"
        assert data["breakdown"]["compute"] == 5.0


class TestScenarioRegistry:
    """Test ScenarioRegistry."""

    def test_registry_creation(self):
        """Test registry singleton."""
        registry = ScenarioRegistry()
        assert registry is not None

        registry2 = ScenarioRegistry()
        assert registry is registry2

    def test_list_scenarios(self):
        """Test listing scenarios."""
        registry = ScenarioRegistry()
        scenarios = registry.list_scenarios()
        assert "training" in scenarios
        assert "autoregressive-inference" in scenarios

    def test_list_by_type(self):
        """Test listing by type."""
        registry = ScenarioRegistry()
        by_type = registry.list_by_type()
        assert "llm_training" in by_type
        assert "llm_inference" in by_type

    def test_get_scenario_info(self):
        """Test getting scenario info."""
        registry = ScenarioRegistry()
        info = registry.get("training")
        assert info.name == "training"
        assert info.scenario_type == ScenarioType.LLM_TRAINING

    def test_is_registered(self):
        """Test checking registration."""
        registry = ScenarioRegistry()
        assert registry.is_registered("training")
        assert not registry.is_registered("nonexistent")


class TestLLMTrainingScenario:
    """Test LLMTrainingScenario."""

    def test_training_config(self):
        """Test training config."""
        config = LLMTrainingConfig(
            name="test-training",
            batch_size=32,
            seq_len=2048,
        )
        assert config.batch_size == 32
        assert config.seq_len == 2048
        assert config.scenario_type == ScenarioType.LLM_TRAINING
        assert config.required_models == ["main"]

    def test_training_scenario_creation(self):
        """Test creating training scenario."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMTrainingConfig(name="test-training")
        scenario = LLMTrainingScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )
        assert scenario is not None
        assert scenario.get_model("main") is model

    def test_training_scenario_analyze(self):
        """Test training scenario analysis."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMTrainingConfig(name="test-training")
        scenario = LLMTrainingScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        result = scenario.analyze(batch_size=32, seq_len=2048)

        assert result.samples_per_sec > 0
        assert result.tokens_per_sec > 0
        assert result.time_per_step_sec > 0
        assert result.memory_per_device_gb > 0

    def test_training_scenario_get_analyzer(self):
        """Test getting analyzer."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMTrainingConfig(name="test-training")
        scenario = LLMTrainingScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        analyzer = scenario.get_analyzer()
        assert analyzer is not None

    def test_training_scenario_to_dict(self):
        """Test scenario serialization."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMTrainingConfig(name="test-training")
        scenario = LLMTrainingScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        data = scenario.to_dict()
        assert "config" in data
        assert "models" in data
        assert "strategy" in data

    def test_training_result_to_dict(self):
        """Test training result serialization."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMTrainingConfig(name="test-training")
        scenario = LLMTrainingScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        result = scenario.analyze(batch_size=32, seq_len=2048)
        data = result.to_dict()

        assert data["samples_per_sec"] > 0
        assert data["tokens_per_sec"] > 0
        assert "unified_result" in data


class TestLLMInferenceScenario:
    """Test LLMInferenceScenario."""

    def test_inference_config(self):
        """Test inference config."""
        config = LLMInferenceConfig(
            name="test-inference",
            batch_size=1,
            prompt_len=512,
            generation_len=128,
        )
        assert config.batch_size == 1
        assert config.prompt_len == 512
        assert config.generation_len == 128
        assert config.scenario_type == ScenarioType.LLM_INFERENCE

    def test_inference_scenario_creation(self):
        """Test creating inference scenario."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMInferenceConfig(name="test-inference")
        scenario = LLMInferenceScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )
        assert scenario is not None

    def test_inference_scenario_analyze(self):
        """Test inference scenario analysis."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMInferenceConfig(name="test-inference")
        scenario = LLMInferenceScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        result = scenario.analyze(
            batch_size=8,
            prompt_len=512,
            generation_len=128,
        )

        assert result.prefill_time_sec > 0
        assert result.decode_time_per_step_sec > 0
        assert result.prefill_tokens_per_sec > 0
        assert result.decode_tokens_per_sec > 0
        assert result.memory_per_device_gb > 0

    def test_inference_estimate_latency(self):
        """Test latency estimation."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMInferenceConfig(name="test-inference")
        scenario = LLMInferenceScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        latency = scenario.estimate_latency(batch_size=1, prompt_len=512, generation_len=128)

        assert latency["ttft_sec"] > 0
        assert latency["ttft_ms"] > 0
        assert latency["tpot_sec"] > 0
        assert latency["tpot_ms"] > 0

    def test_inference_estimate_memory(self):
        """Test memory estimation."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMInferenceConfig(name="test-inference")
        scenario = LLMInferenceScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        memory = scenario.estimate_memory(batch_size=1, max_seq_len=1024)

        assert "total_memory_gb" in memory
        assert memory["total_memory_gb"] > 0

    def test_inference_result_to_dict(self):
        """Test inference result serialization."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        config = LLMInferenceConfig(name="test-inference")
        scenario = LLMInferenceScenario(
            config=config,
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        result = scenario.analyze(batch_size=1, prompt_len=512, generation_len=128)
        data = result.to_dict()

        assert "prefill" in data
        assert "decode" in data
        assert data["prefill"]["tokens_per_sec"] > 0
        assert data["decode"]["tokens_per_sec"] > 0


class TestRegistryCreateScenario:
    """Test registry create_scenario method."""

    def test_create_training_scenario(self):
        """Test creating scenario via registry."""
        model = create_model_from_config({"preset": "llama-7b"})
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        registry = ScenarioRegistry()
        scenario = registry.create_scenario(
            "training",
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        assert scenario is not None
        assert isinstance(scenario, LLMTrainingScenario)

        result = scenario.analyze(batch_size=32, seq_len=2048)
        assert result.samples_per_sec > 0

    def test_create_inference_scenario(self):
        """Test creating inference scenario via registry."""
        model = create_model_from_config({"preset": "llama-7b"})
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        registry = ScenarioRegistry()
        scenario = registry.create_scenario(
            "autoregressive-inference",
            models={"main": model},
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

        assert scenario is not None
        assert isinstance(scenario, LLMInferenceScenario)

        result = scenario.analyze(batch_size=1, prompt_len=512, generation_len=128)
        assert result.prefill_tokens_per_sec > 0

    def test_create_nonexistent_scenario(self):
        """Test creating nonexistent scenario."""
        model = create_model_from_config({"preset": "llama-7b"})
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        registry = ScenarioRegistry()
        with pytest.raises(KeyError):
            registry.create_scenario(
                "nonexistent",
                models={"main": model},
                device=device,
                cluster=cluster,
                strategy=strategy,
            )

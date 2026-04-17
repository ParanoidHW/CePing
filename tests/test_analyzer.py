"""Tests for new analyzer module."""

import pytest

from llm_perf.modeling import LlamaModel, create_model_from_config
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer import TrainingAnalyzer, InferenceAnalyzer


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


class TestTrainingAnalyzer:
    """Test TrainingAnalyzer."""

    def test_analyzer_creation(self):
        """Test creating analyzer."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        assert analyzer is not None

    def test_training_analysis(self):
        """Test training analysis."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(batch_size=32, seq_len=2048)

        assert result.samples_per_sec > 0
        assert result.tokens_per_sec > 0
        assert result.time_per_step_sec > 0
        assert result.memory_per_gpu_gb > 0
        assert result.breakdown is not None

    def test_training_result_to_dict(self):
        """Test result serialization."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 1)
        strategy = StrategyConfig(tp_degree=1)

        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(batch_size=1, seq_len=128)

        result_dict = result.to_dict()
        assert "throughput" in result_dict
        assert "time" in result_dict
        assert "memory" in result_dict


class TestInferenceAnalyzer:
    """Test InferenceAnalyzer."""

    def test_inference_analysis(self):
        """Test inference analysis."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = InferenceAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            batch_size=8,
            prompt_len=512,
            generation_len=128,
        )

        assert result.prefill_time_sec > 0
        assert result.decode_time_per_step_sec > 0
        assert result.prefill_tokens_per_sec > 0
        assert result.decode_tokens_per_sec > 0
        assert result.memory_per_gpu_gb > 0

    def test_inference_result_to_dict(self):
        """Test inference result serialization."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 1)
        strategy = StrategyConfig(tp_degree=1)

        analyzer = InferenceAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(batch_size=1, prompt_len=128, generation_len=32)

        result_dict = result.to_dict()
        assert "prefill" in result_dict
        assert "decode" in result_dict


class TestAnalyzerWithPreset:
    """Test analyzer with preset models."""

    def test_analyzer_with_llama_preset(self):
        """Test analyzer with preset."""
        model = create_model_from_config({"preset": "llama-7b"})
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(batch_size=32, seq_len=4096)

        assert result is not None

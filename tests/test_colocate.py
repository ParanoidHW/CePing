"""Tests for ColocateAnalyzer."""

from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.device import Device
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.modeling import LlamaModel
from llm_perf.scenarios import ColocateAnalyzer, ModelAllocation
from llm_perf.strategy.base import StrategyConfig


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


class TestColocateAnalyzer:
    """Test ColocateAnalyzer."""

    def test_colocate_two_models(self):
        """Test colocating two models."""
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)

        model1 = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        model2 = LlamaModel(vocab_size=32000, hidden_size=2048, num_layers=4, num_heads=16)

        allocations = [
            ModelAllocation(
                name="llama-4k",
                model=model1,
                strategy=StrategyConfig(tp_degree=4),
                workload="training",
            ),
            ModelAllocation(
                name="llama-2k",
                model=model2,
                strategy=StrategyConfig(tp_degree=2),
                workload="training",
            ),
        ]

        analyzer = ColocateAnalyzer(device, cluster)
        result = analyzer.analyze(allocations, batch_size=32, seq_len=2048)

        assert len(result.model_results) == 2
        assert "llama-4k" in result.model_results
        assert "llama-2k" in result.model_results
        assert result.total_devices == 8
        assert result.total_utilization > 0

    def test_colocate_result_to_dict(self):
        """Test ColocateResult serialization."""
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)

        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)

        allocations = [
            ModelAllocation(
                name="test-model",
                model=model,
                strategy=StrategyConfig(tp_degree=8),
                workload="training",
            ),
        ]

        analyzer = ColocateAnalyzer(device, cluster)
        result = analyzer.analyze(allocations, batch_size=32, seq_len=2048)

        data = result.to_dict()
        assert "model_results" in data
        assert "test-model" in data["model_results"]
        assert data["total_devices"] == 8

    def test_colocate_with_different_workloads(self):
        """Test colocating models with different workloads."""
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)

        model1 = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        model2 = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)

        allocations = [
            ModelAllocation(
                name="training-model",
                model=model1,
                strategy=StrategyConfig(tp_degree=8),
                workload="training",
            ),
            ModelAllocation(
                name="inference-model",
                model=model2,
                strategy=StrategyConfig(tp_degree=8),
                workload="autoregressive-inference",
            ),
        ]

        analyzer = ColocateAnalyzer(device, cluster)
        result = analyzer.analyze(
            allocations,
            batch_size=32,
            seq_len=2048,
            prompt_len=512,
            generation_len=128,
        )

        assert len(result.model_results) == 2

        training_result = result.model_results["training-model"]
        inference_result = result.model_results["inference-model"]

        assert training_result.workload_name == "training"
        assert inference_result.workload_name == "autoregressive-inference"

    def test_model_allocation_defaults(self):
        """Test ModelAllocation defaults."""
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 1)

        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=2, num_heads=32)

        allocation = ModelAllocation(
            name="default-test",
            model=model,
            strategy=StrategyConfig(tp_degree=1),
            workload="training",
        )

        assert allocation.allocated_ratio == 1.0

        analyzer = ColocateAnalyzer(device, cluster)
        result = analyzer.analyze([allocation], batch_size=1, seq_len=512)

        assert result.total_utilization == 1.0

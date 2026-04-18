"""Tests for UnifiedAnalyzer module."""

import pytest

from llm_perf.modeling import LlamaModel, create_model_from_config
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer import (
    UnifiedAnalyzer,
    UnifiedResult,
    WorkloadConfig,
    Phase,
    PhaseResult,
    ComputeType,
    WorkloadType,
    ThroughputMetric,
    get_workload,
    list_workloads,
    infer_workload,
    register_workload,
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


class TestEnums:
    """Test enum types."""

    def test_compute_type_values(self):
        assert ComputeType.FORWARD.value == "forward"
        assert ComputeType.BACKWARD.value == "backward"
        assert ComputeType.OPTIMIZER.value == "optimizer"

    def test_workload_type_values(self):
        assert WorkloadType.TRAINING.value == "training"
        assert WorkloadType.INFERENCE.value == "inference"
        assert WorkloadType.MIXED.value == "mixed"

    def test_throughput_metric_values(self):
        assert ThroughputMetric.TOKENS_PER_SEC.value == "tokens_per_sec"
        assert ThroughputMetric.PIXELS_PER_SEC.value == "pixels_per_sec"


class TestPhase:
    """Test Phase dataclass."""

    def test_phase_creation(self):
        phase = Phase("prefill", ComputeType.FORWARD, component="main", repeat=1)
        assert phase.name == "prefill"
        assert phase.compute_type == ComputeType.FORWARD
        assert phase.component == "main"
        assert phase.repeat == 1

    def test_phase_with_dynamic_repeat(self):
        phase = Phase("decode", ComputeType.FORWARD, repeat="generation_len")
        assert phase.repeat == "generation_len"

    def test_phase_to_dict(self):
        phase = Phase("forward", ComputeType.FORWARD, repeat=1, seq_len_factor=1.0)
        data = phase.to_dict()
        assert data["name"] == "forward"
        assert data["compute_type"] == "forward"


class TestPhaseResult:
    """Test PhaseResult dataclass."""

    def test_phase_result_creation(self):
        result = PhaseResult(
            name="prefill",
            component="main",
            compute_type=ComputeType.FORWARD,
            single_time_sec=0.1,
            repeat_count=1,
            total_time_sec=0.1,
            memory_gb=10.0,
        )
        assert result.name == "prefill"
        assert result.single_time_sec == 0.1
        assert result.total_time_sec == 0.1

    def test_phase_result_to_dict(self):
        result = PhaseResult(
            name="forward",
            component="main",
            compute_type=ComputeType.FORWARD,
            single_time_sec=0.5,
            repeat_count=10,
            total_time_sec=5.0,
            memory_gb=20.0,
        )
        data = result.to_dict()
        assert data["single_time_ms"] == 500.0
        assert data["total_time_ms"] == 5000.0


class TestUnifiedResult:
    """Test UnifiedResult dataclass."""

    def test_result_creation(self):
        phase1 = PhaseResult("prefill", "main", ComputeType.FORWARD, single_time_sec=0.1)
        phase2 = PhaseResult("decode", "main", ComputeType.FORWARD, single_time_sec=0.05, repeat_count=128)

        result = UnifiedResult(
            workload_name="llm-inference",
            workload_type=WorkloadType.INFERENCE,
            phases=[phase1, phase2],
            total_time_sec=phase1.total_time_sec + phase2.total_time_sec,
            peak_memory_gb=10.0,
            throughput={"tokens_per_sec": 1000.0},
        )

        assert result.workload_name == "llm-inference"
        assert len(result.phases) == 2

    def test_result_get_phase(self):
        phase = PhaseResult("prefill", "main", ComputeType.FORWARD)
        result = UnifiedResult(
            workload_name="test",
            workload_type=WorkloadType.INFERENCE,
            phases=[phase],
        )

        found = result.get_phase("prefill")
        assert found is not None
        assert found.name == "prefill"

        not_found = result.get_phase("nonexistent")
        assert not_found is None

    def test_result_to_dict(self):
        result = UnifiedResult(
            workload_name="llm-training",
            workload_type=WorkloadType.TRAINING,
            phases=[],
            total_time_sec=1.0,
            peak_memory_gb=50.0,
            throughput={"tokens_per_sec": 100000.0},
        )
        data = result.to_dict()
        assert data["workload_name"] == "llm-training"
        assert data["total_time_ms"] == 1000.0


class TestWorkloadConfig:
    """Test WorkloadConfig dataclass."""

    def test_config_creation(self):
        config = WorkloadConfig(
            name="custom-workload",
            description="Custom workload for testing",
            workload_type=WorkloadType.TRAINING,
            phases=[
                Phase("forward", ComputeType.FORWARD),
                Phase("backward", ComputeType.BACKWARD),
            ],
        )
        assert config.name == "custom-workload"
        assert len(config.phases) == 2

    def test_config_get_required_params(self):
        config = WorkloadConfig(
            name="test",
            phases=[
                Phase("decode", ComputeType.FORWARD, repeat="generation_len"),
                Phase("dit", ComputeType.FORWARD, repeat="num_inference_steps"),
            ],
        )
        required = config.get_required_params()
        assert "generation_len" in required
        assert "num_inference_steps" in required


class TestPresets:
    """Test workload presets."""

    def test_get_workload(self):
        workload = get_workload("llm-training")
        assert workload.name == "llm-training"
        assert workload.workload_type == WorkloadType.TRAINING

    def test_get_workload_not_found(self):
        with pytest.raises(KeyError):
            get_workload("nonexistent-workload")

    def test_list_workloads(self):
        workloads = list_workloads()
        assert "llm-training" in workloads
        assert "llm-inference" in workloads
        assert "diffusion-inference" in workloads

    def test_infer_workload_llm(self):
        workload_name = infer_workload("llama", "training")
        assert workload_name == "llm-training"

        workload_name = infer_workload("llama", "inference")
        assert workload_name == "llm-inference"

    def test_infer_workload_diffusion(self):
        workload_name = infer_workload("wan-dit", "inference")
        assert workload_name == "diffusion-inference"

    def test_register_workload(self):
        config = WorkloadConfig(
            name="test-custom",
            description="Test custom workload",
            phases=[Phase("forward", ComputeType.FORWARD)],
        )
        register_workload(config)

        workload = get_workload("test-custom")
        assert workload.name == "test-custom"


class TestUnifiedAnalyzer:
    """Test UnifiedAnalyzer."""

    def test_analyzer_creation(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        assert analyzer is not None

    def test_analyze_llm_training(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("llm-training", batch_size=32, seq_len=2048)

        assert result.workload_name == "llm-training"
        assert result.total_time_sec > 0
        assert result.peak_memory_gb > 0
        assert "tokens_per_sec" in result.throughput

    def test_analyze_llm_inference(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            "llm-inference",
            batch_size=8,
            prompt_len=512,
            generation_len=128,
        )

        assert result.workload_name == "llm-inference"
        assert len(result.phases) >= 2

        prefill = result.get_phase("prefill")
        decode = result.get_phase("decode")

        assert prefill is not None
        assert decode is not None
        assert prefill.total_time_sec > 0
        assert decode.single_time_sec > 0

    def test_analyze_with_custom_workload(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        custom_workload = WorkloadConfig(
            name="custom-test",
            phases=[
                Phase("forward", ComputeType.FORWARD, repeat=1),
                Phase("backward", ComputeType.BACKWARD, repeat=1),
            ],
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(custom_workload, batch_size=32, seq_len=2048)

        assert result.workload_name == "custom-test"
        assert len(result.phases) == 2

    def test_analyze_moe_training(self):
        model = create_model_from_config({"preset": "llama-7b"})
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("moe-training", batch_size=32, seq_len=2048)

        assert result.workload_name == "moe-training"
        assert result.total_time_sec > 0

    def test_phase_breakdown(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("llm-training", batch_size=32, seq_len=2048)

        forward_phase = result.get_phase("forward")
        backward_phase = result.get_phase("backward")
        optimizer_phase = result.get_phase("optimizer")

        assert forward_phase is not None
        assert backward_phase is not None
        assert optimizer_phase is not None

        assert backward_phase.single_time_sec > forward_phase.single_time_sec

    def test_dynamic_param_resolution(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result1 = analyzer.analyze("llm-inference", batch_size=1, generation_len=10)
        result2 = analyzer.analyze("llm-inference", batch_size=1, generation_len=100)

        decode1 = result1.get_phase("decode")
        decode2 = result2.get_phase("decode")

        assert decode2.repeat_count == 100
        assert decode2.total_time_sec > decode1.total_time_sec

    def test_result_to_dict_complete(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("llm-inference", batch_size=1, prompt_len=512, generation_len=128)

        data = result.to_dict()

        assert "workload_name" in data
        assert "phases" in data
        assert "total_time_sec" in data
        assert "throughput" in data
        assert "params" in data

        assert data["params"]["generation_len"] == 128


class TestAnalyzerWithPreset:
    """Test analyzer with preset models."""

    def test_analyzer_with_llama_preset(self):
        model = create_model_from_config({"preset": "llama-7b"})
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("llm-inference", batch_size=1)

        assert result.total_time_sec > 0
        assert result.peak_memory_gb > 0

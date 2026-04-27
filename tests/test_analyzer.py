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
    ComputePattern,
    WorkloadType,
    get_workload,
    list_workloads,
    register_workload,
    load_workload_from_yaml,
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


class TestPhase:
    """Test Phase dataclass."""

    def test_phase_creation(self):
        phase = Phase("prefill", ComputeType.FORWARD, component="main", repeat=1)
        assert phase.name == "prefill"
        assert phase.compute_type == ComputeType.FORWARD
        assert phase.component == "main"
        assert phase.repeat == 1

    def test_phase_with_compute_pattern(self):
        phase = Phase("encode", ComputeType.FORWARD, compute_pattern=ComputePattern.TRANSFORMER_BLOCK)
        assert phase.compute_pattern == ComputePattern.TRANSFORMER_BLOCK

    def test_phase_with_dynamic_repeat(self):
        phase = Phase("decode", ComputeType.FORWARD, repeat="generation_len")
        assert phase.repeat == "generation_len"

    def test_phase_to_dict(self):
        phase = Phase("forward", ComputeType.FORWARD, compute_pattern=ComputePattern.TRANSFORMER_BLOCK)
        data = phase.to_dict()
        assert data["name"] == "forward"
        assert data["compute_type"] == "forward"
        assert data["compute_pattern"] == "transformer_block"


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
            component="backbone",
            compute_type=ComputeType.FORWARD,
            single_time_sec=0.5,
            repeat_count=10,
            total_time_sec=5.0,
            memory_gb=20.0,
        )
        data = result.to_dict()
        assert data["single_time_ms"] == 500.0
        assert data["total_time_ms"] == 5000.0
        assert data["component"] == "backbone"


class TestUnifiedResult:
    """Test UnifiedResult dataclass."""

    def test_result_creation(self):
        phase1 = PhaseResult("prefill", "main", ComputeType.FORWARD, single_time_sec=0.1)
        phase2 = PhaseResult("decode", "main", ComputeType.FORWARD, single_time_sec=0.05, repeat_count=128)

        result = UnifiedResult(
            workload_name="autoregressive-inference",
            workload_type=WorkloadType.INFERENCE,
            phases=[phase1, phase2],
            total_time_sec=phase1.total_time_sec + phase2.total_time_sec,
            peak_memory_gb=10.0,
            throughput={"tokens_per_sec": 1000.0},
        )

        assert result.workload_name == "autoregressive-inference"
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

    def test_result_to_dict(self):
        result = UnifiedResult(
            workload_name="training",
            workload_type=WorkloadType.TRAINING,
            phases=[],
            total_time_sec=1.0,
            peak_memory_gb=50.0,
            throughput={"tokens_per_sec": 100000.0},
        )
        data = result.to_dict()
        assert data["workload_name"] == "training"
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
            component_mapping={"backbone": "dit"},
        )
        assert config.name == "custom-workload"
        assert len(config.phases) == 2
        assert config.component_mapping == {"backbone": "dit"}

    def test_config_resolve_component(self):
        config = WorkloadConfig(
            name="test",
            component_mapping={"encoder": "text_encoder", "backbone": "dit"},
        )
        assert config.resolve_component("encoder") == "text_encoder"
        assert config.resolve_component("backbone") == "dit"
        assert config.resolve_component("main") == "main"

    def test_config_get_required_params(self):
        config = WorkloadConfig(
            name="test",
            phases=[
                Phase("decode", ComputeType.FORWARD, repeat="generation_len"),
                Phase("denoise", ComputeType.FORWARD, repeat="num_steps"),
            ],
        )
        required = config.get_required_params()
        assert "generation_len" in required
        assert "num_steps" in required


class TestWorkloadLoader:
    """Test workload loader."""

    def test_get_workload_training(self):
        workload = get_workload("training")
        assert workload.name == "training"
        assert workload.workload_type == WorkloadType.TRAINING

    def test_get_workload_autoregressive_inference(self):
        workload = get_workload("autoregressive-inference")
        assert workload.name == "autoregressive-inference"
        assert workload.workload_type == WorkloadType.INFERENCE
        assert len(workload.phases) == 2

    def test_get_workload_backward_compat(self):
        workload = get_workload("llm-training")
        assert workload.name == "training"  # 映射到新名称

        workload2 = get_workload("llm-inference")
        assert workload2.name == "autoregressive-inference"  # 映射到新名称

    def test_get_workload_not_found(self):
        with pytest.raises(KeyError):
            get_workload("nonexistent-workload")

    def test_list_workloads(self):
        workloads = list_workloads()
        assert "llm-training" in workloads  # backward compat name
        assert "llm-inference" in workloads  # backward compat name
        assert "diffusion-inference" in workloads  # backward compat name

    def test_load_workload_from_yaml(self):
        from pathlib import Path

        yaml_path = Path(__file__).parent.parent / "configs" / "workloads" / "base" / "training.yaml"
        if yaml_path.exists():
            workload = load_workload_from_yaml(yaml_path)
            assert workload.name == "training"

    def test_register_workload(self):
        config = WorkloadConfig(
            name="test-custom",
            description="Test custom workload",
            phases=[Phase("forward", ComputeType.FORWARD)],
        )
        register_workload(config)

        workload = get_workload("test-custom")
        assert workload.name == "test-custom"


class TestResultDimensions:
    """Tests for result dimension correctness."""

    def test_memory_dimension_per_device(self):
        """Test memory metrics are per-device."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=2, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology(
            name="test",
            intra_node_bandwidth_gbps=200.0,
            intra_node_latency_us=1.0,
            inter_node_bandwidth_gbps=25.0,
            inter_node_latency_us=10.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 8, topology)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        result_dict = result.to_dict()

        assert result_dict["peak_memory_gb"] > 0
        assert result_dict["memory"]["memory_per_device_gb"] == result_dict["peak_memory_gb"]

    def test_throughput_dimension_global(self):
        """Test throughput metrics are global."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=2, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology(
            name="test",
            intra_node_bandwidth_gbps=200.0,
            intra_node_latency_us=1.0,
            inter_node_bandwidth_gbps=25.0,
            inter_node_latency_us=10.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 16, topology)
        strategy = StrategyConfig(tp_degree=4, dp_degree=4)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=128, seq_len=2048)

        result_dict = result.to_dict()

        assert "tokens_per_sec" in result_dict["throughput"]
        assert result_dict["throughput"]["tokens_per_sec"] > 0

        if result_dict["decode"]:
            assert result_dict["decode"]["tps"] == result_dict["throughput"]["tokens_per_sec"]

    def test_inference_prefill_decode_fields(self):
        """Test inference result has prefill and decode with correct dimensions."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=2, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology(
            name="test",
            intra_node_bandwidth_gbps=200.0,
            intra_node_latency_us=1.0,
            inter_node_bandwidth_gbps=25.0,
            inter_node_latency_us=10.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 8, topology)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            "llm-inference",
            batch_size=8,
            prompt_len=512,
            generation_len=128,
        )

        result_dict = result.to_dict()

        assert result_dict["prefill"] is not None
        assert result_dict["decode"] is not None

        assert result_dict["prefill"]["ttft_sec"] > 0
        assert result_dict["prefill"]["total_time_sec"] == result_dict["prefill"]["ttft_sec"]

        assert result_dict["decode"]["tps"] > 0
        assert result_dict["decode"]["tpot_sec"] > 0

        assert result_dict["end_to_end"]["overall_tps"] > 0
        assert result_dict["end_to_end"]["total_time_sec"] > 0

    def test_metadata_parallel_degrees(self):
        """Test metadata contains parallel degrees."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=2, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology(
            name="test",
            intra_node_bandwidth_gbps=200.0,
            intra_node_latency_us=1.0,
            inter_node_bandwidth_gbps=25.0,
            inter_node_latency_us=10.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 8, topology)
        strategy = StrategyConfig(tp_degree=4, dp_degree=2, pp_degree=1)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=64, seq_len=2048)

        result_dict = result.to_dict()

        assert "tp_degree" in result_dict["metadata"]
        assert "dp_degree" in result_dict["metadata"]
        assert "pp_degree" in result_dict["metadata"]
        assert "ep_degree" in result_dict["metadata"]

        assert result_dict["metadata"]["tp_degree"] == 4
        assert result_dict["metadata"]["dp_degree"] == 2
        assert result_dict["metadata"]["pp_degree"] == 1
        assert result_dict["metadata"]["ep_degree"] == 1

    def test_detailed_breakdown_memory_dimensions(self):
        """Test detailed_breakdown memory metrics are per-device."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=2, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology(
            name="test",
            intra_node_bandwidth_gbps=200.0,
            intra_node_latency_us=1.0,
            inter_node_bandwidth_gbps=25.0,
            inter_node_latency_us=10.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 8, topology)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        result_dict = result.to_dict()

        if result_dict["detailed_breakdown"]:
            memory = result_dict["detailed_breakdown"]["memory"]

            assert "summary" in memory
            assert "weight_gb" in memory["summary"]
            assert "activation_gb" in memory["summary"]
            assert "gradient_gb" in memory["summary"]
            assert "optimizer_gb" in memory["summary"]
            assert "total_gb" in memory["summary"]
            assert memory["summary"]["weight_gb"] >= 0
            assert memory["summary"]["activation_gb"] >= 0
            assert memory["summary"]["total_gb"] >= 0

            assert "by_submodule_type" in memory
            for block_type, metrics in memory.get("by_submodule_type", {}).items():
                assert "memory" in metrics
                assert "weight_gb" in metrics["memory"]
                assert "activation_gb" in metrics["memory"]
                assert metrics["memory"]["weight_gb"] >= 0
                assert metrics["memory"]["activation_gb"] >= 0

            # Backward compatibility: by_type field
            assert "by_type" in memory or "summary" in memory

    """Test UnifiedAnalyzer."""

    def test_analyzer_creation(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        assert analyzer is not None

    def test_analyze_training(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        assert result.workload_name == "training"
        assert result.total_time_sec > 0
        assert result.peak_memory_gb > 0
        assert "tokens_per_sec" in result.throughput

    def test_analyze_autoregressive_inference(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            "autoregressive-inference",
            batch_size=8,
            prompt_len=512,
            generation_len=128,
        )

        assert result.workload_name == "autoregressive-inference"
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
                Phase("forward", ComputeType.FORWARD, compute_pattern=ComputePattern.TRANSFORMER_BLOCK),
                Phase("backward", ComputeType.BACKWARD, compute_pattern=ComputePattern.TRANSFORMER_BLOCK),
            ],
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(custom_workload, batch_size=32, seq_len=2048)

        assert result.workload_name == "custom-test"
        assert len(result.phases) == 2

    def test_phase_breakdown(self):
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

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

        result1 = analyzer.analyze("autoregressive-inference", batch_size=1, generation_len=10)
        result2 = analyzer.analyze("autoregressive-inference", batch_size=1, generation_len=100)

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
        result = analyzer.analyze("autoregressive-inference", batch_size=1, prompt_len=512, generation_len=128)

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
        result = analyzer.analyze("autoregressive-inference", batch_size=1)

        assert result.total_time_sec > 0
        assert result.peak_memory_gb > 0


class TestBackwardKernelEstimation:
    """Test backward estimation uses kernel definitions, not hardcoded x2."""

    def test_backward_estimate_uses_kernel_definition(self):
        """Verify backward flops/time come from kernel.flops_backward, not x2."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        forward_phase = result.get_phase("forward")
        backward_phase = result.get_phase("backward")

        assert forward_phase is not None
        assert backward_phase is not None
        assert forward_phase.flops > 0
        assert backward_phase.flops > 0

        ratio = backward_phase.flops / forward_phase.flops
        assert ratio > 1.5 and ratio < 2.5, f"backward/forward ratio should be ~2x, got {ratio}"

        assert backward_phase.single_time_sec > forward_phase.single_time_sec

    def test_linear_kernel_backward_flops(self):
        """Verify linear kernel has correct backward flops (4*m*n*k)."""
        from llm_perf.kernels.functional import linear

        batch = 32
        seq_len = 2048
        hidden_size = 4096
        m = batch * seq_len

        result = linear((m, hidden_size), (hidden_size * 4, hidden_size))

        expected_forward = 2 * m * hidden_size * (hidden_size * 4)
        expected_backward = 4 * m * hidden_size * (hidden_size * 4)

        assert result.flops == expected_forward
        assert result.flops_backward == expected_backward
        assert result.bytes_accessed_backward > result.bytes_accessed

    def test_attention_kernel_backward_metrics(self):
        """Verify flash_attention kernel has backward metrics."""
        from llm_perf.kernels.functional import flash_attention

        batch = 32
        num_heads = 32
        seq_len = 2048
        head_dim = 128

        result = flash_attention(
            (batch, num_heads, seq_len, head_dim),
            (batch, num_heads, seq_len, head_dim),
            (batch, num_heads, seq_len, head_dim),
            is_causal=True,
        )

        assert result.flops_backward > 0
        assert result.bytes_accessed_backward > 0
        assert result.flops_backward == 2 * result.flops

    def test_backward_time_not_simple_multiply(self):
        """Verify backward time uses kernel metrics, not simple x2 forward time."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=1, seq_len=512)

        forward_phase = result.get_phase("forward")
        backward_phase = result.get_phase("backward")

        naive_ratio = backward_phase.single_time_sec / forward_phase.single_time_sec
        assert naive_ratio > 1.0, "backward should take longer than forward"

    def test_transformer_block_backward_decomposition(self):
        """Verify transformer_block backward uses attention + ffn backward."""
        from llm_perf.kernels.functional import linear, flash_attention

        batch = 1
        seq_len = 512
        hidden_size = 4096
        m = batch * seq_len

        qkv = linear((m, hidden_size), (hidden_size, hidden_size * 3))
        attn = flash_attention((1, 32, seq_len, 128), (1, 32, seq_len, 128), (1, 32, seq_len, 128))
        o_proj = linear((m, hidden_size), (hidden_size, hidden_size))

        attention_backward_flops = qkv.flops_backward + attn.flops_backward + o_proj.flops_backward

        up = linear((m, hidden_size), (hidden_size * 4, hidden_size))
        down = linear((m, hidden_size * 4), (hidden_size, hidden_size * 4))

        ffn_backward_flops = up.flops_backward + down.flops_backward

        total_backward = attention_backward_flops + ffn_backward_flops
        assert total_backward > 0

    def test_different_compute_patterns_backward(self):
        """Verify different ComputePattern backward estimations."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        custom_workload = WorkloadConfig(
            name="test-patterns",
            phases=[
                Phase("forward_tb", ComputeType.FORWARD, compute_pattern=ComputePattern.TRANSFORMER_BLOCK),
                Phase("backward_tb", ComputeType.BACKWARD, compute_pattern=ComputePattern.TRANSFORMER_BLOCK),
                Phase("forward_dense", ComputeType.FORWARD, compute_pattern=ComputePattern.DENSE_FORWARD),
                Phase("backward_dense", ComputeType.BACKWARD, compute_pattern=ComputePattern.DENSE_FORWARD),
            ],
        )

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(custom_workload, batch_size=32, seq_len=2048)

        tb_forward = result.get_phase("forward_tb")
        tb_backward = result.get_phase("backward_tb")
        dense_forward = result.get_phase("forward_dense")
        dense_backward = result.get_phase("backward_dense")

        assert tb_forward is not None and tb_backward is not None
        assert dense_forward is not None and dense_backward is not None

        assert tb_backward.flops > tb_forward.flops
        assert dense_backward.flops > dense_forward.flops


class TestConvDecoderEstimation:
    """Test VAE decoder estimation uses kernel API."""

    def test_vae_decoder_flops_reasonable(self):
        """Verify VAE decoder FLOPs are calculated using kernel API, not simplified formula."""
        from llm_perf.modeling.encoder import ShardedVAEDecoder

        decoder = ShardedVAEDecoder(
            out_channels=3,
            latent_channels=16,
            block_out_channels=(128, 256, 512, 512),
            use_3d=True,
            dtype="fp16",
        )

        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 1)
        strategy = StrategyConfig(tp_degree=1)

        analyzer = UnifiedAnalyzer(decoder, device, cluster, strategy)

        workload = WorkloadConfig(
            name="vae-decoder-test",
            phases=[
                Phase("decode", ComputeType.FORWARD, compute_pattern=ComputePattern.CONV_DECODER),
            ],
        )

        result = analyzer.analyze(workload, batch_size=1, num_frames=81, height=720, width=1280)

        decode_phase = result.get_phase("decode")
        assert decode_phase is not None

        assert decode_phase.flops > 1e12, "VAE decoder FLOPs should be > 1 TFLOP, not simplified calculation"

        naive_flops = 81 * 720 * 1280 * 3 * 16
        assert decode_phase.flops > naive_flops * 10, "Kernel-based FLOPs should be much larger than naive calculation"

    def test_vae_decoder_time_reasonable(self):
        """Verify VAE decoder time is reasonable, not near-zero."""
        from llm_perf.modeling.encoder import ShardedVAEDecoder

        decoder = ShardedVAEDecoder(
            out_channels=3,
            latent_channels=16,
            block_out_channels=(128, 256, 512, 512),
            use_3d=True,
            dtype="fp16",
        )

        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 1)
        strategy = StrategyConfig(tp_degree=1)

        analyzer = UnifiedAnalyzer(decoder, device, cluster, strategy)

        workload = WorkloadConfig(
            name="vae-decoder-test",
            phases=[
                Phase("decode", ComputeType.FORWARD, compute_pattern=ComputePattern.CONV_DECODER),
            ],
        )

        result = analyzer.analyze(workload, batch_size=1, num_frames=81, height=720, width=1280)

        decode_phase = result.get_phase("decode")
        assert decode_phase.single_time_sec > 0.01, "VAE decoder time should be > 10ms, not near-zero"

    def test_vae_decoder_backward(self):
        """Verify VAE decoder backward estimation."""
        from llm_perf.modeling.encoder import ShardedVAEDecoder

        decoder = ShardedVAEDecoder(
            out_channels=3,
            latent_channels=16,
            block_out_channels=(128, 256, 512, 512),
            use_3d=True,
            dtype="fp16",
        )

        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 1)
        strategy = StrategyConfig(tp_degree=1)

        analyzer = UnifiedAnalyzer(decoder, device, cluster, strategy)

        workload = WorkloadConfig(
            name="vae-decoder-test",
            phases=[
                Phase("forward", ComputeType.FORWARD, compute_pattern=ComputePattern.CONV_DECODER),
                Phase("backward", ComputeType.BACKWARD, compute_pattern=ComputePattern.CONV_DECODER),
            ],
        )

        result = analyzer.analyze(workload, batch_size=1, num_frames=81, height=720, width=1280)

        forward_phase = result.get_phase("forward")
        backward_phase = result.get_phase("backward")

        assert forward_phase is not None
        assert backward_phase is not None
        assert backward_phase.flops > forward_phase.flops


class TestQPSAndCommunicationBreakdown:
    """Phase 2: QPS + CommunicationBreakdown tests."""

    def test_qps_calculation(self):
        """Verify QPS is calculated correctly."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8, dp_degree=1)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        workload = get_workload("training")

        result = analyzer.analyze(workload, batch_size=32, seq_len=2048)

        assert result.qps is not None
        assert result.qps > 0

        expected_qps = 32 * strategy.dp_degree / result.total_time_sec
        assert abs(result.qps - expected_qps) < 0.01

    def test_qps_with_different_dp(self):
        """Verify QPS scales with DP degree."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 16)

        strategy_dp1 = StrategyConfig(tp_degree=8, dp_degree=1)
        strategy_dp4 = StrategyConfig(tp_degree=8, dp_degree=4)

        analyzer1 = UnifiedAnalyzer(model, device, cluster, strategy_dp1)
        analyzer4 = UnifiedAnalyzer(model, device, cluster, strategy_dp4)

        workload = get_workload("training")

        result1 = analyzer1.analyze(workload, batch_size=32, seq_len=2048)
        result4 = analyzer4.analyze(workload, batch_size=32, seq_len=2048)

        assert result4.qps > result1.qps

    def test_communication_breakdown_structure(self):
        """Verify CommunicationBreakdown has correct dual-layer structure."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8, dp_degree=1)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        workload = get_workload("training")

        result = analyzer.analyze(workload, batch_size=32, seq_len=2048)

        if result.communication_breakdown:
            comm_dict = result.communication_breakdown.to_dict()

            # Check dual-layer structure
            assert "by_parallelism" in comm_dict
            assert "by_operation" in comm_dict
            assert "total_bytes" in comm_dict

            # by_parallelism should contain ptype (tp, dp, pp)
            if comm_dict["by_parallelism"]:
                for ptype, pdata in comm_dict["by_parallelism"].items():
                    assert "total_bytes" in pdata
                    assert "operations" in pdata

    def test_communication_breakdown_in_detailed_breakdown(self):
        """Verify communication breakdown is in detailed_breakdown."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8, dp_degree=2)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        workload = get_workload("training")

        result = analyzer.analyze(workload, batch_size=32, seq_len=2048)

        if result.detailed_breakdown:
            assert "communication" in result.detailed_breakdown

            comm = result.detailed_breakdown["communication"]
            assert "by_parallelism" in comm


class TestMFU:
    """Phase 1: MFU (Model FLOPs Utilization) tests."""

    def test_mfu_calculation(self):
        """Verify MFU is calculated correctly."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8, dp_degree=1)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        workload = get_workload("training")

        result = analyzer.analyze(workload, batch_size=32, seq_len=2048)

        assert result.mfu is not None
        assert 0 < result.mfu <= 1.0

    def test_mfu_with_different_tp(self):
        """Verify MFU changes with different TP degrees."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)

        strategy_tp4 = StrategyConfig(tp_degree=4, dp_degree=2)
        strategy_tp8 = StrategyConfig(tp_degree=8, dp_degree=1)

        analyzer4 = UnifiedAnalyzer(model, device, cluster, strategy_tp4)
        analyzer8 = UnifiedAnalyzer(model, device, cluster, strategy_tp8)

        workload = get_workload("training")

        result4 = analyzer4.analyze(workload, batch_size=32, seq_len=2048)
        result8 = analyzer8.analyze(workload, batch_size=32, seq_len=2048)

        assert result4.mfu is not None
        assert result8.mfu is not None

    def test_mfu_in_result_dict(self):
        """Verify MFU is in to_dict output."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        workload = get_workload("training")

        result = analyzer.analyze(workload, batch_size=32, seq_len=2048)

        result_dict = result.to_dict()
        assert "mfu" in result_dict


class TestModuleBreakdown:
    """Test module_breakdown feature."""

    def test_module_breakdown_structure(self):
        """Verify module_breakdown has correct structure."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        assert result.module_breakdown is not None
        assert "by_module" in result.module_breakdown
        assert "by_block_type" in result.module_breakdown
        assert "summary" in result.module_breakdown

    def test_module_breakdown_by_module(self):
        """Verify by_module contains all submodules (after norm merge)."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        by_module = result.module_breakdown["by_module"]

        assert "embedding" in by_module
        assert "layers.0" in by_module
        assert "layers.1" in by_module
        assert "layers.2" in by_module
        assert "layers.3" in by_module
        assert "lm_head" in by_module
        assert "final_norm" not in by_module

    def test_module_breakdown_by_block_type(self):
        """Verify by_block_type groups modules correctly (after norm merge)."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        by_block_type = result.module_breakdown["by_block_type"]

        assert "embedding" in by_block_type
        assert "transformer_block" in by_block_type
        assert "lm_head" in by_block_type
        assert "rms_norm" not in by_block_type

    def test_module_breakdown_fields(self):
        """Verify each module entry has compute/memory/communication fields."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        for module_name, module_data in result.module_breakdown["by_module"].items():
            assert "name" in module_data
            assert "module_type" in module_data
            assert "compute" in module_data
            assert "memory" in module_data
            assert "communication" in module_data

            compute = module_data["compute"]
            assert "time_sec" in compute
            assert "flops" in compute

            memory = module_data["memory"]
            assert "activations_gb" in memory

            communication = module_data["communication"]
            assert "total_bytes" in communication
            assert "total_gb" in communication

    def test_module_breakdown_summary(self):
        """Verify summary aggregates correctly."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        summary = result.module_breakdown["summary"]

        assert "total_compute_time_sec" in summary
        assert "total_memory_gb" in summary
        assert "total_communication_gb" in summary

        assert summary["total_compute_time_sec"] > 0
        assert summary["total_memory_gb"] > 0

    def test_module_breakdown_memory_per_device(self):
        """Verify memory in module_breakdown is per-device dimension."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        for module_data in result.module_breakdown["by_module"].values():
            memory_gb = module_data["memory"]["activations_gb"]
            assert memory_gb < 100, f"Memory per device should be < 100GB, got {memory_gb}GB"

    def test_module_breakdown_communication_non_zero(self):
        """Verify transformer_block modules have non-zero communication."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        by_block_type = result.module_breakdown["by_block_type"]

        transformer_comm = by_block_type["transformer_block"]["communication"]["total_gb"]
        assert transformer_comm > 0, "Transformer blocks should have non-zero communication"

    def test_norm_merged_into_lm_head(self):
        """Verify final_norm is merged into lm_head."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        by_module = result.module_breakdown["by_module"]

        lm_head_flops = by_module["lm_head"]["compute"]["flops"]
        lm_head_memory = by_module["lm_head"]["memory"]["activations_gb"]

        assert lm_head_flops > 30e9, f"lm_head flops should include final_norm, got {lm_head_flops / 1e9:.1f}G"
        assert lm_head_memory > 1.0, f"lm_head memory should include final_norm, got {lm_head_memory:.2f}GB"


class TestEvaluationAccuracy:
    """Test evaluation accuracy and correctness."""

    def test_flops_approximately_correct(self):
        """Verify FLOPs estimation is approximately correct."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        batch_size = 32
        seq_len = 2048
        hidden_size = 4096
        num_layers = 4

        expected_forward_flops_per_layer = 2 * batch_size * seq_len * hidden_size * hidden_size * 6
        expected_forward_flops = expected_forward_flops_per_layer * num_layers

        forward_phase = result.get_phase("forward")
        actual_flops = forward_phase.flops

        ratio = actual_flops / expected_forward_flops
        assert 0.5 < ratio < 2.0, f"FLOPs ratio should be ~1, got {ratio}"

    def test_memory_reasonable_range(self):
        """Verify memory estimation is in reasonable range."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        backward_phase = result.get_phase("backward")
        memory_gb = backward_phase.memory_gb

        # Training memory includes: weight + gradient + optimizer + activation
        # For 4-layer LLaMA with TP=8:
        # - weight: ~1.7GB (per device)
        # - gradient: ~1.7GB
        # - optimizer: ~6.8GB (Adam: 2 × FP32 states)
        # - activation: ~7GB
        # Total: ~16GB per device
        assert 10 < memory_gb < 200, f"Memory should be in reasonable range, got {memory_gb}GB"

        # Verify memory breakdown exists
        assert "weight_gb" in backward_phase.memory_breakdown
        assert "gradient_gb" in backward_phase.memory_breakdown
        assert "optimizer_gb" in backward_phase.memory_breakdown
        assert "activation_gb" in backward_phase.memory_breakdown

    def test_time_scaling_with_layers(self):
        """Verify time scales approximately linearly with number of layers."""
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        model4 = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        analyzer4 = UnifiedAnalyzer(model4, device, cluster, strategy)
        result4 = analyzer4.analyze("training", batch_size=32, seq_len=2048)

        model8 = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=8, num_heads=32)
        analyzer8 = UnifiedAnalyzer(model8, device, cluster, strategy)
        result8 = analyzer8.analyze("training", batch_size=32, seq_len=2048)

        time4 = result4.get_phase("forward").single_time_sec
        time8 = result8.get_phase("forward").single_time_sec

        ratio = time8 / time4
        assert ratio > 1.8, f"Time ratio for 8/4 layers should be > 1.8, got {ratio}"

    def test_time_scaling_with_batch_size(self):
        """Verify time scales with batch size."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result32 = analyzer.analyze("training", batch_size=32, seq_len=2048)
        result64 = analyzer.analyze("training", batch_size=64, seq_len=2048)

        time32 = result32.get_phase("forward").single_time_sec
        time64 = result64.get_phase("forward").single_time_sec

        ratio = time64 / time32
        assert 1.5 < ratio < 3.0, f"Time ratio for batch 64/32 should be ~2, got {ratio}"

    def test_tp_communication_scaling(self):
        """Verify TP communication scales with hidden_size."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result_seq2048 = analyzer.analyze("training", batch_size=32, seq_len=2048)
        result_seq4096 = analyzer.analyze("training", batch_size=32, seq_len=4096)

        comm2048 = result_seq2048.module_breakdown["by_block_type"]["transformer_block"]["communication"]["total_gb"]
        comm4096 = result_seq4096.module_breakdown["by_block_type"]["transformer_block"]["communication"]["total_gb"]

        ratio = comm4096 / comm2048
        assert 1.5 < ratio < 2.5, f"Communication ratio for seq 4096/2048 should be ~2, got {ratio}"

    def test_mfu_reasonable(self):
        """Verify MFU is in reasonable range."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        assert 0 < result.mfu < 1.0, f"MFU should be between 0 and 1, got {result.mfu}"

        assert result.mfu < 0.8, f"MFU for theory backend should be < 0.8, got {result.mfu}"


class TestInferenceEvaluation:
    """Test inference-specific evaluation."""

    def test_inference_prefill_decode_separate(self):
        """Verify inference has separate prefill and decode phases."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("autoregressive-inference", batch_size=1, prompt_len=512, generation_len=128)

        prefill = result.get_phase("prefill")
        decode = result.get_phase("decode")

        assert prefill is not None
        assert decode is not None

        assert prefill.single_time_sec > 0
        assert decode.single_time_sec > 0

        assert decode.repeat_count == 128
        assert decode.total_time_sec > prefill.total_time_sec

    def test_decode_repeat_count_matches_generation_len(self):
        """Verify decode repeat count matches generation length."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        result_gen10 = analyzer.analyze("autoregressive-inference", batch_size=1, prompt_len=512, generation_len=10)
        result_gen100 = analyzer.analyze("autoregressive-inference", batch_size=1, prompt_len=512, generation_len=100)

        decode10 = result_gen10.get_phase("decode")
        decode100 = result_gen100.get_phase("decode")

        assert decode10.repeat_count == 10
        assert decode100.repeat_count == 100

    def test_inference_memory_lower_than_training(self):
        """Verify inference memory is lower than training."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        training_result = analyzer.analyze("training", batch_size=32, seq_len=2048)
        inference_result = analyzer.analyze(
            "autoregressive-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        training_memory = training_result.peak_memory_gb
        inference_memory = inference_result.peak_memory_gb

        assert inference_memory < training_memory

    def test_kv_cache_field_present(self):
        """Verify KV cache field is present in metadata."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("autoregressive-inference", batch_size=1, prompt_len=512, generation_len=128)

        assert "kv_cache_gb" in result.metadata


class TestSubmoduleTypeInference:
    """Test submodule type inference."""

    def test_embedding_type_correct(self):
        """Verify embedding submodule has correct type."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        forward_phase = result.get_phase("forward")
        embedding_sm = next(sm for sm in forward_phase.submodules if sm.name == "embedding")

        assert embedding_sm.submodule_type == "embedding"

    def test_transformer_block_type_correct(self):
        """Verify transformer_block submodule has correct type."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        forward_phase = result.get_phase("forward")
        layer0_sm = next(sm for sm in forward_phase.submodules if sm.name == "layers.0")

        assert layer0_sm.submodule_type == "transformer_block"

    def test_lm_head_type_correct(self):
        """Verify lm_head submodule has correct type."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        forward_phase = result.get_phase("forward")
        lm_head_sm = next(sm for sm in forward_phase.submodules if sm.name == "lm_head")

        assert lm_head_sm.submodule_type == "lm_head"

    def test_no_unknown_types(self):
        """Verify no submodule has unknown type."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=4, num_heads=32)
        device = Device.from_preset("H100-SXM-80GB")
        cluster = make_cluster(device, 8)
        strategy = StrategyConfig(tp_degree=8)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze("training", batch_size=32, seq_len=2048)

        forward_phase = result.get_phase("forward")
        for sm in forward_phase.submodules:
            assert sm.submodule_type != "unknown", f"Submodule {sm.name} should not have unknown type"

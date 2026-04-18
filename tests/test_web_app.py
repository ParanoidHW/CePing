"""Tests for web application API endpoints."""

from llm_perf.app import Evaluator


class TestWebAppAPI:
    """Test web app API responses."""

    def test_evaluator_training_wan_dit(self):
        """Test /api/evaluate/training endpoint with wan-dit preset."""
        evaluator = Evaluator()
        result = evaluator.evaluate(
            "wan-dit",
            "H100-SXM-80GB",
            "llm-training",
            "tp8",
            batch_size=1,
            seq_len=2048,
        )

        result_dict = result.to_dict()

        assert result_dict["total_time_sec"] > 0
        assert result_dict["peak_memory_gb"] > 0
        assert "phases" in result_dict
        assert len(result_dict["phases"]) > 0

    def test_evaluator_pipeline_diffusion_video(self):
        """Test /api/evaluate/pipeline/diffusion-video endpoint."""
        from llm_perf.analyzer.unified import UnifiedAnalyzer
        from llm_perf.analyzer.workload_loader import get_workload
        from llm_perf.hardware.cluster import Cluster
        from llm_perf.hardware.device import Device
        from llm_perf.hardware.topology import NetworkTopology
        from llm_perf.modeling import create_model_from_config
        from llm_perf.strategy.base import StrategyConfig

        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology(
            name="default",
            intra_node_bandwidth_gbps=200.0,
            intra_node_latency_us=1.0,
            inter_node_bandwidth_gbps=25.0,
            inter_node_latency_us=10.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 8, topology)
        strategy = StrategyConfig(tp_degree=8)

        models = {
            "encoder": create_model_from_config({"type": "wan-text-encoder"}),
            "backbone": create_model_from_config({"type": "wan-dit"}),
            "decoder": create_model_from_config({"type": "wan-vae"}),
        }

        analyzer = UnifiedAnalyzer(models, device, cluster, strategy)
        workload = get_workload("diffusion-pipeline")
        result = analyzer.analyze(
            workload,
            num_frames=81,
            height=720,
            width=1280,
            num_steps=50,
            use_cfg=True,
        )

        result_dict = result.to_dict()

        assert result_dict["total_time_sec"] > 0
        assert result_dict["peak_memory_gb"] > 0
        assert "phases" in result_dict
        assert len(result_dict["phases"]) >= 3

        phase_names = [p["name"] for p in result_dict["phases"]]
        assert "encode" in phase_names
        assert "denoise" in phase_names
        assert "decode" in phase_names

        denoise_phase = next(p for p in result_dict["phases"] if p["name"] == "denoise")
        assert denoise_phase["repeat_count"] == 50
        assert denoise_phase["total_time_sec"] > 0

        encode_phase = next(p for p in result_dict["phases"] if p["name"] == "encode")
        assert encode_phase["single_time_sec"] > 0

        decode_phase = next(p for p in result_dict["phases"] if p["name"] == "decode")
        assert decode_phase["single_time_sec"] > 0

    def test_evaluator_pipeline_result_format(self):
        """Test that pipeline result contains required fields for frontend."""
        from llm_perf.analyzer.unified import UnifiedAnalyzer
        from llm_perf.analyzer.workload_loader import get_workload
        from llm_perf.hardware.cluster import Cluster
        from llm_perf.hardware.device import Device
        from llm_perf.hardware.topology import NetworkTopology
        from llm_perf.modeling import create_model_from_config
        from llm_perf.strategy.base import StrategyConfig

        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology(
            name="default",
            intra_node_bandwidth_gbps=200.0,
            intra_node_latency_us=1.0,
            inter_node_bandwidth_gbps=25.0,
            inter_node_latency_us=10.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 8, topology)
        strategy = StrategyConfig(tp_degree=8)

        models = {
            "encoder": create_model_from_config({"type": "wan-text-encoder"}),
            "backbone": create_model_from_config({"type": "wan-dit"}),
            "decoder": create_model_from_config({"type": "wan-vae"}),
        }

        analyzer = UnifiedAnalyzer(models, device, cluster, strategy)
        workload = get_workload("diffusion-pipeline")
        result = analyzer.analyze(
            workload,
            num_frames=81,
            height=720,
            width=1280,
            num_steps=50,
            use_cfg=True,
        )

        result_dict = result.to_dict()

        assert "total_time_sec" in result_dict
        assert "peak_memory_gb" in result_dict
        assert "throughput" in result_dict
        assert "pixels_per_sec" in result_dict["throughput"]
        assert "params" in result_dict
        assert "num_frames" in result_dict["params"]
        assert "height" in result_dict["params"]
        assert "width" in result_dict["params"]

        for phase in result_dict["phases"]:
            assert "name" in phase
            assert "component" in phase
            assert "single_time_sec" in phase
            assert "total_time_sec" in phase
            assert "repeat_count" in phase

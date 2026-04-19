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

    def test_training_result_has_breakdown_fields(self):
        """Test training result contains breakdown and detailed_breakdown fields."""
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

        assert "breakdown" in result_dict
        assert result_dict["breakdown"] is not None
        assert "overview" in result_dict["breakdown"]
        assert "time_breakdown" in result_dict["breakdown"]
        assert "layers" in result_dict["breakdown"]
        assert "total_time_sec" in result_dict["breakdown"]["overview"]
        assert "compute_sec" in result_dict["breakdown"]["time_breakdown"]
        assert len(result_dict["breakdown"]["layers"]) > 0

        assert "detailed_breakdown" in result_dict
        assert result_dict["detailed_breakdown"] is not None
        assert "submodels" in result_dict["detailed_breakdown"]
        assert "memory" in result_dict["detailed_breakdown"]
        assert "by_type" in result_dict["detailed_breakdown"]["memory"]

    def test_pipeline_phases_not_zero(self):
        """Test diffusion-video pipeline phases have non-zero time values."""
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

        for phase in result_dict["phases"]:
            assert phase["total_time_sec"] > 0, f"Phase {phase['name']} has zero total_time_sec"
            assert phase["single_time_sec"] > 0, f"Phase {phase['name']} has zero single_time_sec"

    def test_component_presets_filtered_for_inference(self):
        """Verify component presets are filtered for inference mode."""
        from llm_perf.modeling.registry import get_model_presets

        presets = get_model_presets()

        component_presets = [key for key, preset in presets.items() if preset.get("preset_type") == "component"]
        assert len(component_presets) > 0, "Should have at least one component preset"

        model_presets = [key for key, preset in presets.items() if preset.get("preset_type") in ("model", "pipeline")]
        assert len(model_presets) > 0, "Should have at least one model/pipeline preset"

    def test_frontend_compatibility_fields(self):
        """Test that result.to_dict() contains frontend compatibility fields."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()

        assert "time" in result_dict
        assert "time_per_step_sec" in result_dict["time"]
        assert "memory" in result_dict
        assert "memory_per_gpu_gb" in result_dict["memory"]
        assert "prefill" in result_dict
        assert "decode" in result_dict
        assert "end_to_end" in result_dict

    def test_memory_is_per_gpu(self):
        """Test that memory metrics are per-GPU."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()

        assert result_dict["peak_memory_gb"] > 0
        assert result_dict["memory"]["memory_per_gpu_gb"] == result_dict["peak_memory_gb"]

        if result_dict["detailed_breakdown"]:
            memory = result_dict["detailed_breakdown"]["memory"]
            assert "by_type" in memory
            assert "activations_gb" in memory["by_type"]

    def test_throughput_is_global(self):
        """Test that throughput metrics are global."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8_dp4", batch_size=128)

        result_dict = result.to_dict()

        assert "throughput" in result_dict
        assert "tokens_per_sec" in result_dict["throughput"]

        assert "decode" in result_dict
        if result_dict["decode"]:
            assert result_dict["decode"]["tps"] == result_dict["throughput"]["tokens_per_sec"]

    def test_inference_result_fields(self):
        """Test inference result contains prefill and decode fields."""
        evaluator = Evaluator()
        result = evaluator.evaluate(
            "llama-7b", "H100-SXM-80GB", "llm-inference", "tp8", batch_size=8, prompt_len=512, generation_len=128
        )

        result_dict = result.to_dict()

        assert result_dict["prefill"] is not None
        assert result_dict["decode"] is not None

        assert result_dict["prefill"]["ttft_sec"] > 0
        assert result_dict["decode"]["tps"] > 0
        assert result_dict["decode"]["tpot_sec"] > 0

        assert result_dict["end_to_end"]["overall_tps"] > 0

    def test_metadata_has_parallel_degrees(self):
        """Test that metadata contains parallel degrees."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp4_dp2", batch_size=64)

        result_dict = result.to_dict()

        assert "metadata" in result_dict
        assert "tp_degree" in result_dict["metadata"]
        assert "dp_degree" in result_dict["metadata"]
        assert "pp_degree" in result_dict["metadata"]

        assert result_dict["metadata"]["tp_degree"] == 4
        assert result_dict["metadata"]["dp_degree"] == 2

    def test_detailed_breakdown_by_block_type(self):
        """Test that detailed_breakdown contains by_block_type."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()

        if result_dict["detailed_breakdown"]:
            memory = result_dict["detailed_breakdown"]["memory"]
            assert "by_block_type" in memory

            if memory["by_block_type"]:
                for block_type, metrics in memory["by_block_type"].items():
                    assert "activations_gb" in metrics
                    assert "compute" in metrics
                    assert "flops" in metrics["compute"]
                    assert metrics["activations_gb"] >= 0

    def test_communication_breakdown_per_gpu(self):
        """Test that communication breakdown is per-GPU."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()

        if result_dict["communication_breakdown"]:
            for comm_type, data in result_dict["communication_breakdown"].items():
                if data and "total_bytes" in data:
                    assert data["total_bytes"] >= 0

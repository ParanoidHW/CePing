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
        assert "summary" in result_dict["detailed_breakdown"]["memory"]

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
            assert "summary" in memory
            assert "total_gb" in memory["summary"]
            # Backward compatibility
            assert "by_type" in memory or "summary" in memory

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
            detailed = result_dict["detailed_breakdown"]
            memory = detailed.get("memory", {})

            # Check by_submodule_type
            assert "by_submodule_type" in memory

            if memory["by_submodule_type"]:
                for block_type, metrics in memory["by_submodule_type"].items():
                    assert "activations_gb" in metrics
                    assert metrics["activations_gb"] >= 0

            # Check unified by_submodule_type structure
            assert "by_submodule_type" in detailed
            for submodule_type, data in detailed["by_submodule_type"].items():
                assert "memory" in data
                assert "compute" in data
                assert "communication" in data
                assert "activations_gb" in data["memory"]
                assert "flops" in data["compute"]
                assert "gb" in data["communication"]

    def test_communication_breakdown_per_gpu(self):
        """Test that communication breakdown is per-GPU."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()

        if result_dict["communication_breakdown"]:
            for comm_type, data in result_dict["communication_breakdown"].items():
                if data and "total_bytes" in data:
                    assert data["total_bytes"] >= 0


class TestFrontendCompatibility:
    """Test frontend JavaScript compatibility - ensure fields match what frontend expects."""

    def test_detailed_breakdown_all_fields_types(self):
        """Test detailed_breakdown fields have correct types for frontend toFixed."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()
        detailed = result_dict.get("detailed_breakdown")

        if not detailed:
            return

        # Test memory.summary - all values should be numbers
        mem_summary = detailed.get("memory", {}).get("summary", {})
        for key, value in mem_summary.items():
            assert isinstance(value, (int, float)), f"memory.summary[{key}] should be number, got {type(value)}"

        # Test memory.by_submodule - all nested values should be numbers
        mem_by_submodel = detailed.get("memory", {}).get("by_submodel", {})
        for name, mems in mem_by_submodel.items():
            for key, value in mems.items():
                assert isinstance(value, (int, float)), f"memory.by_submodel[{name}][{key}] should be number"

        # Test memory.by_submodule_type - activations_gb should be number
        mem_by_submodule_type = detailed.get("memory", {}).get("by_submodule_type", {})
        for submodule_type, data in mem_by_submodule_type.items():
            assert "activation_gb" in data or "activations_gb" in data, (
                f"memory.by_submodule_type[{submodule_type}] missing activation fields"
            )
            activation_val = data.get("activation_gb", data.get("activations_gb", 0))
            assert isinstance(activation_val, (int, float)), f"activation should be number"

        # Test unified by_submodule_type structure
        by_submodule_type = detailed.get("by_submodule_type", {})
        for submodule_type, data in by_submodule_type.items():
            # memory nested fields should be numbers
            if "memory" in data:
                for key, value in data["memory"].items():
                    assert isinstance(value, (int, float)), f"memory[{key}] should be number"

            # compute nested fields should be numbers
            if "compute" in data:
                for key, value in data["compute"].items():
                    assert isinstance(value, (int, float)), f"compute[{key}] should be number"

            # communication nested fields should be numbers
            if "communication" in data:
                for key, value in data["communication"].items():
                    assert isinstance(value, (int, float)), f"communication[{key}] should be number"

    def test_communication_breakdown_fields_types(self):
        """Test communication breakdown fields for frontend."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()

        # Test detailed_breakdown.communication.by_parallelism
        detailed = result_dict.get("detailed_breakdown")
        if detailed:
            comm_by_para = detailed.get("communication", {}).get("by_parallelism", {})
            for comm_type, data in comm_by_para.items():
                if data:
                    # Frontend uses total_bytes and total_time_sec
                    if "total_bytes" in data:
                        assert isinstance(data["total_bytes"], (int, float)), f"total_bytes should be number"
                    if "total_time_sec" in data:
                        assert isinstance(data["total_time_sec"], (int, float)), f"total_time_sec should be number"

        # Test top-level communication_breakdown
        comm_breakdown = result_dict.get("communication_breakdown")
        if comm_breakdown:
            for comm_type, data in comm_breakdown.items():
                if data and "total_bytes" in data:
                    assert isinstance(data["total_bytes"], (int, float))

    def test_no_flops_at_top_level_of_memory(self):
        """Ensure flops is NOT at top level of memory.by_submodule_type (prevent frontend misreading)."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()
        detailed = result_dict.get("detailed_breakdown")

        if not detailed:
            return

        mem_by_submodule_type = detailed.get("memory", {}).get("by_submodule_type", {})
        for submodule_type, data in mem_by_submodule_type.items():
            # flops should NOT be at top level of memory
            assert "flops" not in data, (
                f"flops should not be at top level of memory.by_submodule_type[{submodule_type}]"
            )
            assert "time_sec" not in data, (
                f"time_sec should not be at top level of memory.by_submodule_type[{submodule_type}]"
            )

    def test_submodels_memory_fields(self):
        """Test submodels memory fields for frontend."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()
        detailed = result_dict.get("detailed_breakdown")

        if not detailed:
            return

        submodels = detailed.get("submodels", [])
        for sm in submodels:
            mem_summary = sm.get("memory", {}).get("summary", {})
            for key, value in mem_summary.items():
                assert isinstance(value, (int, float)), f"submodel.memory.summary[{key}] should be number"

    def test_inference_detailed_breakdown_fields(self):
        """Test inference detailed_breakdown fields."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "inference", "tp8", batch_size=1)

        result_dict = result.to_dict()
        detailed = result_dict.get("detailed_breakdown")

        if not detailed:
            return

        # Same checks as training
        mem_by_submodule_type = detailed.get("memory", {}).get("by_submodule_type", {})
        for submodule_type, data in mem_by_submodule_type.items():
            assert "activation_gb" in data or "activations_gb" in data
            activation_val = data.get("activation_gb", data.get("activations_gb", 0))
            assert isinstance(activation_val, (int, float))

        # Check by_submodule_type has complete structure
        by_submodule_type = detailed.get("by_submodule_type", {})
        for submodule_type, data in by_submodule_type.items():
            assert "memory" in data
            assert "compute" in data
            assert "communication" in data

    def test_memory_by_type_fields_exist(self):
        """Test memory.by_type and by_submodel fields exist for frontend table display."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()
        detailed = result_dict.get("detailed_breakdown")

        if not detailed:
            return

        mem = detailed.get("memory", {})

        # Check by_type exists with correct fields
        by_type = mem.get("by_type", {})
        assert len(by_type) > 0, "memory.by_type should not be empty"
        assert "weight" in by_type, "memory.by_type should have 'weight'"
        assert "gradient" in by_type, "memory.by_type should have 'gradient'"
        assert "optimizer" in by_type, "memory.by_type should have 'optimizer'"
        assert "activation" in by_type, "memory.by_type should have 'activation'"
        assert "total" in by_type, "memory.by_type should have 'total'"

        # Check values are numbers and non-negative
        for key, value in by_type.items():
            assert isinstance(value, (int, float)), f"by_type[{key}] should be number"
            assert value >= 0, f"by_type[{key}] should be non-negative"

        # Check weight > 0 for training
        assert by_type["weight"] > 0, "weight should be > 0 for training"
        assert by_type["gradient"] > 0, "gradient should be > 0 for training"
        assert by_type["optimizer"] > 0, "optimizer should be > 0 for training"

        # Check by_submodel exists
        by_submodel = mem.get("by_submodel", {})
        assert len(by_submodel) > 0, "memory.by_submodel should not be empty"

        for name, data in by_submodel.items():
            assert "activations_gb" in data, f"by_submodel[{name}] should have activations_gb"
            assert isinstance(data["activations_gb"], (int, float)), "activations_gb should be number"
            assert data["activations_gb"] > 0, f"by_submodel[{name}] activations_gb should be > 0"

"""Tests for web application API endpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_perf.app import Evaluator
from web.app import app


class TestWorkloadScenarios:
    """Test workload scenario selection and parameter handling."""

    def test_workload_training_scenario(self):
        """Test training workload scenario via /api/evaluate."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": {"scenario": "training", "batch_size": 32, "seq_len": 4096},
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["result"]["peak_memory_gb"] > 0

    def test_workload_inference_scenario(self):
        """Test inference workload scenario via /api/evaluate."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": {"scenario": "inference", "batch_size": 1, "input_tokens": 1000, "output_tokens": 100},
                "batch_size": 1,
                "prompt_len": 1000,
                "generation_len": 100,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["result"]["prefill"] is not None
        assert data["result"]["decode"] is not None

    def test_workload_rl_training_scenario(self):
        """Test rl_training workload scenario via /api/evaluate."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": {"scenario": "rl_training", "batch_size": 32, "seq_len": 4096},
                "batch_size": 32,
                "seq_len": 4096,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["result"]["peak_memory_gb"] > 0

    def test_workload_string_format(self):
        """Test workload as string (legacy format)."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["result"]["peak_memory_gb"] > 0


class TestWebHTTPAPI:
    """Test web app HTTP API responses using Flask test client."""

    def test_http_api_memory_by_type_exists(self):
        """Test that /api/evaluate returns memory.by_type for frontend table."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        detailed = result.get("detailed_breakdown")
        assert detailed is not None

        mem = detailed.get("memory", {})
        by_type = mem.get("by_type", {})

        # Verify by_type has all required fields
        assert len(by_type) > 0, "memory.by_type should not be empty"
        assert "weight" in by_type
        assert "gradient" in by_type
        assert "optimizer" in by_type
        assert "activation" in by_type
        assert "total" in by_type

        # Verify values are numbers
        for key, value in by_type.items():
            assert isinstance(value, (int, float)), f"by_type[{key}] should be number"
            assert value >= 0, f"by_type[{key}] should be non-negative"

        # Verify training has gradient and optimizer
        assert by_type["weight"] > 0, "weight should be > 0"
        assert by_type["gradient"] > 0, "gradient should be > 0 for training"
        assert by_type["optimizer"] > 0, "optimizer should be > 0 for training"

    def test_http_api_memory_by_submodel_exists(self):
        """Test that /api/evaluate returns memory.by_submodel."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        detailed = result.get("detailed_breakdown")

        mem = detailed.get("memory", {})
        by_submodel = mem.get("by_submodel", {})

        assert len(by_submodel) > 0, "memory.by_submodel should not be empty"

        for name, data in by_submodel.items():
            assert "activations_gb" in data, f"by_submodel[{name}] missing activations_gb"
            assert isinstance(data["activations_gb"], (int, float))
            assert data["activations_gb"] > 0

    def test_http_api_memory_by_submodule_type_has_data(self):
        """Test that memory.by_submodule_type has actual data, not empty."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        detailed = result.get("detailed_breakdown")

        mem = detailed.get("memory", {})
        by_submodule_type = mem.get("by_submodule_type", {})

        # Should have at least embedding, transformer_block, lm_head
        assert len(by_submodule_type) >= 3, "by_submodule_type should have at least 3 module types"

        for submodule_type, data in by_submodule_type.items():
            assert "memory" in data
            assert "weight_gb" in data["memory"]
            assert isinstance(data["memory"]["weight_gb"], (int, float))
            assert data["memory"]["weight_gb"] >= 0


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
        assert "memory_per_device_gb" in result_dict["memory"]
        assert "prefill" in result_dict
        assert "decode" in result_dict
        assert "end_to_end" in result_dict

    def test_memory_is_per_device(self):
        """Test that memory metrics are per-device."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()

        assert result_dict["peak_memory_gb"] > 0
        assert result_dict["memory"]["memory_per_device_gb"] == result_dict["peak_memory_gb"]

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
                    assert "memory" in metrics
                    assert "activations_gb" in metrics["memory"]
                    assert metrics["memory"]["activations_gb"] >= 0

            # Check unified by_submodule_type structure
            assert "by_submodule_type" in detailed
            for submodule_type, data in detailed["by_submodule_type"].items():
                assert "memory" in data
                assert "compute" in data
                assert "communication" in data
                assert "activations_gb" in data["memory"]
                assert "flops" in data["compute"]
                assert "gb" in data["communication"]

    def test_communication_breakdown_per_device(self):
        """Test that communication breakdown is per-device."""
        evaluator = Evaluator()
        result = evaluator.evaluate("llama-7b", "H100-SXM-80GB", "training", "tp8", batch_size=32)

        result_dict = result.to_dict()

        if result_dict["communication_breakdown"]:
            comm_breakdown = result_dict["communication_breakdown"]
            # Check new structure: by_parallelism and by_operation
            if "by_parallelism" in comm_breakdown:
                for ptype, data in comm_breakdown["by_parallelism"].items():
                    if data and "total_bytes" in data:
                        assert data["total_bytes"] >= 0
            if "total_bytes" in comm_breakdown:
                assert comm_breakdown["total_bytes"] >= 0


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
            assert "memory" in data, f"memory.by_submodule_type[{submodule_type}] missing memory field"
            assert "activation_gb" in data["memory"] or "activations_gb" in data["memory"], (
                f"memory.by_submodule_type[{submodule_type}] missing activation fields"
            )
            activation_val = data["memory"].get("activation_gb", data["memory"].get("activations_gb", 0))
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
            # Skip top-level numeric fields (total_bytes, total_time_sec)
            for comm_type, data in comm_breakdown.items():
                if isinstance(data, dict) and "total_bytes" in data:
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
            assert "memory" in data
            assert "activation_gb" in data["memory"] or "activations_gb" in data["memory"]
            activation_val = data["memory"].get("activation_gb", data["memory"].get("activations_gb", 0))
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

    def test_http_api_memory_by_type_total_equals_sum(self):
        """Test that memory.by_type total equals sum of breakdown items."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        detailed = result.get("detailed_breakdown")
        assert detailed is not None

        mem = detailed.get("memory", {})
        by_type = mem.get("by_type", {})

        # Verify total exists
        assert "total" in by_type, "memory.by_type should have 'total'"
        total = by_type["total"]

        # Verify sum of breakdown items equals total
        breakdown_keys = ["weight", "gradient", "optimizer", "activation"]
        sum_of_breakdown = sum(by_type.get(key, 0) for key in breakdown_keys)

        # Allow small floating point tolerance
        assert abs(total - sum_of_breakdown) < 0.01, (
            f"total ({total}) should equal sum of breakdown items ({sum_of_breakdown})"
        )

    def test_http_api_deepseek_v3_has_transformer_block(self):
        """Test that DeepSeek V3 breakdown includes transformer_block."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "deepseek-v3"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 1,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        detailed = result.get("detailed_breakdown")

        by_submodule_type = detailed.get("memory", {}).get("by_submodule_type", {})
        assert "transformer_block" in by_submodule_type, "Should have transformer_block"

        by_submodule_type_full = detailed.get("by_submodule_type", {})
        assert "transformer_block" in by_submodule_type_full, "Should have transformer_block in full breakdown"

        transformer_full = by_submodule_type_full["transformer_block"]
        assert "nested_breakdown" in transformer_full, "Should have nested_breakdown"
        nested = transformer_full["nested_breakdown"]
        assert "attention" in nested, "Should have attention in nested_breakdown"

    def test_http_api_communication_by_parallelism(self):
        """Test that communication breakdown shows TP/DP/PP."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        comm_breakdown = result.get("communication_breakdown", {})

        by_parallelism = comm_breakdown.get("by_parallelism", {})

        if by_parallelism:
            assert "tp" in by_parallelism or "tensor_parallel" in by_parallelism, (
                "by_parallelism should contain tp for TP=8"
            )

            for ptype, pdata in by_parallelism.items():
                assert "total_bytes" in pdata, f"by_parallelism[{ptype}] should have total_bytes"
                assert isinstance(pdata["total_bytes"], (int, float))
                assert pdata["total_bytes"] >= 0

    def test_http_api_weight_equals_by_submodule_type_sum(self):
        """Test that by_type.weight equals sum of by_submodule_type weights."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "deepseek-v3"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 1,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        detailed = result.get("detailed_breakdown")

        mem = detailed.get("memory", {})
        by_type_weight = mem.get("by_type", {}).get("weight", 0)
        by_submodule = mem.get("by_submodule_type", {})
        by_submodule_sum = sum(v.get("memory", {}).get("weight_gb", 0) for v in by_submodule.values())

        assert abs(by_type_weight - by_submodule_sum) < 0.1, (
            f"weight mismatch: by_type={by_type_weight}, by_submodule_sum={by_submodule_sum}"
        )

    def test_http_api_compute_time_equals_forward_phase(self):
        """Test that compute_sec equals forward phase total_time_sec."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]

        breakdown = result.get("breakdown", {})
        compute_sec = breakdown.get("time_breakdown", {}).get("compute_sec", 0)

        forward_phases = [p for p in result["phases"] if "forward" in p["name"].lower()]
        forward_time = sum(p.get("total_time_sec", 0) for p in forward_phases)

        assert abs(compute_sec - forward_time) < 0.01, (
            f"compute_sec mismatch: {compute_sec} vs forward_time {forward_time}"
        )


class TestMultipleModelsHTTPAPI:
    """Test multiple models consistency via HTTP API after topology migration."""

    def _verify_weight_memory_consistency(self, result):
        """Verify weight memory consistency between by_type and by_submodule_type."""
        detailed = result.get("detailed_breakdown")
        assert detailed is not None

        mem = detailed.get("memory", {})
        by_type_weight = mem.get("by_type", {}).get("weight", 0)
        by_submodule = mem.get("by_submodule_type", {})
        by_submodule_sum = sum(v.get("memory", {}).get("weight_gb", 0) for v in by_submodule.values())

        assert abs(by_type_weight - by_submodule_sum) < 0.1, (
            f"Weight memory inconsistency: by_type.weight={by_type_weight}, by_submodule_sum={by_submodule_sum}"
        )

    def _verify_compute_time_consistency(self, result):
        """Verify compute_sec equals sum of forward phases."""
        breakdown = result.get("breakdown", {})
        compute_sec = breakdown.get("time_breakdown", {}).get("compute_sec", 0)

        forward_phases = [p for p in result["phases"] if "forward" in p["name"].lower()]
        forward_time = sum(p.get("total_time_sec", 0) for p in forward_phases)

        assert abs(compute_sec - forward_time) < 0.01, (
            f"Compute time inconsistency: compute_sec={compute_sec}, forward_time={forward_time}"
        )

    def _verify_communication_breakdown(self, result):
        """Verify communication breakdown structure and consistency."""
        comm_breakdown = result.get("communication_breakdown")
        if not comm_breakdown:
            return

        assert "by_parallelism" in comm_breakdown
        if comm_breakdown["by_parallelism"]:
            for ptype, data in comm_breakdown["by_parallelism"].items():
                assert "total_bytes" in data, f"{ptype} missing total_bytes"
                assert "operations" in data, f"{ptype} missing operations"

                ops_sum = sum(op.get("total_bytes", 0) for op in data["operations"].values())
                assert abs(ops_sum - data["total_bytes"]) < 0.01, (
                    f"{ptype}: operations sum {ops_sum} != total_bytes {data['total_bytes']}"
                )

    def _verify_topology_bandwidth(self, result):
        """Verify topology has valid bandwidth values."""
        metadata = result.get("metadata", {})
        topology = metadata.get("topology")

        assert topology is not None, "Topology should not be None"
        assert "levels" in topology, "Topology should have levels"

        levels = topology["levels"]
        assert len(levels) > 0, "Topology levels should not be empty"

        for level in levels:
            if "bandwidth_gbps" in level:
                assert level["bandwidth_gbps"] > 0, "Bandwidth should be positive"

    def _verify_all_checks(self, result, model_name):
        """Run all consistency checks on result."""
        self._verify_weight_memory_consistency(result)
        self._verify_compute_time_consistency(result)
        self._verify_communication_breakdown(result)
        self._verify_topology_bandwidth(result)

    def test_llama_7b_evaluation_consistency(self):
        """Test LLaMA 7B (TP=8) evaluation consistency."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        self._verify_all_checks(result, "llama-7b")

    def test_llama_13b_evaluation_consistency(self):
        """Test LLaMA 13B (TP=8) evaluation consistency."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-13b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        self._verify_all_checks(result, "llama-13b")

    def test_llama_70b_gqa_evaluation_consistency(self):
        """Test LLaMA 70B (TP=8, GQA) evaluation consistency."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-70b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 8,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        self._verify_all_checks(result, "llama-70b")

    def test_deepseek_v3_evaluation_consistency(self):
        """Test DeepSeek V3 (TP=8, MoE) evaluation consistency."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "deepseek-v3"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 1,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        self._verify_all_checks(result, "deepseek-v3")

    def test_wan_dit_evaluation_consistency(self):
        """Test Wan-DiT (video generation model) evaluation consistency."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "wan-dit"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 1,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        self._verify_all_checks(result, "wan-dit")


class TestFirstEvaluateDeepSeekV3:
    """Test first evaluation parameter passing for DeepSeek V3."""

    def test_first_evaluate_deepseek_v3_params(self):
        """Test first evaluation of deepseek-v3 has correct parameters."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate/training",
            json={
                "model": {
                    "preset": "deepseek-v3",
                    "type": "deepseek",
                    "sparse_type": "deepseek_moe",
                    "hidden_size": 7168,
                    "num_layers": 61,
                    "num_heads": 128,
                    "num_kv_heads": 128,
                    "intermediate_size": 18432,
                    "vocab_size": 129280,
                    "max_seq_len": 163840,
                    "dtype": "fp16",
                    "first_k_dense_layers": 1,
                    "num_experts": 256,
                    "num_experts_per_token": 8,
                },
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "training": {
                    "batch_size": 32,
                    "seq_len": 4096,
                },
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        detailed = result.get("detailed_breakdown")
        assert detailed is not None

        mem = detailed.get("memory", {})
        by_type = mem.get("by_type", {})

        assert by_type["weight"] > 150, f"weight memory should be > 150GB for DSv3, got {by_type['weight']}"

        metadata = result.get("metadata", {})
        assert metadata.get("tp_degree") == 8

    def test_deepseek_v3_preset_config(self):
        """Test deepseek-v3 preset has correct config."""
        client = app.test_client()
        response = client.get("/api/model/presets")
        assert response.status_code == 200

        data = response.get_json()
        presets = data["presets"]

        assert "deepseek-v3" in presets
        dsv3 = presets["deepseek-v3"]
        assert dsv3["num_layers"] == 61, f"num_layers should be 61, got {dsv3.get('num_layers')}"
        assert dsv3["architecture"] == "deepseek", f"architecture should be 'deepseek', got {dsv3.get('architecture')}"

    def test_first_evaluate_with_preset_field(self):
        """Test evaluation with preset field correctly resolves to deepseek architecture."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate/training",
            json={
                "model": {
                    "preset": "deepseek-v3",
                    "type": "deepseek",
                },
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "training": {
                    "batch_size": 1,
                    "seq_len": 2048,
                },
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        assert result.get("peak_memory_gb") > 0

        detailed = result.get("detailed_breakdown")
        if detailed:
            by_submodule_type = detailed.get("memory", {}).get("by_submodule_type", {})
            assert "transformer_block" in by_submodule_type

    def test_batch_size_fallback_to_training_params(self):
        """Test that batch_size from training params is used correctly."""
        client = app.test_client()

        response = client.post(
            "/api/evaluate/training",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "training": {
                    "batch_size": 64,
                    "seq_len": 2048,
                },
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        assert result.get("peak_memory_gb") > 0


class TestWebFrontendDisplay:
    """测试前端显示功能"""

    def test_model_spec_display_with_preset(self):
        """测试预设模型选择后返回规格参数"""
        client = app.test_client()

        response = client.get("/api/model/presets")
        assert response.status_code == 200

        data = response.get_json()
        presets = data.get("presets", {})
        assert len(presets) > 0

        llama7b = presets.get("llama-7b")
        assert llama7b is not None
        assert llama7b.get("hidden_size") == 4096
        assert llama7b.get("num_layers") == 32
        assert llama7b.get("num_heads") == 32

    def test_time_breakdown_absolute_values(self):
        """测试耗时分解返回绝对时间"""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        result = data["result"]
        detailed = result.get("detailed_breakdown")
        assert detailed is not None

        by_submodule_type = detailed.get("by_submodule_type", {})
        assert len(by_submodule_type) > 0

        total_compute_sec = 0
        total_comm_sec = 0

        for submodule_type, data in by_submodule_type.items():
            compute_time = data.get("compute", {}).get("time_sec", 0)
            comm_time = data.get("communication", {}).get("time_sec", 0)

            assert isinstance(compute_time, (int, float))
            assert isinstance(comm_time, (int, float))
            assert compute_time >= 0
            assert comm_time >= 0

            total_compute_sec += compute_time
            total_comm_sec += comm_time

        assert total_compute_sec > 0, "计算时间总和应 > 0"
        assert total_comm_sec >= 0, "通信时间总和应 >= 0"

    def test_time_breakdown_percentage_sum(self):
        """测试占比总和应为100%"""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]

        breakdown = result.get("breakdown", {})
        time_breakdown = breakdown.get("time_breakdown", {})

        if time_breakdown:
            percentages = [
                time_breakdown.get("compute_percent", 0),
                time_breakdown.get("backward_percent", 0),
                time_breakdown.get("optimizer_percent", 0),
                time_breakdown.get("communication_percent", 0),
            ]
            total_pct = sum(percentages)
            assert abs(total_pct - 100.0) < 1.0, f"占比总和应为100%，实际{total_pct}"

    def test_detailed_breakdown_table_structure(self):
        """测试分解表格结构包含所有列"""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        detailed = result.get("detailed_breakdown")

        assert detailed is not None
        by_submodule_type = detailed.get("by_submodule_type", {})
        assert len(by_submodule_type) > 0

        required_fields = {
            "memory": ["activations_gb"],
            "compute": ["time_sec", "flops"],
            "communication": ["time_sec", "bytes"],
        }

        for submodule_type, data in by_submodule_type.items():
            for category, fields in required_fields.items():
                assert category in data, f"{submodule_type} 缺少 {category}"
                for field in fields:
                    assert field in data[category], f"{submodule_type}.{category} 缺少 {field}"

    def test_memory_by_type_total_matches_sum(self):
        """测试内存分解总和匹配"""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        detailed = result.get("detailed_breakdown")

        mem = detailed.get("memory", {})
        by_type = mem.get("by_type", {})

        if by_type:
            breakdown_keys = ["weight", "gradient", "optimizer", "activation"]
            breakdown_sum = sum(by_type.get(k, 0) for k in breakdown_keys)
            total = by_type.get("total", 0)

            assert abs(total - breakdown_sum) < 0.01, (
                f"内存总和 {total} != 分解总和 {breakdown_sum}"
            )

    def test_compute_time_greater_than_zero(self):
        """测试计算时间不为零"""
        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-training",
                "batch_size": 32,
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        detailed = result.get("detailed_breakdown")

        by_submodule_type = detailed.get("by_submodule_type", {})

        transformer_block = by_submodule_type.get("transformer_block")
        if transformer_block:
            compute_time = transformer_block.get("compute", {}).get("time_sec", 0)
            assert compute_time > 0, "transformer_block计算时间应 > 0"

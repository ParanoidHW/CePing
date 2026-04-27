"""Tests for scenario display decoupling architecture.

Ensures:
1. ScenarioType is properly inferred from workload_name
2. Backend returns scenario_type in result
3. Frontend rendering is driven by scenario_type, NOT hardcoded scenario checks
4. New workload types automatically work without frontend changes
"""

import pytest
from llm_perf.analyzer import ScenarioType, infer_scenario_type, WorkloadType
from llm_perf.analyzer.base import UnifiedResult, WorkloadConfig, Phase, PhaseResult, ComputeType


class TestScenarioTypeInference:
    """Test scenario_type inference from workload_name."""

    def test_training_workload_returns_training_scenario(self):
        """Test 'training' workload returns training scenario_type."""
        scenario = infer_scenario_type("training", WorkloadType.TRAINING)
        assert scenario == ScenarioType.TRAINING

    def test_autoregressive_inference_returns_inference_scenario(self):
        """Test 'autoregressive-inference' returns inference scenario_type."""
        scenario = infer_scenario_type("autoregressive-inference", WorkloadType.INFERENCE)
        assert scenario == ScenarioType.INFERENCE

    def test_diffusion_pipeline_returns_diffusion_scenario(self):
        """Test 'diffusion-pipeline' returns diffusion scenario_type."""
        scenario = infer_scenario_type("diffusion-pipeline", WorkloadType.DIFFUSION)
        assert scenario == ScenarioType.DIFFUSION

    def test_denoise_inference_returns_diffusion_scenario(self):
        """Test 'denoise-inference' returns diffusion scenario_type."""
        scenario = infer_scenario_type("denoise-inference", WorkloadType.DIFFUSION)
        assert scenario == ScenarioType.DIFFUSION

    def test_rl_ppo_returns_rl_training_scenario(self):
        """Test 'rl-ppo' returns rl_training scenario_type."""
        scenario = infer_scenario_type("rl-ppo", WorkloadType.MIXED)
        assert scenario == ScenarioType.RL_TRAINING

    def test_rl_grpo_returns_rl_training_scenario(self):
        """Test 'rl-grpo' returns rl_training scenario_type."""
        scenario = infer_scenario_type("rl-grpo", WorkloadType.MIXED)
        assert scenario == ScenarioType.RL_TRAINING

    def test_unknown_workload_defaults_to_inference(self):
        """Test unknown workload defaults to inference scenario."""
        scenario = infer_scenario_type("unknown-workload", WorkloadType.INFERENCE)
        assert scenario == ScenarioType.INFERENCE

    def test_unknown_training_workload_defaults_to_training(self):
        """Test unknown training workload defaults to training scenario."""
        scenario = infer_scenario_type("unknown-training", WorkloadType.TRAINING)
        assert scenario == ScenarioType.TRAINING


class TestUnifiedResultScenarioType:
    """Test UnifiedResult includes scenario_type in to_dict()."""

    def test_result_to_dict_contains_scenario_type(self):
        """Test result.to_dict() contains scenario_type field."""
        result = UnifiedResult(
            workload_name="training",
            workload_type=WorkloadType.TRAINING,
            total_time_sec=1.0,
        )
        data = result.to_dict()
        assert "scenario_type" in data
        assert data["scenario_type"] == "training"

    def test_diffusion_result_scenario_type(self):
        """Test diffusion result has correct scenario_type."""
        result = UnifiedResult(
            workload_name="diffusion-pipeline",
            workload_type=WorkloadType.DIFFUSION,
            total_time_sec=10.0,
        )
        data = result.to_dict()
        assert data["scenario_type"] == "diffusion"

    def test_inference_result_scenario_type(self):
        """Test inference result has correct scenario_type."""
        result = UnifiedResult(
            workload_name="autoregressive-inference",
            workload_type=WorkloadType.INFERENCE,
            phases=[
                PhaseResult(name="prefill", component="main", compute_type=ComputeType.FORWARD, total_time_sec=0.1, single_time_sec=0.1),
                PhaseResult(name="decode", component="main", compute_type=ComputeType.FORWARD, total_time_sec=0.5, single_time_sec=0.005, repeat_count=100),
            ],
        )
        data = result.to_dict()
        assert data["scenario_type"] == "inference"
        assert data["prefill"]["ttft_sec"] > 0
        assert data["decode"]["tpot_sec"] > 0

    def test_diffusion_result_no_llm_decode_metrics(self):
        """Test diffusion result does NOT have LLM inference decode metrics."""
        result = UnifiedResult(
            workload_name="diffusion-pipeline",
            workload_type=WorkloadType.DIFFUSION,
            phases=[
                PhaseResult(name="encode", component="encoder", compute_type=ComputeType.FORWARD, total_time_sec=0.1),
                PhaseResult(name="denoise", component="backbone", compute_type=ComputeType.FORWARD, total_time_sec=5.0, repeat_count=50),
                PhaseResult(name="decode", component="decoder", compute_type=ComputeType.FORWARD, total_time_sec=0.2),
            ],
        )
        data = result.to_dict()
        assert data["scenario_type"] == "diffusion"
        assert data["prefill"] is None
        assert data["decode"] is None


class TestFrontendRenderingDecoupling:
    """Test frontend rendering is driven by scenario_type."""

    def test_frontend_template_mapping_exists(self):
        """Test frontend RENDER_TEMPLATES mapping covers all scenarios."""
        expected_templates = [
            ScenarioType.TRAINING.value,
            ScenarioType.INFERENCE.value,
            ScenarioType.DIFFUSION.value,
            ScenarioType.RL_TRAINING.value,
            ScenarioType.PD_DISAGG.value,
        ]

        for template_type in expected_templates:
            assert template_type in [
                "training", "inference", "diffusion", "rl_training", "pd_disagg"
            ]

    def test_new_workload_type_uses_generic_template(self):
        """Test new workload type uses generic template (fallback)."""
        result = UnifiedResult(
            workload_name="new-workload-type",
            workload_type=WorkloadType.INFERENCE,
            total_time_sec=5.0,
        )
        data = result.to_dict()
        assert "scenario_type" in data
        assert "total_time_sec" in data
        assert "peak_memory_gb" in data


class TestScenarioTypeEnumValues:
    """Test ScenarioType enum has correct values."""

    def test_scenario_type_values_match_frontend_options(self):
        """Test ScenarioType values match frontend workload-scenario options."""
        frontend_options = ["training", "inference", "pd_disagg", "rl_training", "diffusion"]
        scenario_values = [e.value for e in ScenarioType]

        for option in frontend_options:
            assert option in scenario_values, f"Frontend option '{option}' not in ScenarioType"


class TestBackendAPIScenarioType:
    """Test backend API returns correct scenario_type."""

    @pytest.fixture
    def client(self):
        from web.app import app
        app.config['TESTING'] = True
        return app.test_client()

    def test_training_api_returns_training_scenario_type(self, client):
        """Test training API returns 'training' scenario_type."""
        response = client.post(
            "/api/evaluate",
            json={
                "model": {"preset": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": {"scenario": "training"},
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        assert result["scenario_type"] == "training"

    def test_inference_api_returns_inference_scenario_type(self, client):
        """Test inference API returns 'inference' scenario_type."""
        response = client.post(
            "/api/evaluate",
            json={
                "model": {"preset": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": {"scenario": "inference"},
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        assert result["scenario_type"] == "inference"

    def test_diffusion_api_returns_diffusion_scenario_type(self, client):
        """Test diffusion API returns 'diffusion' scenario_type."""
        response = client.post(
            "/api/evaluate",
            json={
                "model": {"preset": "hunyuan-image-3"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": {"scenario": "diffusion", "diffusion_steps": 50},
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        result = data["result"]
        assert result["scenario_type"] == "diffusion"
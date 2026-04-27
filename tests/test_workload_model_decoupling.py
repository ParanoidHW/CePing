"""Tests for workload and model decoupling architecture.

Ensures:
1. Frontend filters models by workload type
2. Backend API returns filtered models based on workload
3. Sparse type unification to dense/sparse
"""

from llm_perf.modeling import (
    get_model_presets,
    get_presets_by_workload,
    get_presets_by_workload_grouped,
)


class TestWorkloadModelDecouplingRegistry:
    """Test registry functions for workload filtering."""

    def test_get_presets_by_workload_training(self):
        """Test get_presets_by_workload returns training-supported models."""
        presets = get_presets_by_workload("training")

        assert "llama-7b" in presets
        assert "deepseek-v3" in presets
        assert "qwen3-5" in presets

    def test_get_presets_by_workload_inference(self):
        """Test get_presets_by_workload returns inference-supported models."""
        presets = get_presets_by_workload("inference")

        assert "llama-7b" in presets
        assert "hunyuan-image-3" in presets

    def test_get_presets_by_workload_diffusion(self):
        """Test get_presets_by_workload returns diffusion-supported models."""
        presets = get_presets_by_workload("diffusion")

        assert "hunyuan-image-3" in presets
        assert "wan-dit" in presets
        assert "wan-t2v-14b" in presets

        assert "llama-7b" not in presets

    def test_get_presets_by_workload_grouped_diffusion(self):
        """Test get_presets_by_workload_grouped returns grouped presets."""
        grouped = get_presets_by_workload_grouped("diffusion")

        assert "dense" in grouped
        assert "sparse" in grouped

        sparse_presets = grouped["sparse"]
        sparse_names = [p["name"] for p in sparse_presets]
        assert "hunyuan-image-3" in sparse_names

    def test_hunyuan_image_3_no_duplicate_architectures(self):
        """Test hunyuan-image-3 does not have duplicate architecture names."""
        presets = get_model_presets()

        architectures = []
        for name, preset in presets.items():
            if name.startswith("hunyuan"):
                architectures.append(preset.get("architecture"))

        assert len(architectures) == 1, "Should only have one hunyuan-image-3 architecture"


class TestWorkloadModelDecouplingAPI:
    """Test API endpoints for workload filtering."""

    def test_api_models_endpoint_with_workload_type(self):
        """Test API returns filtered models based on workload_type."""
        from web.app import app

        client = app.test_client()

        response = client.get("/api/models?workload_type=diffusion")
        assert response.status_code == 200

        data = response.get_json()
        assert "presets" in data
        assert "by_sparse_type" in data

        presets = data["presets"]
        assert "hunyuan-image-3" in presets
        assert "llama-7b" not in presets

    def test_api_models_endpoint_without_workload_type(self):
        """Test API returns all models when workload_type not specified."""
        from web.app import app

        client = app.test_client()

        response = client.get("/api/models")
        assert response.status_code == 200

        data = response.get_json()
        presets = data["presets"]
        assert "llama-7b" in presets
        assert "hunyuan-image-3" in presets


class TestWorkloadTypesDefinition:
    """Test workload_types.py definitions."""

    def test_workload_type_enum_exists(self):
        """Test WorkloadType enum exists."""
        from llm_perf.utils.workload_types import WorkloadType

        assert WorkloadType.TRAINING.value == "training"
        assert WorkloadType.INFERENCE.value == "inference"
        assert WorkloadType.DIFFUSION.value == "diffusion"
        assert WorkloadType.RL_TRAINING.value == "rl_training"
        assert WorkloadType.PD_DISAGG.value == "pd_disagg"

    def test_workload_params_defaults(self):
        """Test workload params have defaults."""
        from llm_perf.utils.workload_types import get_workload_defaults

        training_defaults = get_workload_defaults(
            __import__("llm_perf.utils.workload_types", fromlist=["WorkloadType"]).WorkloadType.TRAINING
        )
        assert "batch_size" in training_defaults
        assert "seq_len" in training_defaults

    def test_default_supported_workloads_mapping(self):
        """Test default supported workloads mapping."""
        from llm_perf.utils.workload_types import get_supported_workloads

        llama_workloads = get_supported_workloads("llama")
        assert "training" in [w.value for w in llama_workloads]
        assert "inference" in [w.value for w in llama_workloads]

        wan_dit_workloads = get_supported_workloads("wan_dit")
        assert "diffusion" in [w.value for w in wan_dit_workloads]


class TestSparseTypeUnification:
    """Test sparse_type unification to dense/sparse."""

    def test_get_presets_by_sparse_type_no_hardcoding(self):
        """Test get_presets_by_sparse_type uses auto-grouping, no hardcoded types."""
        from llm_perf.modeling import get_presets_by_sparse_type

        result = get_presets_by_sparse_type()
        
        assert len(result) == 2
        assert "dense" in result
        assert "sparse" in result
        
        dense_names = [p["name"] for p in result["dense"]]
        assert "llama-7b" in dense_names
        assert "llama-13b" in dense_names
        assert "qwen3-5" in dense_names
        
        sparse_names = [p["name"] for p in result["sparse"]]
        assert "deepseek-v3" in sparse_names
        assert "hunyuan-image-3" in sparse_names
        assert "qwen3-5-moe" in sparse_names
        assert "mixtral-8x7b" in sparse_names

    def test_get_presets_by_workload_grouped_auto_grouping(self):
        """Test get_presets_by_workload_grouped uses auto-grouping."""
        from llm_perf.modeling import get_presets_by_workload_grouped

        training_grouped = get_presets_by_workload_grouped("training")
        assert len(training_grouped) == 2
        assert "dense" in training_grouped
        assert "sparse" in training_grouped
        
        inference_grouped = get_presets_by_workload_grouped("inference")
        assert len(inference_grouped) == 2
        
        diffusion_grouped = get_presets_by_workload_grouped("diffusion")
        assert len(diffusion_grouped) == 2

    def test_all_sparse_models_grouped_as_sparse(self):
        """Test all sparse-type models are unified under 'sparse'."""
        from llm_perf.modeling import get_presets_by_sparse_type, get_model_presets

        presets = get_model_presets()
        sparse_result = get_presets_by_sparse_type()
        
        for name, config in presets.items():
            sparse_type = config.get("sparse_type", "dense")
            if sparse_type != "dense":
                sparse_names = [p["name"] for p in sparse_result["sparse"]]
                assert name in sparse_names, f"{name} should be in sparse group"

    def test_api_returns_unified_sparse_groups(self):
        """Test API returns unified sparse groups."""
        from web.app import app

        client = app.test_client()
        
        response = client.get("/api/models?workload_type=training")
        data = response.get_json()
        
        by_sparse_type = data.get("by_sparse_type", {})
        assert len(by_sparse_type) == 2
        assert "dense" in by_sparse_type
        assert "sparse" in by_sparse_type
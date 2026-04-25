"""Tests for workload and model decoupling architecture.

Ensures:
1. Model config files have supported_workloads metadata
2. hunyuan-image-3-text.yaml and hunyuan-image-3-diffusion.yaml are deleted
3. hunyuan-image-3.yaml has unified config with supported_workloads
4. Frontend filters models by workload type
5. Backend API returns filtered models based on workload
"""

import pytest
from pathlib import Path
import yaml

from llm_perf.modeling import (
    get_model_presets,
    get_presets_by_workload,
    get_presets_by_workload_grouped,
)


class TestWorkloadModelDecouplingFiles:
    """Test correct file structure after refactoring."""

    def test_hunyuan_image_3_text_yaml_deleted(self):
        """Test hunyuan-image-3-text.yaml is deleted."""
        config_path = Path("configs/models/hunyuan-image-3-text.yaml")
        assert not config_path.exists(), "hunyuan-image-3-text.yaml should be deleted"

    def test_hunyuan_image_3_diffusion_yaml_deleted(self):
        """Test hunyuan-image-3-diffusion.yaml is deleted."""
        config_path = Path("configs/models/hunyuan-image-3-diffusion.yaml")
        assert not config_path.exists(), "hunyuan-image-3-diffusion.yaml should be deleted"

    def test_hunyuan_image_3_yaml_exists(self):
        """Test hunyuan-image-3.yaml exists with unified config."""
        config_path = Path("configs/models/hunyuan-image-3.yaml")
        assert config_path.exists(), "hunyuan-image-3.yaml should exist"

    def test_hunyuan_image_3_yaml_has_supported_workloads(self):
        """Test hunyuan-image-3.yaml has supported_workloads field."""
        config_path = Path("configs/models/hunyuan-image-3.yaml")
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "supported_workloads" in config, "hunyuan-image-3.yaml should have supported_workloads"
        assert "inference" in config["supported_workloads"]
        assert "diffusion" in config["supported_workloads"]

    def test_llama_config_has_supported_workloads(self):
        """Test llama-7b.yaml has supported_workloads."""
        config_path = Path("configs/models/llama-7b.yaml")
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "supported_workloads" in config
        assert "training" in config["supported_workloads"]
        assert "inference" in config["supported_workloads"]

    def test_wan_dit_config_has_supported_workloads(self):
        """Test wan-dit.yaml has supported_workloads."""
        config_path = Path("configs/models/wan-dit.yaml")
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "supported_workloads" in config
        assert "diffusion" in config["supported_workloads"]


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
        assert "sparse_hunyuan_moe" in grouped

        hunyuan_presets = grouped["sparse_hunyuan_moe"]
        hunyuan_names = [p["name"] for p in hunyuan_presets]
        assert "hunyuan-image-3" in hunyuan_names

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
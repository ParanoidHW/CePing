"""Tests for llm_perf.workload module.

Tests:
- Schema definitions
- Loader functionality
- Registry functionality
- Validator functionality
"""

import pytest
from pathlib import Path

from llm_perf.workload.schema import (
    WorkloadSchema,
    StageSchema,
    HardwareSchema,
    StrategySchema,
    ModelSchema,
    ParamSchemaItem,
    WorkloadCategory,
)
from llm_perf.workload.loader import WorkloadLoader, get_loader
from llm_perf.workload.registry import WorkloadRegistry, ModelRegistry, get_workload_registry, get_model_registry
from llm_perf.workload.validator import WorkloadValidator, ValidationResult


class TestParamSchemaItem:
    """Test ParamSchemaItem."""

    def test_basic_param(self):
        """Test basic parameter schema."""
        param = ParamSchemaItem(
            name="batch_size",
            label="Batch Size",
            type="number",
            default=32,
            required=True,
        )
        assert param.name == "batch_size"
        assert param.label == "Batch Size"
        assert param.type == "number"
        assert param.default == 32
        assert param.required == True

    def test_to_dict(self):
        """Test to_dict conversion."""
        param = ParamSchemaItem(
            name="seq_len",
            label="Sequence Length",
            type="number",
            default=2048,
            min=1,
            max=8192,
        )
        d = param.to_dict()
        assert d["name"] == "seq_len"
        assert d["min"] == 1
        assert d["max"] == 8192


class TestStageSchema:
    """Test StageSchema."""

    def test_basic_stage(self):
        """Test basic stage schema."""
        stage = StageSchema(
            name="prefill",
            compute_type="forward",
            component="main",
            repeat=1,
            compute_pattern="transformer_block",
        )
        assert stage.name == "prefill"
        assert stage.compute_type == "forward"
        assert stage.component == "main"

    def test_to_dict(self):
        """Test to_dict conversion."""
        stage = StageSchema(
            name="decode",
            compute_type="forward",
            component="main",
            repeat="generation_len",
            extra_params={"seq_len": 1},
        )
        d = stage.to_dict()
        assert d["name"] == "decode"
        assert d["repeat"] == "generation_len"
        assert d["extra_params"]["seq_len"] == 1


class TestHardwareSchema:
    """Test HardwareSchema."""

    def test_basic_hardware(self):
        """Test basic hardware schema."""
        hw = HardwareSchema(
            device_preset="H100-SXM-80GB",
            num_devices=8,
        )
        assert hw.device_preset == "H100-SXM-80GB"
        assert hw.num_devices == 8

    def test_to_dict(self):
        """Test to_dict conversion."""
        hw = HardwareSchema(
            device_preset="H100-SXM-80GB",
            num_devices=16,
            topology_type="homogeneous",
        )
        d = hw.to_dict()
        assert d["device_preset"] == "H100-SXM-80GB"
        assert d["num_devices"] == 16


class TestStrategySchema:
    """Test StrategySchema."""

    def test_basic_strategy(self):
        """Test basic strategy schema."""
        strategy = StrategySchema(
            tp_degree=8,
            pp_degree=1,
            dp_degree=1,
        )
        assert strategy.tp_degree == 8
        assert strategy.pp_degree == 1

    def test_to_dict(self):
        """Test to_dict conversion."""
        strategy = StrategySchema(
            tp_degree=8,
            pp_degree=2,
            activation_checkpointing=True,
            zero_stage=1,
        )
        d = strategy.to_dict()
        assert d["tp_degree"] == 8
        assert d["activation_checkpointing"] == True


class TestWorkloadSchema:
    """Test WorkloadSchema."""

    def test_basic_workload(self):
        """Test basic workload schema."""
        params = {
            "batch_size": ParamSchemaItem(
                name="batch_size",
                label="Batch Size",
                default=1,
            ),
        }
        workload = WorkloadSchema(
            name="autoregressive-inference",
            workload_name="inference/autoregressive",
            description="Autoregressive inference",
            category=WorkloadCategory.INFERENCE,
            parameters=params,
        )
        assert workload.name == "autoregressive-inference"
        assert workload.category == WorkloadCategory.INFERENCE

    def test_get_defaults(self):
        """Test get_defaults."""
        params = {
            "batch_size": ParamSchemaItem(name="batch_size", label="Batch Size", default=8),
            "seq_len": ParamSchemaItem(name="seq_len", label="Seq Len", default=1024),
        }
        workload = WorkloadSchema(
            name="test",
            workload_name="test",
            parameters=params,
        )
        defaults = workload.get_defaults()
        assert defaults["batch_size"] == 8
        assert defaults["seq_len"] == 1024

    def test_get_required_params(self):
        """Test get_required_params."""
        params = {
            "batch_size": ParamSchemaItem(name="batch_size", label="Batch Size", default=8, required=True),
            "seq_len": ParamSchemaItem(name="seq_len", label="Seq Len", default=1024, required=False),
        }
        workload = WorkloadSchema(
            name="test",
            workload_name="test",
            parameters=params,
        )
        required = workload.get_required_params()
        assert "batch_size" in required
        assert "seq_len" not in required


class TestModelSchema:
    """Test ModelSchema."""

    def test_basic_model(self):
        """Test basic model schema."""
        model = ModelSchema(
            name="llama-7b",
            description="LLaMA 7B",
            architecture="llama",
            sparse_type="dense",
        )
        assert model.name == "llama-7b"
        assert model.architecture == "llama"

    def test_param_schema(self):
        """Test param_schema access."""
        param_schema = {
            "training": [
                ParamSchemaItem(name="batch_size", label="Batch Size", default=32),
            ],
            "inference": [
                ParamSchemaItem(name="batch_size", label="Batch Size", default=8),
            ],
        }
        model = ModelSchema(
            name="test",
            description="Test",
            architecture="test",
            param_schema=param_schema,
        )
        training_params = model.get_param_schema("training")
        assert len(training_params) == 1
        assert training_params[0].default == 32


class TestWorkloadLoader:
    """Test WorkloadLoader."""

    def test_list_workloads(self):
        """Test list_workloads."""
        loader = WorkloadLoader()
        workloads = loader.list_workloads()
        assert len(workloads) > 0
        assert "inference/autoregressive" in workloads

    def test_list_workload_categories(self):
        """Test list_workload_categories."""
        loader = WorkloadLoader()
        categories = loader.list_workload_categories()
        assert "training" in categories
        assert "inference" in categories

    def test_list_models(self):
        """Test list_models."""
        loader = WorkloadLoader()
        models = loader.list_models()
        assert len(models) > 0
        assert "llama-7b" in models

    def test_load_workload_yaml(self):
        """Test load_workload_yaml."""
        loader = WorkloadLoader()
        config = loader.load_workload_yaml("inference/autoregressive")
        assert config["name"] == "autoregressive-inference"
        assert config["workload_type"] == "inference"

    def test_load_model_yaml(self):
        """Test load_model_yaml."""
        loader = WorkloadLoader()
        config = loader.load_model_yaml("llama-7b")
        assert config["description"] == "LLaMA 7B"
        assert config["architecture"] == "llama"

    def test_get_workload_schema(self):
        """Test get_workload_schema."""
        loader = WorkloadLoader()
        schema = loader.get_workload_schema("inference/autoregressive")
        assert schema.name == "autoregressive"
        assert schema.workload_name == "inference/autoregressive"
        assert schema.category == WorkloadCategory.INFERENCE
        assert len(schema.stages) > 0

    def test_get_model_schema(self):
        """Test get_model_schema."""
        loader = WorkloadLoader()
        schema = loader.get_model_schema("llama-7b")
        assert schema.name == "llama-7b"
        assert schema.architecture == "llama"

    def test_get_loader_singleton(self):
        """Test get_loader singleton."""
        loader1 = get_loader()
        loader2 = get_loader()
        assert loader1 is loader2


class TestWorkloadRegistry:
    """Test WorkloadRegistry."""

    def test_list_workloads(self):
        """Test list_workloads."""
        registry = WorkloadRegistry()
        workloads = registry.list_workloads()
        assert len(workloads) > 0

    def test_list_workload_categories(self):
        """Test list_workload_categories."""
        registry = WorkloadRegistry()
        categories = registry.list_workload_categories()
        assert "training" in categories

    def test_get_workload_schema(self):
        """Test get_workload_schema."""
        registry = WorkloadRegistry()
        schema = registry.get_workload_schema("inference/autoregressive")
        assert schema.name == "autoregressive"
        assert schema.workload_name == "inference/autoregressive"

    def test_is_valid_workload(self):
        """Test is_valid_workload."""
        registry = WorkloadRegistry()
        assert registry.is_valid_workload("inference/autoregressive") == True
        assert registry.is_valid_workload("invalid/workload") == False


class TestModelRegistry:
    """Test ModelRegistry."""

    def test_list_models(self):
        """Test list_models."""
        registry = ModelRegistry()
        models = registry.list_models()
        assert len(models) > 0

    def test_get_model_schema(self):
        """Test get_model_schema."""
        registry = ModelRegistry()
        schema = registry.get_model_schema("llama-7b")
        assert schema.name == "llama-7b"

    def test_is_valid_model(self):
        """Test is_valid_model."""
        registry = ModelRegistry()
        assert registry.is_valid_model("llama-7b") == True
        assert registry.is_valid_model("invalid-model") == False

    def test_supports_workload(self):
        """Test supports_workload."""
        registry = ModelRegistry()
        assert registry.supports_workload("llama-7b", "training") == True
        assert registry.supports_workload("llama-7b", "diffusion") == False


class TestWorkloadValidator:
    """Test WorkloadValidator."""

    def test_validate_workload_yaml_valid(self):
        """Test validate_workload_yaml with valid config."""
        validator = WorkloadValidator()
        config = {
            "name": "test-workload",
            "workload_type": "inference",
            "phases": [
                {
                    "name": "forward",
                    "compute_type": "forward",
                    "component": "main",
                }
            ],
        }
        result = validator.validate_workload_yaml(config)
        assert result.is_valid == True

    def test_validate_workload_yaml_invalid(self):
        """Test validate_workload_yaml with invalid config."""
        validator = WorkloadValidator()
        config = {
            "name": "test-workload",
        }
        result = validator.validate_workload_yaml(config)
        assert result.is_valid == False
        assert len(result.errors) > 0

    def test_validate_model_yaml_valid(self):
        """Test validate_model_yaml with valid config."""
        validator = WorkloadValidator()
        config = {
            "description": "Test Model",
            "architecture": "llama",
            "config": {
                "hidden_size": 4096,
                "num_layers": 32,
                "num_heads": 32,
                "vocab_size": 32000,
            },
        }
        result = validator.validate_model_yaml(config)
        assert result.is_valid == True

    def test_validate_model_yaml_invalid(self):
        """Test validate_model_yaml with invalid config."""
        validator = WorkloadValidator()
        config = {
            "description": "Test Model",
        }
        result = validator.validate_model_yaml(config)
        assert result.is_valid == False

    def test_validation_result_to_dict(self):
        """Test ValidationResult.to_dict."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )
        d = result.to_dict()
        assert d["is_valid"] == False
        assert len(d["errors"]) == 2
        assert len(d["warnings"]) == 1
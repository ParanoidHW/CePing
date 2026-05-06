"""Tests for Pydantic schema definitions."""

import pytest
from pydantic import ValidationError

from llm_perf.workload.schema_pydantic import (
    WorkloadSchema,
    ModelSchema,
    HardwareSchema,
    StrategySchema,
    ParamSchemaItem,
    StageSchema,
    WorkloadCategory,
)
from llm_perf.workload.registry_pydantic import (
    WorkloadInfo,
    ModelInfo,
)
from llm_perf.workload.engine_pydantic import (
    EvaluationRequest,
    EvaluationResult,
)


class TestParamSchemaItem:
    """Test ParamSchemaItem Pydantic model."""

    def test_basic_field(self):
        """Test basic field creation."""
        item = ParamSchemaItem(
            name="batch_size",
            label="Batch Size",
            type="number",
            default=32,
            min=1,
            max=1024,
            required=True,
            description="Batch size for training",
        )
        assert item.name == "batch_size"
        assert item.type == "number"
        assert item.default == 32

    def test_optional_fields(self):
        """Test optional fields are None by default."""
        item = ParamSchemaItem(name="seq_len", label="Sequence Length")
        assert item.min is None
        assert item.max is None
        assert item.description == ""

    def test_extra_fields_forbidden(self):
        """Test extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ParamSchemaItem(
                name="test",
                label="Test",
                extra_field="should_fail",
            )

    def test_model_dump(self):
        """Test model_dump() replaces to_dict()."""
        item = ParamSchemaItem(name="test", label="Test", default=10)
        data = item.model_dump()
        assert data["name"] == "test"
        assert data["default"] == 10


class TestStageSchema:
    """Test StageSchema Pydantic model."""

    def test_basic_stage(self):
        """Test basic stage creation."""
        stage = StageSchema(
            name="prefill",
            compute_type="forward",
            component="main",
            repeat=1,
        )
        assert stage.name == "prefill"
        assert stage.compute_type == "forward"

    def test_dynamic_repeat(self):
        """Test repeat can be string (dynamic param name)."""
        stage = StageSchema(
            name="decode",
            compute_type="forward",
            repeat="num_tokens",
        )
        assert stage.repeat == "num_tokens"


class TestHardwareSchema:
    """Test HardwareSchema Pydantic model."""

    def test_basic_hardware(self):
        """Test basic hardware creation."""
        hw = HardwareSchema(
            device_preset="H100-SXM-80GB",
            num_devices=8,
            topology_type="homogeneous",
        )
        assert hw.device_preset == "H100-SXM-80GB"
        assert hw.num_devices == 8

    def test_defaults(self):
        """Test default values."""
        hw = HardwareSchema(device_preset="A100")
        assert hw.num_devices == 1
        assert hw.topology_type == "homogeneous"


class TestStrategySchema:
    """Test StrategySchema Pydantic model."""

    def test_basic_strategy(self):
        """Test basic strategy creation."""
        strategy = StrategySchema(
            tp_degree=8,
            pp_degree=1,
            dp_degree=1,
        )
        assert strategy.tp_degree == 8
        assert strategy.pp_degree == 1

    def test_defaults(self):
        """Test default values."""
        strategy = StrategySchema()
        assert strategy.tp_degree == 1
        assert strategy.activation_checkpointing is False


class TestWorkloadInfo:
    """Test WorkloadInfo Pydantic model."""

    def test_basic_workload_info(self):
        """Test basic workload info."""
        info = WorkloadInfo(
            name="autoregressive",
            category="inference",
            description="Autoregressive inference",
            workload_type="inference",
        )
        assert info.name == "autoregressive"
        assert info.category == "inference"

    def test_model_dump(self):
        """Test model_dump() serialization."""
        info = WorkloadInfo(
            name="test",
            category="training",
            description="Test workload",
            workload_type="training",
        )
        data = info.model_dump()
        assert data["name"] == "test"
        assert data["category"] == "training"


class TestModelInfo:
    """Test ModelInfo Pydantic model."""

    def test_basic_model_info(self):
        """Test basic model info."""
        info = ModelInfo(
            name="llama-7b",
            display_name="LLaMA 7B",
            architecture="llama",
            sparse_type="dense",
            supported_workloads=["training", "inference"],
        )
        assert info.name == "llama-7b"
        assert info.sparse_type == "dense"

    def test_optional_supported_workloads(self):
        """Test supported_workloads default to empty list."""
        info = ModelInfo(
            name="test",
            display_name="Test",
            architecture="test",
            sparse_type="dense",
        )
        assert info.supported_workloads == []


class TestEvaluationRequest:
    """Test EvaluationRequest Pydantic model."""

    def test_basic_request(self):
        """Test basic request creation."""
        hw = HardwareSchema(device_preset="H100", num_devices=8)
        strategy = StrategySchema(tp_degree=8)
        request = EvaluationRequest(
            workload_name="inference/autoregressive",
            model_name="llama-7b",
            hardware=hw,
            strategy=strategy,
            params={"batch_size": 32},
        )
        assert request.workload_name == "inference/autoregressive"
        assert request.hardware.num_devices == 8

    def test_nested_model_dump(self):
        """Test nested model serialization."""
        hw = HardwareSchema(device_preset="H100", num_devices=4)
        strategy = StrategySchema(tp_degree=4)
        request = EvaluationRequest(
            workload_name="test",
            model_name="test",
            hardware=hw,
            strategy=strategy,
        )
        data = request.model_dump()
        assert data["hardware"]["num_devices"] == 4
        assert data["strategy"]["tp_degree"] == 4


class TestEvaluationResult:
    """Test EvaluationResult Pydantic model."""

    def test_success_result(self):
        """Test successful result."""
        result = EvaluationResult(
            success=True,
            result={"time_sec": 10.0},
        )
        assert result.success is True
        assert result.result["time_sec"] == 10.0

    def test_failure_result(self):
        """Test failed result."""
        result = EvaluationResult(
            success=False,
            error="Validation failed",
        )
        assert result.success is False
        assert result.error == "Validation failed"


class TestModelSchema:
    """Test ModelSchema Pydantic model."""

    def test_basic_model_schema(self):
        """Test basic model schema."""
        schema = ModelSchema(
            name="llama-7b",
            description="LLaMA 7B model",
            architecture="llama",
            sparse_type="dense",
        )
        assert schema.name == "llama-7b"
        assert schema.architecture == "llama"

    def test_get_param_schema(self):
        """Test get_param_schema method."""
        param = ParamSchemaItem(name="batch_size", label="Batch Size")
        schema = ModelSchema(
            name="test",
            description="Test",
            architecture="test",
            param_schema={"inference": [param]},
        )
        params = schema.get_param_schema("inference")
        assert len(params) == 1
        assert params[0].name == "batch_size"


class TestWorkloadSchema:
    """Test WorkloadSchema Pydantic model."""

    def test_basic_workload_schema(self):
        """Test basic workload schema."""
        schema = WorkloadSchema(
            name="autoregressive",
            workload_name="inference/autoregressive",
            description="Autoregressive inference",
            category=WorkloadCategory.INFERENCE,
        )
        assert schema.name == "autoregressive"
        assert schema.category == WorkloadCategory.INFERENCE

    def test_get_defaults(self):
        """Test get_defaults method."""
        param1 = ParamSchemaItem(name="batch_size", label="Batch", default=32)
        param2 = ParamSchemaItem(name="seq_len", label="Seq Len", default=1024)
        schema = WorkloadSchema(
            name="test",
            parameters={"batch_size": param1, "seq_len": param2},
        )
        defaults = schema.get_defaults()
        assert defaults["batch_size"] == 32
        assert defaults["seq_len"] == 1024

    def test_get_required_params(self):
        """Test get_required_params method."""
        param1 = ParamSchemaItem(name="batch_size", label="Batch", required=True)
        param2 = ParamSchemaItem(name="optional", label="Opt", required=False)
        schema = WorkloadSchema(
            name="test",
            parameters={"batch_size": param1, "optional": param2},
        )
        required = schema.get_required_params()
        assert "batch_size" in required
        assert "optional" not in required
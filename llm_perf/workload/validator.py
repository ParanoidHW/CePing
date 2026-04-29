"""Configuration validator for workload evaluation.

Validates:
1. Workload configuration
2. Model configuration
3. Hardware configuration
4. Strategy configuration
5. Memory constraints
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from llm_perf.modeling import ShardedModule
from llm_perf.hardware.device import Device
from llm_perf.utils.constants import DTYPE_SIZES

from .schema import (
    WorkloadSchema,
    ModelSchema,
    HardwareSchema,
    StrategySchema,
    ParamSchemaItem,
)

if TYPE_CHECKING:
    from .engine import EvaluationRequest


@dataclass
class ValidationResult:
    """Validation result.

    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        memory_warning: Memory constraint warning (if applicable)
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    memory_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "memory_warning": self.memory_warning,
        }


class WorkloadValidator:
    """Validator for workload configurations.

    Validates:
    - Workload parameters against schema
    - Model parameters against schema
    - Hardware configuration
    - Strategy configuration
    - Memory constraints
    """

    REQUIRED_WORKLOAD_FIELDS = ["name", "workload_type", "phases"]
    REQUIRED_MODEL_FIELDS = ["description", "architecture", "config"]

    VALID_WORKLOAD_TYPES = ["training", "inference", "diffusion", "mixed"]
    VALID_COMPUTE_TYPES = ["forward", "backward", "optimizer"]
    VALID_COMPUTE_PATTERNS = [
        "transformer_block",
        "conv_encoder",
        "conv_decoder",
        "attention_only",
        "dense_forward",
    ]

    def validate_request(
        self, request: "EvaluationRequest", model: ShardedModule
    ) -> ValidationResult:
        """Validate EvaluationRequest.

        Args:
            request: EvaluationRequest
            model: ShardedModule instance

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        workload_schema = self.loader.get_workload_schema(request.workload_name)
        model_schema = self.loader.get_model_schema(request.model_name)

        param_errors = self._validate_params(request.params, workload_schema.parameters)
        errors.extend(param_errors)

        hardware_errors = self._validate_hardware(request.hardware)
        errors.extend(hardware_errors)

        strategy_errors = self._validate_strategy(request.strategy)
        errors.extend(strategy_errors)

        support_errors = self._validate_model_workload_support(
            request.model_name, request.workload_name
        )
        errors.extend(support_errors)

        memory_warning = None
        if len(errors) == 0:
            memory_warning = self._check_memory_constraint(
                request.hardware, request.strategy, model
            )
            if memory_warning:
                warnings.append(memory_warning)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            memory_warning=memory_warning,
        )

    def __init__(self, loader=None):
        from .loader import get_loader
        self.loader = loader or get_loader()

    def validate_workload_yaml(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate workload YAML configuration.

        Args:
            config: Workload config dict

        Returns:
            ValidationResult
        """
        errors = []

        for field in self.REQUIRED_WORKLOAD_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        if "workload_type" in config:
            if config["workload_type"] not in self.VALID_WORKLOAD_TYPES:
                errors.append(f"Invalid workload_type: {config['workload_type']}")

        if "phases" in config:
            for i, phase in enumerate(config["phases"]):
                phase_errors = self._validate_phase(phase, i)
                errors.extend(phase_errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_model_yaml(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate model YAML configuration.

        Args:
            config: Model config dict

        Returns:
            ValidationResult
        """
        errors = []

        for field in self.REQUIRED_MODEL_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        if "config" in config:
            config_errors = self._validate_model_config(config["config"])
            errors.extend(config_errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_phase(self, phase: Dict[str, Any], index: int) -> List[str]:
        """Validate a phase configuration."""
        errors = []

        if "name" not in phase:
            errors.append(f"Phase {index}: missing 'name'")

        if "compute_type" not in phase:
            errors.append(f"Phase {index}: missing 'compute_type'")
        elif phase["compute_type"] not in self.VALID_COMPUTE_TYPES:
            errors.append(f"Phase {index}: invalid compute_type '{phase['compute_type']}'")

        if "component" not in phase:
            errors.append(f"Phase {index}: missing 'component'")

        if "compute_pattern" in phase and phase["compute_pattern"] not in self.VALID_COMPUTE_PATTERNS:
            errors.append(f"Phase {index}: invalid compute_pattern '{phase['compute_pattern']}'")

        return errors

    def _validate_model_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate model config."""
        errors = []

        required = ["hidden_size", "num_layers", "num_heads", "vocab_size"]
        for field in required:
            if field not in config:
                errors.append(f"Model config missing: {field}")

        return errors

    def _validate_params(
        self, params: Dict[str, Any], param_schema: Dict[str, ParamSchemaItem]
    ) -> List[str]:
        """Validate parameters against schema."""
        errors = []

        for name, schema_item in param_schema.items():
            if schema_item.required and name not in params:
                errors.append(f"Missing required parameter: {name}")

            if name in params:
                value = params[name]
                if schema_item.type == "number":
                    if not isinstance(value, (int, float)):
                        errors.append(f"Parameter {name} should be number")
                    elif schema_item.min is not None and value < schema_item.min:
                        errors.append(f"Parameter {name} below min: {schema_item.min}")
                    elif schema_item.max is not None and value > schema_item.max:
                        errors.append(f"Parameter {name} above max: {schema_item.max}")

        return errors

    def _validate_hardware(self, hardware: HardwareSchema) -> List[str]:
        """Validate hardware configuration."""
        errors = []

        if not hardware.device_preset:
            errors.append("Missing device_preset")

        if hardware.num_devices < 1:
            errors.append("num_devices must be >= 1")

        return errors

    def _validate_strategy(self, strategy: StrategySchema) -> List[str]:
        """Validate strategy configuration."""
        errors = []

        if strategy.tp_degree < 1:
            errors.append("tp_degree must be >= 1")
        if strategy.pp_degree < 1:
            errors.append("pp_degree must be >= 1")
        if strategy.dp_degree < 1:
            errors.append("dp_degree must be >= 1")

        if strategy.zero_stage not in [0, 1, 2, 3]:
            errors.append("zero_stage must be 0, 1, 2, or 3")

        return errors

    def _validate_model_workload_support(
        self, model_name: str, workload_name: str
    ) -> List[str]:
        """Validate that model supports workload."""
        errors = []

        try:
            supported = self.loader.get_supported_workloads_for_model(model_name)
            category = workload_name.split("/")[0] if "/" in workload_name else workload_name

            if category not in supported:
                errors.append(f"Model {model_name} does not support workload {category}")
        except FileNotFoundError:
            errors.append(f"Cannot verify workload support: model or workload not found")

        return errors

    def _check_memory_constraint(
        self, hardware: HardwareSchema, strategy: StrategySchema, model: ShardedModule
    ) -> Optional[str]:
        """Check if estimated memory exceeds device memory.

        Returns:
            Warning message if memory may exceed, None otherwise
        """
        try:
            device = Device.from_preset(hardware.device_preset)
            device_memory_gb = device.config.memory_gb

            hidden_size = getattr(model, "hidden_size", 4096)
            num_layers = getattr(model, "num_layers", 32)
            dtype = getattr(model, "dtype", "fp16")
            dtype_size = DTYPE_SIZES.get(dtype, 2)

            params_count = num_layers * 2 * hidden_size * hidden_size
            params_count += hidden_size * 100000
            params_per_device = params_count // strategy.tp_degree // strategy.dp_degree

            weight_memory_gb = params_per_device * dtype_size / 1e9

            if weight_memory_gb > device_memory_gb * 0.8:
                return f"Estimated weight memory ({weight_memory_gb:.2f} GB) may exceed device memory ({device_memory_gb} GB)"

        except Exception:
            pass

        return None
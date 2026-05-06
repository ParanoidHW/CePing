"""Pydantic schema definitions for Web2 API.

These schemas are designed for:
1. Frontend form rendering (schema-driven)
2. API request/response structure
3. Configuration validation
4. Automatic JSON schema generation

Pydantic provides:
- Runtime validation
- Type coercion
- JSON schema generation
- Strict field checking

Migration from dataclass:
- Replace @dataclass with BaseModel
- Replace field(default_factory=dict) with Field(default_factory=dict)
- Remove to_dict() methods (use model_dump() instead)
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field


class WorkloadCategory(str, Enum):
    """Workload categories for frontend grouping."""

    TRAINING = "training"
    RL_TRAINING = "rl_training"
    INFERENCE = "inference"
    DIFFUSION = "diffusion"
    CONV = "conv"
    MULTIMODAL = "multimodal"


class ParamSchemaItem(BaseModel):
    """Parameter schema item for frontend form rendering.

    Attributes:
        name: Parameter name (e.g., "batch_size")
        label: Display label (e.g., "Batch Size")
        type: Parameter type ("number" or "string")
        default: Default value
        min: Minimum value (optional)
        max: Maximum value (optional)
        required: Whether this parameter is required
        description: Parameter description
    """

    name: str
    label: str
    type: str = "number"
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    required: bool = True
    description: str = ""

    model_config = {"extra": "forbid"}


class StageSchema(BaseModel):
    """Stage schema for frontend rendering.

    A stage represents a compute phase in workload.

    Attributes:
        name: Stage name (e.g., "prefill", "decode", "forward")
        compute_type: Compute type ("forward", "backward", "optimizer")
        component: Component identifier (e.g., "main", "encoder", "decoder")
        repeat: Repeat count (int or dynamic param name)
        compute_pattern: Compute pattern (e.g., "transformer_block")
        extra_params: Stage-specific parameters
    """

    name: str
    compute_type: str
    component: str = "main"
    repeat: Union[int, str] = 1
    compute_pattern: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class HardwareSchema(BaseModel):
    """Hardware schema for frontend rendering.

    Attributes:
        device_preset: Device preset name (e.g., "H100-SXM-80GB")
        num_devices: Number of devices
        topology_type: Topology type (e.g., "homogeneous", "pd_disagg")
        custom_topology: Custom topology config (optional)
    """

    device_preset: str
    num_devices: int = 1
    topology_type: str = "homogeneous"
    custom_topology: Optional[Dict[str, Any]] = None

    model_config = {"extra": "forbid"}


class StrategySchema(BaseModel):
    """Strategy schema for frontend rendering.

    Attributes:
        tp_degree: Tensor parallelism degree
        pp_degree: Pipeline parallelism degree
        dp_degree: Data parallelism degree
        ep_degree: Expert parallelism degree (for MoE)
        sp_degree: Sequence parallelism degree
        activation_checkpointing: Whether to use activation checkpointing
        zero_stage: ZeRO optimization stage (0, 1, 2, 3)
    """

    tp_degree: int = 1
    pp_degree: int = 1
    dp_degree: int = 1
    ep_degree: int = 1
    sp_degree: int = 1
    activation_checkpointing: bool = False
    zero_stage: int = 0

    model_config = {"extra": "forbid"}


class ModelSchema(BaseModel):
    """Model schema for frontend rendering.

    Attributes:
        name: Model name (e.g., "llama-7b")
        description: Model description
        architecture: Architecture type (e.g., "llama", "deepseek")
        sparse_type: Sparse type (e.g., "dense", "moe")
        attention_features: Attention features (e.g., ["gqa", "mla"])
        supported_workloads: Supported workload categories
        config: Model configuration dict
        param_schema: Parameter schema by workload type
    """

    name: str
    description: str
    architecture: str
    sparse_type: str = "dense"
    attention_features: List[str] = Field(default_factory=list)
    supported_workloads: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    param_schema: Dict[str, List[ParamSchemaItem]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def get_param_schema(self, workload_type: str) -> List[ParamSchemaItem]:
        """Get parameter schema for a specific workload type."""
        return self.param_schema.get(workload_type, [])


class WorkloadSchema(BaseModel):
    """Workload schema for frontend rendering.

    This is the complete schema for a workload, used to:
    1. Render frontend form
    2. Validate API request
    3. Generate default values

    Attributes:
        name: Workload name (e.g., "autoregressive-inference")
        display_name: User-friendly display name
        workload_name: Internal workload identifier (e.g., "inference/autoregressive")
        description: Workload description
        category: Workload category (for frontend grouping)
        workload_type: Workload type (training/inference/diffusion)
        compute_mode: Compute mode description
        stages: List of stage schemas
        parameters: Parameter schema dict
        throughput_metric: Primary throughput metric
        supported_models: List of supported model names
    """

    name: str
    display_name: str = ""
    workload_name: str = ""
    description: str = ""
    category: WorkloadCategory = WorkloadCategory.INFERENCE
    workload_type: str = "inference"
    compute_mode: str = ""
    stages: List[StageSchema] = Field(default_factory=list)
    parameters: Dict[str, ParamSchemaItem] = Field(default_factory=dict)
    throughput_metric: str = "tokens_per_sec"
    supported_models: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    def get_defaults(self) -> Dict[str, Any]:
        """Get default parameter values."""
        return {k: v.default for k, v in self.parameters.items() if v.default is not None}

    def get_required_params(self) -> List[str]:
        """Get required parameter names."""
        return [k for k, v in self.parameters.items() if v.required]
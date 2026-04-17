"""Base classes for Scenario abstraction.

A Scenario represents a complete performance evaluation context, including:
- Models involved
- Parallelism strategies
- Analysis pipelines
- Result types

This abstraction allows supporting different workload types:
- LLM Training (single model, forward+backward)
- LLM Inference (single model, prefill+decode)
- PD Disaggregation (prefill and decode nodes separate)
- RL Training (policy/reward/reference models)
- Diffusion Generation (text encoder, dit, vae)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from .models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig


class ScenarioType(str, Enum):
    """Types of evaluation scenarios."""

    LLM_TRAINING = "llm_training"
    LLM_INFERENCE = "llm_inference"
    PD_DISAGG = "pd_disagg"
    RL_TRAINING = "rl_training"
    DIFFUSION = "diffusion"


class ParallelismType(str, Enum):
    """Supported parallelism types for scenarios."""

    TP = "tp"
    PP = "pp"
    DP = "dp"
    EP = "ep"
    SP = "sp"
    CP = "cp"


@dataclass
class ScenarioConfig:
    """Configuration for a performance evaluation scenario.

    Attributes:
        name: Unique identifier for this scenario
        description: Human-readable description
        scenario_type: Type of scenario (training, inference, etc.)
        required_models: List of model roles required (e.g., ["policy", "reward"])
        supported_parallelisms: List of parallelism types this scenario supports
        default_strategy: Default parallelism strategy configuration
        additional_config: Scenario-specific configuration
    """

    name: str
    description: str = ""
    scenario_type: ScenarioType = ScenarioType.LLM_INFERENCE
    required_models: List[str] = field(default_factory=list)
    supported_parallelisms: List[ParallelismType] = field(
        default_factory=lambda: [
            ParallelismType.TP,
            ParallelismType.PP,
            ParallelismType.DP,
        ]
    )
    default_strategy: Optional[Dict[str, Any]] = None
    additional_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type.value,
            "required_models": self.required_models,
            "supported_parallelisms": [p.value for p in self.supported_parallelisms],
            "default_strategy": self.default_strategy or {},
            "additional_config": self.additional_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioConfig":
        """Create from dictionary."""
        scenario_type_str = data.get("scenario_type", "llm_inference")
        try:
            scenario_type = ScenarioType(scenario_type_str)
        except ValueError:
            scenario_type = ScenarioType.LLM_INFERENCE

        parallelisms = data.get("supported_parallelisms", ["tp", "pp", "dp"])
        supported_parallelisms = [ParallelismType(p) for p in parallelisms if p in [e.value for e in ParallelismType]]

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            scenario_type=scenario_type,
            required_models=data.get("required_models", []),
            supported_parallelisms=supported_parallelisms,
            default_strategy=data.get("default_strategy"),
            additional_config=data.get("additional_config", {}),
        )


@dataclass
class ScenarioResult:
    """Base result class for scenario evaluation.

    Attributes:
        scenario_name: Name of the scenario that produced this result
        total_time_sec: Total execution time
        throughput: Primary throughput metric
        memory_peak_gb: Peak memory usage
        breakdown: Detailed breakdown by component/model
        metadata: Additional metadata
    """

    scenario_name: str
    total_time_sec: float = 0.0
    throughput: float = 0.0
    memory_peak_gb: float = 0.0
    breakdown: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "scenario_name": self.scenario_name,
            "total_time_sec": self.total_time_sec,
            "throughput": self.throughput,
            "memory_peak_gb": self.memory_peak_gb,
            "breakdown": self.breakdown,
            "metadata": self.metadata,
        }


class Scenario(ABC):
    """Abstract base class for performance evaluation scenarios.

    A Scenario encapsulates:
    - Model(s) configuration
    - Hardware configuration
    - Strategy configuration
    - Analysis pipeline construction
    - Result generation

    Subclasses implement specific scenario types:
    - LLMTrainingScenario: Single model training
    - LLMInferenceScenario: Single model inference
    - PDDisaggScenario: Prefill-Decode disaggregated inference
    - RLTrainingScenario: RL post-training (PPO, etc.)
    - DiffusionScenario: Diffusion-based generation
    """

    def __init__(
        self,
        config: ScenarioConfig,
        models: Dict[str, BaseModel],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize scenario.

        Args:
            config: Scenario configuration
            models: Dictionary of models by role (e.g., {"policy": model, "reward": model})
            device: Device configuration
            cluster: Cluster configuration
            strategy: Parallelism strategy
        """
        self.config = config
        self.models = models
        self.device = device
        self.cluster = cluster
        self.strategy = strategy
        self._validate_models()

    def _validate_models(self) -> None:
        """Validate that required models are provided."""
        missing = []
        for required_role in self.config.required_models:
            if required_role not in self.models:
                missing.append(required_role)
        if missing:
            raise ValueError(f"Scenario '{self.config.name}' requires models for roles: {missing}")

    @abstractmethod
    def analyze(self, **kwargs) -> ScenarioResult:
        """Run performance analysis for this scenario.

        Args:
            **kwargs: Scenario-specific parameters

        Returns:
            ScenarioResult with performance metrics
        """
        pass

    @abstractmethod
    def get_analyzer(self) -> Any:
        """Get the analyzer instance for this scenario.

        Returns:
            Analyzer instance (TrainingAnalyzer, InferenceAnalyzer, etc.)
        """
        pass

    def get_model(self, role: str) -> Optional[BaseModel]:
        """Get model by role.

        Args:
            role: Model role (e.g., "policy", "reward", "main")

        Returns:
            Model if found, None otherwise
        """
        return self.models.get(role)

    def supports_parallelism(self, parallelism_type: ParallelismType) -> bool:
        """Check if this scenario supports a specific parallelism type.

        Args:
            parallelism_type: Type of parallelism to check

        Returns:
            True if supported, False otherwise
        """
        return parallelism_type in self.config.supported_parallelisms

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary representation."""
        return {
            "config": self.config.to_dict(),
            "models": {role: model.config.name for role, model in self.models.items()},
            "device": self.device.name,
            "cluster": {
                "num_nodes": self.cluster.num_nodes,
                "devices_per_node": self.cluster.devices_per_node,
            },
            "strategy": self.strategy.to_dict(),
        }

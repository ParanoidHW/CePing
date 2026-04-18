"""Scenario Registry implementation.

Provides centralized registration and factory pattern for scenarios.
Supports dynamic scenario discovery and instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .base import Scenario, ScenarioConfig, ScenarioType
from llm_perf.modeling import ShardedModule
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig


@dataclass
class ScenarioInfo:
    """Information about a registered scenario."""

    name: str
    config_class: Type[ScenarioConfig]
    scenario_class: Type[Scenario]
    description: str
    scenario_type: ScenarioType
    required_models: List[str] = field(default_factory=list)
    supported_parallelisms: List[str] = field(default_factory=list)
    default_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type.value,
            "required_models": self.required_models,
            "supported_parallelisms": self.supported_parallelisms,
            "config_class": self.config_class.__name__,
            "scenario_class": self.scenario_class.__name__,
            "default_config": self.default_config or {},
        }


class ScenarioRegistry:
    """Central registry for all scenarios.

    Provides factory pattern for scenario creation and supports dynamic scenario discovery.
    """

    _instance: Optional[ScenarioRegistry] = None
    _scenarios: Dict[str, ScenarioInfo]

    def __new__(cls) -> ScenarioRegistry:
        """Singleton pattern to ensure single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._scenarios = {}
        return cls._instance

    def register(
        self,
        name: str,
        scenario_class: Type[Scenario],
        config_class: Optional[Type[ScenarioConfig]] = None,
        description: str = "",
        scenario_type: Optional[ScenarioType] = None,
        required_models: Optional[List[str]] = None,
        supported_parallelisms: Optional[List[str]] = None,
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a scenario with the registry."""
        if name in self._scenarios:
            raise ValueError(f"Scenario '{name}' is already registered")

        config_class = config_class or ScenarioConfig
        scenario_type = scenario_type or ScenarioType.LLM_INFERENCE
        required_models = required_models or []
        supported_parallelisms = supported_parallelisms or ["tp", "pp", "dp"]

        self._scenarios[name] = ScenarioInfo(
            name=name,
            config_class=config_class,
            scenario_class=scenario_class,
            description=description,
            scenario_type=scenario_type,
            required_models=required_models,
            supported_parallelisms=supported_parallelisms,
            default_config=default_config,
        )

    def unregister(self, name: str) -> None:
        """Unregister a scenario."""
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found in registry")
        del self._scenarios[name]

    def get(self, name: str) -> ScenarioInfo:
        """Get scenario information."""
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found in registry")
        return self._scenarios[name]

    def create_scenario(
        self,
        name: str,
        models: Dict[str, ShardedModule],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
        **config_kwargs: Any,
    ) -> Scenario:
        """Create a scenario instance by name."""
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found in registry")

        info = self._scenarios[name]

        merged_config = dict(info.default_config or {})
        merged_config.update(config_kwargs)
        merged_config.setdefault("name", name)
        merged_config.setdefault("description", info.description)
        merged_config.setdefault("scenario_type", info.scenario_type)
        merged_config.setdefault("required_models", info.required_models)

        config = info.config_class(**merged_config)

        return info.scenario_class(
            config=config,
            models=models,
            device=device,
            cluster=cluster,
            strategy=strategy,
        )

    def list_scenarios(self, scenario_type: Optional[ScenarioType] = None) -> List[str]:
        """List all registered scenario names."""
        if scenario_type:
            return [name for name, info in self._scenarios.items() if info.scenario_type == scenario_type]
        return list(self._scenarios.keys())

    def list_by_type(self) -> Dict[str, List[str]]:
        """List scenarios grouped by type."""
        result: Dict[str, List[str]] = {}
        for name, info in self._scenarios.items():
            type_name = info.scenario_type.value
            if type_name not in result:
                result[type_name] = []
            result[type_name].append(name)
        return result

    def get_all_infos(self) -> Dict[str, ScenarioInfo]:
        """Get all registered scenario informations."""
        return dict(self._scenarios)

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary representation."""
        return {
            "scenarios": {name: info.to_dict() for name, info in self._scenarios.items()},
            "by_type": self.list_by_type(),
        }

    def clear(self) -> None:
        """Clear all registered scenarios."""
        self._scenarios.clear()

    def is_registered(self, name: str) -> bool:
        """Check if a scenario is registered."""
        return name in self._scenarios

    def get_scenarios_for_model(self, model_type: str) -> List[str]:
        """Get scenarios that can work with a specific model type."""
        compatible = []
        for name, info in self._scenarios.items():
            if model_type in info.required_models or "main" in info.required_models:
                compatible.append(name)
        return compatible


def register_all_scenarios() -> None:
    """Register all built-in scenarios with the ScenarioRegistry."""
    from .llm_training import LLMTrainingScenario, LLMTrainingConfig
    from .llm_inference import LLMInferenceScenario, LLMInferenceConfig
    from .base import ScenarioType

    registry = ScenarioRegistry()

    registry.register(
        name="training",
        scenario_class=LLMTrainingScenario,
        config_class=LLMTrainingConfig,
        description="Standard LLM training scenario (forward + backward)",
        scenario_type=ScenarioType.LLM_TRAINING,
        required_models=["main"],
        supported_parallelisms=["tp", "pp", "dp", "ep", "sp", "cp"],
        default_config={
            "batch_size": 1,
            "seq_len": 2048,
        },
    )

    registry.register(
        name="autoregressive-inference",
        scenario_class=LLMInferenceScenario,
        config_class=LLMInferenceConfig,
        description="Standard LLM inference scenario (prefill + decode)",
        scenario_type=ScenarioType.LLM_INFERENCE,
        required_models=["main"],
        supported_parallelisms=["tp", "pp", "dp", "ep", "sp", "cp"],
        default_config={
            "batch_size": 1,
            "prompt_len": 512,
            "generation_len": 128,
        },
    )


register_all_scenarios()

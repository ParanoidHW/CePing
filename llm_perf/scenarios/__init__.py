"""Scenario Layer for LLM Performance Evaluator.

A Scenario represents a complete performance evaluation context:
- Traditional LLM Training: Single model with forward+backward passes
- Traditional LLM Inference: Single model with prefill+decode phases
- PD Disaggregation: Prefill and decode nodes separately
- RL Post-Training: Policy/Reward/Reference models (PPO-style)
- Diffusion Generation: Text Encoder -> DiT -> VAE pipeline

The Scenario abstraction provides:
- Unified interface for different workload types
- Extensible design for new scenarios
- Component-based analysis with detailed breakdowns

Example:
    >>> from llm_perf.scenarios import ScenarioRegistry, ScenarioType
    >>> from llm_perf.models.registry import create_model_from_config
    >>> from llm_perf.hardware.device import Device
    >>> from llm_perf.hardware.cluster import Cluster
    >>> from llm_perf.strategy.base import StrategyConfig
    >>>
    >>> # Create registry and get scenario
    >>> registry = ScenarioRegistry()
    >>> model = create_model_from_config({"preset": "llama-7b"})
    >>> device = Device(name="H100", memory_gb=80)
    >>> cluster = Cluster(num_nodes=1, devices_per_node=8)
    >>> strategy = StrategyConfig(tp_degree=8)
    >>>
    >>> # Create and run LLM inference scenario
    >>> scenario = registry.create_scenario(
    ...     "llm-inference",
    ...     models={"main": model},
    ...     device=device,
    ...     cluster=cluster,
    ...     strategy=strategy,
    ... )
    >>> result = scenario.analyze(batch_size=1, prompt_len=512, generation_len=128)
    >>> print(result.to_dict())
"""

from .base import (
    Scenario,
    ScenarioConfig,
    ScenarioResult,
    ScenarioType,
    ParallelismType,
)
from .registry import (
    ScenarioRegistry,
    ScenarioInfo,
    register_all_scenarios,
)
from .llm_training import (
    LLMTrainingScenario,
    LLMTrainingConfig,
    LLMTrainingResult,
)
from .llm_inference import (
    LLMInferenceScenario,
    LLMInferenceConfig,
    LLMInferenceResult,
)
from .pd_disagg import (
    PDDisaggScenario,
    PDDisaggConfig,
    PDDisaggResult,
    PDNodeConfig,
    PDNodeResult,
)
from .rl_training import (
    RLTrainingScenario,
    RLTrainingConfig,
    RLTrainingResult,
    RLModelConfig,
    RLModelResult,
)
from .diffusion import (
    DiffusionScenario,
    DiffusionConfig,
    DiffusionResult,
    DiffusionComponentConfig,
    ComponentResult,
)

__all__ = [
    "Scenario",
    "ScenarioConfig",
    "ScenarioResult",
    "ScenarioType",
    "ParallelismType",
    "ScenarioRegistry",
    "ScenarioInfo",
    "register_all_scenarios",
    "LLMTrainingScenario",
    "LLMTrainingConfig",
    "LLMTrainingResult",
    "LLMInferenceScenario",
    "LLMInferenceConfig",
    "LLMInferenceResult",
    "PDDisaggScenario",
    "PDDisaggConfig",
    "PDDisaggResult",
    "PDNodeConfig",
    "PDNodeResult",
    "RLTrainingScenario",
    "RLTrainingConfig",
    "RLTrainingResult",
    "RLModelConfig",
    "RLModelResult",
    "DiffusionScenario",
    "DiffusionConfig",
    "DiffusionResult",
    "DiffusionComponentConfig",
    "ComponentResult",
]

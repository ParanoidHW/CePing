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
from .colocate import (
    ColocateAnalyzer,
    ColocateResult,
    ModelAllocation,
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
    "ColocateAnalyzer",
    "ColocateResult",
    "ModelAllocation",
]

"""RL Training Scenario implementation.

RL post-training scenario for models like PPO, which involves:
- Policy model: The model being trained
- Reward model: Computes reward signals
- Reference model: Prevents policy from deviating too much

This scenario models the compute and memory requirements for
RL training workflows.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Scenario, ScenarioConfig, ScenarioResult, ScenarioType, ParallelismType
from llm_perf.legacy.models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig


@dataclass
class RLModelConfig:
    """Configuration for a model in RL training."""

    role: str
    tp_degree: int = 1
    pp_degree: int = 1
    dp_degree: int = 1
    weight_sharing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "tp_degree": self.tp_degree,
            "pp_degree": self.pp_degree,
            "dp_degree": self.dp_degree,
            "weight_sharing": self.weight_sharing,
        }


@dataclass
class RLTrainingConfig(ScenarioConfig):
    """Configuration for RL training scenario.

    Attributes:
        policy_config: Policy model configuration
        reward_config: Reward model configuration
        reference_config: Reference model configuration
        batch_size: PPO batch size
        seq_len: Sequence length
        num_ppo_steps: Number of PPO optimization steps
        num_rollouts: Number of rollouts per iteration
        clip_ratio: PPO clip ratio
    """

    policy_config: RLModelConfig = None
    reward_config: RLModelConfig = None
    reference_config: RLModelConfig = None
    batch_size: int = 1
    seq_len: int = 512
    num_ppo_steps: int = 100
    num_rollouts: int = 4
    clip_ratio: float = 0.2

    def __post_init__(self):
        """Set scenario type and required models."""
        self.scenario_type = ScenarioType.RL_TRAINING
        if not self.required_models:
            self.required_models = ["policy", "reward", "reference"]
        if not self.supported_parallelisms:
            self.supported_parallelisms = [
                ParallelismType.TP,
                ParallelismType.PP,
                ParallelismType.DP,
            ]
        if self.policy_config is None:
            self.policy_config = RLModelConfig(role="policy", tp_degree=2)
        if self.reward_config is None:
            self.reward_config = RLModelConfig(role="reward", tp_degree=1)
        if self.reference_config is None:
            self.reference_config = RLModelConfig(role="reference", tp_degree=1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "policy_config": self.policy_config.to_dict(),
                "reward_config": self.reward_config.to_dict(),
                "reference_config": self.reference_config.to_dict(),
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "num_ppo_steps": self.num_ppo_steps,
                "num_rollouts": self.num_rollouts,
                "clip_ratio": self.clip_ratio,
            }
        )
        return base


@dataclass
class RLModelResult:
    """Result for a single model in RL training."""

    role: str
    forward_time_sec: float = 0.0
    backward_time_sec: float = 0.0
    total_time_sec: float = 0.0
    memory_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "forward_time_sec": self.forward_time_sec,
            "backward_time_sec": self.backward_time_sec,
            "total_time_sec": self.total_time_sec,
            "memory_gb": self.memory_gb,
        }


@dataclass
class RLTrainingResult(ScenarioResult):
    """Result of RL training scenario evaluation.

    Attributes:
        policy_result: Policy model performance
        reward_result: Reward model performance
        reference_result: Reference model performance
        total_iteration_time_sec: Time for one PPO iteration
        samples_per_sec: Training throughput
        memory_total_gb: Total memory across all models
    """

    policy_result: RLModelResult = None
    reward_result: RLModelResult = None
    reference_result: RLModelResult = None
    total_iteration_time_sec: float = 0.0
    samples_per_sec: float = 0.0
    memory_total_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "policy": self.policy_result.to_dict() if self.policy_result else {},
                "reward": self.reward_result.to_dict() if self.reward_result else {},
                "reference": self.reference_result.to_dict() if self.reference_result else {},
                "total_iteration_time_sec": self.total_iteration_time_sec,
                "samples_per_sec": self.samples_per_sec,
                "memory_total_gb": self.memory_total_gb,
            }
        )
        return base


class RLTrainingScenario(Scenario):
    """RL post-training performance scenario.

    Models the performance of RL training workflows like PPO:
    - Rollout generation (policy forward)
    - Reward computation (reward forward)
    - Reference comparison (reference forward)
    - Policy optimization (policy forward + backward)

    Example:
        >>> from llm_perf.scenarios.registry import ScenarioRegistry
        >>> from llm_perf.modeling import LlamaModel, LlamaConfig
        >>>
        >>> registry = ScenarioRegistry()
        >>> policy = LlamaModel(LlamaConfig(...))
        >>> reward = LlamaModel(LlamaConfig(...))
        >>> reference = LlamaModel(LlamaConfig(...))
        >>>
        >>> scenario = registry.create_scenario(
        ...     "rl-training",
        ...     models={"policy": policy, "reward": reward, "reference": reference},
        ...     device=device,
        ...     cluster=cluster,
        ...     strategy=strategy,
        ... )
        >>> result = scenario.analyze(batch_size=4, seq_len=512)
    """

    def __init__(
        self,
        config: RLTrainingConfig,
        models: Dict[str, BaseModel],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize RL training scenario.

        Args:
            config: RL training configuration
            models: Dictionary with "policy", "reward", "reference" models
            device: Device configuration
            cluster: Cluster configuration
            strategy: Base strategy (used as template)
        """
        super().__init__(config, models, device, cluster, strategy)
        self._policy_analyzer: Any = None
        self._reward_analyzer: Any = None
        self._reference_analyzer: Any = None

    def _create_model_strategy(
        self,
        model_config: RLModelConfig,
    ) -> StrategyConfig:
        """Create strategy for a specific model.

        Args:
            model_config: Model configuration

        Returns:
            StrategyConfig for the model
        """
        return StrategyConfig(
            model_name=self.strategy.model_name,
            tp_degree=model_config.tp_degree,
            pp_degree=model_config.pp_degree,
            dp_degree=model_config.dp_degree,
            sp_degree=1,
            ep_degree=1,
        )

    def _create_analyzer(self, role: str) -> Any:
        """Create analyzer for a specific model role.

        Args:
            role: Model role ("policy", "reward", "reference")

        Returns:
            TrainingAnalyzer for the model
        """
        model = self.models.get(role)
        if model is None:
            return None

        config_map = {
            "policy": self.config.policy_config,
            "reward": self.config.reward_config,
            "reference": self.config.reference_config,
        }
        model_config = config_map.get(role)
        if model_config is None:
            return None

        strategy = self._create_model_strategy(model_config)

        from ..analyzer.training import TrainingAnalyzer

        return TrainingAnalyzer(
            model=model,
            device=self.device,
            cluster=self.cluster,
            strategy=strategy,
        )

    def _estimate_model_forward(
        self,
        role: str,
        batch_size: int,
        seq_len: int,
    ) -> float:
        """Estimate forward pass time for a model.

        Args:
            role: Model role
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Forward pass time in seconds
        """
        analyzer = self._create_analyzer(role)
        if analyzer is None:
            return 0.0

        result = analyzer.analyze(batch_size=batch_size, seq_len=seq_len, num_steps=1)

        forward_time = result.time_per_step_sec * 0.33
        return forward_time

    def _estimate_model_backward(
        self,
        role: str,
        batch_size: int,
        seq_len: int,
    ) -> float:
        """Estimate backward pass time for a model.

        Args:
            role: Model role
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Backward pass time in seconds
        """
        analyzer = self._create_analyzer(role)
        if analyzer is None:
            return 0.0

        result = analyzer.analyze(batch_size=batch_size, seq_len=seq_len, num_steps=1)

        backward_time = result.time_per_step_sec * 0.67
        return backward_time

    def _estimate_model_memory(
        self,
        role: str,
        batch_size: int,
        seq_len: int,
    ) -> float:
        """Estimate memory for a model.

        Args:
            role: Model role
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Memory in GB
        """
        analyzer = self._create_analyzer(role)
        if analyzer is None:
            return 0.0

        result = analyzer.analyze(batch_size=batch_size, seq_len=seq_len, num_steps=1)
        return result.memory_per_gpu_gb

    def analyze(
        self,
        batch_size: int = None,
        seq_len: int = None,
        num_ppo_steps: int = None,
        **kwargs,
    ) -> RLTrainingResult:
        """Run RL training analysis.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_ppo_steps: Number of PPO steps
            **kwargs: Additional parameters

        Returns:
            RLTrainingResult with performance metrics
        """
        batch_size = batch_size or self.config.batch_size
        seq_len = seq_len or self.config.seq_len
        num_ppo_steps = num_ppo_steps or self.config.num_ppo_steps

        policy_forward = self._estimate_model_forward("policy", batch_size, seq_len)
        policy_backward = self._estimate_model_backward("policy", batch_size, seq_len)
        reward_forward = self._estimate_model_forward("reward", batch_size, seq_len)
        reference_forward = self._estimate_model_forward("reference", batch_size, seq_len)

        rollout_time = policy_forward * self.config.num_rollouts
        reward_time = reward_forward * self.config.num_rollouts
        reference_time = reference_forward * self.config.num_rollouts
        optimization_time = (policy_forward + policy_backward) * num_ppo_steps

        total_iteration_time = rollout_time + reward_time + reference_time + optimization_time
        samples_per_sec = batch_size / total_iteration_time if total_iteration_time > 0 else 0

        policy_memory = self._estimate_model_memory("policy", batch_size, seq_len)
        reward_memory = self._estimate_model_memory("reward", batch_size, seq_len)
        reference_memory = self._estimate_model_memory("reference", batch_size, seq_len)

        total_memory = policy_memory + reward_memory + reference_memory
        if self.config.reference_config.weight_sharing:
            total_memory = policy_memory + reward_memory

        policy_result = RLModelResult(
            role="policy",
            forward_time_sec=policy_forward,
            backward_time_sec=policy_backward,
            total_time_sec=policy_forward + policy_backward,
            memory_gb=policy_memory,
        )

        reward_result = RLModelResult(
            role="reward",
            forward_time_sec=reward_forward,
            backward_time_sec=0.0,
            total_time_sec=reward_forward,
            memory_gb=reward_memory,
        )

        reference_result = RLModelResult(
            role="reference",
            forward_time_sec=reference_forward,
            backward_time_sec=0.0,
            total_time_sec=reference_forward,
            memory_gb=reference_memory if not self.config.reference_config.weight_sharing else 0.0,
        )

        return RLTrainingResult(
            scenario_name=self.config.name,
            total_time_sec=total_iteration_time,
            throughput=samples_per_sec,
            memory_peak_gb=total_memory,
            policy_result=policy_result,
            reward_result=reward_result,
            reference_result=reference_result,
            total_iteration_time_sec=total_iteration_time,
            samples_per_sec=samples_per_sec,
            memory_total_gb=total_memory,
            breakdown={
                "rollout_time_sec": rollout_time,
                "reward_time_sec": reward_time,
                "reference_time_sec": reference_time,
                "optimization_time_sec": optimization_time,
                "rollout_percent": rollout_time / total_iteration_time * 100 if total_iteration_time > 0 else 0,
                "reward_percent": reward_time / total_iteration_time * 100 if total_iteration_time > 0 else 0,
                "reference_percent": reference_time / total_iteration_time * 100 if total_iteration_time > 0 else 0,
                "optimization_percent": optimization_time / total_iteration_time * 100
                if total_iteration_time > 0
                else 0,
            },
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_ppo_steps": num_ppo_steps,
                "num_rollouts": self.config.num_rollouts,
                "clip_ratio": self.config.clip_ratio,
            },
        )

    def get_analyzer(self) -> Any:
        """RL training uses multiple analyzers.

        Returns:
            None (this scenario doesn't have a single analyzer)
        """
        return None

    def get_policy_analyzer(self) -> Any:
        """Get analyzer for policy model.

        Returns:
            TrainingAnalyzer for policy
        """
        if self._policy_analyzer is None:
            self._policy_analyzer = self._create_analyzer("policy")
        return self._policy_analyzer

    def get_reward_analyzer(self) -> Any:
        """Get analyzer for reward model.

        Returns:
            TrainingAnalyzer for reward
        """
        if self._reward_analyzer is None:
            self._reward_analyzer = self._create_analyzer("reward")
        return self._reward_analyzer

    def get_reference_analyzer(self) -> Any:
        """Get analyzer for reference model.

        Returns:
            TrainingAnalyzer for reference
        """
        if self._reference_analyzer is None:
            self._reference_analyzer = self._create_analyzer("reference")
        return self._reference_analyzer

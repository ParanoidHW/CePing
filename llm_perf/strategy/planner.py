"""Strategy planner for automatic parallelism strategy selection."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .base import StrategyConfig, ParallelStrategy


@dataclass
class StrategyConstraints:
    """Constraints for strategy planning."""
    max_tp_degree: int = 8
    max_pp_degree: int = 16
    max_dp_degree: int = 256
    min_batch_per_dp: int = 1
    prefer_memory_efficient: bool = True


class StrategyPlanner:
    """
    Automatically plans parallelism strategy based on model and hardware constraints.
    
    This class provides intelligent strategy recommendations considering:
    - Model size and architecture
    - Available hardware resources
    - Memory constraints
    - Communication costs
    """
    
    def __init__(self, constraints: Optional[StrategyConstraints] = None):
        self.constraints = constraints or StrategyConstraints()
    
    def plan_strategy(
        self,
        model_name: str,
        model_params_b: float,
        world_size: int,
        batch_size: int = 1,
        sequence_length: int = 4096,
        memory_per_device_gb: float = 80.0,
        is_moe: bool = False,
        num_experts: int = 1,
    ) -> StrategyConfig:
        """
        Plan an optimal parallelism strategy.
        
        Args:
            model_name: Name of the model
            model_params_b: Model size in billions of parameters
            world_size: Total number of devices available
            batch_size: Global batch size
            sequence_length: Sequence length
            memory_per_device_gb: Memory per device in GB
            is_moe: Whether it's a MoE model
            num_experts: Number of experts (for MoE)
        
        Returns:
            Optimal StrategyConfig
        """
        # Estimate memory requirements
        param_memory_gb = model_params_b * 2  # FP16 parameters (2 bytes per param)
        activation_memory_gb = self._estimate_activation_memory(
            model_params_b, batch_size, sequence_length, is_moe
        )
        
        total_memory_needed = param_memory_gb + activation_memory_gb
        
        # Start with TP to fit model on single device or reduce per-device memory
        tp_degree = 1
        if param_memory_gb > memory_per_device_gb * 0.8:
            # Model doesn't fit on single device, need TP
            tp_degree = min(
                8,  # Max TP degree
                self._next_power_of_2(int(param_memory_gb / (memory_per_device_gb * 0.7)) + 1),
                world_size,
                self.constraints.max_tp_degree
            )
        
        # Calculate remaining devices after TP
        remaining_devices = world_size // tp_degree
        
        # Determine DP degree based on batch size
        dp_degree = min(
            batch_size,  # At least 1 sample per DP replica
            remaining_devices,
            self.constraints.max_dp_degree
        )
        
        # Use PP if we have more devices after TP+DP
        remaining_after_dp = remaining_devices // dp_degree
        pp_degree = min(
            remaining_after_dp,
            self.constraints.max_pp_degree
        )
        
        # For MoE models, add EP
        ep_degree = 1
        if is_moe and num_experts > 1:
            # EP degree should divide num_experts and not exceed available devices
            ep_degree = min(
                num_experts,
                world_size // (tp_degree * pp_degree * dp_degree),
                8  # Reasonable max EP degree
            )
            if ep_degree < 2:
                ep_degree = 1
        
        config = StrategyConfig(
            model_name=model_name,
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            dp_degree=dp_degree,
            ep_degree=ep_degree,
            zero_stage=1 if dp_degree > 1 else 0,
            activation_checkpointing=pp_degree > 1,
        )
        
        return config
    
    def _estimate_activation_memory(
        self,
        model_params_b: float,
        batch_size: int,
        sequence_length: int,
        is_moe: bool
    ) -> float:
        """Estimate activation memory in GB."""
        # Simplified estimation
        base_activation = batch_size * sequence_length * model_params_b * 0.001
        if is_moe:
            base_activation *= 1.5  # MoE has more activations
        return base_activation
    
    def _next_power_of_2(self, n: int) -> int:
        """Get next power of 2 >= n."""
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()
    
    def recommend_strategy(
        self,
        model_params_b: float,
        world_size: int,
        target_batch_size: int = 1,
        is_inference: bool = False
    ) -> Dict[str, Any]:
        """
        Get strategy recommendations with explanations.
        
        Args:
            model_params_b: Model size in billions
            world_size: Total number of devices
            target_batch_size: Target batch size
            is_inference: Whether for inference
        
        Returns:
            Dict with recommended config and explanations
        """
        config = self.plan_strategy(
            model_name="unknown",
            model_params_b=model_params_b,
            world_size=world_size,
            batch_size=target_batch_size,
        )
        
        explanations = []
        
        if config.tp_degree > 1:
            explanations.append(
                f"TP={config.tp_degree}: Model size ({model_params_b}B) requires "
                f"tensor parallelism to fit in memory"
            )
        
        if config.pp_degree > 1:
            explanations.append(
                f"PP={config.pp_degree}: Pipeline parallelism for efficient "
                f"utilization with {world_size} devices"
            )
        
        if config.dp_degree > 1:
            explanations.append(
                f"DP={config.dp_degree}: Data parallelism for batch size {target_batch_size}"
            )
        
        if config.zero_stage > 0:
            explanations.append(
                f"ZeRO-{config.zero_stage}: Memory optimization for data parallelism"
            )
        
        return {
            "config": config.to_dict(),
            "explanations": explanations,
            "memory_estimate_gb": model_params_b * 2 / config.tp_degree,
        }

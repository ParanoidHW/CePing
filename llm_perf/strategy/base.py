"""Base strategy classes."""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum


class ParallelType(Enum):
    """Types of parallelism."""
    TENSOR = "tp"          # Tensor Parallelism
    PIPELINE = "pp"        # Pipeline Parallelism
    DATA = "dp"            # Data Parallelism
    EXPERT = "ep"          # Expert Parallelism (MoE)
    SEQUENCE = "sp"        # Sequence Parallelism
    CONTEXT = "cp"         # Context Parallelism


class SPType(Enum):
    """Types of sequence parallelism."""
    ULYSSES = "ulysses"
    RING_P2P = "ring_p2p"
    RING_ALLGATHER = "ring_allgather"
    UNIFIED_2D = "unified_2d"
    MEGATRON = "megatron"


@dataclass
class ParallelConfig:
    """Configuration for a specific parallelism type."""
    enabled: bool = False
    degree: int = 1  # Parallelism degree (e.g., TP=4 means 4 GPUs)
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Complete parallelism strategy configuration."""
    
    # Model configuration reference
    model_name: str = ""
    
    # Parallelism degrees
    tp_degree: int = 1   # Tensor Parallelism
    pp_degree: int = 1   # Pipeline Parallelism
    dp_degree: int = 1   # Data Parallelism
    ep_degree: int = 1   # Expert Parallelism
    sp_degree: int = 1   # Sequence Parallelism
    cp_degree: int = 1   # Context Parallelism
    
    # Sequence parallelism configuration
    sp_type: SPType = SPType.ULYSSES
    ulysses_degree: int = 1
    ring_degree: int = 1
    
    # Scheduling options
    pipeline_schedule: str = "1f1b"  # 1F1B, GPipe, etc.
    micro_batch_size: int = 1
    
    # Optimization flags
    activation_checkpointing: bool = False
    sequence_parallel: bool = False
    use_megatron: bool = True
    
    # Memory optimization
    zero_stage: int = 0  # ZeRO stage (0-3)
    
    @property
    def world_size(self) -> int:
        """Total number of GPUs needed."""
        return (
            self.tp_degree * 
            self.pp_degree * 
            self.dp_degree * 
            self.ep_degree * 
            self.sp_degree
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "parallelism": {
                "tp": self.tp_degree,
                "pp": self.pp_degree,
                "dp": self.dp_degree,
                "ep": self.ep_degree,
                "sp": self.sp_degree,
                "cp": self.cp_degree,
            },
            "sequence_parallelism": {
                "sp_type": self.sp_type.value,
                "ulysses_degree": self.ulysses_degree,
                "ring_degree": self.ring_degree,
            },
            "scheduling": {
                "pipeline_schedule": self.pipeline_schedule,
                "micro_batch_size": self.micro_batch_size,
            },
            "optimization": {
                "activation_checkpointing": self.activation_checkpointing,
                "sequence_parallel": self.sequence_parallel,
                "use_megatron": self.use_megatron,
                "zero_stage": self.zero_stage,
            },
            "world_size": self.world_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Create from dictionary."""
        parallel = data.get("parallelism", {})
        scheduling = data.get("scheduling", {})
        optimization = data.get("optimization", {})
        sp_config = data.get("sequence_parallelism", {})
        
        sp_type_str = sp_config.get("sp_type", "ulysses")
        try:
            sp_type = SPType(sp_type_str)
        except ValueError:
            sp_type = SPType.ULYSSES
        
        return cls(
            model_name=data.get("model_name", ""),
            tp_degree=parallel.get("tp", 1),
            pp_degree=parallel.get("pp", 1),
            dp_degree=parallel.get("dp", 1),
            ep_degree=parallel.get("ep", 1),
            sp_degree=parallel.get("sp", 1),
            cp_degree=parallel.get("cp", 1),
            sp_type=sp_type,
            ulysses_degree=sp_config.get("ulysses_degree", 1),
            ring_degree=sp_config.get("ring_degree", 1),
            pipeline_schedule=scheduling.get("pipeline_schedule", "1f1b"),
            micro_batch_size=scheduling.get("micro_batch_size", 1),
            activation_checkpointing=optimization.get("activation_checkpointing", False),
            sequence_parallel=optimization.get("sequence_parallel", False),
            use_megatron=optimization.get("use_megatron", True),
            zero_stage=optimization.get("zero_stage", 0),
        )


class ParallelStrategy:
    """
    Represents a parallel execution strategy.
    
    This class tracks how different parts of the model are distributed
    across devices and what communication patterns are needed.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self._layer_assignment: Dict[str, int] = {}  # layer -> pp_stage
        self._tensor_sharding: Dict[str, int] = {}   # layer -> tp_group
        self._expert_assignment: Dict[str, int] = {} # expert -> ep_group
    
    def assign_layer_to_stage(self, layer_name: str, pp_stage: int):
        """Assign a layer to a pipeline stage."""
        self._layer_assignment[layer_name] = pp_stage
    
    def get_layer_stage(self, layer_name: str) -> int:
        """Get pipeline stage for a layer."""
        return self._layer_assignment.get(layer_name, 0)
    
    def is_tp_enabled(self) -> bool:
        """Check if tensor parallelism is enabled."""
        return self.config.tp_degree > 1
    
    def is_pp_enabled(self) -> bool:
        """Check if pipeline parallelism is enabled."""
        return self.config.pp_degree > 1
    
    def is_dp_enabled(self) -> bool:
        """Check if data parallelism is enabled."""
        return self.config.dp_degree > 1
    
    def is_ep_enabled(self) -> bool:
        """Check if expert parallelism is enabled."""
        return self.config.ep_degree > 1
    
    def get_tp_group(self, rank: int) -> List[int]:
        """Get all ranks in the same TP group as given rank."""
        if not self.is_tp_enabled():
            return [rank]
        
        # TP groups are formed within each DP group
        tp_size = self.config.tp_degree
        
        # Find which DP replica this rank belongs to
        dp_replica = rank // tp_size
        
        # Ranks in the same TP group
        start = dp_replica * tp_size
        return list(range(start, start + tp_size))
    
    def get_dp_group(self, rank: int) -> List[int]:
        """Get all ranks in the same DP group as given rank."""
        if not self.is_dp_enabled():
            return [rank]
        
        tp_size = self.config.tp_degree
        world_size = self.config.world_size
        
        # DP groups span across TP groups
        # For each TP position, collect ranks across DP replicas
        tp_position = rank % tp_size
        return list(range(tp_position, world_size, tp_size))
    
    def get_pp_group(self, rank: int) -> List[int]:
        """Get all ranks in the same PP group (all stages) as given rank."""
        if not self.is_pp_enabled():
            return [rank]
        
        # PP groups are formed within each TP+DP combination
        # This is a simplification - actual PP groups depend on implementation
        pp_size = self.config.pp_degree
        
        # Find position within PP stage
        ranks_per_stage = self.config.world_size // pp_size
        
        # All ranks in all stages of this pipeline
        result = []
        for s in range(pp_size):
            start = s * ranks_per_stage
            result.extend(range(start, start + ranks_per_stage))
        return result
    
    def get_ep_group(self, rank: int) -> List[int]:
        """Get all ranks in the same EP group as given rank."""
        if not self.is_ep_enabled():
            return [rank]
        
        # EP groups typically span across TP
        ep_size = self.config.ep_degree
        tp_size = self.config.tp_degree
        
        # Simplified: EP groups are formed within each TP group
        tp_position = rank % tp_size
        start = tp_position * ep_size
        return list(range(start, start + ep_size))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "layer_assignment": self._layer_assignment,
            "tensor_sharding": self._tensor_sharding,
            "expert_assignment": self._expert_assignment,
        }

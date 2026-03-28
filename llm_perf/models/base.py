"""Base model classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    name: str
    input_shape: tuple  # (batch_size, seq_len, hidden_dim) or similar
    output_shape: tuple
    params_count: int  # Number of parameters
    flops: int  # FLOPs for forward pass
    activation_bytes: int  # Activation memory in bytes
    is_moe: bool = False  # Whether this is an MoE layer


@dataclass  
class ModelConfig:
    """Base model configuration."""
    name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None  # For GQA
    intermediate_size: int = 0  # FFN intermediate size
    max_seq_len: int = 4096
    dtype: str = "fp16"
    # MoE specific
    num_experts: Optional[int] = None
    num_experts_per_token: Optional[int] = None


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._layers: List[LayerConfig] = []
    
    @property
    def layers(self) -> List[LayerConfig]:
        """Get all layer configurations."""
        return self._layers
    
    @abstractmethod
    def build_layers(self) -> List[LayerConfig]:
        """Build and return layer configurations."""
        pass
    
    @property
    def total_params(self) -> int:
        """Total number of parameters."""
        return sum(layer.params_count for layer in self._layers)
    
    @property
    def total_flops_forward(self) -> int:
        """Total FLOPs for forward pass."""
        return sum(layer.flops for layer in self._layers)
    
    @property
    def total_flops_backward(self) -> int:
        """Total FLOPs for backward pass (typically 2x forward)."""
        return self.total_flops_forward * 2
    
    @property
    def activation_memory(self) -> int:
        """Total activation memory in bytes."""
        return sum(layer.activation_bytes for layer in self._layers)
    
    def get_layer_by_name(self, name: str) -> Optional[LayerConfig]:
        """Get layer configuration by name."""
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "name": self.config.name,
            "config": {
                "vocab_size": self.config.vocab_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "intermediate_size": self.config.intermediate_size,
                "max_seq_len": self.config.max_seq_len,
                "dtype": self.config.dtype,
                "num_experts": self.config.num_experts,
                "num_experts_per_token": self.config.num_experts_per_token,
            },
            "total_params": self.total_params,
            "total_flops_forward": self.total_flops_forward,
            "activation_memory": self.activation_memory,
            "layers": [
                {
                    "name": layer.name,
                    "input_shape": layer.input_shape,
                    "output_shape": layer.output_shape,
                    "params_count": layer.params_count,
                    "flops": layer.flops,
                    "activation_bytes": layer.activation_bytes,
                    "is_moe": layer.is_moe,
                }
                for layer in self._layers
            ]
        }

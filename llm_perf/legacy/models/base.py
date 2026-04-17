"""Base model classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .sharding import ShardingInfo, ShardedLayerConfig
    from ..strategy.base import StrategyConfig


class SubmoduleType(str, Enum):
    """Submodule type for transformer components.

    Note: NORM layers are now merged into their serving modules:
    - input_norm/pre_attn_norm -> ATTENTION
    - post_attn_norm/pre_ffn_norm/ffn_norm -> FFN/MOE
    - final_norm -> LM_HEAD
    NORM is kept for backward compatibility but not actively used.
    """

    EMBEDDING = "embedding"
    ATTENTION = "attention"
    FFN = "ffn"
    NORM = "norm"  # Deprecated: merged into serving modules
    LM_HEAD = "lm_head"
    MOE = "moe"
    OTHER = "other"


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
    submodule_type: SubmoduleType = SubmoduleType.OTHER  # Submodule type for transformer components
    sharding_info: Optional["ShardingInfo"] = None  # Sharding metadata for parallelism


@dataclass
class MemoryCalibrationConfig:
    """Memory calibration configuration for fine-tuning memory estimates.

    These factors account for various sources of memory overhead that are not
    captured by the basic calculation, such as:
    - Memory fragmentation
    - CUDA context overhead
    - Framework overhead (PyTorch/TensorFlow)
    - Communication buffers (for distributed training)
    - Extra workspace for kernels
    """

    # Fragmentation overhead (typically 10-20%)
    fragmentation_factor: float = 1.15  # 15% overhead

    # CUDA context and framework overhead (MB)
    cuda_context_mb: float = 500.0

    # Communication buffer overhead for distributed training
    # (scaled by number of GPUs and communication pattern)
    comm_buffer_factor: float = 1.05  # 5% overhead

    # Kernel workspace overhead (temporary buffers for ops like softmax, etc.)
    kernel_workspace_factor: float = 1.02  # 2% overhead

    # Extra safety margin for unexpected allocations
    safety_margin_factor: float = 1.05  # 5% margin

    def apply(self, base_memory_bytes: int, is_distributed: bool = False) -> int:
        """Apply calibration factors to base memory estimate.

        Args:
            base_memory_bytes: Base memory estimate without calibration
            is_distributed: Whether running in distributed mode

        Returns:
            Calibrated memory estimate in bytes
        """
        # Start with base memory
        memory = base_memory_bytes

        # Apply kernel workspace factor
        memory = int(memory * self.kernel_workspace_factor)

        # Apply communication buffer factor if distributed
        if is_distributed:
            memory = int(memory * self.comm_buffer_factor)

        # Apply fragmentation factor
        memory = int(memory * self.fragmentation_factor)

        # Add CUDA context overhead
        memory += int(self.cuda_context_mb * 1024 * 1024)

        # Apply safety margin
        memory = int(memory * self.safety_margin_factor)

        return memory


@dataclass
class ModelConfig:
    """Base model configuration."""

    # Required fields (no defaults)
    name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int

    # Optional fields (with defaults)
    num_key_value_heads: Optional[int] = None  # For GQA
    intermediate_size: int = 0  # FFN intermediate size
    max_seq_len: int = 4096
    dtype: str = "fp16"
    # MoE specific
    num_experts: Optional[int] = None
    num_experts_per_token: Optional[int] = None
    # Memory calibration config
    memory_calibration: MemoryCalibrationConfig = field(default_factory=MemoryCalibrationConfig)


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
        """Build and return layer configurations (original, unsharded)."""
        pass

    def build_sharded_layers(
        self,
        strategy: "StrategyConfig",
    ) -> List["ShardedLayerConfig"]:
        """Build and return sharded layer configurations.

        This method creates a view of the model after applying parallelism
        sharding. Each ShardedLayerConfig represents what a single GPU
        actually computes.

        Args:
            strategy: Parallelism strategy configuration

        Returns:
            List of ShardedLayerConfig, one per original layer, representing
            the computation on a single GPU rank

        Note:
            Subclasses should override this method to provide accurate sharding
            implementation. The default implementation provides a simple
            approximation that divides FLOPs by TP degree.
        """
        from .sharding import ShardedLayerConfig

        sharded_layers = []
        tp = strategy.tp_degree
        sp = strategy.sp_degree

        for idx, layer in enumerate(self._layers):
            # Simple approximation: divide by TP
            sharded_flops = layer.flops // max(tp, 1)
            sharded_params = layer.params_count // max(tp, 1)
            sharded_activation = layer.activation_bytes // max(tp, 1)

            # Adjust for SP if sequence-related
            if sp > 1 and "attention" in layer.name:
                sharded_flops = sharded_flops // max(sp, 1)
                sharded_activation = sharded_activation // max(sp, 1)

            sharded_layer = ShardedLayerConfig(
                name=layer.name,
                original_layer_idx=idx,
                sharded_input_shape=layer.input_shape,
                sharded_output_shape=layer.output_shape,
                sharded_flops=sharded_flops,
                sharded_params=sharded_params,
                sharded_activation_bytes=sharded_activation,
                sharding_info=layer.sharding_info,
            )
            sharded_layers.append(sharded_layer)

        return sharded_layers

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
        """Total activation memory in bytes.

        Warning: This sums all layer activations, which is correct for training
        but overestimates inference memory. Use estimate_memory() for accurate
        estimates based on training/inference mode.
        """
        return sum(layer.activation_bytes for layer in self._layers)

    def estimate_memory(
        self,
        inference_mode: bool = True,
        batch_size: int = 1,
        is_distributed: bool = False,
        apply_calibration: bool = True,
    ) -> int:
        """Estimate peak memory usage for the model.

        This method calculates memory usage considering layer lifecycle:
        - Training: All layer activations are saved for backward pass
        - Inference: Activations are computed layer-by-layer and reused

        Args:
            inference_mode: If True, use inference memory estimation (layer-wise
                activation reuse). If False, use training memory (all layers saved).
            batch_size: Batch size for the computation
            is_distributed: Whether running in distributed mode (affects comm buffers)
            apply_calibration: Whether to apply calibration factors

        Returns:
            Estimated peak memory in bytes
        """
        # Calculate parameter memory
        dtype_size = self._get_dtype_size()
        param_memory = self.total_params * dtype_size

        # Calculate activation memory based on mode
        if inference_mode:
            # Inference: find max single layer activation (layer-wise reuse)
            activation_memory = self._estimate_inference_activation_memory(batch_size)
        else:
            # Training: sum all layer activations (need for backward)
            activation_memory = self._estimate_training_activation_memory(batch_size)

        # Total base memory
        base_memory = param_memory + activation_memory

        # Apply calibration if requested
        if apply_calibration:
            calib = self.config.memory_calibration
            base_memory = calib.apply(base_memory, is_distributed)

        return base_memory

    def _get_dtype_size(self) -> int:
        """Get dtype size in bytes."""
        dtype_sizes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
        }
        return dtype_sizes.get(self.config.dtype, 2)

    def _estimate_inference_activation_memory(self, batch_size: int) -> int:
        """Estimate activation memory for inference mode.

        In inference, activations are computed layer-by-layer and can be reused.
        Peak memory is determined by the layer with maximum activation.

        Args:
            batch_size: Batch size for the computation

        Returns:
            Estimated activation memory in bytes
        """
        if not self._layers:
            return 0

        # Find max single layer activation
        max_layer_activation = max(layer.activation_bytes for layer in self._layers)

        # Scale by batch size (assuming linear scaling)
        # Note: This is a simplification; actual scaling depends on layer type
        max_layer_activation *= batch_size

        # Add residual stream (input that needs to be kept for residual connections)
        # This is typically the largest layer's input
        residual_stream = self._estimate_residual_stream(batch_size)

        return max_layer_activation + residual_stream

    def _estimate_training_activation_memory(self, batch_size: int) -> int:
        """Estimate activation memory for training mode.

        In training, all layer activations must be saved for backward pass
        (unless activation checkpointing is used).

        Args:
            batch_size: Batch size for the computation

        Returns:
            Estimated activation memory in bytes
        """
        if not self._layers:
            return 0

        # Sum all layer activations
        total_activation = sum(layer.activation_bytes for layer in self._layers)

        # Scale by batch size
        total_activation *= batch_size

        return total_activation

    def _estimate_residual_stream(self, batch_size: int) -> int:
        """Estimate residual stream memory.

        This is the memory needed to keep the input tensor for residual connections.

        Args:
            batch_size: Batch size

        Returns:
            Residual stream memory in bytes
        """
        # Default: use first layer's input shape
        # Subclasses can override for more accurate estimates
        if not self._layers:
            return 0

        first_layer = self._layers[0]
        input_shape = first_layer.input_shape

        # Calculate bytes from shape
        # Shape is typically (batch, seq_len, hidden) or similar
        num_elements = 1
        for dim in input_shape:
            if isinstance(dim, int) and dim > 0:
                num_elements *= dim

        # Scale by actual batch size
        if batch_size != 1 and input_shape and input_shape[0] == 1:
            num_elements *= batch_size

        dtype_size = self._get_dtype_size()
        return num_elements * dtype_size

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
            "estimated_memory": {
                "inference_bytes": self.estimate_memory(inference_mode=True),
                "training_bytes": self.estimate_memory(inference_mode=False),
                "inference_gb": self.estimate_memory(inference_mode=True) / 1024**3,
                "training_gb": self.estimate_memory(inference_mode=False) / 1024**3,
            },
            "layers": [
                {
                    "name": layer.name,
                    "input_shape": layer.input_shape,
                    "output_shape": layer.output_shape,
                    "params_count": layer.params_count,
                    "flops": layer.flops,
                    "activation_bytes": layer.activation_bytes,
                    "is_moe": layer.is_moe,
                    "submodule_type": layer.submodule_type.value,
                }
                for layer in self._layers
            ],
        }

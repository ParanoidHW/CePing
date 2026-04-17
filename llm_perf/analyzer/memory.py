"""Memory estimation utilities."""

from typing import Tuple

from llm_perf.legacy.models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig
from ..utils.constants import DTYPE_SIZES


class MemoryEstimator:
    """Estimates memory requirements for training and inference."""

    def __init__(
        self,
        model: BaseModel,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        self.model = model
        self.device = device
        self.cluster = cluster
        self.strategy = strategy

    def estimate_training_memory(self, batch_size: int, seq_len: int) -> int:
        """Estimate memory per GPU for training.

        Accounts for:
        - TP: shards QKV/FFN weights and activations
        - PP: distributes layers
        - SP: shards sequence dimension
        - EP: shards expert parameters (for MoE)
        - ZeRO: shards optimizer states and/or gradients

        Returns:
            Estimated memory per GPU in bytes
        """
        effective_seq_len = self._get_effective_seq_len(seq_len)
        effective_num_layers = self._get_effective_num_layers()

        param_memory = self._estimate_param_memory()
        grad_memory = self._estimate_grad_memory(param_memory)
        optimizer_memory = self._estimate_optimizer_memory(param_memory)
        activation_memory = self._estimate_activation_memory_training(
            batch_size, seq_len, effective_seq_len, effective_num_layers
        )
        kv_cache_memory = self._estimate_kv_cache_memory(batch_size, effective_seq_len)

        total_memory = param_memory + activation_memory + grad_memory + optimizer_memory + kv_cache_memory

        calib = self.model.config.memory_calibration
        is_distributed = self._is_distributed()
        return calib.apply(total_memory, is_distributed)

    def estimate_inference_memory(self, batch_size: int, max_seq_len: int) -> Tuple[int, int]:
        """Estimate memory per GPU for inference.

        Returns:
            Tuple of (total_memory, kv_cache_memory) in bytes
        """
        param_memory = self._estimate_param_memory_inference()
        kv_cache_memory = self._estimate_kv_cache_memory_inference(batch_size, max_seq_len)
        activation_memory = self._estimate_activation_memory_inference(batch_size)

        total_memory = param_memory + kv_cache_memory + activation_memory

        calib = self.model.config.memory_calibration
        is_distributed = self._is_distributed()
        return calib.apply(total_memory, is_distributed), kv_cache_memory

    def _estimate_param_memory(self) -> int:
        """Estimate parameter memory per GPU (training)."""
        dtype_size = DTYPE_SIZES.get(self.model.config.dtype, 2)
        total_params = self.model.total_params
        param_memory = total_params * dtype_size

        if self.strategy.tp_degree > 1:
            param_memory = param_memory // self.strategy.tp_degree

        if self.strategy.pp_degree > 1:
            param_memory = param_memory // self.strategy.pp_degree

        return param_memory

    def _estimate_param_memory_inference(self) -> int:
        """Estimate parameter memory per GPU (inference)."""
        dtype_size = DTYPE_SIZES.get(self.model.config.dtype, 2)
        param_memory = self.model.total_params * dtype_size

        if self.strategy.tp_degree > 1:
            param_memory //= self.strategy.tp_degree

        if self.strategy.pp_degree > 1:
            param_memory //= self.strategy.pp_degree

        return param_memory

    def _estimate_grad_memory(self, param_memory: int) -> int:
        """Estimate gradient memory per GPU."""
        grad_memory = param_memory

        if self.strategy.zero_stage >= 2:
            grad_memory = grad_memory // self.strategy.dp_degree

        return grad_memory

    def _estimate_optimizer_memory(self, param_memory: int) -> int:
        """Estimate optimizer state memory (Adam: momentum + variance in fp32)."""
        optimizer_memory = param_memory * 2 * 4

        if self.strategy.zero_stage >= 1:
            optimizer_memory = optimizer_memory // self.strategy.dp_degree

        return optimizer_memory

    def _estimate_activation_memory_training(
        self,
        batch_size: int,
        seq_len: int,
        effective_seq_len: int,
        effective_num_layers: int,
    ) -> int:
        """Estimate activation memory for training."""
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        is_distributed = self._is_distributed()

        base_memory = self.model.estimate_memory(
            inference_mode=False,
            batch_size=batch_size,
            is_distributed=is_distributed,
            apply_calibration=False,
        )

        param_memory_total = self.model.total_params * dtype_size
        activation_memory = base_memory - param_memory_total

        seq_ratio = seq_len // max(self.model.config.max_seq_len, 1)
        if seq_ratio > 0:
            activation_memory *= seq_ratio

        if self.strategy.tp_degree > 1:
            activation_memory = activation_memory // self.strategy.tp_degree

        if self.strategy.sp_degree > 1:
            activation_memory = activation_memory // self.strategy.sp_degree

        if self.strategy.pp_degree > 1:
            activation_memory = activation_memory // self.strategy.pp_degree

        if self.strategy.activation_checkpointing:
            max_layer_activation = (
                max(layer.activation_bytes for layer in self.model.layers[:effective_num_layers])
                if self.model.layers
                else 0
            )

            max_layer_activation *= batch_size * effective_seq_len // max(self.model.config.max_seq_len, 1)

            if self.strategy.tp_degree > 1:
                max_layer_activation = max_layer_activation // self.strategy.tp_degree

            activation_memory = max_layer_activation

        return activation_memory

    def _estimate_activation_memory_inference(self, batch_size: int) -> int:
        """Estimate activation memory for inference."""
        dtype_size = DTYPE_SIZES.get(self.model.config.dtype, 2)

        is_distributed = self._is_distributed()
        base_memory = self.model.estimate_memory(
            inference_mode=True,
            batch_size=batch_size,
            is_distributed=is_distributed,
            apply_calibration=False,
        )

        param_memory_total = self.model.total_params * dtype_size
        activation_memory = base_memory - param_memory_total

        if self.strategy.tp_degree > 1:
            activation_memory //= self.strategy.tp_degree

        seq_parallel_degree = max(self.strategy.sp_degree, self.strategy.cp_degree)
        if seq_parallel_degree > 1:
            activation_memory //= seq_parallel_degree

        if self.strategy.pp_degree > 1:
            activation_memory //= self.strategy.pp_degree

        return activation_memory

    def _estimate_kv_cache_memory(self, batch_size: int, effective_seq_len: int) -> int:
        """Estimate KV cache memory for training."""
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // self.model.config.num_attention_heads

        num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)

        kv_cache_memory = batch_size * effective_seq_len * num_kv_heads * head_dim * 2 * dtype_size

        if self.strategy.tp_degree > 1:
            effective_kv_heads = self._get_effective_num_kv_heads()
            kv_cache_memory = batch_size * effective_seq_len * effective_kv_heads * head_dim * 2 * dtype_size

        return kv_cache_memory

    def _estimate_kv_cache_memory_inference(self, batch_size: int, max_seq_len: int) -> int:
        """Estimate KV cache memory for inference."""
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // self.model.config.num_attention_heads

        _, effective_num_kv_heads = self._get_effective_num_heads()
        effective_num_layers = self._get_effective_num_layers()

        kv_per_token = 2 * effective_num_kv_heads * head_dim * dtype_size
        kv_cache_memory = batch_size * max_seq_len * effective_num_layers * kv_per_token

        return kv_cache_memory

    def _get_effective_seq_len(self, seq_len: int) -> int:
        """Get effective sequence length after SP/CP sharding."""
        seq_parallel_degree = max(self.strategy.sp_degree, self.strategy.cp_degree)
        if seq_parallel_degree > 1:
            return max(1, seq_len // seq_parallel_degree)
        return seq_len

    def _get_effective_num_layers(self) -> int:
        """Get effective number of layers after PP sharding."""
        num_layers = self.model.config.num_layers
        if self.strategy.pp_degree > 1:
            return num_layers // self.strategy.pp_degree
        return num_layers

    def _get_effective_num_heads(self) -> Tuple[int, int]:
        """Get effective attention heads after TP sharding."""
        tp_degree = self.strategy.tp_degree
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = self.model.config.num_key_value_heads or num_heads

        if tp_degree > 1:
            effective_num_heads = max(1, num_heads // tp_degree)
            effective_num_kv_heads = max(1, num_kv_heads // tp_degree)
            return effective_num_heads, effective_num_kv_heads
        return num_heads, num_kv_heads

    def _get_effective_num_kv_heads(self) -> int:
        """Get effective number of KV heads after TP sharding."""
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        if self.strategy.tp_degree > 1:
            return max(1, num_kv_heads // self.strategy.tp_degree)
        return num_kv_heads

    def _is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return (
            self.strategy.tp_degree > 1
            or self.strategy.dp_degree > 1
            or self.strategy.pp_degree > 1
            or self.strategy.sp_degree > 1
        )

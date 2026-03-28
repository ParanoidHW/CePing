"""Llama model implementation."""

from dataclasses import dataclass
from typing import List

from .base import BaseModel, ModelConfig, LayerConfig
from ..utils.constants import DTYPE_SIZES


@dataclass
class LlamaConfig(ModelConfig):
    """Llama-specific configuration with sensible defaults."""
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.intermediate_size == 0:
            # Default Llama ratio (about 2.67x)
            self.intermediate_size = int(self.hidden_size * 8 / 3)


class LlamaModel(BaseModel):
    """Llama model implementation."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """Build Llama layer configurations."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        
        # Embedding layer
        embed_params = cfg.vocab_size * cfg.hidden_size
        layers.append(LayerConfig(
            name="embedding",
            input_shape=(1, 1),  # (batch, seq_len) - token ids
            output_shape=(1, 1, cfg.hidden_size),
            params_count=embed_params,
            flops=0,  # Lookup table, negligible compute
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # Build each transformer layer
        for i in range(cfg.num_layers):
            layers.extend(self._build_transformer_layer(i, dtype_size))
        
        # Final norm and LM head
        layers.append(LayerConfig(
            name="final_norm",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=cfg.hidden_size,  # weight only
            flops=cfg.hidden_size * 5,  # rmsnorm: square, mean, add, rsqrt, mul
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        layers.append(LayerConfig(
            name="lm_head",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.vocab_size),
            params_count=cfg.hidden_size * cfg.vocab_size,
            flops=cfg.hidden_size * cfg.vocab_size * 2,
            activation_bytes=cfg.vocab_size * dtype_size,
        ))
        
        return layers
    
    def _build_transformer_layer(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a single transformer layer (attention + FFN)."""
        layers = []
        cfg = self.config
        prefix = f"layer_{layer_idx}"
        
        # === Attention Components ===
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        
        # Q, K, V projections
        q_params = cfg.hidden_size * (q_heads * head_dim)
        kv_params = cfg.hidden_size * (kv_heads * head_dim)
        
        layers.append(LayerConfig(
            name=f"{prefix}_q_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, q_heads * head_dim),
            params_count=q_params,
            flops=q_params * 2,
            activation_bytes=q_heads * head_dim * dtype_size,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_k_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, kv_heads * head_dim),
            params_count=kv_params,
            flops=kv_params * 2,
            activation_bytes=kv_heads * head_dim * dtype_size,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_v_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, kv_heads * head_dim),
            params_count=kv_params,
            flops=kv_params * 2,
            activation_bytes=kv_heads * head_dim * dtype_size,
        ))
        
        # Attention computation (Q*K^T, softmax, *V)
        # Simplified estimation
        seq_len = cfg.max_seq_len
        attn_flops = (
            seq_len * seq_len * q_heads * head_dim * 2 +  # QK^T
            seq_len * seq_len * q_heads * 5 +              # softmax (exp, sum, div)
            seq_len * seq_len * q_heads * head_dim * 2     # *V
        )
        
        layers.append(LayerConfig(
            name=f"{prefix}_attention",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=0,  # No params, just computation
            flops=attn_flops,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
        ))
        
        # O projection
        o_params = cfg.hidden_size * cfg.hidden_size
        layers.append(LayerConfig(
            name=f"{prefix}_o_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=o_params,
            flops=o_params * 2,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # Attention output norm/residual (simplified)
        layers.append(LayerConfig(
            name=f"{prefix}_attn_norm",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=cfg.hidden_size,
            flops=cfg.hidden_size * 5,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # === FFN Components ===
        # Llama uses SwiGLU: up_proj, gate_proj, down_proj
        ffn_intermediate = cfg.intermediate_size
        
        layers.append(LayerConfig(
            name=f"{prefix}_up_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, ffn_intermediate),
            params_count=cfg.hidden_size * ffn_intermediate,
            flops=cfg.hidden_size * ffn_intermediate * 2,
            activation_bytes=ffn_intermediate * dtype_size,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_gate_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, ffn_intermediate),
            params_count=cfg.hidden_size * ffn_intermediate,
            flops=cfg.hidden_size * ffn_intermediate * 2,
            activation_bytes=ffn_intermediate * dtype_size,
        ))
        
        # SwiGLU activation (multiply + swish)
        layers.append(LayerConfig(
            name=f"{prefix}_swiglu",
            input_shape=(1, 1, ffn_intermediate),
            output_shape=(1, 1, ffn_intermediate),
            params_count=0,
            flops=ffn_intermediate * 8,  # sigmoid, mul, mul
            activation_bytes=ffn_intermediate * dtype_size,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_down_proj",
            input_shape=(1, 1, ffn_intermediate),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=ffn_intermediate * cfg.hidden_size,
            flops=ffn_intermediate * cfg.hidden_size * 2,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # FFN norm/residual
        layers.append(LayerConfig(
            name=f"{prefix}_ffn_norm",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=cfg.hidden_size,
            flops=cfg.hidden_size * 5,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        return layers

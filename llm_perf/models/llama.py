"""Llama model implementation using kernel API."""

from dataclasses import dataclass
from typing import List

from .base import BaseModel, ModelConfig, LayerConfig
from ..utils.constants import DTYPE_SIZES
from ..kernels import linear, rms_norm, silu, scaled_dot_product_attention, embedding
from ..kernels.utils import kernel_result_to_layer


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
    """Llama model implementation using kernel API."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """Build Llama layer configurations using kernel API."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        
        # Embedding layer using embedding kernel
        emb_result = embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.hidden_size,
            input_shape=(1, cfg.max_seq_len),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="embedding",
            result=emb_result))
        
        # Build each transformer layer
        for i in range(cfg.num_layers):
            layers.extend(self._build_transformer_layer(i, dtype_size))
        
        # Final norm using rms_norm kernel
        final_norm_result = rms_norm(
            input=(1, cfg.max_seq_len, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="final_norm",
            result=final_norm_result))
        
        # LM head using linear kernel
        lm_head_result = linear(
            input=(cfg.max_seq_len, cfg.hidden_size),
            weight=(cfg.vocab_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="lm_head",
            result=lm_head_result))
        
        return layers
    
    def _build_transformer_layer(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a single transformer layer (attention + FFN) using kernel API."""
        layers = []
        cfg = self.config
        prefix = f"layer_{layer_idx}"
        seq_len = cfg.max_seq_len
        m = seq_len  # Flattened batch*seq for linear ops
        
        # === Attention Components ===
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        
        # Pre-attention RMSNorm
        input_norm_result = rms_norm(
            input=(1, seq_len, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_input_norm",
            result=input_norm_result))
        
        # Q projection using linear kernel
        q_result = linear(
            input=(m, cfg.hidden_size),
            weight=(q_heads * head_dim, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_q_proj",
            result=q_result))
        
        # K projection using linear kernel
        k_result = linear(
            input=(m, cfg.hidden_size),
            weight=(kv_heads * head_dim, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_k_proj",
            result=k_result))
        
        # V projection using linear kernel
        v_result = linear(
            input=(m, cfg.hidden_size),
            weight=(kv_heads * head_dim, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_v_proj",
            result=v_result))
        
        # Attention computation using scaled_dot_product_attention kernel
        attn_result = scaled_dot_product_attention(
            query=(1, q_heads, seq_len, head_dim),
            key=(1, kv_heads, seq_len, head_dim),
            value=(1, kv_heads, seq_len, head_dim),
            is_causal=True,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_attention",
            result=attn_result))
        
        # O projection using linear kernel
        o_result = linear(
            input=(m, cfg.hidden_size),
            weight=(cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_o_proj",
            result=o_result))
        
        # Post-attention RMSNorm
        attn_norm_result = rms_norm(
            input=(1, seq_len, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_post_attn_norm",
            result=attn_norm_result))
        
        # === FFN Components ===
        # Llama uses SwiGLU: up_proj, gate_proj, down_proj
        ffn_intermediate = cfg.intermediate_size
        
        # Up projection using linear kernel
        up_result = linear(
            input=(m, cfg.hidden_size),
            weight=(ffn_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_up_proj",
            result=up_result))
        
        # Gate projection using linear kernel
        gate_result = linear(
            input=(m, cfg.hidden_size),
            weight=(ffn_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_gate_proj",
            result=gate_result))
        
        # SwiGLU activation using silu kernel (x * sigmoid(x))
        swiglu_result = silu(
            input=(1, seq_len, ffn_intermediate),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_swiglu",
            result=swiglu_result))
        
        # Down projection using linear kernel
        down_result = linear(
            input=(m, ffn_intermediate),
            weight=(cfg.hidden_size, ffn_intermediate),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_down_proj",
            result=down_result))
        
        # Post-FFN RMSNorm
        ffn_norm_result = rms_norm(
            input=(1, seq_len, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_ffn_norm",
            result=ffn_norm_result))
        
        return layers

"""Mixture of Experts (MoE) model implementation."""

from dataclasses import dataclass
from typing import List

from .base import BaseModel, LayerConfig
from .llama import LlamaConfig
from ..utils.constants import DTYPE_SIZES


@dataclass
class MoEConfig(LlamaConfig):
    """MoE-specific configuration."""
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_intermediate_size: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if self.expert_intermediate_size == 0:
            self.expert_intermediate_size = self.intermediate_size


class MoEModel(BaseModel):
    """MoE model implementation with EP (Expert Parallelism) support."""
    
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """Build MoE layer configurations."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        
        # Embedding layer (same as Llama)
        embed_params = cfg.vocab_size * cfg.hidden_size
        layers.append(LayerConfig(
            name="embedding",
            input_shape=(1, 1),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=embed_params,
            flops=0,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # Build each transformer layer (alternating dense and MoE)
        for i in range(cfg.num_layers):
            # Every 4th layer is MoE (typical configuration)
            if i % 4 == 3 and cfg.num_experts > 0:
                layers.extend(self._build_moe_layer(i, dtype_size))
            else:
                layers.extend(self._build_dense_layer(i, dtype_size))
        
        # Final norm and LM head
        layers.append(LayerConfig(
            name="final_norm",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=cfg.hidden_size,
            flops=cfg.hidden_size * 5,
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
    
    def _build_dense_layer(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a standard dense transformer layer."""
        from .llama import LlamaModel, LlamaConfig
        
        # Reuse Llama layer building logic
        llama_cfg = LlamaConfig(
            name="llama_dense",
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            intermediate_size=self.config.intermediate_size,
            dtype=self.config.dtype,
        )
        llama = LlamaModel(llama_cfg)
        layers = llama._build_transformer_layer(layer_idx, dtype_size)
        
        # Mark as non-MoE
        for layer in layers:
            layer.is_moe = False
        
        return layers
    
    def _build_moe_layer(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build an MoE transformer layer."""
        layers = []
        cfg = self.config
        prefix = f"layer_{layer_idx}"
        
        # === Attention (same as dense) ===
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
            is_moe=False,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_k_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, kv_heads * head_dim),
            params_count=kv_params,
            flops=kv_params * 2,
            activation_bytes=kv_heads * head_dim * dtype_size,
            is_moe=False,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_v_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, kv_heads * head_dim),
            params_count=kv_params,
            flops=kv_params * 2,
            activation_bytes=kv_heads * head_dim * dtype_size,
            is_moe=False,
        ))
        
        seq_len = cfg.max_seq_len
        attn_flops = (
            seq_len * seq_len * q_heads * head_dim * 2 +
            seq_len * seq_len * q_heads * 5 +
            seq_len * seq_len * q_heads * head_dim * 2
        )
        
        layers.append(LayerConfig(
            name=f"{prefix}_attention",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=0,
            flops=attn_flops,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
            is_moe=False,
        ))
        
        o_params = cfg.hidden_size * cfg.hidden_size
        layers.append(LayerConfig(
            name=f"{prefix}_o_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=o_params,
            flops=o_params * 2,
            activation_bytes=cfg.hidden_size * dtype_size,
            is_moe=False,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_attn_norm",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=cfg.hidden_size,
            flops=cfg.hidden_size * 5,
            activation_bytes=cfg.hidden_size * dtype_size,
            is_moe=False,
        ))
        
        # === MoE Components ===
        # Router / Gate
        router_params = cfg.hidden_size * cfg.num_experts
        layers.append(LayerConfig(
            name=f"{prefix}_router",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.num_experts),
            params_count=router_params,
            flops=router_params * 2 + cfg.num_experts * 10,  # matmul + softmax
            activation_bytes=cfg.num_experts * dtype_size,
            is_moe=True,
        ))
        
        # Each expert's FFN (only active ones computed per token)
        expert_scale = cfg.num_experts_per_token / cfg.num_experts
        ffn_intermediate = cfg.expert_intermediate_size
        
        # Up proj (per active expert)
        expert_up_params = cfg.hidden_size * ffn_intermediate
        layers.append(LayerConfig(
            name=f"{prefix}_expert_up",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, ffn_intermediate),
            params_count=expert_up_params * cfg.num_experts,  # Total across all experts
            flops=int(expert_up_params * 2 * cfg.num_experts_per_token),
            activation_bytes=int(ffn_intermediate * dtype_size * cfg.num_experts_per_token),
            is_moe=True,
        ))
        
        # Gate proj
        expert_gate_params = cfg.hidden_size * ffn_intermediate
        layers.append(LayerConfig(
            name=f"{prefix}_expert_gate",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, ffn_intermediate),
            params_count=expert_gate_params * cfg.num_experts,
            flops=int(expert_gate_params * 2 * cfg.num_experts_per_token),
            activation_bytes=int(ffn_intermediate * dtype_size * cfg.num_experts_per_token),
            is_moe=True,
        ))
        
        # SwiGLU
        layers.append(LayerConfig(
            name=f"{prefix}_expert_swiglu",
            input_shape=(1, 1, ffn_intermediate),
            output_shape=(1, 1, ffn_intermediate),
            params_count=0,
            flops=int(ffn_intermediate * 8 * cfg.num_experts_per_token),
            activation_bytes=int(ffn_intermediate * dtype_size * cfg.num_experts_per_token),
            is_moe=True,
        ))
        
        # Down proj
        expert_down_params = ffn_intermediate * cfg.hidden_size
        layers.append(LayerConfig(
            name=f"{prefix}_expert_down",
            input_shape=(1, 1, ffn_intermediate),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=expert_down_params * cfg.num_experts,
            flops=int(expert_down_params * 2 * cfg.num_experts_per_token),
            activation_bytes=cfg.hidden_size * dtype_size,
            is_moe=True,
        ))
        
        # All-to-all communication (dispatch and combine)
        layers.append(LayerConfig(
            name=f"{prefix}_alltoall_dispatch",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=0,
            flops=0,
            activation_bytes=cfg.hidden_size * dtype_size * cfg.num_experts_per_token,
            is_moe=True,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_alltoall_combine",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=0,
            flops=0,
            activation_bytes=cfg.hidden_size * dtype_size,
            is_moe=True,
        ))
        
        # MoE norm/residual
        layers.append(LayerConfig(
            name=f"{prefix}_moe_norm",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=cfg.hidden_size,
            flops=cfg.hidden_size * 5,
            activation_bytes=cfg.hidden_size * dtype_size,
            is_moe=True,
        ))
        
        return layers

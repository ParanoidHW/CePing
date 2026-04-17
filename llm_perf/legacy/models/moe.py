"""Mixture of Experts (MoE) model implementation using kernel API."""

from dataclasses import dataclass
from typing import List

from .base import BaseModel, LayerConfig, SubmoduleType
from .llama import LlamaConfig
from llm_perf.utils.constants import DTYPE_SIZES
from llm_perf.kernels import linear, rms_norm, silu, scaled_dot_product_attention, embedding, softmax
from llm_perf.kernels.utils import kernel_result_to_layer


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
    """MoE model implementation with EP (Expert Parallelism) support using kernel API."""
    
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """Build MoE layer configurations using kernel API."""
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        seq_len = cfg.max_seq_len
        
        # Embedding layer using embedding kernel
        emb_result = embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.hidden_size,
            input_shape=(1, seq_len),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="embedding",
            result=emb_result,
            submodule_type=SubmoduleType.EMBEDDING))
        
        # Build each transformer layer (alternating dense and MoE)
        for i in range(cfg.num_layers):
            # Every 4th layer is MoE (typical configuration)
            if i % 4 == 3 and cfg.num_experts > 0:
                layers.extend(self._build_moe_layer(i, dtype_size))
            else:
                layers.extend(self._build_dense_layer(i, dtype_size))
        
        # Final norm using rms_norm kernel (merged into LM_HEAD)
        final_norm_result = rms_norm(
            input=(1, seq_len, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="final_norm",
            result=final_norm_result,
            submodule_type=SubmoduleType.LM_HEAD))
        
        # LM head using linear kernel
        lm_head_result = linear(
            input=(seq_len, cfg.hidden_size),
            weight=(cfg.vocab_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="lm_head",
            result=lm_head_result,
            submodule_type=SubmoduleType.LM_HEAD))
        
        return layers
    
    def _build_dense_layer(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a standard dense transformer layer using Llama's kernel-based implementation."""
        from .llama import LlamaModel, LlamaConfig
        
        # Reuse Llama layer building logic (now kernel-based)
        llama_cfg = LlamaConfig(
            name="llama_dense",
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            intermediate_size=self.config.intermediate_size,
            max_seq_len=self.config.max_seq_len,
            dtype=self.config.dtype,
        )
        llama = LlamaModel(llama_cfg)
        layers = llama._build_transformer_layer(layer_idx, dtype_size)
        
        # Mark as non-MoE
        for layer in layers:
            layer.is_moe = False
        
        return layers
    
    def _build_moe_layer(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build an MoE transformer layer using kernel API."""
        layers = []
        cfg = self.config
        prefix = f"layer_{layer_idx}"
        seq_len = cfg.max_seq_len
        m = seq_len
        
        # === Attention (same as dense) ===
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        
        # Pre-attention RMSNorm (merged into ATTENTION)
        input_norm_result = rms_norm(
            input=(1, seq_len, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_input_norm",
            result=input_norm_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # Q, K, V projections using linear kernel
        for proj_name, out_dim, is_q in [
            ("q_proj", q_heads * head_dim, True),
            ("k_proj", kv_heads * head_dim, False),
            ("v_proj", kv_heads * head_dim, False)
        ]:
            proj_result = linear(
                input=(m, cfg.hidden_size),
                weight=(out_dim, cfg.hidden_size),
                bias=None,
                dtype=cfg.dtype
            )
            layers.append(kernel_result_to_layer(
                name=f"{prefix}_{proj_name}",
                result=proj_result,
                submodule_type=SubmoduleType.ATTENTION))
        
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
            result=attn_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # O projection using linear kernel
        o_result = linear(
            input=(m, cfg.hidden_size),
            weight=(cfg.hidden_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_o_proj",
            result=o_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # Post-attention RMSNorm (merged into MOE)
        attn_norm_result = rms_norm(
            input=(1, seq_len, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_post_attn_norm",
            result=attn_norm_result,
            submodule_type=SubmoduleType.MOE))
        
        # === MoE Components ===
        # Router / Gate using linear kernel
        router_linear_result = linear(
            input=(m, cfg.hidden_size),
            weight=(cfg.num_experts, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        # Softmax for routing
        router_softmax_result = softmax(
            input=(1, seq_len, cfg.num_experts),
            dim=-1,
            dtype=cfg.dtype
        )
        
        router_layer = kernel_result_to_layer(
            name=f"{prefix}_router",
            result=router_linear_result,
            is_moe=True,
            submodule_type=SubmoduleType.MOE)
        router_layer.flops = router_linear_result.flops + router_softmax_result.flops
        layers.append(router_layer)
        
        # Each expert's FFN (only active ones computed per token)
        ffn_intermediate = cfg.expert_intermediate_size
        
        # Up proj (per active expert) using linear kernel
        expert_up_result = linear(
            input=(m, cfg.hidden_size),
            weight=(ffn_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        expert_up_layer = kernel_result_to_layer(
            name=f"{prefix}_expert_up",
            result=expert_up_result,
            submodule_type=SubmoduleType.MOE)
        expert_up_layer.flops = int(expert_up_result.flops * cfg.num_experts_per_token)
        layers.append(expert_up_layer)
        
        # Gate proj using linear kernel
        expert_gate_result = linear(
            input=(m, cfg.hidden_size),
            weight=(ffn_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        expert_gate_layer = kernel_result_to_layer(
            name=f"{prefix}_expert_gate",
            result=expert_gate_result,
            submodule_type=SubmoduleType.MOE)
        expert_gate_layer.flops = int(expert_gate_result.flops * cfg.num_experts_per_token)
        layers.append(expert_gate_layer)
        
        # SwiGLU using silu kernel
        swiglu_result = silu(
            input=(1, seq_len, ffn_intermediate),
            dtype=cfg.dtype
        )
        swiglu_layer = kernel_result_to_layer(
            name=f"{prefix}_expert_swiglu",
            result=swiglu_result,
            submodule_type=SubmoduleType.MOE)
        swiglu_layer.flops = int(swiglu_result.flops * cfg.num_experts_per_token)
        layers.append(swiglu_layer)
        
        # Down proj using linear kernel
        expert_down_result = linear(
            input=(m, ffn_intermediate),
            weight=(cfg.hidden_size, ffn_intermediate),
            bias=None,
            dtype=cfg.dtype
        )
        expert_down_layer = kernel_result_to_layer(
            name=f"{prefix}_expert_down",
            result=expert_down_result,
            submodule_type=SubmoduleType.MOE)
        expert_down_layer.flops = int(expert_down_result.flops * cfg.num_experts_per_token)
        layers.append(expert_down_layer)
        
        # All-to-all communication (dispatch and combine)
        # NOTE: Manual calculation for communication layer (alltoall)
        layers.append(LayerConfig(
            name=f"{prefix}_alltoall_dispatch",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=0,
            flops=0,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size * cfg.num_experts_per_token,
            is_moe=True,
            submodule_type=SubmoduleType.MOE,
        ))
        
        # NOTE: Manual calculation for communication layer (alltoall)
        layers.append(LayerConfig(
            name=f"{prefix}_alltoall_combine",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=0,
            flops=0,
            activation_bytes=seq_len * cfg.hidden_size * dtype_size,
            is_moe=True,
            submodule_type=SubmoduleType.MOE,
        ))
        
        # MoE norm/residual using rms_norm kernel (merged into MOE)
        moe_norm_result = rms_norm(
            input=(1, seq_len, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        moe_norm_layer = kernel_result_to_layer(
            name=f"{prefix}_moe_norm",
            result=moe_norm_result,
            submodule_type=SubmoduleType.MOE)
        moe_norm_layer.is_moe = True
        layers.append(moe_norm_layer)
        
        return layers

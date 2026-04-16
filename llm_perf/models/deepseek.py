"""DeepSeek V2/V3 model implementation with MLA (Multi-head Latent Attention) using kernel API.

DeepSeek-V2 and DeepSeek-V3 use Multi-head Latent Attention (MLA) which
compresses the Key-Value cache into a latent vector, significantly reducing
memory consumption during inference.

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V2
           https://huggingface.co/deepseek-ai/DeepSeek-V3
"""

from dataclasses import dataclass
from typing import List

from .base import BaseModel, ModelConfig, LayerConfig, SubmoduleType
from ..utils.constants import DTYPE_SIZES
from ..kernels import (
    linear, rms_norm, silu, scaled_dot_product_attention,
    embedding
)
from ..kernels.utils import kernel_result_to_layer


@dataclass
class DeepSeekConfig(ModelConfig):
    """DeepSeek V2/V3 configuration with MLA support.
    
    Based on official HuggingFace config.json:
    - DeepSeek-V2: https://huggingface.co/deepseek-ai/DeepSeek-V2
    - DeepSeek-V3: https://huggingface.co/deepseek-ai/DeepSeek-V3
    
    Args:
        kv_lora_rank: KV compression dimension (latent vector size)
        q_lora_rank: Query compression dimension (0 means no compression)
        qk_nope_head_dim: Dimension of non-positional query/key per head
        qk_rope_head_dim: Dimension of RoPE-encoded query/key per head
        v_head_dim: Value head dimension
        moe_intermediate_size: Intermediate size for MoE experts
        n_routed_experts: Number of routed experts in MoE
        n_shared_experts: Number of shared experts in MoE
        first_k_dense_replace: First N layers use dense FFN instead of MoE
        n_group: Number of expert groups for routing
        topk_group: Number of groups to select experts from
        topk_method: Expert routing method (e.g., "group_limited_greedy")
        routed_scaling_factor: Scaling factor for routed experts
        aux_loss_alpha: Auxiliary loss coefficient for load balancing
        seq_aux: Whether to use sequence-level auxiliary loss
        norm_topk_prob: Whether to normalize top-k probabilities
        rope_theta: RoPE base frequency
    """
    # MLA-specific parameters
    kv_lora_rank: int = 512  # KV cache compression dimension
    q_lora_rank: int = 1536  # Query compression dimension (0 = no compression)
    qk_nope_head_dim: int = 128  # Non-RoPE head dim for Q/K
    qk_rope_head_dim: int = 64   # RoPE head dim for Q/K
    v_head_dim: int = 128  # Value head dimension
    
    # MoE-specific parameters
    moe_intermediate_size: int = 1536
    n_routed_experts: int = 160
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    
    # MoE layer configuration
    first_k_dense_replace: int = 1  # First N layers use dense FFN
    n_group: int = 8  # Number of expert groups
    topk_group: int = 3  # Top-k groups to route from
    topk_method: str = "group_limited_greedy"
    routed_scaling_factor: float = 16.0
    
    # Training stability
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    norm_topk_prob: bool = False
    
    # RoPE configuration
    rope_theta: float = 10000.0
    max_position_embeddings: int = 163840
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.intermediate_size == 0:
            # DeepSeek uses ~2.4x ratio
            self.intermediate_size = int(self.hidden_size * 12 / 5)


@dataclass
class DeepSeekV3Config(DeepSeekConfig):
    """DeepSeek-V3 specific configuration.
    
    DeepSeek-V3 increases model size and expert count compared to V2:
    - hidden_size: 5120 -> 7168
    - num_layers: 60 -> 61
    - n_routed_experts: 160 -> 256
    - vocab_size: 102400 -> 129280
    
    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3
    """
    # V3 uses larger hidden dimension
    hidden_size: int = 7168
    num_layers: int = 61
    num_attention_heads: int = 128
    intermediate_size: int = 18432
    vocab_size: int = 129280
    
    # V3 increases expert count
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048  # Estimated from V3 config
    
    # MLA dimensions (V3 specific)
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    
    # V3 uses larger context window capability
    max_position_embeddings: int = 163840
    max_seq_len: int = 163840


class DeepSeekModel(BaseModel):
    """DeepSeek V2/V3 model implementation with MLA using kernel API.
    
    Implements Multi-head Latent Attention (MLA) which compresses the KV cache
    through low-rank projection matrices:
    
        latent_kv = W_DK · concat([q; k]) · W_DV
    
    This reduces inference memory from O(seq_len × num_heads × head_dim) to
    O(seq_len × kv_lora_rank), where kv_lora_rank << num_heads × head_dim.
    
    The model also uses DeepSeekMoE architecture for efficient training.
    """
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__(config)
        self._layers = self.build_layers()
    
    def build_layers(self) -> List[LayerConfig]:
        """Build DeepSeek layer configurations with MLA and MoE using kernel API.
        
        Returns:
            List of layer configurations including:
            - Embedding layer
            - Transformer layers with MLA attention
            - MoE or Dense FFN layers (alternating)
            - Final normalization and output projection
        """
        layers = []
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        
        # Embedding layer using embedding kernel
        emb_result = embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.hidden_size,
            input_shape=(1, 1),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="embedding",
            result=emb_result,
            submodule_type=SubmoduleType.EMBEDDING))
        
        # Build each transformer layer
        for i in range(cfg.num_layers):
            layers.extend(self._build_transformer_layer(i, dtype_size))
        
        # Final norm using rms_norm kernel
        final_norm_result = rms_norm(
            input=(1, 1, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="final_norm",
            result=final_norm_result,
            submodule_type=SubmoduleType.NORM))
        
        # LM head using linear kernel
        lm_head_result = linear(
            input=(1, cfg.hidden_size),
            weight=(cfg.vocab_size, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name="lm_head",
            result=lm_head_result,
            submodule_type=SubmoduleType.LM_HEAD))
        
        return layers
    
    def _build_transformer_layer(self, layer_idx: int, dtype_size: int) -> List[LayerConfig]:
        """Build a single transformer layer with MLA and MoE/Dense FFN.
        
        Args:
            layer_idx: Index of the transformer layer
            dtype_size: Size of data type in bytes
            
        Returns:
            List of layer configurations for this transformer layer
        """
        layers = []
        cfg = self.config
        prefix = f"layer_{layer_idx}"
        
        # === MLA Attention Components ===
        # MLA reduces KV cache memory through compression
        layers.extend(self._build_mla_attention(prefix, dtype_size))
        
        # === FFN Components (MoE or Dense) ===
        # First k layers use dense FFN, rest use MoE
        if layer_idx < cfg.first_k_dense_replace:
            layers.extend(self._build_dense_ffn(prefix, dtype_size))
        else:
            layers.extend(self._build_moe_ffn(prefix, dtype_size))
        
        return layers
    
    def _build_mla_attention(self, prefix: str, dtype_size: int) -> List[LayerConfig]:
        """Build MLA (Multi-head Latent Attention) components using kernel API.
        
        MLA Architecture:
        1. Query compression (if q_lora_rank > 0): h -> c_q
        2. KV compression: h -> c_kv (latent vector)
        3. Decompress c_kv -> k, v for attention
        4. Output projection
        
        Args:
            prefix: Layer name prefix
            dtype_size: Size of data type in bytes
            
        Returns:
            List of attention layer configurations
        """
        layers = []
        cfg = self.config
        
        # Query compression (latent space)
        if cfg.q_lora_rank > 0:
            # W_DQ: Down-projection to latent space
            q_down_result = linear(
                input=(1, cfg.hidden_size),
                weight=(cfg.q_lora_rank, cfg.hidden_size),
                bias=None,
                dtype=cfg.dtype
            )
            layers.append(kernel_result_to_layer(
                name=f"{prefix}_q_down_proj",
                result=q_down_result,
                submodule_type=SubmoduleType.ATTENTION))
            
            # W_UQ: Up-projection to Q heads
            q_up_dim = cfg.num_attention_heads * cfg.qk_nope_head_dim
            q_up_result = linear(
                input=(1, cfg.q_lora_rank),
                weight=(q_up_dim, cfg.q_lora_rank),
                bias=None,
                dtype=cfg.dtype
            )
            layers.append(kernel_result_to_layer(
                name=f"{prefix}_q_up_proj",
                result=q_up_result,
                submodule_type=SubmoduleType.ATTENTION))
        
        # KV compression (the key innovation of MLA)
        # W_DKV: Projects hidden state to compressed KV representation
        kv_down_result = linear(
            input=(1, cfg.hidden_size),
            weight=(cfg.kv_lora_rank, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_kv_down_proj",
            result=kv_down_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # W_UK: Decompress latent KV to key heads
        uk_dim = cfg.num_key_value_heads * cfg.qk_nope_head_dim
        uk_result = linear(
            input=(1, cfg.kv_lora_rank),
            weight=(uk_dim, cfg.kv_lora_rank),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_k_up_proj",
            result=uk_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # W_UV: Decompress latent KV to value heads
        uv_dim = cfg.num_key_value_heads * cfg.v_head_dim
        uv_result = linear(
            input=(1, cfg.kv_lora_rank),
            weight=(uv_dim, cfg.kv_lora_rank),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_v_up_proj",
            result=uv_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # RoPE projections for position encoding
        q_rope_result = linear(
            input=(1, cfg.hidden_size),
            weight=(cfg.qk_rope_head_dim, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_q_rope_proj",
            result=q_rope_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        k_rope_result = linear(
            input=(1, cfg.hidden_size),
            weight=(cfg.qk_rope_head_dim, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_k_rope_proj",
            result=k_rope_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # Attention computation using scaled_dot_product_attention kernel
        # For inference, we use cache_len=1, but for training use full seq_len
        seq_len = cfg.max_position_embeddings
        kv_heads = cfg.num_key_value_heads
        q_heads = cfg.num_attention_heads
        head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        
        attn_result = scaled_dot_product_attention(
            query=(1, q_heads, 1, head_dim),
            key=(1, kv_heads, seq_len, head_dim),
            value=(1, kv_heads, seq_len, cfg.v_head_dim),
            is_causal=True,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_mla_attention",
            result=attn_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # Output projection
        o_dim = cfg.num_attention_heads * cfg.v_head_dim
        o_result = linear(
            input=(1, o_dim),
            weight=(cfg.hidden_size, o_dim),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_o_proj",
            result=o_result,
            submodule_type=SubmoduleType.ATTENTION))
        
        # Attention norm/residual using rms_norm kernel
        attn_norm_result = rms_norm(
            input=(1, 1, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_attn_norm",
            result=attn_norm_result,
            submodule_type=SubmoduleType.NORM))
        
        return layers
    
    def _build_dense_ffn(self, prefix: str, dtype_size: int) -> List[LayerConfig]:
        """Build dense FFN layers (used for first k layers) using kernel API.
        
        Args:
            prefix: Layer name prefix
            dtype_size: Size of data type in bytes
            
        Returns:
            List of dense FFN layer configurations
        """
        layers = []
        cfg = self.config
        
        # SwiGLU FFN: up_proj, gate_proj, down_proj
        ffn_intermediate = cfg.intermediate_size
        
        # Up projection
        up_result = linear(
            input=(1, cfg.hidden_size),
            weight=(ffn_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_up_proj",
            result=up_result,
            submodule_type=SubmoduleType.FFN))
        
        # Gate projection
        gate_result = linear(
            input=(1, cfg.hidden_size),
            weight=(ffn_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_gate_proj",
            result=gate_result,
            submodule_type=SubmoduleType.FFN))
        
        # SwiGLU activation using silu kernel
        swiglu_result = silu(
            input=(1, 1, ffn_intermediate),
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_swiglu",
            result=swiglu_result,
            submodule_type=SubmoduleType.FFN))
        
        # Down projection
        down_result = linear(
            input=(1, ffn_intermediate),
            weight=(cfg.hidden_size, ffn_intermediate),
            bias=None,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_down_proj",
            result=down_result,
            submodule_type=SubmoduleType.FFN))
        
        # FFN norm using rms_norm kernel
        ffn_norm_result = rms_norm(
            input=(1, 1, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        layers.append(kernel_result_to_layer(
            name=f"{prefix}_ffn_norm",
            result=ffn_norm_result,
            submodule_type=SubmoduleType.NORM))
        
        return layers
    
    def _build_moe_ffn(self, prefix: str, dtype_size: int) -> List[LayerConfig]:
        """Build MoE FFN layers with DeepSeekMoE architecture using kernel API.
        
        DeepSeekMoE uses:
        - Routed experts: Selected via top-k routing
        - Shared experts: Always active
        - Group-limited routing for efficiency
        
        Args:
            prefix: Layer name prefix
            dtype_size: Size of data type in bytes
            
        Returns:
            List of MoE layer configurations
        """
        layers = []
        cfg = self.config
        
        # Router / Gate using linear kernel
        router_result = linear(
            input=(1, cfg.hidden_size),
            weight=(cfg.n_routed_experts, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )

        router_layer = kernel_result_to_layer(
            name=f"{prefix}_router",
            result=router_result,
            submodule_type=SubmoduleType.MOE)
        router_layer.is_moe = True
        layers.append(router_layer)
        
        # Routed experts (only active ones computed per token)
        ffn_intermediate = cfg.moe_intermediate_size
        
        # Up projection for routed experts
        up_result = linear(
            input=(1, cfg.hidden_size),
            weight=(ffn_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        up_layer = kernel_result_to_layer(
            name=f"{prefix}_routed_up",
            result=up_result,
            submodule_type=SubmoduleType.MOE)
        up_layer.flops = int(up_result.flops * cfg.num_experts_per_tok)
        up_layer.is_moe = True
        layers.append(up_layer)
        
        # Gate projection for routed experts
        gate_result = linear(
            input=(1, cfg.hidden_size),
            weight=(ffn_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        gate_layer = kernel_result_to_layer(
            name=f"{prefix}_routed_gate",
            result=gate_result,
            submodule_type=SubmoduleType.MOE)
        gate_layer.flops = int(gate_result.flops * cfg.num_experts_per_tok)
        gate_layer.is_moe = True
        layers.append(gate_layer)
        
        # SwiGLU activation for routed experts using silu kernel
        swiglu_result = silu(
            input=(1, 1, ffn_intermediate),
            dtype=cfg.dtype
        )
        swiglu_layer = kernel_result_to_layer(
            name=f"{prefix}_routed_swiglu",
            result=swiglu_result,
            submodule_type=SubmoduleType.MOE)
        swiglu_layer.flops = int(swiglu_result.flops * cfg.num_experts_per_tok)
        swiglu_layer.is_moe = True
        layers.append(swiglu_layer)
        
        # Down projection for routed experts
        down_result = linear(
            input=(1, ffn_intermediate),
            weight=(cfg.hidden_size, ffn_intermediate),
            bias=None,
            dtype=cfg.dtype
        )
        down_layer = kernel_result_to_layer(
            name=f"{prefix}_routed_down",
            result=down_result,
            submodule_type=SubmoduleType.MOE)
        down_layer.flops = int(down_result.flops * cfg.num_experts_per_tok)
        down_layer.is_moe = True
        layers.append(down_layer)
        
        # Shared experts (always active)
        shared_intermediate = cfg.moe_intermediate_size * cfg.n_shared_experts
        
        # Shared up projection
        shared_up_result = linear(
            input=(1, cfg.hidden_size),
            weight=(shared_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        shared_up_layer = kernel_result_to_layer(
            name=f"{prefix}_shared_up",
            result=shared_up_result,
            submodule_type=SubmoduleType.MOE)
        shared_up_layer.is_moe = True
        layers.append(shared_up_layer)
        
        # Shared gate projection
        shared_gate_result = linear(
            input=(1, cfg.hidden_size),
            weight=(shared_intermediate, cfg.hidden_size),
            bias=None,
            dtype=cfg.dtype
        )
        shared_gate_layer = kernel_result_to_layer(
            name=f"{prefix}_shared_gate",
            result=shared_gate_result,
            submodule_type=SubmoduleType.MOE)
        shared_gate_layer.is_moe = True
        layers.append(shared_gate_layer)
        
        # Shared SwiGLU
        shared_swiglu_result = silu(
            input=(1, 1, shared_intermediate),
            dtype=cfg.dtype
        )
        shared_swiglu_layer = kernel_result_to_layer(
            name=f"{prefix}_shared_swiglu",
            result=shared_swiglu_result,
            submodule_type=SubmoduleType.MOE)
        shared_swiglu_layer.is_moe = True
        layers.append(shared_swiglu_layer)
        
        # Shared down projection
        shared_down_result = linear(
            input=(1, shared_intermediate),
            weight=(cfg.hidden_size, shared_intermediate),
            bias=None,
            dtype=cfg.dtype
        )
        shared_down_layer = kernel_result_to_layer(
            name=f"{prefix}_shared_down",
            result=shared_down_result,
            submodule_type=SubmoduleType.MOE)
        shared_down_layer.is_moe = True
        layers.append(shared_down_layer)
        
        # All-to-all communication for expert parallelism
        # NOTE: Manual calculation for communication layer (alltoall)
        layers.append(LayerConfig(
            name=f"{prefix}_alltoall_dispatch",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=0,
            flops=0,
            activation_bytes=cfg.hidden_size * dtype_size * cfg.num_experts_per_tok,
            is_moe=True,
            submodule_type=SubmoduleType.MOE,
        ))
        
        # NOTE: Manual calculation for communication layer (alltoall)
        layers.append(LayerConfig(
            name=f"{prefix}_alltoall_combine",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=0,
            flops=0,
            activation_bytes=cfg.hidden_size * dtype_size,
            is_moe=True,
            submodule_type=SubmoduleType.MOE,
        ))
        
        # MoE norm/residual using rms_norm kernel
        moe_norm_result = rms_norm(
            input=(1, 1, cfg.hidden_size),
            dim=-1,
            dtype=cfg.dtype
        )
        moe_norm_layer = kernel_result_to_layer(
            name=f"{prefix}_moe_norm",
            result=moe_norm_result,
            submodule_type=SubmoduleType.NORM)
        moe_norm_layer.is_moe = True
        layers.append(moe_norm_layer)
        
        return layers
    
    def get_mla_kv_cache_size(self, batch_size: int, seq_len: int) -> int:
        """Calculate the MLA KV cache size in bytes.
        
        This is the key metric for MLA efficiency - it shows how much
        memory is saved compared to standard MHA/GQA.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            KV cache size in bytes
        """
        dtype_size = DTYPE_SIZES.get(self.config.dtype, 2)
        # MLA stores only the compressed latent vectors
        return batch_size * seq_len * self.config.kv_lora_rank * dtype_size
    
    def get_standard_kv_cache_size(
        self, batch_size: int, seq_len: int
    ) -> int:
        """Calculate what the KV cache size would be without MLA.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Standard KV cache size in bytes
        """
        cfg = self.config
        dtype_size = DTYPE_SIZES.get(cfg.dtype, 2)
        kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        # Standard MHA/GQA stores full K and V tensors
        return batch_size * seq_len * kv_heads * head_dim * 2 * dtype_size
    
    def get_kv_cache_compression_ratio(self) -> float:
        """Calculate the KV cache compression ratio of MLA.
        
        Returns:
            Compression ratio (standard_size / mla_size)
        """
        cfg = self.config
        kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        standard_size = kv_heads * head_dim * 2  # K + V
        mla_size = cfg.kv_lora_rank
        return standard_size / mla_size


class DeepSeekV2Model(DeepSeekModel):
    """DeepSeek-V2 specific model implementation.
    
    Pre-configured with DeepSeek-V2 official parameters.
    """
    
    def __init__(self):
        config = DeepSeekConfig(
            name="deepseek-v2",
            vocab_size=102400,
            hidden_size=5120,
            num_layers=60,
            num_attention_heads=128,
            intermediate_size=12288,
            num_key_value_heads=128,
            max_seq_len=163840,
            # MLA params
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            # MoE params
            moe_intermediate_size=1536,
            n_routed_experts=160,
            n_shared_experts=2,
            num_experts_per_tok=6,
            first_k_dense_replace=1,
            n_group=8,
            topk_group=3,
            routed_scaling_factor=16.0,
        )
        super().__init__(config)


class DeepSeekV3Model(DeepSeekModel):
    """DeepSeek-V3 specific model implementation.
    
    Pre-configured with DeepSeek-V3 official parameters.
    """
    
    def __init__(self):
        config = DeepSeekV3Config(
            name="deepseek-v3",
            vocab_size=129280,
            hidden_size=7168,
            num_layers=61,
            num_attention_heads=128,
            intermediate_size=18432,
            num_key_value_heads=128,
            max_seq_len=163840,
            # MLA params
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            # MoE params
            moe_intermediate_size=2048,
            n_routed_experts=256,
            n_shared_experts=1,
            num_experts_per_tok=8,
            first_k_dense_replace=1,
            n_group=8,
            topk_group=4,
            routed_scaling_factor=2.5,
        )
        super().__init__(config)

"""DeepSeek V2/V3 model implementation with MLA (Multi-head Latent Attention).

DeepSeek-V2 and DeepSeek-V3 use Multi-head Latent Attention (MLA) which
compresses the Key-Value cache into a latent vector, significantly reducing
memory consumption during inference.

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V2
           https://huggingface.co/deepseek-ai/DeepSeek-V3
"""

from dataclasses import dataclass
from typing import List, Optional

from .base import BaseModel, ModelConfig, LayerConfig
from ..utils.constants import DTYPE_SIZES


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
    """DeepSeek V2/V3 model implementation with MLA.
    
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
        """Build DeepSeek layer configurations with MLA and MoE.
        
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
        
        # Embedding layer
        embed_params = cfg.vocab_size * cfg.hidden_size
        layers.append(LayerConfig(
            name="embedding",
            input_shape=(1, 1),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=embed_params,
            flops=0,
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
            params_count=cfg.hidden_size,
            flops=cfg.hidden_size * 5,  # RMSNorm
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
        """Build MLA (Multi-head Latent Attention) components.
        
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
            q_down_params = cfg.hidden_size * cfg.q_lora_rank
            layers.append(LayerConfig(
                name=f"{prefix}_q_down_proj",
                input_shape=(1, 1, cfg.hidden_size),
                output_shape=(1, 1, cfg.q_lora_rank),
                params_count=q_down_params,
                flops=q_down_params * 2,
                activation_bytes=cfg.q_lora_rank * dtype_size,
            ))
            
            # W_UQ: Up-projection to Q heads
            q_up_params = cfg.q_lora_rank * (
                cfg.num_attention_heads * cfg.qk_nope_head_dim
            )
            layers.append(LayerConfig(
                name=f"{prefix}_q_up_proj",
                input_shape=(1, 1, cfg.q_lora_rank),
                output_shape=(1, 1, cfg.num_attention_heads * cfg.qk_nope_head_dim),
                params_count=q_up_params,
                flops=q_up_params * 2,
                activation_bytes=cfg.num_attention_heads * cfg.qk_nope_head_dim * dtype_size,
            ))
        
        # KV compression (the key innovation of MLA)
        # W_DKV: Projects hidden state to compressed KV representation
        kv_down_params = cfg.hidden_size * cfg.kv_lora_rank
        layers.append(LayerConfig(
            name=f"{prefix}_kv_down_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.kv_lora_rank),
            params_count=kv_down_params,
            flops=kv_down_params * 2,
            activation_bytes=cfg.kv_lora_rank * dtype_size,
        ))
        
        # W_UK: Decompress latent KV to key heads
        uk_params = cfg.kv_lora_rank * (
            cfg.num_key_value_heads * cfg.qk_nope_head_dim
        )
        layers.append(LayerConfig(
            name=f"{prefix}_k_up_proj",
            input_shape=(1, 1, cfg.kv_lora_rank),
            output_shape=(1, 1, cfg.num_key_value_heads * cfg.qk_nope_head_dim),
            params_count=uk_params,
            flops=uk_params * 2,
            activation_bytes=cfg.num_key_value_heads * cfg.qk_nope_head_dim * dtype_size,
        ))
        
        # W_UV: Decompress latent KV to value heads
        uv_params = cfg.kv_lora_rank * (
            cfg.num_key_value_heads * cfg.v_head_dim
        )
        layers.append(LayerConfig(
            name=f"{prefix}_v_up_proj",
            input_shape=(1, 1, cfg.kv_lora_rank),
            output_shape=(1, 1, cfg.num_key_value_heads * cfg.v_head_dim),
            params_count=uv_params,
            flops=uv_params * 2,
            activation_bytes=cfg.num_key_value_heads * cfg.v_head_dim * dtype_size,
        ))
        
        # RoPE projections for position encoding
        qk_rope_params = cfg.hidden_size * cfg.qk_rope_head_dim
        layers.append(LayerConfig(
            name=f"{prefix}_q_rope_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.qk_rope_head_dim),
            params_count=qk_rope_params,
            flops=qk_rope_params * 2,
            activation_bytes=cfg.qk_rope_head_dim * dtype_size,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_k_rope_proj",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.qk_rope_head_dim),
            params_count=qk_rope_params,
            flops=qk_rope_params * 2,
            activation_bytes=cfg.qk_rope_head_dim * dtype_size,
        ))
        
        # Attention computation
        seq_len = cfg.max_seq_len
        attn_flops = self._compute_mla_attention_flops(seq_len)
        layers.append(LayerConfig(
            name=f"{prefix}_mla_attention",
            input_shape=(1, seq_len, cfg.hidden_size),
            output_shape=(1, seq_len, cfg.hidden_size),
            params_count=0,
            flops=attn_flops,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # Output projection
        o_params = (cfg.num_attention_heads * cfg.v_head_dim) * cfg.hidden_size
        layers.append(LayerConfig(
            name=f"{prefix}_o_proj",
            input_shape=(1, 1, cfg.num_attention_heads * cfg.v_head_dim),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=o_params,
            flops=o_params * 2,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        # Attention norm/residual
        layers.append(LayerConfig(
            name=f"{prefix}_attn_norm",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=cfg.hidden_size,
            flops=cfg.hidden_size * 5,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        return layers
    
    def _compute_mla_attention_flops(self, seq_len: int) -> int:
        """Compute FLOPs for MLA attention computation.
        
        MLA uses compressed representations, so attention FLOPs are
        similar to standard attention but with different dimensions.
        
        Args:
            seq_len: Sequence length for attention computation
            
        Returns:
            Estimated FLOPs for attention computation
        """
        cfg = self.config
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        
        # QK^T computation
        qk_flops = seq_len * seq_len * q_heads * head_dim * 2
        
        # Softmax (exp, sum, div)
        softmax_flops = seq_len * seq_len * q_heads * 5
        
        # Attention * V
        attn_v_flops = seq_len * seq_len * q_heads * cfg.v_head_dim * 2
        
        return qk_flops + softmax_flops + attn_v_flops
    
    def _build_dense_ffn(self, prefix: str, dtype_size: int) -> List[LayerConfig]:
        """Build dense FFN layers (used for first k layers).
        
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
        
        layers.append(LayerConfig(
            name=f"{prefix}_swiglu",
            input_shape=(1, 1, ffn_intermediate),
            output_shape=(1, 1, ffn_intermediate),
            params_count=0,
            flops=ffn_intermediate * 8,
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
        
        layers.append(LayerConfig(
            name=f"{prefix}_ffn_norm",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=cfg.hidden_size,
            flops=cfg.hidden_size * 5,
            activation_bytes=cfg.hidden_size * dtype_size,
        ))
        
        return layers
    
    def _build_moe_ffn(self, prefix: str, dtype_size: int) -> List[LayerConfig]:
        """Build MoE FFN layers with DeepSeekMoE architecture.
        
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
        
        # Router / Gate
        router_params = cfg.hidden_size * cfg.n_routed_experts
        layers.append(LayerConfig(
            name=f"{prefix}_router",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.n_routed_experts),
            params_count=router_params,
            flops=router_params * 2 + cfg.n_routed_experts * 10,
            activation_bytes=cfg.n_routed_experts * dtype_size,
            is_moe=True,
        ))
        
        # Routed experts (only active ones computed per token)
        expert_scale = cfg.num_experts_per_tok / cfg.n_routed_experts
        ffn_intermediate = cfg.moe_intermediate_size
        
        # Up projection for routed experts
        expert_up_params = cfg.hidden_size * ffn_intermediate
        layers.append(LayerConfig(
            name=f"{prefix}_routed_up",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, ffn_intermediate),
            params_count=expert_up_params * cfg.n_routed_experts,
            flops=int(expert_up_params * 2 * cfg.num_experts_per_tok),
            activation_bytes=int(ffn_intermediate * dtype_size * cfg.num_experts_per_tok),
            is_moe=True,
        ))
        
        # Gate projection for routed experts
        layers.append(LayerConfig(
            name=f"{prefix}_routed_gate",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, ffn_intermediate),
            params_count=expert_up_params * cfg.n_routed_experts,
            flops=int(expert_up_params * 2 * cfg.num_experts_per_tok),
            activation_bytes=int(ffn_intermediate * dtype_size * cfg.num_experts_per_tok),
            is_moe=True,
        ))
        
        # SwiGLU activation for routed experts
        layers.append(LayerConfig(
            name=f"{prefix}_routed_swiglu",
            input_shape=(1, 1, ffn_intermediate),
            output_shape=(1, 1, ffn_intermediate),
            params_count=0,
            flops=int(ffn_intermediate * 8 * cfg.num_experts_per_tok),
            activation_bytes=int(ffn_intermediate * dtype_size * cfg.num_experts_per_tok),
            is_moe=True,
        ))
        
        # Down projection for routed experts
        expert_down_params = ffn_intermediate * cfg.hidden_size
        layers.append(LayerConfig(
            name=f"{prefix}_routed_down",
            input_shape=(1, 1, ffn_intermediate),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=expert_down_params * cfg.n_routed_experts,
            flops=int(expert_down_params * 2 * cfg.num_experts_per_tok),
            activation_bytes=cfg.hidden_size * dtype_size,
            is_moe=True,
        ))
        
        # Shared experts (always active)
        shared_intermediate = cfg.moe_intermediate_size * cfg.n_shared_experts
        
        layers.append(LayerConfig(
            name=f"{prefix}_shared_up",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, shared_intermediate),
            params_count=cfg.hidden_size * shared_intermediate,
            flops=cfg.hidden_size * shared_intermediate * 2,
            activation_bytes=shared_intermediate * dtype_size,
            is_moe=True,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_shared_gate",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, shared_intermediate),
            params_count=cfg.hidden_size * shared_intermediate,
            flops=cfg.hidden_size * shared_intermediate * 2,
            activation_bytes=shared_intermediate * dtype_size,
            is_moe=True,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_shared_swiglu",
            input_shape=(1, 1, shared_intermediate),
            output_shape=(1, 1, shared_intermediate),
            params_count=0,
            flops=shared_intermediate * 8,
            activation_bytes=shared_intermediate * dtype_size,
            is_moe=True,
        ))
        
        layers.append(LayerConfig(
            name=f"{prefix}_shared_down",
            input_shape=(1, 1, shared_intermediate),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=shared_intermediate * cfg.hidden_size,
            flops=shared_intermediate * cfg.hidden_size * 2,
            activation_bytes=cfg.hidden_size * dtype_size,
            is_moe=True,
        ))
        
        # All-to-all communication for expert parallelism
        layers.append(LayerConfig(
            name=f"{prefix}_alltoall_dispatch",
            input_shape=(1, 1, cfg.hidden_size),
            output_shape=(1, 1, cfg.hidden_size),
            params_count=0,
            flops=0,
            activation_bytes=cfg.hidden_size * dtype_size * cfg.num_experts_per_tok,
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

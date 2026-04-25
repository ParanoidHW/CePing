"""HunyuanImage 3.0 Models with MoE and QK Norm.

HunyuanImage 3.0 family includes:
- Text model: Text autoregressive generation
- Diffusion model: Image diffusion generation

Key features:
- MoE FFN: 64 experts + 1 shared expert, top-8 routing
- QK Norm: Attention uses QK normalization
- 2D RoPE: Image spatial position encoding (not implemented in modeling)
- SwiGLU activation in experts

Reference: HunyuanImage technical report
"""

import logging
from typing import Optional

from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor, ShardedParameter
from llm_perf.modeling.layers import (
    ShardedEmbedding,
    ShardedMoE,
    ShardedLMHead,
)
from llm_perf.kernels.op import RMSNormOp
from llm_perf.modeling.config_compat import SimpleModelConfig

logger = logging.getLogger(__name__)


class ShardedHunyuanAttention(ShardedModule):
    """Attention with QK Normalization.

    HunyuanImage uses QK normalization to stabilize attention.
    This is unique to Hunyuan models.

    Sharding:
    - QKV weights: heads dimension TP-sharded (column sharding)
    - O weight: heads dimension TP-sharded (row sharding)
    - QK Norm weights: not sharded (replicated, head_dim dimension)

    Args:
        hidden_size: Hidden size
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (GQA)
        head_dim: Head dimension
        use_qk_norm: Whether to use QK normalization
        dtype: Data type
    """

    _submodule_name = "attention"

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_qk_norm: bool = True,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.use_qk_norm = use_qk_norm

        qkv_dim = self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim

        self.qkv_weight = ShardedParameter(
            shape=(hidden_size, qkv_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="qkv_weight",
        )

        if use_qk_norm:
            self.query_norm_weight = ShardedParameter(
                shape=(self.head_dim,),
                shardable={},
                dtype=dtype,
                name="query_norm_weight",
            )
            self.key_norm_weight = ShardedParameter(
                shape=(self.head_dim,),
                shardable={},
                dtype=dtype,
                name="key_norm_weight",
            )

        o_dim = self.num_heads * self.head_dim
        self.o_weight = ShardedParameter(
            shape=(o_dim, hidden_size),
            shardable={0: "tp"},
            dtype=dtype,
            name="o_weight",
        )

    def forward(
        self,
        hidden: ShardedTensor,
        is_causal: bool = True,
    ) -> ShardedTensor:
        """Attention forward with QK Norm.

        Args:
            hidden: (batch, seq, hidden_size)
            is_causal: Whether causal mask

        Returns:
            output: (batch, seq, hidden_size)
        """
        batch = hidden.shape[0] if len(hidden.shape) >= 1 else 1
        seq = hidden.shape[1] if len(hidden.shape) >= 2 else 1

        qkv_proj = self._track_intermediate("qkv_proj", hidden @ self.qkv_weight)

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        q_proj = ShardedTensor(
            shape=(batch, seq, q_dim),
            shardable={2: "tp"},
            dtype=hidden.dtype,
            name="q_proj",
        )
        k_proj = ShardedTensor(
            shape=(batch, seq, kv_dim),
            shardable={2: "tp"},
            dtype=hidden.dtype,
            name="k_proj",
        )
        v_proj = ShardedTensor(
            shape=(batch, seq, kv_dim),
            shardable={2: "tp"},
            dtype=hidden.dtype,
            name="v_proj",
        )

        if self.use_qk_norm:
            q_normed = self._rms_norm_per_head(q_proj, self.query_norm_weight)
            k_normed = self._rms_norm_per_head(k_proj, self.key_norm_weight)
        else:
            q_normed = q_proj
            k_normed = k_proj

        q = q_normed.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_normed.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_proj.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        from llm_perf.modeling.layers import flash_attention

        attn_out = flash_attention(q, k, v, is_causal=is_causal)

        attn_flat = attn_out.transpose(1, 2).view(batch, seq, self.num_heads * self.head_dim)

        output = self._track_intermediate("output", attn_flat @ self.o_weight)

        self._activations["qkv_proj"] = qkv_proj
        self._activations["output"] = output

        return output

    def _rms_norm_per_head(
        self,
        input_tensor: ShardedTensor,
        weight: ShardedParameter,
    ) -> ShardedTensor:
        """RMS normalization per head dimension.

        Args:
            input_tensor: (batch, seq, heads * head_dim)
            weight: (head_dim,)

        Returns:
            output: (batch, seq, heads * head_dim)
        """
        output = ShardedTensor(
            shape=input_tensor.shape,
            shardable=input_tensor.shardable,
            dtype=input_tensor.dtype,
            name="qk_norm_output",
        )
        output._op_history = input_tensor._op_history + [
            RMSNormOp(
                dtype=input_tensor.dtype,
                input=input_tensor,
                weight=weight,
                output=output,
            )
        ]
        return output


class ShardedHunyuanMoEBlock(ShardedModule):
    """Transformer Block for HunyuanImage with MoE and QK Norm.

    Structure:
    - Input RMSNorm
    - Attention (with QK Norm)
    - Post-Attention RMSNorm
    - MoE FFN (routed + shared experts)

    Args:
        hidden_size: Hidden size
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (GQA)
        head_dim: Head dimension
        moe_intermediate_size: MoE intermediate size
        num_experts: Number of routed experts
        num_experts_per_token: Number of active experts
        num_shared_experts: Number of shared experts
        use_qk_norm: Whether to use QK normalization
        dtype: Data type
    """

    _submodule_name = "transformer_block"

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        num_experts: int = 64,
        num_experts_per_token: int = 8,
        num_shared_experts: int = 1,
        use_qk_norm: bool = True,
        dtype: str = "fp16",
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        if moe_intermediate_size is None:
            moe_intermediate_size = hidden_size * 4

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_shared_experts = num_shared_experts
        self.use_qk_norm = use_qk_norm
        self.dtype = dtype

        self.input_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="input_norm_weight",
        )

        self.attention = ShardedHunyuanAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
            dtype=dtype,
        )

        self.post_attn_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="post_attn_norm_weight",
        )

        shared_expert_intermediate = moe_intermediate_size * num_shared_experts

        self.moe = ShardedMoE(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            shared_expert_intermediate=shared_expert_intermediate,
            dtype=dtype,
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """HunyuanImage MoE block forward.

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            output: (batch, seq, hidden_size)
        """
        norm_out = self._rms_norm(hidden, self.input_norm_weight)
        attn_out = self.attention(norm_out)
        hidden = hidden + attn_out

        norm_out = self._rms_norm(hidden, self.post_attn_norm_weight)
        moe_out = self.moe(norm_out)
        output = hidden + moe_out

        self._activations["attn_out"] = attn_out
        self._activations["moe_out"] = moe_out

        return output

    def _rms_norm(self, input_tensor: ShardedTensor, weight: ShardedParameter) -> ShardedTensor:
        """RMS normalization."""
        output = ShardedTensor(
            shape=input_tensor.shape,
            shardable=input_tensor.shardable,
            dtype=input_tensor.dtype,
            name="rmsnorm_output",
        )
        output._op_history = input_tensor._op_history + [
            RMSNormOp(
                dtype=input_tensor.dtype,
                input=input_tensor,
                weight=weight,
                output=output,
            )
        ]
        self._track_intermediate("rmsnorm_output", output)
        return output


class HunyuanImage3TextModel(ShardedModule):
    """HunyuanImage 3.0 Text Model for autoregressive generation.

    Structure:
    - Embedding
    - N x ShardedHunyuanMoEBlock
    - Final RMSNorm
    - LM Head

    Key features:
    - MoE: 64 experts + 1 shared expert, top-8 routing
    - QK Norm: Attention normalization for stability
    - SwiGLU: Expert activation

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden size
        num_layers: Number of transformer layers
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (GQA)
        head_dim: Head dimension
        moe_intermediate_size: MoE intermediate size
        num_experts: Number of routed experts
        num_experts_per_token: Number of active experts
        num_shared_experts: Number of shared experts
        use_qk_norm: Whether to use QK normalization
        max_seq_len: Maximum sequence length
        dtype: Data type
    """

    def __init__(
        self,
        vocab_size: int = 133120,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        num_experts: int = 64,
        num_experts_per_token: int = 8,
        num_shared_experts: int = 1,
        use_qk_norm: bool = True,
        max_seq_len: int = 4096,
        dtype: str = "fp16",
    ):
        super().__init__()
        logger.debug(
            f"[HUNYUAN_TEXT_INIT] num_layers={num_layers}, num_experts={num_experts}, "
            f"use_qk_norm={use_qk_norm}"
        )

        if num_kv_heads is None:
            num_kv_heads = 8
        if head_dim is None:
            head_dim = 128
        if moe_intermediate_size is None:
            moe_intermediate_size = 3072

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_shared_experts = num_shared_experts
        self.use_qk_norm = use_qk_norm
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.config = SimpleModelConfig(
            name="hunyuan_image_3_text",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            intermediate_size=moe_intermediate_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

        self.embedding = ShardedEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
        )

        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                ShardedHunyuanMoEBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    moe_intermediate_size=moe_intermediate_size,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    num_shared_experts=num_shared_experts,
                    use_qk_norm=use_qk_norm,
                    dtype=dtype,
                )
            )
        self.layers = self.layers

        self.final_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="final_norm_weight",
        )

        self.lm_head = ShardedLMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dtype=dtype,
        )

    def forward(self, input_ids: ShardedTensor) -> ShardedTensor:
        """HunyuanImage Text forward.

        Args:
            input_ids: (batch, seq)

        Returns:
            logits: (batch, seq, vocab_size)
        """
        hidden = self.embedding(input_ids)
        self._activations["embedding_output"] = hidden

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            self._activations[f"layer_{i}_output"] = hidden

        hidden = self._rms_norm(hidden, self.final_norm_weight)
        self._activations["final_norm_output"] = hidden

        logits = self.lm_head(hidden)
        self._activations["lm_head_output"] = logits

        return logits

    def _rms_norm(self, input_tensor: ShardedTensor, weight: ShardedParameter) -> ShardedTensor:
        """RMS normalization."""
        output = ShardedTensor(
            shape=input_tensor.shape,
            shardable=input_tensor.shardable,
            dtype=input_tensor.dtype,
            name="rmsnorm_output",
        )
        output._op_history = input_tensor._op_history + [
            RMSNormOp(
                dtype=input_tensor.dtype,
                input=input_tensor,
                weight=weight,
                output=output,
            )
        ]
        self._track_intermediate("rmsnorm_output", output)
        return output


class HunyuanImage3DiffusionModel(ShardedModule):
    """HunyuanImage 3.0 Diffusion Model for image generation.

    Structure:
    - Embedding (learned positional embedding for image patches)
    - N x ShardedHunyuanMoEBlock
    - Final RMSNorm
    - Output projection (no LM head for diffusion)

    Key features:
    - Same backbone as Text model
    - Timestep embedding for diffusion
    - 2D RoPE for spatial position (not implemented in modeling)

    Args:
        hidden_size: Hidden size
        num_layers: Number of transformer layers
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (GQA)
        head_dim: Head dimension
        moe_intermediate_size: MoE intermediate size
        num_experts: Number of routed experts
        num_experts_per_token: Number of active experts
        num_shared_experts: Number of shared experts
        use_qk_norm: Whether to use QK normalization
        image_height: Image height (for latent)
        image_width: Image width (for latent)
        latent_channels: Latent channels
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        num_experts: int = 64,
        num_experts_per_token: int = 8,
        num_shared_experts: int = 1,
        use_qk_norm: bool = True,
        image_height: int = 64,
        image_width: int = 64,
        latent_channels: int = 16,
        dtype: str = "fp16",
    ):
        super().__init__()
        logger.debug(
            f"[HUNYUAN_DIFF_INIT] num_layers={num_layers}, num_experts={num_experts}, "
            f"image_height={image_height}, image_width={image_width}"
        )

        if num_kv_heads is None:
            num_kv_heads = 8
        if head_dim is None:
            head_dim = 128
        if moe_intermediate_size is None:
            moe_intermediate_size = 3072

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_shared_experts = num_shared_experts
        self.use_qk_norm = use_qk_norm
        self.image_height = image_height
        self.image_width = image_width
        self.latent_channels = latent_channels
        self.dtype = dtype

        self.seq_len = image_height * image_width

        self.pos_embedding = ShardedEmbedding(
            num_embeddings=self.seq_len,
            embedding_dim=hidden_size,
            dtype=dtype,
        )

        self.timestep_in_weight = ShardedParameter(
            shape=(256, hidden_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="timestep_in_weight",
        )
        self.timestep_out_weight = ShardedParameter(
            shape=(hidden_size, hidden_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="timestep_out_weight",
        )

        self.input_proj_weight = ShardedParameter(
            shape=(latent_channels, hidden_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="input_proj_weight",
        )

        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                ShardedHunyuanMoEBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    moe_intermediate_size=moe_intermediate_size,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    num_shared_experts=num_shared_experts,
                    use_qk_norm=use_qk_norm,
                    dtype=dtype,
                )
            )
        self.layers = self.layers

        self.final_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="final_norm_weight",
        )

        self.output_proj_weight = ShardedParameter(
            shape=(hidden_size, latent_channels),
            shardable={1: "tp"},
            dtype=dtype,
            name="output_proj_weight",
        )

    def forward(
        self,
        latent: ShardedTensor,
        timestep: ShardedTensor,
    ) -> ShardedTensor:
        """HunyuanImage Diffusion forward.

        Args:
            latent: (batch, seq, latent_channels) where seq = h * w
            timestep: (batch,) or (batch, 1)

        Returns:
            output: (batch, seq, latent_channels)
        """
        batch = latent.shape[0] if len(latent.shape) >= 1 else 1
        seq = latent.shape[1] if len(latent.shape) >= 2 else self.seq_len

        hidden = latent @ self.input_proj_weight
        self._activations["input_proj_output"] = hidden

        pos_ids = ShardedTensor(shape=(batch, seq), dtype="int32", name="pos_ids")
        pos_embed = self.pos_embedding(pos_ids)
        hidden = hidden + pos_embed
        self._activations["pos_embedding_output"] = hidden

        time_embed = timestep @ self.timestep_in_weight
        from llm_perf.modeling.layers import silu
        time_embed = silu(time_embed)
        time_embed = time_embed @ self.timestep_out_weight
        self._activations["timestep_embedding"] = time_embed

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            self._activations[f"layer_{i}_output"] = hidden

        hidden = self._rms_norm(hidden, self.final_norm_weight)
        self._activations["final_norm_output"] = hidden

        output = hidden @ self.output_proj_weight
        self._activations["output_proj_output"] = output

        return output

    def _rms_norm(self, input_tensor: ShardedTensor, weight: ShardedParameter) -> ShardedTensor:
        """RMS normalization."""
        output = ShardedTensor(
            shape=input_tensor.shape,
            shardable=input_tensor.shardable,
            dtype=input_tensor.dtype,
            name="rmsnorm_output",
        )
        output._op_history = input_tensor._op_history + [
            RMSNormOp(
                dtype=input_tensor.dtype,
                input=input_tensor,
                weight=weight,
                output=output,
            )
        ]
        self._track_intermediate("rmsnorm_output", output)
        return output
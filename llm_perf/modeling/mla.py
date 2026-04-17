"""MLA (Multi-head Latent Attention) implementation for DeepSeek.

MLA compresses KV cache through low-rank projection, reducing memory from
O(seq_len × num_heads × head_dim) to O(seq_len × kv_lora_rank).

Architecture:
1. KV compression: hidden -> latent_kv (kv_lora_rank)
2. KV decompression: latent_kv -> k, v
3. Q compression (optional): hidden -> latent_q -> q
4. RoPE projections: q_rope, k_rope
5. Output projection

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V2
"""

from typing import Optional
from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor
from llm_perf.modeling.layers import flash_attention
from llm_perf.kernels.op import MatmulOp


class ShardedMLA(ShardedModule):
    """Multi-head Latent Attention (MLA) layer.

    MLA reduces KV cache memory through compression:
    - KV compression: hidden_size -> kv_lora_rank (latent vector)
    - KV decompression: latent -> num_kv_heads × head_dim

    Sharding:
    - KV down weight: replicated (no sharding on latent dimension)
    - KV up weights: TP-sharded on heads dimension
    - Q weights: TP-sharded (if q_lora_rank > 0)
    - O weight: TP-sharded on heads dimension (row sharding)
    - Output needs AllReduce aggregation

    Args:
        hidden_size: Hidden size
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads
        kv_lora_rank: KV compression dimension (latent vector size)
        q_lora_rank: Query compression dimension (0 means no compression)
        qk_nope_head_dim: Non-RoPE query/key head dimension
        qk_rope_head_dim: RoPE query/key head dimension
        v_head_dim: Value head dimension
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim

        if q_lora_rank > 0:
            self.q_down_weight = ShardedTensor(
                shape=(hidden_size, q_lora_rank),
                shardable={},
                dtype=dtype,
                name="q_down_weight",
            )

            q_up_dim = num_heads * qk_nope_head_dim
            self.q_up_weight = ShardedTensor(
                shape=(q_lora_rank, q_up_dim),
                shardable={1: "tp"},
                dtype=dtype,
                name="q_up_weight",
            )
        else:
            self.q_down_weight = None
            self.q_up_weight = None

        self.kv_down_weight = ShardedTensor(
            shape=(hidden_size, kv_lora_rank),
            shardable={},
            dtype=dtype,
            name="kv_down_weight",
        )

        k_up_dim = self.num_kv_heads * qk_nope_head_dim
        self.k_up_weight = ShardedTensor(
            shape=(kv_lora_rank, k_up_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="k_up_weight",
        )

        v_up_dim = self.num_kv_heads * v_head_dim
        self.v_up_weight = ShardedTensor(
            shape=(kv_lora_rank, v_up_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="v_up_weight",
        )

        self.q_rope_weight = ShardedTensor(
            shape=(hidden_size, num_heads * qk_rope_head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="q_rope_weight",
        )

        self.k_rope_weight = ShardedTensor(
            shape=(hidden_size, self.num_kv_heads * qk_rope_head_dim),
            shardable={1: "tp"},
            dtype=dtype,
            name="k_rope_weight",
        )

        o_dim = num_heads * v_head_dim
        self.o_weight = ShardedTensor(
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
        """MLA forward.

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            output: (batch, seq, hidden_size)
        """
        batch = hidden.shape[0] if len(hidden.shape) >= 1 else 1
        seq = hidden.shape[1] if len(hidden.shape) >= 2 else 1

        if self.q_lora_rank > 0:
            q_down = hidden @ self.q_down_weight
            q_up = q_down @ self.q_up_weight
            q_nope = q_up
        else:
            q_nope_weight = ShardedTensor(
                shape=(self.hidden_size, self.num_heads * self.qk_nope_head_dim),
                shardable={1: "tp"},
                dtype=hidden.dtype,
                name="q_nope_weight",
            )
            q_nope = hidden @ q_nope_weight

        q_rope = hidden @ self.q_rope_weight

        kv_down = hidden @ self.kv_down_weight

        k_up = kv_down @ self.k_up_weight
        v_up = kv_down @ self.v_up_weight

        k_rope = hidden @ self.k_rope_weight

        q_nope_reshaped = q_nope.view(batch, seq, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        q_rope_reshaped = q_rope.view(batch, seq, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        q = ShardedTensor(
            shape=(batch, self.num_heads, seq, self.head_dim),
            shardable=q_nope_reshaped.shardable,
            dtype=hidden.dtype,
            name="q_concat",
        )

        k_nope_reshaped = k_up.view(batch, seq, self.num_kv_heads, self.qk_nope_head_dim).transpose(1, 2)
        k_rope_reshaped = k_rope.view(batch, seq, self.num_kv_heads, self.qk_rope_head_dim).transpose(1, 2)
        k = ShardedTensor(
            shape=(batch, self.num_kv_heads, seq, self.head_dim),
            shardable=k_nope_reshaped.shardable,
            dtype=hidden.dtype,
            name="k_concat",
        )

        v_reshaped = v_up.view(batch, seq, self.num_kv_heads, self.v_head_dim).transpose(1, 2)

        attn_out = flash_attention(q, k, v_reshaped, is_causal=is_causal)

        attn_flat = attn_out.transpose(1, 2).view(batch, seq, self.num_heads * self.v_head_dim)

        output = attn_flat @ self.o_weight

        self._activations["kv_down"] = kv_down
        self._activations["attn_out"] = attn_out

        return output

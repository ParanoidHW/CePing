"""Qwen3.5 MoE Model with Hybrid Attention.

Qwen3.5-35B-A3B features:
- Hybrid attention: 3 linear_attention + 1 full_attention per 4-layer cycle
- MoE FFN: 256 experts, top-8 routing, shared expert
- 40 transformer layers
- Linear attention: O(seq) complexity with kernel_dim=4

Reference: Qwen3.5 technical report
"""

import logging
from typing import List, Optional

from llm_perf.modeling.module import ShardedModule
from llm_perf.modeling.tensor import ShardedTensor, ShardedParameter
from llm_perf.modeling.layers import (
    ShardedEmbedding,
    ShardedAttention,
    ShardedLinearAttention,
    ShardedMoE,
    ShardedLMHead,
)
from llm_perf.kernels.op import RMSNormOp
from llm_perf.modeling.config_compat import SimpleModelConfig

logger = logging.getLogger(__name__)


def generate_layer_types(num_layers: int) -> List[str]:
    """Generate layer_types pattern for Qwen3.5.

    Pattern: 3 linear_attention + 1 full_attention per 4-layer cycle.
    Total: 10 cycles for 40 layers.

    Args:
        num_layers: Total number of layers

    Returns:
        List of layer type strings
    """
    pattern = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    num_cycles = num_layers // 4
    remainder = num_layers % 4

    layer_types = pattern * num_cycles + pattern[:remainder]
    return layer_types


class ShardedQwen3_5MoEBlock(ShardedModule):
    """Transformer Block for Qwen3.5 MoE with hybrid attention.

    Supports both linear_attention and full_attention based on layer_type.
    All layers use MoE FFN (routed + shared experts).

    Args:
        hidden_size: Hidden size
        layer_type: "linear_attention" or "full_attention"
        num_heads: Number of query heads (for full attention)
        num_kv_heads: Number of KV heads (for full attention)
        head_dim: Head dimension (for full attention)
        linear_num_heads: Number of linear attention heads
        linear_num_kv_heads: Number of linear attention KV heads
        linear_key_head_dim: Linear attention key head dimension
        linear_value_head_dim: Linear attention value head dimension
        linear_kernel_dim: Linear attention kernel dimension
        intermediate_size: MoE intermediate size
        num_experts: Number of routed experts
        num_experts_per_token: Number of active experts per token
        shared_expert_intermediate: Shared expert intermediate size
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int,
        layer_type: str,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        linear_num_heads: Optional[int] = None,
        linear_num_kv_heads: Optional[int] = None,
        linear_key_head_dim: Optional[int] = None,
        linear_value_head_dim: Optional[int] = None,
        linear_kernel_dim: int = 4,
        intermediate_size: Optional[int] = None,
        num_experts: int = 256,
        num_experts_per_token: int = 8,
        shared_expert_intermediate: Optional[int] = None,
        dtype: str = "fp16",
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.hidden_size = hidden_size
        self.layer_type = layer_type
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.linear_num_heads = linear_num_heads or num_heads
        self.linear_num_kv_heads = linear_num_kv_heads or linear_num_heads
        self.linear_key_head_dim = linear_key_head_dim or head_dim
        self.linear_value_head_dim = linear_value_head_dim or head_dim
        self.linear_kernel_dim = linear_kernel_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.shared_expert_intermediate = shared_expert_intermediate
        self.dtype = dtype

        self.input_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="input_norm_weight",
        )

        if layer_type == "linear_attention":
            self.attention = ShardedLinearAttention(
                hidden_size=hidden_size,
                num_heads=self.linear_num_heads,
                num_kv_heads=self.linear_num_kv_heads,
                head_dim=self.linear_key_head_dim,
                kernel_dim=linear_kernel_dim,
                dtype=dtype,
            )
        else:
            self.attention = ShardedAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            )

        self.post_attn_norm_weight = ShardedParameter(
            shape=(hidden_size,),
            shardable={},
            dtype=dtype,
            name="post_attn_norm_weight",
        )

        self.moe = ShardedMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            shared_expert_intermediate=shared_expert_intermediate,
            dtype=dtype,
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Qwen3.5 MoE block forward.

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


class Qwen3_5MoEModel(ShardedModule):
    """Qwen3.5 MoE Model with Hybrid Attention.

    Structure:
    - Embedding
    - N x ShardedQwen3_5MoEBlock (hybrid linear/full attention + MoE)
    - Final RMSNorm
    - LM Head

    Key features:
    - layer_types: 3 linear + 1 full per 4-layer cycle
    - Linear attention: O(seq) complexity
    - MoE: 256 experts, top-8 routing, shared expert

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden size
        num_layers: Number of transformer layers
        num_heads: Number of full attention heads
        num_kv_heads: Number of full attention KV heads
        head_dim: Full attention head dimension
        linear_num_heads: Linear attention key heads
        linear_num_kv_heads: Linear attention value heads
        linear_key_head_dim: Linear attention key head dimension
        linear_value_head_dim: Linear attention value head dimension
        linear_kernel_dim: Linear attention kernel dimension
        intermediate_size: MoE intermediate size
        num_experts: Number of routed experts
        num_experts_per_token: Number of active experts
        shared_expert_intermediate: Shared expert intermediate size
        layer_types: List of layer types (optional, auto-generated if None)
        max_seq_len: Maximum sequence length
        dtype: Data type
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        linear_num_heads: Optional[int] = None,
        linear_num_kv_heads: Optional[int] = None,
        linear_key_head_dim: Optional[int] = None,
        linear_value_head_dim: Optional[int] = None,
        linear_kernel_dim: int = 4,
        intermediate_size: Optional[int] = None,
        num_experts: int = 256,
        num_experts_per_token: int = 8,
        shared_expert_intermediate: Optional[int] = None,
        layer_types: Optional[List[str]] = None,
        max_seq_len: int = 4096,
        dtype: str = "fp16",
    ):
        super().__init__()
        logger.debug(
            f"[QWEN3_5_INIT] num_layers={num_layers}, num_experts={num_experts}, "
            f"linear_kernel_dim={linear_kernel_dim}"
        )

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        if intermediate_size is None:
            intermediate_size = 512

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.linear_num_heads = linear_num_heads or num_heads
        self.linear_num_kv_heads = linear_num_kv_heads or self.linear_num_heads
        self.linear_key_head_dim = linear_key_head_dim or head_dim
        self.linear_value_head_dim = linear_value_head_dim or head_dim
        self.linear_kernel_dim = linear_kernel_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.shared_expert_intermediate = shared_expert_intermediate
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        if layer_types is None:
            self.layer_types = generate_layer_types(num_layers)
        else:
            self.layer_types = layer_types

        if len(self.layer_types) != num_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) != num_layers ({num_layers})"
            )

        self.config = SimpleModelConfig(
            name="qwen3_5_moe",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            intermediate_size=intermediate_size,
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
        for i, layer_type in enumerate(self.layer_types):
            self.layers.append(
                ShardedQwen3_5MoEBlock(
                    hidden_size=hidden_size,
                    layer_type=layer_type,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    linear_num_heads=self.linear_num_heads,
                    linear_num_kv_heads=self.linear_num_kv_heads,
                    linear_key_head_dim=self.linear_key_head_dim,
                    linear_value_head_dim=self.linear_value_head_dim,
                    linear_kernel_dim=linear_kernel_dim,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    shared_expert_intermediate=shared_expert_intermediate,
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
        """Qwen3.5 MoE forward.

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
"""Complete transformer models.

Includes:
- ShardedTransformerBlock: Transformer block with Attention + FFN + Norms
- LlamaModel: Complete Llama model
"""

from typing import Optional
from .module import ShardedModule
from .tensor import ShardedTensor
from .layers import (
    ShardedEmbedding,
    ShardedRMSNorm,
    ShardedAttention,
    ShardedFFN,
    ShardedLMHead,
)


class ShardedTransformerBlock(ShardedModule):
    """Transformer Block.

    Composition: Attention + FFN + RMSNorms

    Args:
        hidden_size: Hidden size
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (GQA)
        head_dim: Head dimension
        intermediate_size: FFN intermediate size
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        intermediate_size: int = None,
        dtype: str = "fp16",
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.dtype = dtype

        self.input_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        self.attention = ShardedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.post_attn_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        self.ffn = ShardedFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Transformer block forward.

        Args:
            hidden: (batch, seq, hidden_size)

        Returns:
            output: (batch, seq, hidden_size)
        """
        norm_out = self.input_norm(hidden)
        attn_out = self.attention(norm_out)
        hidden = hidden + attn_out

        norm_out = self.post_attn_norm(hidden)
        ffn_out = self.ffn(norm_out)
        output = hidden + ffn_out

        self._activations["attn_out"] = attn_out
        self._activations["ffn_out"] = ffn_out

        return output


class LlamaModel(ShardedModule):
    """Llama model.

    Structure:
    - Embedding
    - N x TransformerBlock
    - Final RMSNorm
    - LM Head

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (GQA)
        intermediate_size: FFN intermediate size
        head_dim: Head dimension
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
        intermediate_size: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_seq_len: int = 4096,
        dtype: str = "fp16",
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            head_dim = hidden_size // num_heads
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.embedding = ShardedEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
        )

        self.layers = [
            ShardedTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        self.final_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        self.lm_head = ShardedLMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dtype=dtype,
        )

    def forward(self, input_ids: ShardedTensor) -> ShardedTensor:
        """Llama forward.

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

        hidden = self.final_norm(hidden)
        self._activations["final_norm_output"] = hidden

        logits = self.lm_head(hidden)
        self._activations["lm_head_output"] = logits

        return logits

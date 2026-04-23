"""Config compatibility layer for modeling framework.

Provides ModelConfig compatibility for new ShardedModule models.
"""

from dataclasses import dataclass


@dataclass
class SimpleModelConfig:
    """Simple config for ShardedModule models.

    Compatible with legacy BaseModel.config interface.
    """

    name: str = "unknown"
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    intermediate_size: int = 0
    max_seq_len: int = 4096
    dtype: str = "fp16"
    num_experts: int = 0
    num_experts_per_token: int = 0
    mtp_num_layers: int = 0
    mtp_share_embeddings: bool = True

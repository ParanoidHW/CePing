"""Wan video generation models using Sharded interface.

Based on: Wan: Open and Advanced Large-Scale Video Generative Models
Paper: https://arxiv.org/abs/2503.20314

Includes:
- ShardedWanTextEncoder: umT5-XXL text encoder
- ShardedWanDiTBlock: DiT transformer block
- ShardedWanDiT: Complete DiT model
"""

from typing import Tuple
from llm_perf.modeling.base.module import ShardedModule
from llm_perf.modeling.base.tensor import ShardedTensor
from llm_perf.modeling.layers import (
    ShardedEmbedding,
    ShardedAttention,
    ShardedFFN,
    ShardedRMSNorm,
    silu,
    flash_attention,
)
from llm_perf.modeling.utils.vision import ShardedConv3d


class ShardedLayerNorm(ShardedModule):
    """Layer Normalization (T5 style).

    Args:
        hidden_size: Hidden size
        elementwise_affine: Whether to learn scale and bias
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = ShardedTensor(
                shape=(hidden_size,),
                shardable={},
                dtype=dtype,
                name="layernorm_weight",
            )
            self.bias = ShardedTensor(
                shape=(hidden_size,),
                shardable={},
                dtype=dtype,
                name="layernorm_bias",
            )

    def forward(self, input_tensor: ShardedTensor) -> ShardedTensor:
        """Layer norm forward."""
        output = ShardedTensor(
            shape=input_tensor.shape,
            shardable=input_tensor.shardable,
            dtype=input_tensor.dtype,
            name="layernorm_output",
        )
        return output


class ShardedT5Block(ShardedModule):
    """T5 Encoder Block.

    Structure: Self-Attention -> LayerNorm -> FFN -> LayerNorm

    Args:
        hidden_size: Hidden size
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate size
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        self.self_attn = ShardedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            dtype=dtype,
        )

        self.post_attn_norm = ShardedLayerNorm(hidden_size, elementwise_affine=True, dtype=dtype)

        self.ffn_gate = ShardedFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )

        self.final_norm = ShardedLayerNorm(hidden_size, elementwise_affine=True, dtype=dtype)

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """T5 block forward."""
        attn_out = self.self_attn(hidden)
        hidden = hidden + attn_out
        hidden = self.post_attn_norm(hidden)

        ffn_out = self.ffn_gate(hidden)
        hidden = hidden + ffn_out
        hidden = self.final_norm(hidden)

        return hidden


class ShardedWanTextEncoder(ShardedModule):
    """Wan Text Encoder (umT5-XXL).

    T5 encoder for encoding text prompts into embeddings.

    Args:
        vocab_size: Vocabulary size (umT5 uses 256384)
        hidden_size: Hidden size (4096 for umT5-XXL)
        num_layers: Number of encoder layers (24)
        num_heads: Number of attention heads (64)
        intermediate_size: FFN intermediate size (10240)
        max_text_len: Maximum text length
        dtype: Data type
    """

    def __init__(
        self,
        vocab_size: int = 256384,
        hidden_size: int = 4096,
        num_layers: int = 24,
        num_heads: int = 64,
        intermediate_size: int = 10240,
        max_text_len: int = 512,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_text_len = max_text_len

        self.embed_tokens = ShardedEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
        )

        self.layers = [
            ShardedT5Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        self.final_norm = ShardedLayerNorm(hidden_size, elementwise_affine=True, dtype=dtype)

    def forward(self, input_ids: ShardedTensor) -> ShardedTensor:
        """Encode text to embeddings."""
        hidden = self.embed_tokens(input_ids)
        self._activations["embedding"] = hidden

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            self._activations[f"layer_{i}"] = hidden

        hidden = self.final_norm(hidden)
        return hidden


class ShardedWanDiTBlock(ShardedModule):
    """Wan DiT Transformer Block.

    Structure:
    1. Modulation (time-dependent)
    2. Self-Attention (spatial-temporal)
    3. Cross-Attention (text conditioning)
    4. FFN

    Args:
        hidden_size: Hidden size
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate size
        text_dim: Text embedding dimension
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        text_dim: int = 4096,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.text_dim = text_dim

        self.norm1 = ShardedLayerNorm(hidden_size, elementwise_affine=False, dtype=dtype)

        self.self_attn_qkv = ShardedModule()
        self.self_attn_qkv.q_weight = ShardedTensor(
            shape=(hidden_size, hidden_size), shardable={}, dtype=dtype, name="q_weight"
        )
        self.self_attn_qkv.k_weight = ShardedTensor(
            shape=(hidden_size, hidden_size), shardable={}, dtype=dtype, name="k_weight"
        )
        self.self_attn_qkv.v_weight = ShardedTensor(
            shape=(hidden_size, hidden_size), shardable={}, dtype=dtype, name="v_weight"
        )
        self.self_attn_qkv.o_weight = ShardedTensor(
            shape=(hidden_size, hidden_size), shardable={}, dtype=dtype, name="o_weight"
        )

        self.self_attn_q_norm = ShardedRMSNorm(hidden_size, dtype=dtype)
        self.self_attn_k_norm = ShardedRMSNorm(hidden_size, dtype=dtype)

        self.norm2 = ShardedLayerNorm(hidden_size, elementwise_affine=False, dtype=dtype)

        self.cross_attn_q_weight = ShardedTensor(
            shape=(hidden_size, hidden_size), shardable={}, dtype=dtype, name="cross_q_weight"
        )
        self.cross_attn_kv_weight = ShardedTensor(
            shape=(text_dim, 2 * hidden_size), shardable={}, dtype=dtype, name="cross_kv_weight"
        )
        self.cross_attn_o_weight = ShardedTensor(
            shape=(hidden_size, hidden_size), shardable={}, dtype=dtype, name="cross_o_weight"
        )

        self.norm3 = ShardedLayerNorm(hidden_size, elementwise_affine=False, dtype=dtype)

        self.ffn = ShardedFFN(hidden_size, intermediate_size, dtype=dtype)

        self.modulation = ShardedTensor(shape=(6, hidden_size), shardable={}, dtype=dtype, name="modulation")

    def forward(
        self,
        hidden: ShardedTensor,
        text_embed: ShardedTensor,
        time_embed: ShardedTensor,
    ) -> ShardedTensor:
        """DiT block forward."""
        norm_hidden = self.norm1(hidden)

        q = norm_hidden @ self.self_attn_qkv.q_weight
        k = norm_hidden @ self.self_attn_qkv.k_weight
        v = norm_hidden @ self.self_attn_qkv.v_weight

        q = self.self_attn_q_norm(q)
        k = self.self_attn_k_norm(k)

        batch = hidden.shape[0] if len(hidden.shape) >= 1 else 1
        seq = hidden.shape[-2] if len(hidden.shape) >= 2 else 1
        head_dim = self.hidden_size // self.num_heads

        q_4d = q.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        k_4d = k.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        v_4d = v.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)

        attn_out = flash_attention(q_4d, k_4d, v_4d)
        attn_out = attn_out.transpose(1, 2).view(batch, seq, self.hidden_size)

        attn_out = attn_out @ self.self_attn_qkv.o_weight
        hidden = hidden + attn_out

        norm_hidden = self.norm2(hidden)
        cross_q = norm_hidden @ self.cross_attn_q_weight

        hidden = hidden + cross_q

        norm_hidden = self.norm3(hidden)
        ffn_out = self.ffn(norm_hidden)
        hidden = hidden + ffn_out

        self._activations["self_attn_out"] = attn_out
        self._activations["ffn_out"] = ffn_out

        return hidden


class ShardedWanDiT(ShardedModule):
    """Wan Diffusion Transformer (DiT) Model.

    Complete DiT model for video generation.

    Args:
        hidden_size: Hidden size (5120 for 14B)
        num_layers: Number of transformer blocks (40)
        num_heads: Number of attention heads (40)
        intermediate_size: FFN intermediate size (13824)
        in_channels: Input latent channels (16)
        out_channels: Output latent channels (16)
        text_dim: Text embedding dimension (4096)
        patch_size: Patch size (1, 2, 2)
        freq_dim: Frequency dimension for time embedding (256)
        latent_num_frames: Number of latent frames
        latent_height: Latent height
        latent_width: Latent width
        dtype: Data type
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        num_layers: int = 40,
        num_heads: int = 40,
        intermediate_size: int = 13824,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        freq_dim: int = 256,
        latent_num_frames: int = 21,
        latent_height: int = 90,
        latent_width: int = 160,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.patch_size = patch_size
        self.freq_dim = freq_dim
        self.latent_num_frames = latent_num_frames
        self.latent_height = latent_height
        self.latent_width = latent_width

        pt, ph, pw = patch_size
        self.patchify = ShardedConv3d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
            padding=(0, 0, 0),
            dtype=dtype,
        )

        self.time_embedding_in = ShardedModule()
        self.time_embedding_in.weight = ShardedTensor(
            shape=(freq_dim, hidden_size), shardable={}, dtype=dtype, name="time_embed_in"
        )

        self.time_embedding_out = ShardedModule()
        self.time_embedding_out.weight = ShardedTensor(
            shape=(hidden_size, hidden_size), shardable={}, dtype=dtype, name="time_embed_out"
        )

        self.time_projection = ShardedModule()
        self.time_projection.weight = ShardedTensor(
            shape=(hidden_size, 6 * hidden_size), shardable={}, dtype=dtype, name="time_proj"
        )

        self.blocks = [
            ShardedWanDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                text_dim=text_dim,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

        self.norm_final = ShardedLayerNorm(hidden_size, elementwise_affine=False, dtype=dtype)

        self.unpatchify = ShardedModule()
        self.unpatchify.weight = ShardedTensor(
            shape=(hidden_size, out_channels), shardable={}, dtype=dtype, name="unpatchify_weight"
        )

    def forward(
        self,
        latent: ShardedTensor,
        text_embed: ShardedTensor,
        time_embed: ShardedTensor,
    ) -> ShardedTensor:
        """DiT forward."""
        patches = self.patchify(latent)

        batch = patches.shape[0] if len(patches.shape) >= 1 else 1
        t = patches.shape[2] if len(patches.shape) >= 3 else self.latent_num_frames
        h = patches.shape[3] if len(patches.shape) >= 4 else self.latent_height // 2
        w = patches.shape[4] if len(patches.shape) >= 5 else self.latent_width // 2
        seq_len = t * h * w

        hidden = ShardedTensor(
            shape=(batch, seq_len, self.hidden_size),
            shardable={},
            dtype=patches.dtype,
            name="hidden",
        )
        hidden._op_history = patches._op_history
        self._activations["patches"] = hidden

        time_hidden = time_embed @ self.time_embedding_in.weight
        time_hidden = silu(time_hidden)
        time_hidden = time_hidden @ self.time_embedding_out.weight
        self._activations["time_hidden"] = time_hidden

        for i, block in enumerate(self.blocks):
            hidden = block(hidden, text_embed, time_hidden)
            self._activations[f"block_{i}"] = hidden

        hidden = self.norm_final(hidden)

        output = hidden @ self.unpatchify.weight

        return output


class ShardedWanVAE(ShardedModule):
    """Wan 3D Causal VAE.

    Simplified version using ShardedVAE components.

    Args:
        in_channels: Input channels (3 for RGB)
        latent_channels: Latent channels (16 for Wan)
        block_out_channels: Channel progression
        dtype: Data type
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        dtype: str = "fp16",
    ):
        super().__init__()

        from llm_perf.modeling.models.vision_models import ShardedVAEEncoder, ShardedVAEDecoder

        self.encoder = ShardedVAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            use_3d=True,
            dtype=dtype,
        )

        self.decoder = ShardedVAEDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            use_3d=True,
            dtype=dtype,
        )

    def encode(self, video: ShardedTensor) -> ShardedTensor:
        """Encode video to latent."""
        return self.encoder(video)

    def decode(self, latent: ShardedTensor) -> ShardedTensor:
        """Decode latent to video."""
        return self.decoder(latent)

    def forward(self, video: ShardedTensor) -> ShardedTensor:
        """VAE forward."""
        latent = self.encode(video)
        reconstructed = self.decode(latent)
        return reconstructed

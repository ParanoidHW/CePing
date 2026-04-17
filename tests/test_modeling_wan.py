"""Tests for Wan video generation models using Sharded interface."""

import pytest
from llm_perf.modeling import (
    ShardedTensor,
    ShardedLayerNorm,
    ShardedT5Block,
    ShardedWanTextEncoder,
    ShardedWanDiTBlock,
    ShardedWanDiT,
    ShardedWanVAE,
)


class TestShardedLayerNorm:
    """Test LayerNorm."""

    def test_layernorm_creation(self):
        """Test creating layer norm."""
        norm = ShardedLayerNorm(hidden_size=4096, elementwise_affine=True)

        assert norm.hidden_size == 4096
        assert norm.elementwise_affine == True

    def test_layernorm_params(self):
        """Test layer norm params count."""
        norm = ShardedLayerNorm(4096, elementwise_affine=True)

        expected = 4096 * 2
        assert norm.params_count() == expected

    def test_layernorm_no_affine(self):
        """Test layer norm without affine."""
        norm = ShardedLayerNorm(4096, elementwise_affine=False)

        assert norm.params_count() == 0

    def test_layernorm_forward(self):
        """Test layer norm forward."""
        norm = ShardedLayerNorm(4096)

        input_tensor = ShardedTensor(shape=(1, 512, 4096))
        output = norm(input_tensor)

        assert output.shape == (1, 512, 4096)


class TestShardedT5Block:
    """Test T5 Block."""

    def test_t5_block_creation(self):
        """Test creating T5 block."""
        block = ShardedT5Block(
            hidden_size=4096,
            num_heads=64,
            intermediate_size=10240,
        )

        assert block.hidden_size == 4096
        assert block.num_heads == 64

    def test_t5_block_submodules(self):
        """Test T5 block has correct submodules."""
        block = ShardedT5Block(4096, 64, 10240)

        assert "self_attn" in block._submodules
        assert "ffn_gate" in block._submodules

    def test_t5_block_forward(self):
        """Test T5 block forward."""
        block = ShardedT5Block(4096, 64, 10240)

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = block(hidden)

        assert output.shape == (1, 512, 4096)


class TestShardedWanTextEncoder:
    """Test Wan Text Encoder."""

    def test_encoder_creation(self):
        """Test creating text encoder."""
        encoder = ShardedWanTextEncoder(
            vocab_size=256384,
            hidden_size=4096,
            num_layers=24,
            num_heads=64,
            intermediate_size=10240,
        )

        assert encoder.vocab_size == 256384
        assert encoder.hidden_size == 4096
        assert encoder.num_layers == 24

    def test_encoder_layers(self):
        """Test encoder has correct number of layers."""
        encoder = ShardedWanTextEncoder(num_layers=4)

        assert len(encoder.layers) == 4

    def test_encoder_params_count(self):
        """Test encoder params count."""
        encoder = ShardedWanTextEncoder(
            vocab_size=256384,
            hidden_size=4096,
            num_layers=2,
            num_heads=64,
            intermediate_size=10240,
        )

        embedding_params = 256384 * 4096
        layer_params = encoder.layers[0].params_count()
        final_norm_params = 4096 * 2

        expected = embedding_params + layer_params * 2 + final_norm_params
        assert encoder.params_count() == expected

    def test_encoder_forward(self):
        """Test encoder forward."""
        encoder = ShardedWanTextEncoder(num_layers=2)

        input_ids = ShardedTensor(shape=(1, 512))
        output = encoder(input_ids)

        assert output.shape == (1, 512, 4096)


class TestShardedWanDiTBlock:
    """Test Wan DiT Block."""

    def test_dit_block_creation(self):
        """Test creating DiT block."""
        block = ShardedWanDiTBlock(
            hidden_size=5120,
            num_heads=40,
            intermediate_size=13824,
            text_dim=4096,
        )

        assert block.hidden_size == 5120
        assert block.num_heads == 40

    def test_dit_block_params(self):
        """Test DiT block params count."""
        block = ShardedWanDiTBlock(5120, 40, 13824)

        assert block.params_count() > 0

    def test_dit_block_forward(self):
        """Test DiT block forward."""
        block = ShardedWanDiTBlock(5120, 40, 13824)

        hidden = ShardedTensor(shape=(1, 1890, 5120))
        text_embed = ShardedTensor(shape=(1, 512, 4096))
        time_embed = ShardedTensor(shape=(1, 256))

        output = block(hidden, text_embed, time_embed)

        assert output.shape == (1, 1890, 5120)


class TestShardedWanDiT:
    """Test Wan DiT Model."""

    def test_dit_creation(self):
        """Test creating DiT model."""
        dit = ShardedWanDiT(
            hidden_size=5120,
            num_layers=40,
            num_heads=40,
            intermediate_size=13824,
        )

        assert dit.hidden_size == 5120
        assert dit.num_layers == 40

    def test_dit_components(self):
        """Test DiT has correct components."""
        dit = ShardedWanDiT(num_layers=4)

        assert "patchify" in dit._submodules
        assert len(dit.blocks) == 4

    def test_dit_params_count(self):
        """Test DiT params count."""
        dit = ShardedWanDiT(num_layers=2)

        assert dit.params_count() > 0

    def test_dit_forward(self):
        """Test DiT forward."""
        dit = ShardedWanDiT(
            hidden_size=5120,
            num_layers=2,
            num_heads=40,
            intermediate_size=13824,
            in_channels=16,
            latent_num_frames=21,
            latent_height=90,
            latent_width=160,
        )

        latent = ShardedTensor(shape=(1, 16, 21, 90, 160))
        text_embed = ShardedTensor(shape=(1, 512, 4096))
        time_embed = ShardedTensor(shape=(1, 256))

        output = dit(latent, text_embed, time_embed)

        assert output.shape[0] == 1


class TestShardedWanVAE:
    """Test Wan VAE."""

    def test_wan_vae_creation(self):
        """Test creating Wan VAE."""
        vae = ShardedWanVAE(
            in_channels=3,
            latent_channels=16,
            block_out_channels=(128, 256, 512, 512),
        )

        assert vae.encoder is not None
        assert vae.decoder is not None

    def test_wan_vae_encode_decode(self):
        """Test Wan VAE encode and decode."""
        vae = ShardedWanVAE(
            in_channels=3,
            latent_channels=16,
            block_out_channels=(128, 256),
        )

        video = ShardedTensor(shape=(1, 3, 16, 64, 64))
        latent = vae.encode(video)

        assert latent.shape[1] == 32

        reconstructed = vae.decode(latent)
        assert reconstructed.shape[1] == 3


class TestWanIntegration:
    """Integration tests for Wan models."""

    def test_text_encoder_to_dit(self):
        """Test text encoder output to DiT."""
        encoder = ShardedWanTextEncoder(
            vocab_size=256384,
            hidden_size=4096,
            num_layers=2,
        )

        dit = ShardedWanDiT(
            hidden_size=5120,
            num_layers=2,
            text_dim=4096,
        )

        input_ids = ShardedTensor(shape=(1, 512))
        text_embed = encoder(input_ids)

        assert text_embed.shape == (1, 512, 4096)

"""Tests for vision/video models using Sharded interface."""

import pytest
from llm_perf.modeling import (
    ShardedTensor,
    ShardedConv2d,
    ShardedConv3d,
    ShardedGroupNorm,
    ShardedResNetBlock2d,
    ShardedResNetBlock3d,
    ShardedVAEEncoder,
    ShardedVAEDecoder,
    ShardedVAE,
    conv2d,
    conv3d,
)


class TestShardedConv2d:
    """Test ShardedConv2d."""

    def test_conv2d_creation(self):
        """Test creating conv2d layer."""
        conv = ShardedConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        assert conv.in_channels == 64
        assert conv.out_channels == 128
        assert conv.kernel_size == (3, 3)

    def test_conv2d_params(self):
        """Test conv2d params count."""
        conv = ShardedConv2d(64, 128, (3, 3))

        expected = 128 * 64 * 3 * 3
        assert conv.params_count() == expected

    def test_conv2d_forward(self):
        """Test conv2d forward."""
        conv = ShardedConv2d(64, 128, (3, 3), (1, 1), (1, 1))

        input_tensor = ShardedTensor(shape=(1, 64, 256, 256))
        output = conv(input_tensor)

        assert output.shape == (1, 128, 256, 256)

    def test_conv2d_stride(self):
        """Test conv2d with stride."""
        conv = ShardedConv2d(64, 128, (3, 3), (2, 2), (1, 1))

        input_tensor = ShardedTensor(shape=(1, 64, 256, 256))
        output = conv(input_tensor)

        assert output.shape == (1, 128, 128, 128)

    def test_conv2d_no_shardable(self):
        """Test conv2d has no shardable dims."""
        conv = ShardedConv2d(64, 128, (3, 3))

        assert conv.weight.shardable == {}


class TestShardedConv3d:
    """Test ShardedConv3d."""

    def test_conv3d_creation(self):
        """Test creating conv3d layer."""
        conv = ShardedConv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )

        assert conv.in_channels == 64
        assert conv.out_channels == 128

    def test_conv3d_params(self):
        """Test conv3d params count."""
        conv = ShardedConv3d(64, 128, (3, 3, 3))

        expected = 128 * 64 * 3 * 3 * 3
        assert conv.params_count() == expected

    def test_conv3d_forward(self):
        """Test conv3d forward."""
        conv = ShardedConv3d(64, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1))

        input_tensor = ShardedTensor(shape=(1, 64, 16, 256, 256))
        output = conv(input_tensor)

        assert output.shape == (1, 128, 16, 256, 256)

    def test_conv3d_stride_spatial(self):
        """Test conv3d with spatial stride."""
        conv = ShardedConv3d(64, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1))

        input_tensor = ShardedTensor(shape=(1, 64, 16, 256, 256))
        output = conv(input_tensor)

        assert output.shape == (1, 128, 16, 128, 128)


class TestShardedGroupNorm:
    """Test ShardedGroupNorm."""

    def test_groupnorm_creation(self):
        """Test creating groupnorm layer."""
        norm = ShardedGroupNorm(num_groups=32, num_channels=512)

        assert norm.num_groups == 32
        assert norm.num_channels == 512

    def test_groupnorm_params(self):
        """Test groupnorm params count."""
        norm = ShardedGroupNorm(32, 512)

        expected = 512 * 2
        assert norm.params_count() == expected

    def test_groupnorm_forward(self):
        """Test groupnorm forward."""
        norm = ShardedGroupNorm(32, 512)

        input_tensor = ShardedTensor(shape=(1, 512, 256, 256))
        output = norm(input_tensor)

        assert output.shape == (1, 512, 256, 256)


class TestFunctionalConv:
    """Test functional conv2d/conv3d."""

    def test_conv2d_functional(self):
        """Test functional conv2d."""
        input_tensor = ShardedTensor(shape=(1, 64, 256, 256))
        weight = ShardedTensor(shape=(128, 64, 3, 3))

        output = conv2d(input_tensor, weight, stride=(1, 1), padding=(1, 1))

        assert output.shape == (1, 128, 256, 256)

    def test_conv3d_functional(self):
        """Test functional conv3d."""
        input_tensor = ShardedTensor(shape=(1, 64, 16, 256, 256))
        weight = ShardedTensor(shape=(128, 64, 3, 3, 3))

        output = conv3d(input_tensor, weight, stride=(1, 1, 1), padding=(1, 1, 1))

        assert output.shape == (1, 128, 16, 256, 256)


class TestShardedResNetBlock:
    """Test ResNet blocks."""

    def test_resnet_block_2d_creation(self):
        """Test creating 2D ResNet block."""
        block = ShardedResNetBlock2d(128, 256)

        assert block.in_channels == 128
        assert block.out_channels == 256

    def test_resnet_block_2d_forward(self):
        """Test 2D ResNet block forward."""
        block = ShardedResNetBlock2d(128, 256)

        input_tensor = ShardedTensor(shape=(1, 128, 64, 64))
        output = block(input_tensor)

        assert output.shape == (1, 256, 64, 64)

    def test_resnet_block_3d_creation(self):
        """Test creating 3D ResNet block."""
        block = ShardedResNetBlock3d(128, 256)

        assert block.in_channels == 128
        assert block.out_channels == 256

    def test_resnet_block_3d_forward(self):
        """Test 3D ResNet block forward."""
        block = ShardedResNetBlock3d(128, 256)

        input_tensor = ShardedTensor(shape=(1, 128, 16, 64, 64))
        output = block(input_tensor)

        assert output.shape == (1, 256, 16, 64, 64)

    def test_resnet_block_same_channels(self):
        """Test ResNet block with same channels (no shortcut)."""
        block = ShardedResNetBlock2d(128, 128)

        input_tensor = ShardedTensor(shape=(1, 128, 64, 64))
        output = block(input_tensor)

        assert output.shape == (1, 128, 64, 64)
        assert "shortcut" not in block._weights


class TestShardedVAEEncoder:
    """Test VAE Encoder."""

    def test_encoder_2d_creation(self):
        """Test creating 2D encoder."""
        encoder = ShardedVAEEncoder(
            in_channels=3,
            latent_channels=4,
            block_out_channels=(128, 256, 512),
            use_3d=False,
        )

        assert encoder.in_channels == 3
        assert encoder.latent_channels == 4

    def test_encoder_3d_creation(self):
        """Test creating 3D encoder."""
        encoder = ShardedVAEEncoder(
            in_channels=3,
            latent_channels=4,
            block_out_channels=(128, 256, 512),
            use_3d=True,
        )

        assert encoder.use_3d

    def test_encoder_params_count(self):
        """Test encoder params count."""
        encoder = ShardedVAEEncoder(3, 4, (128, 256), use_3d=False)

        assert encoder.params_count() > 0

    def test_encoder_forward_2d(self):
        """Test 2D encoder forward."""
        encoder = ShardedVAEEncoder(
            in_channels=3,
            latent_channels=4,
            block_out_channels=(128, 256),
            use_3d=False,
        )

        input_tensor = ShardedTensor(shape=(1, 3, 128, 128))
        latent = encoder(input_tensor)

        assert latent.shape[1] == 8


class TestShardedVAEDecoder:
    """Test VAE Decoder."""

    def test_decoder_2d_creation(self):
        """Test creating 2D decoder."""
        decoder = ShardedVAEDecoder(
            out_channels=3,
            latent_channels=4,
            block_out_channels=(128, 256, 512),
            use_3d=False,
        )

        assert decoder.out_channels == 3
        assert decoder.latent_channels == 4

    def test_decoder_params_count(self):
        """Test decoder params count."""
        decoder = ShardedVAEDecoder(3, 4, (128, 256), use_3d=False)

        assert decoder.params_count() > 0


class TestShardedVAE:
    """Test VAE model."""

    def test_vae_creation(self):
        """Test creating VAE."""
        vae = ShardedVAE(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(128, 256, 512),
            use_3d=False,
        )

        assert vae.in_channels == 3
        assert vae.out_channels == 3

    def test_vae_components(self):
        """Test VAE has encoder and decoder."""
        vae = ShardedVAE()

        assert "encoder" in vae._submodules
        assert "decoder" in vae._submodules

    def test_vae_params_count(self):
        """Test VAE params count."""
        vae = ShardedVAE()

        encoder_params = vae.encoder.params_count()
        decoder_params = vae.decoder.params_count()

        assert vae.params_count() == encoder_params + decoder_params

    def test_vae_encode_decode(self):
        """Test VAE encode and decode."""
        vae = ShardedVAE(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(128, 256),
            use_3d=False,
        )

        video = ShardedTensor(shape=(1, 3, 64, 64))
        latent = vae.encode(video)

        assert latent.shape[1] == 8

        reconstructed = vae.decode(latent)
        assert reconstructed.shape[1] == 3

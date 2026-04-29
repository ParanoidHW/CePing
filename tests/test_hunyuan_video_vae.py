"""Tests for HunyuanVideo VAE."""

from llm_perf.modeling.base.vae_3d import (
    ShardedVideoVAEEncoder,
    ShardedVideoVAEDecoder,
    ShardedVideoVAE,
)
from llm_perf.modeling.tensor import ShardedTensor


class TestShardedVideoVAEEncoder:
    """Tests for ShardedVideoVAEEncoder."""

    def test_weight_shapes(self):
        """Test encoder weight shapes."""
        in_channels = 3
        latent_channels = 16
        base_channels = 128
        channel_multipliers = (1, 2, 4, 4)
        num_res_blocks = 2

        encoder = ShardedVideoVAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            dtype="bf16",
        )

        assert encoder.conv_in_weight.shape == (base_channels, in_channels, 3, 3, 3)
        assert encoder.conv_in_weight.shardable == {0: "tp"}

        final_channels = base_channels * channel_multipliers[-1]
        assert encoder.conv_out_weight.shape == (latent_channels, final_channels, 3, 3, 3)
        assert encoder.conv_out_weight.shardable == {0: "tp"}

    def test_forward_output_shape(self):
        """Test encoder forward output shape (compression ratio)."""
        batch_size = 1
        in_channels = 3
        frames = 16
        height = 256
        width = 256

        encoder = ShardedVideoVAEEncoder(
            in_channels=in_channels,
            latent_channels=16,
            base_channels=128,
            channel_multipliers=(1, 2, 4, 4),
            num_res_blocks=2,
            dtype="bf16",
        )

        video = ShardedTensor(
            shape=(batch_size, in_channels, frames, height, width),
            shardable={},
            dtype="bf16",
            name="video",
        )

        latent = encoder.forward(video)

        expected_frames = frames // 4
        expected_height = height // 8
        expected_width = width // 8

        assert latent.shape[0] == batch_size
        assert latent.shape[1] == 16
        assert latent.shape[2] == expected_frames
        assert latent.shape[3] == expected_height
        assert latent.shape[4] == expected_width
        assert latent.dtype == "bf16"

    def test_parameter_count(self):
        """Test encoder parameter count is reasonable."""
        encoder = ShardedVideoVAEEncoder(
            in_channels=3,
            latent_channels=16,
            base_channels=128,
            channel_multipliers=(1, 2, 4, 4),
            num_res_blocks=2,
            dtype="bf16",
        )

        total_params = 0
        total_params += encoder.conv_in_weight.shape[0] * encoder.conv_in_weight.shape[1] * 27

        for key, param in encoder.down_blocks.items():
            total_params += param.shape[0] * param.shape[1] * 27

        total_params += encoder.conv_out_weight.shape[0] * encoder.conv_out_weight.shape[1] * 27

        assert total_params > 0
        assert total_params < 100_000_000


class TestShardedVideoVAEDecoder:
    """Tests for ShardedVideoVAEDecoder."""

    def test_weight_shapes(self):
        """Test decoder weight shapes."""
        latent_channels = 16
        out_channels = 3
        base_channels = 128
        channel_multipliers = (1, 2, 4, 4)
        num_res_blocks = 3

        decoder = ShardedVideoVAEDecoder(
            latent_channels=latent_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            dtype="bf16",
        )

        final_channels = base_channels * channel_multipliers[-1]
        assert decoder.conv_in_weight.shape == (final_channels, latent_channels, 3, 3, 3)
        assert decoder.conv_in_weight.shardable == {0: "tp"}

        assert decoder.conv_out_weight.shape == (out_channels, base_channels, 3, 3, 3)
        assert decoder.conv_out_weight.shardable == {0: "tp"}

    def test_forward_output_shape(self):
        """Test decoder forward output shape (reconstruction ratio)."""
        batch_size = 1
        latent_channels = 16
        frames_latent = 4
        height_latent = 16
        width_latent = 16

        decoder = ShardedVideoVAEDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            base_channels=128,
            channel_multipliers=(1, 2, 4, 4),
            num_res_blocks=3,
            dtype="bf16",
        )

        latent = ShardedTensor(
            shape=(batch_size, latent_channels, frames_latent, height_latent, width_latent),
            shardable={},
            dtype="bf16",
            name="latent",
        )

        video = decoder.forward(latent)

        assert video.shape[0] == batch_size
        assert video.shape[1] == 3
        assert video.dtype == "bf16"

    def test_parameter_count(self):
        """Test decoder parameter count is reasonable."""
        decoder = ShardedVideoVAEDecoder(
            latent_channels=16,
            out_channels=3,
            base_channels=128,
            channel_multipliers=(1, 2, 4, 4),
            num_res_blocks=3,
            dtype="bf16",
        )

        total_params = 0
        total_params += decoder.conv_in_weight.shape[0] * decoder.conv_in_weight.shape[1] * 27

        for key, param in decoder.up_blocks.items():
            if "up" in key:
                total_params += param.shape[0] * param.shape[1] * 64
            else:
                total_params += param.shape[0] * param.shape[1] * 27

        total_params += decoder.conv_out_weight.shape[0] * decoder.conv_out_weight.shape[1] * 27

        assert total_params > 0
        assert total_params < 200_000_000


class TestShardedVideoVAE:
    """Tests for complete VAE."""

    def test_initialization(self):
        """Test VAE can be initialized with configs."""
        encoder_config = {
            "in_channels": 3,
            "latent_channels": 16,
            "base_channels": 128,
        }
        decoder_config = {
            "latent_channels": 16,
            "out_channels": 3,
            "base_channels": 128,
        }

        vae = ShardedVideoVAE(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

        assert vae.encoder is not None
        assert vae.decoder is not None
        assert isinstance(vae.encoder, ShardedVideoVAEEncoder)
        assert isinstance(vae.decoder, ShardedVideoVAEDecoder)

    def test_encode(self):
        """Test encode method."""
        vae = ShardedVideoVAE()

        video = ShardedTensor(
            shape=(1, 3, 16, 256, 256),
            shardable={},
            dtype="bf16",
            name="video",
        )

        latent = vae.encode(video)

        assert latent.shape[0] == 1
        assert latent.shape[1] == 16
        assert latent.dtype == "bf16"

    def test_decode(self):
        """Test decode method."""
        vae = ShardedVideoVAE()

        latent = ShardedTensor(
            shape=(1, 16, 4, 16, 16),
            shardable={},
            dtype="bf16",
            name="latent",
        )

        video = vae.decode(latent)

        assert video.shape[0] == 1
        assert video.shape[1] == 3
        assert video.dtype == "bf16"

    def test_forward(self):
        """Test full forward pass."""
        vae = ShardedVideoVAE()

        video = ShardedTensor(
            shape=(1, 3, 16, 256, 256),
            shardable={},
            dtype="bf16",
            name="video",
        )

        reconstructed = vae.forward(video)

        assert reconstructed.dtype == "bf16"


def test_encoder_decoder_consistency():
    """Test that encoder and decoder have matching configurations."""
    encoder = ShardedVideoVAEEncoder(
        latent_channels=16,
        base_channels=128,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
        dtype="bf16",
    )

    decoder = ShardedVideoVAEDecoder(
        latent_channels=16,
        base_channels=128,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=3,
        dtype="bf16",
    )

    assert encoder.latent_channels == decoder.latent_channels
    assert encoder.base_channels == decoder.base_channels
    assert encoder.channel_multipliers == decoder.channel_multipliers


def test_import_from_module():
    """Test that VAE classes can be imported from base module."""
    from llm_perf.modeling.base.vae_3d import (
        ShardedVideoVAEEncoder,
        ShardedVideoVAEDecoder,
        ShardedVideoVAE,
    )

    assert ShardedVideoVAEEncoder is not None
    assert ShardedVideoVAEDecoder is not None
    assert ShardedVideoVAE is not None
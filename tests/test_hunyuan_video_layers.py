"""Tests for HunyuanVideo base layers."""

from llm_perf.modeling.hunyuan_video.layers import (
    ShardedModulateDiT,
    ShardedPatchEmbed3D,
    ShardedTimestepEmbedder,
)
from llm_perf.modeling.tensor import ShardedTensor


class TestShardedModulateDiT:
    """Tests for ShardedModulateDiT."""

    def test_weight_shape(self):
        """Test modulation weight shape."""
        hidden_size = 3072
        modulate = ShardedModulateDiT(hidden_size=hidden_size, dtype="bf16")

        assert modulate.modulation_weight.shape == (hidden_size, hidden_size * 6)
        assert modulate.modulation_weight.shardable == {1: "tp"}
        assert modulate.modulation_weight.dtype == "bf16"

    def test_forward_output_shapes(self):
        """Test forward output shapes."""
        hidden_size = 3072
        batch_size = 2

        modulate = ShardedModulateDiT(hidden_size=hidden_size, dtype="bf16")

        vec = ShardedTensor(
            shape=(batch_size, hidden_size),
            shardable={},
            dtype="bf16",
            name="vec",
        )

        shift1, scale1, gate1, shift2, scale2, gate2 = modulate.forward(vec)

        for output in [shift1, scale1, gate1, shift2, scale2, gate2]:
            assert output.shape == (batch_size, hidden_size)
            assert output.dtype == "bf16"


class TestShardedPatchEmbed3D:
    """Tests for ShardedPatchEmbed3D."""

    def test_weight_shape(self):
        """Test projection weight shape."""
        in_channels = 3
        hidden_size = 3072
        patch_size = (1, 2, 2)

        patch_embed = ShardedPatchEmbed3D(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            dtype="bf16",
        )

        assert patch_embed.proj_weight.shape == (hidden_size, in_channels, *patch_size)
        assert patch_embed.proj_weight.shardable == {0: "tp"}
        assert patch_embed.proj_weight.dtype == "bf16"

    def test_forward_output_shape(self):
        """Test forward output shape."""
        in_channels = 3
        hidden_size = 3072
        patch_size = (1, 2, 2)
        batch_size = 1
        frames = 21
        height = 180
        width = 320

        patch_embed = ShardedPatchEmbed3D(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            dtype="bf16",
        )

        video = ShardedTensor(
            shape=(batch_size, in_channels, frames, height, width),
            shardable={},
            dtype="bf16",
            name="video",
        )

        tokens = patch_embed.forward(video)

        pt, ph, pw = patch_size
        expected_seq = (frames // pt) * (height // ph) * (width // pw)

        assert tokens.shape == (batch_size, expected_seq, hidden_size)
        assert tokens.dtype == "bf16"


class TestShardedTimestepEmbedder:
    """Tests for ShardedTimestepEmbedder."""

    def test_weight_shapes(self):
        """Test MLP weight shapes."""
        hidden_size = 3072
        frequency_embedding_size = 256

        embedder = ShardedTimestepEmbedder(
            hidden_size=hidden_size,
            frequency_embedding_size=frequency_embedding_size,
            dtype="bf16",
        )

        assert embedder.mlp_fc1_weight.shape == (frequency_embedding_size * 4, hidden_size)
        assert embedder.mlp_fc1_weight.shardable == {1: "tp"}

        assert embedder.mlp_fc2_weight.shape == (hidden_size, hidden_size)
        assert embedder.mlp_fc2_weight.shardable == {1: "tp"}

    def test_forward_output_shape(self):
        """Test forward output shape."""
        hidden_size = 3072
        batch_size = 2

        embedder = ShardedTimestepEmbedder(hidden_size=hidden_size, dtype="bf16")

        timestep = ShardedTensor(
            shape=(batch_size,),
            shardable={},
            dtype="bf16",
            name="timestep",
        )

        embed = embedder.forward(timestep)

        assert embed.shape == (batch_size, hidden_size)
        assert embed.dtype == "bf16"


def test_all_layers_creation():
    """Test that all layers can be instantiated."""
    layers = [
        ShardedModulateDiT(hidden_size=3072, dtype="bf16"),
        ShardedPatchEmbed3D(in_channels=3, hidden_size=3072, dtype="bf16"),
        ShardedTimestepEmbedder(hidden_size=3072, dtype="bf16"),
    ]

    for layer in layers:
        assert hasattr(layer, "_submodule_name")
        assert isinstance(layer._submodule_name, str)


def test_import_from_module():
    """Test that layers can be imported from hunyuan_video module."""
    from llm_perf.modeling.hunyuan_video import (
        ShardedModulateDiT,
        ShardedPatchEmbed3D,
        ShardedTimestepEmbedder,
    )

    assert ShardedModulateDiT is not None
    assert ShardedPatchEmbed3D is not None
    assert ShardedTimestepEmbedder is not None
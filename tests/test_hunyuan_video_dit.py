"""Tests for HunyuanVideo complete DiT model."""

from llm_perf.modeling.models.hunyuan_video import ShardedHYVideoDiT
from llm_perf.modeling.tensor import ShardedTensor


class TestShardedHYVideoDiT:
    """Tests for ShardedHYVideoDiT."""

    def test_weight_shapes(self):
        """Test all weight parameter shapes."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128
        mlp_width_ratio = 4.0
        in_channels = 16
        out_channels = 16
        text_states_dim = 4096
        patch_size = (1, 2, 2)

        model = ShardedHYVideoDiT(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            double_blocks_depth=20,
            single_blocks_depth=40,
            mlp_width_ratio=mlp_width_ratio,
            in_channels=in_channels,
            out_channels=out_channels,
            text_states_dim=text_states_dim,
            patch_size=patch_size,
            qk_norm=True,
            dtype="bf16",
        )

        assert model.txt_in_weight.shape == (text_states_dim, hidden_size)
        assert model.txt_in_weight.shardable == {0: "tp"}

        assert model.vector_in_weight.shape == (hidden_size, hidden_size)
        assert model.vector_in_weight.shardable == {1: "tp"}

        assert model.merge_weight.shape == (hidden_size, hidden_size)
        assert model.merge_weight.shardable == {0: "tp"}

        assert model.final_layer_norm.shape == (hidden_size,)
        assert model.final_layer_norm.shardable == {}

        pt, ph, pw = patch_size
        expected_out_features = pt * ph * pw * out_channels
        assert model.final_layer_linear.shape == (hidden_size, expected_out_features)
        assert model.final_layer_linear.shardable == {1: "tp"}

    def test_blocks_count(self):
        """Test correct number of blocks."""
        model = ShardedHYVideoDiT(
            hidden_size=3072,
            heads_num=24,
            head_dim=128,
            double_blocks_depth=20,
            single_blocks_depth=40,
            dtype="bf16",
        )

        assert len(model.double_blocks) == 20
        assert len(model.single_blocks) == 40

        for block in model.double_blocks:
            assert hasattr(block, "_submodule_name")
            assert block._submodule_name == "double_stream_block"

        for block in model.single_blocks:
            assert hasattr(block, "_submodule_name")
            assert block._submodule_name == "single_stream_block"

    def test_params_count_full_model(self):
        """Test parameter count for full model (~7.1B)."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128
        mlp_width_ratio = 4.0
        in_channels = 16
        out_channels = 16
        text_states_dim = 4096
        patch_size = (1, 2, 2)

        model = ShardedHYVideoDiT(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            double_blocks_depth=20,
            single_blocks_depth=40,
            mlp_width_ratio=mlp_width_ratio,
            in_channels=in_channels,
            out_channels=out_channels,
            text_states_dim=text_states_dim,
            patch_size=patch_size,
            dtype="bf16",
        )

        params_count = model.params_count()
        params_gb = params_count * 2 / 1e9

        assert 10e9 < params_count < 20e9
        assert 20.0 < params_gb < 40.0

    def test_forward_output_shape(self):
        """Test forward output shape."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128
        in_channels = 16
        out_channels = 16
        text_states_dim = 4096
        patch_size = (1, 2, 2)
        batch_size = 1
        frames = 21
        height = 180
        width = 320
        txt_seq_len = 256

        model = ShardedHYVideoDiT(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_states_dim=text_states_dim,
            patch_size=patch_size,
            dtype="bf16",
        )

        video = ShardedTensor(
            shape=(batch_size, in_channels, frames, height, width),
            shardable={},
            dtype="bf16",
            name="video",
        )

        timestep = ShardedTensor(
            shape=(batch_size,),
            shardable={},
            dtype="bf16",
            name="timestep",
        )

        text_states = ShardedTensor(
            shape=(batch_size, txt_seq_len, text_states_dim),
            shardable={},
            dtype="bf16",
            name="text_states",
        )

        output = model.forward(video, timestep, text_states)

        pt, ph, pw = patch_size
        expected_shape = (batch_size, out_channels, frames // pt, height // ph, width // pw)

        assert output.shape == expected_shape
        assert output.dtype == "bf16"

    def test_forward_with_guidance(self):
        """Test forward with guidance vector."""
        hidden_size = 3072
        batch_size = 1
        frames = 21
        height = 180
        width = 320
        txt_seq_len = 256
        in_channels = 16
        out_channels = 16
        text_states_dim = 4096

        model = ShardedHYVideoDiT(
            hidden_size=hidden_size,
            heads_num=24,
            head_dim=128,
            in_channels=in_channels,
            out_channels=out_channels,
            text_states_dim=text_states_dim,
            dtype="bf16",
        )

        video = ShardedTensor(
            shape=(batch_size, in_channels, frames, height, width),
            shardable={},
            dtype="bf16",
            name="video",
        )

        timestep = ShardedTensor(
            shape=(batch_size,),
            shardable={},
            dtype="bf16",
            name="timestep",
        )

        text_states = ShardedTensor(
            shape=(batch_size, txt_seq_len, text_states_dim),
            shardable={},
            dtype="bf16",
            name="text_states",
        )

        guidance_vec = ShardedTensor(
            shape=(batch_size, hidden_size),
            shardable={},
            dtype="bf16",
            name="guidance_vec",
        )

        output = model.forward(video, timestep, text_states, guidance_vec=guidance_vec)

        pt, ph, pw = model.patch_size
        expected_shape = (batch_size, out_channels, frames // pt, height // ph, width // pw)

        assert output.shape == expected_shape

    def test_configurable_depth(self):
        """Test model with configurable depth."""
        model_small = ShardedHYVideoDiT(
            hidden_size=1024,
            heads_num=8,
            head_dim=128,
            double_blocks_depth=4,
            single_blocks_depth=8,
            dtype="bf16",
        )

        assert len(model_small.double_blocks) == 4
        assert len(model_small.single_blocks) == 8

        params_small = model_small.params_count()
        assert params_small < 1e9

    def test_submodule_registration(self):
        """Test that submodules are correctly registered."""
        model = ShardedHYVideoDiT(
            hidden_size=3072,
            double_blocks_depth=20,
            single_blocks_depth=40,
            dtype="bf16",
        )

        assert "img_in" in model._submodules
        assert "time_in" in model._submodules

        assert any("double_blocks" in name for name in model._submodules)
        assert any("single_blocks" in name for name in model._submodules)

        double_block_count = sum(1 for name in model._submodules if "double_blocks" in name)
        single_block_count = sum(1 for name in model._submodules if "single_blocks" in name)

        assert double_block_count == 20
        assert single_block_count == 40


def test_import_from_module():
    """Test that model can be imported from hunyuan_video module."""
    from llm_perf.modeling.models.hunyuan_video import ShardedHYVideoDiT

    model = ShardedHYVideoDiT(hidden_size=3072, dtype="bf16")
    assert model is not None
    assert hasattr(model, "_submodule_name")


def test_model_creation_default():
    """Test model creation with default parameters."""
    model = ShardedHYVideoDiT()

    assert model.hidden_size == 3072
    assert model.heads_num == 24
    assert model.head_dim == 128
    assert model.double_blocks_depth == 20
    assert model.single_blocks_depth == 40
    assert model.mlp_width_ratio == 4.0
    assert model.in_channels == 16
    assert model.out_channels == 16
    assert model.text_states_dim == 4096
    assert model.patch_size == (1, 2, 2)
    assert model.qk_norm is True
    assert model.dtype == "bf16"
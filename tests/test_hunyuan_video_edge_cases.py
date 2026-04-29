"""Edge case tests for HunyuanVideo layers and DiT blocks."""

from llm_perf.modeling.hunyuan_video.layers import (
    ShardedModulateDiT,
    ShardedPatchEmbed3D,
    ShardedTimestepEmbedder,
)
from llm_perf.modeling.hunyuan_video.dit_blocks import (
    ShardedMMDoubleStreamBlock,
    ShardedMMSingleStreamBlock,
)
from llm_perf.modeling.tensor import ShardedTensor


class TestEdgeCases:
    """Edge case tests for HunyuanVideo components."""

    def test_modulate_dit_small_hidden_size(self):
        """Test ModulateDiT with hidden_size=1024 (small model)."""
        hidden_size = 1024
        batch_size = 1

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

    def test_double_stream_block_small_hidden_size(self):
        """Test DoubleStreamBlock with hidden_size=1024."""
        hidden_size = 1024
        heads_num = 8
        head_dim = 128
        batch_size = 1
        img_seq_len = 10
        txt_seq_len = 20

        block = ShardedMMDoubleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            qk_norm=True,
            dtype="bf16",
        )

        img = ShardedTensor(
            shape=(batch_size, img_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="img",
        )

        txt = ShardedTensor(
            shape=(batch_size, txt_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="txt",
        )

        vec = ShardedTensor(
            shape=(batch_size, hidden_size),
            shardable={},
            dtype="bf16",
            name="vec",
        )

        img_out, txt_out = block.forward(img, txt, vec)

        assert img_out.shape == (batch_size, img_seq_len, hidden_size)
        assert txt_out.shape == (batch_size, txt_seq_len, hidden_size)

    def test_double_stream_block_heads_num_1(self):
        """Test DoubleStreamBlock with heads_num=1 (minimum heads)."""
        hidden_size = 128
        heads_num = 1
        head_dim = 128
        batch_size = 1
        img_seq_len = 5
        txt_seq_len = 10

        block = ShardedMMDoubleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            qk_norm=True,
            dtype="bf16",
        )

        img = ShardedTensor(
            shape=(batch_size, img_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="img",
        )

        txt = ShardedTensor(
            shape=(batch_size, txt_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="txt",
        )

        vec = ShardedTensor(
            shape=(batch_size, hidden_size),
            shardable={},
            dtype="bf16",
            name="vec",
        )

        img_out, txt_out = block.forward(img, txt, vec)

        assert img_out.shape == (batch_size, img_seq_len, hidden_size)
        assert txt_out.shape == (batch_size, txt_seq_len, hidden_size)

    def test_double_stream_block_without_qk_norm(self):
        """Test DoubleStreamBlock with qk_norm=False."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128
        batch_size = 2
        img_seq_len = 100
        txt_seq_len = 256

        block = ShardedMMDoubleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            qk_norm=False,
            dtype="bf16",
        )

        assert block.img_attn_q_norm is None
        assert block.img_attn_k_norm is None
        assert block.txt_attn_q_norm is None
        assert block.txt_attn_k_norm is None

        img = ShardedTensor(
            shape=(batch_size, img_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="img",
        )

        txt = ShardedTensor(
            shape=(batch_size, txt_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="txt",
        )

        vec = ShardedTensor(
            shape=(batch_size, hidden_size),
            shardable={},
            dtype="bf16",
            name="vec",
        )

        img_out, txt_out = block.forward(img, txt, vec)

        assert img_out.shape == (batch_size, img_seq_len, hidden_size)
        assert txt_out.shape == (batch_size, txt_seq_len, hidden_size)

    def test_single_stream_block_small_hidden_size(self):
        """Test SingleStreamBlock with hidden_size=1024."""
        hidden_size = 1024
        heads_num = 8
        head_dim = 128
        batch_size = 1
        img_seq_len = 10
        txt_seq_len = 20
        total_seq_len = img_seq_len + txt_seq_len

        block = ShardedMMSingleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            qk_norm=True,
            dtype="bf16",
        )

        x = ShardedTensor(
            shape=(batch_size, total_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="x",
        )

        vec = ShardedTensor(
            shape=(batch_size, hidden_size),
            shardable={},
            dtype="bf16",
            name="vec",
        )

        x_out = block.forward(x, vec, txt_seq_len=txt_seq_len)

        assert x_out.shape == (batch_size, total_seq_len, hidden_size)

    def test_single_stream_block_heads_num_1(self):
        """Test SingleStreamBlock with heads_num=1."""
        hidden_size = 128
        heads_num = 1
        head_dim = 128
        batch_size = 1
        img_seq_len = 5
        txt_seq_len = 10
        total_seq_len = img_seq_len + txt_seq_len

        block = ShardedMMSingleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            qk_norm=True,
            dtype="bf16",
        )

        x = ShardedTensor(
            shape=(batch_size, total_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="x",
        )

        vec = ShardedTensor(
            shape=(batch_size, hidden_size),
            shardable={},
            dtype="bf16",
            name="vec",
        )

        x_out = block.forward(x, vec, txt_seq_len=txt_seq_len)

        assert x_out.shape == (batch_size, total_seq_len, hidden_size)

    def test_single_stream_block_without_qk_norm(self):
        """Test SingleStreamBlock with qk_norm=False."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128
        batch_size = 2
        img_seq_len = 100
        txt_seq_len = 256
        total_seq_len = img_seq_len + txt_seq_len

        block = ShardedMMSingleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            qk_norm=False,
            dtype="bf16",
        )

        assert block.q_norm is None
        assert block.k_norm is None

        x = ShardedTensor(
            shape=(batch_size, total_seq_len, hidden_size),
            shardable={},
            dtype="bf16",
            name="x",
        )

        vec = ShardedTensor(
            shape=(batch_size, hidden_size),
            shardable={},
            dtype="bf16",
            name="vec",
        )

        x_out = block.forward(x, vec, txt_seq_len=txt_seq_len)

        assert x_out.shape == (batch_size, total_seq_len, hidden_size)

    def test_patch_embed_3d_various_video_sizes(self):
        """Test PatchEmbed3D with various video sizes."""
        in_channels = 3
        hidden_size = 3072
        patch_size = (1, 2, 2)

        patch_embed = ShardedPatchEmbed3D(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            dtype="bf16",
        )

        test_cases = [
            (1, 1, 21, 180, 320),
            (2, 3, 53, 360, 640),
            (1, 3, 1, 16, 16),
        ]

        for batch, channels, frames, height, width in test_cases:
            video = ShardedTensor(
                shape=(batch, channels, frames, height, width),
                shardable={},
                dtype="bf16",
                name="video",
            )

            tokens = patch_embed.forward(video)

            pt, ph, pw = patch_size
            expected_seq = (frames // pt) * (height // ph) * (width // pw)

            assert tokens.shape == (batch, expected_seq, hidden_size)

    def test_timestep_embedder_small_hidden_size(self):
        """Test TimestepEmbedder with hidden_size=1024."""
        hidden_size = 1024
        batch_size = 1

        embedder = ShardedTimestepEmbedder(hidden_size=hidden_size, dtype="bf16")

        timestep = ShardedTensor(
            shape=(batch_size,),
            shardable={},
            dtype="bf16",
            name="timestep",
        )

        embed = embedder.forward(timestep)

        assert embed.shape == (batch_size, hidden_size)


class TestGeneralizationCases:
    """Generalization tests for various configurations."""

    def test_modulate_dit_various_hidden_sizes(self):
        """Test ModulateDiT with various hidden sizes."""
        hidden_sizes = [512, 1024, 2048, 3072, 4096]

        for hidden_size in hidden_sizes:
            batch_size = 2
            modulate = ShardedModulateDiT(hidden_size=hidden_size, dtype="bf16")

            vec = ShardedTensor(
                shape=(batch_size, hidden_size),
                shardable={},
                dtype="bf16",
                name="vec",
            )

            outputs = modulate.forward(vec)

            for output in outputs:
                assert output.shape == (batch_size, hidden_size)

    def test_double_stream_block_various_heads(self):
        """Test DoubleStreamBlock with various heads_num."""
        configs = [
            (3072, 24, 128),
            (2048, 16, 128),
            (1024, 8, 128),
            (512, 4, 128),
        ]

        for hidden_size, heads_num, head_dim in configs:
            batch_size = 1
            img_seq_len = 10
            txt_seq_len = 20

            block = ShardedMMDoubleStreamBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                head_dim=head_dim,
                qk_norm=True,
                dtype="bf16",
            )

            img = ShardedTensor(
                shape=(batch_size, img_seq_len, hidden_size),
                shardable={},
                dtype="bf16",
                name="img",
            )

            txt = ShardedTensor(
                shape=(batch_size, txt_seq_len, hidden_size),
                shardable={},
                dtype="bf16",
                name="txt",
            )

            vec = ShardedTensor(
                shape=(batch_size, hidden_size),
                shardable={},
                dtype="bf16",
                name="vec",
            )

            img_out, txt_out = block.forward(img, txt, vec)

            assert img_out.shape == (batch_size, img_seq_len, hidden_size)
            assert txt_out.shape == (batch_size, txt_seq_len, hidden_size)

    def test_single_stream_block_various_heads(self):
        """Test SingleStreamBlock with various heads_num."""
        configs = [
            (3072, 24, 128),
            (2048, 16, 128),
            (1024, 8, 128),
            (512, 4, 128),
        ]

        for hidden_size, heads_num, head_dim in configs:
            batch_size = 1
            img_seq_len = 10
            txt_seq_len = 20
            total_seq_len = img_seq_len + txt_seq_len

            block = ShardedMMSingleStreamBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                head_dim=head_dim,
                qk_norm=True,
                dtype="bf16",
            )

            x = ShardedTensor(
                shape=(batch_size, total_seq_len, hidden_size),
                shardable={},
                dtype="bf16",
                name="x",
            )

            vec = ShardedTensor(
                shape=(batch_size, hidden_size),
                shardable={},
                dtype="bf16",
                name="vec",
            )

            x_out = block.forward(x, vec, txt_seq_len=txt_seq_len)

            assert x_out.shape == (batch_size, total_seq_len, hidden_size)
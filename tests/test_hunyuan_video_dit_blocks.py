"""Tests for HunyuanVideo DiT blocks."""

from llm_perf.modeling.base.dit_blocks import (
    ShardedMMDoubleStreamBlock,
    ShardedMMSingleStreamBlock,
)
from llm_perf.modeling.tensor import ShardedTensor


class TestShardedMMDoubleStreamBlock:
    """Tests for ShardedMMDoubleStreamBlock."""

    def test_weight_shapes(self):
        """Test all weight parameter shapes."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128
        mlp_width_ratio = 4.0
        intermediate_size = int(hidden_size * mlp_width_ratio)

        block = ShardedMMDoubleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            mlp_width_ratio=mlp_width_ratio,
            qk_norm=True,
            dtype="bf16",
        )

        # Image branch weights
        assert block.img_attn_qkv.shape == (hidden_size, hidden_size * 3)
        assert block.img_attn_qkv.shardable == {1: "tp"}

        assert block.img_attn_q_norm.shape == (head_dim,)
        assert block.img_attn_q_norm.shardable == {}

        assert block.img_attn_k_norm.shape == (head_dim,)
        assert block.img_attn_k_norm.shardable == {}

        assert block.img_attn_proj.shape == (hidden_size, hidden_size)
        assert block.img_attn_proj.shardable == {0: "tp"}

        assert block.img_mlp_gate.shape == (hidden_size, intermediate_size)
        assert block.img_mlp_gate.shardable == {1: "tp"}

        assert block.img_mlp_up.shape == (hidden_size, intermediate_size)
        assert block.img_mlp_up.shardable == {1: "tp"}

        assert block.img_mlp_down.shape == (intermediate_size, hidden_size)
        assert block.img_mlp_down.shardable == {0: "tp"}

        # Text branch weights
        assert block.txt_attn_qkv.shape == (hidden_size, hidden_size * 3)
        assert block.txt_attn_qkv.shardable == {1: "tp"}

        assert block.txt_attn_q_norm.shape == (head_dim,)
        assert block.txt_attn_q_norm.shardable == {}

        assert block.txt_attn_k_norm.shape == (head_dim,)
        assert block.txt_attn_k_norm.shardable == {}

        assert block.txt_attn_proj.shape == (hidden_size, hidden_size)
        assert block.txt_attn_proj.shardable == {0: "tp"}

        assert block.txt_mlp_gate.shape == (hidden_size, intermediate_size)
        assert block.txt_mlp_gate.shardable == {1: "tp"}

        assert block.txt_mlp_up.shape == (hidden_size, intermediate_size)
        assert block.txt_mlp_up.shardable == {1: "tp"}

        assert block.txt_mlp_down.shape == (intermediate_size, hidden_size)
        assert block.txt_mlp_down.shardable == {0: "tp"}

    def test_weight_shapes_without_qk_norm(self):
        """Test weight shapes when qk_norm=False."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128

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

    def test_forward_output_shapes(self):
        """Test forward output shapes."""
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
        assert img_out.dtype == "bf16"
        assert txt_out.dtype == "bf16"


class TestShardedMMSingleStreamBlock:
    """Tests for ShardedMMSingleStreamBlock."""

    def test_weight_shapes(self):
        """Test all weight parameter shapes."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128
        mlp_width_ratio = 4.0
        intermediate_size = int(hidden_size * mlp_width_ratio)

        block = ShardedMMSingleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            mlp_width_ratio=mlp_width_ratio,
            qk_norm=True,
            dtype="bf16",
        )

        # linear1: (hidden_size, hidden_size * 3 + intermediate_size)
        assert block.linear1.shape == (hidden_size, hidden_size * 3 + intermediate_size)
        assert block.linear1.shardable == {1: "tp"}

        # QK-Norm weights
        assert block.q_norm.shape == (head_dim,)
        assert block.q_norm.shardable == {}

        assert block.k_norm.shape == (head_dim,)
        assert block.k_norm.shardable == {}

        # linear2: (hidden_size + intermediate_size, hidden_size)
        assert block.linear2.shape == (hidden_size + intermediate_size, hidden_size)
        assert block.linear2.shardable == {0: "tp"}

    def test_weight_shapes_without_qk_norm(self):
        """Test weight shapes when qk_norm=False."""
        hidden_size = 3072
        heads_num = 24
        head_dim = 128

        block = ShardedMMSingleStreamBlock(
            hidden_size=hidden_size,
            heads_num=heads_num,
            head_dim=head_dim,
            qk_norm=False,
            dtype="bf16",
        )

        assert block.q_norm is None
        assert block.k_norm is None

    def test_forward_output_shape(self):
        """Test forward output shape."""
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
        assert x_out.dtype == "bf16"


def test_all_blocks_creation():
    """Test that all blocks can be instantiated."""
    blocks = [
        ShardedMMDoubleStreamBlock(hidden_size=3072, dtype="bf16"),
        ShardedMMSingleStreamBlock(hidden_size=3072, dtype="bf16"),
    ]

    for block in blocks:
        assert hasattr(block, "_submodule_name")
        assert isinstance(block._submodule_name, str)


def test_import_from_module():
    """Test that blocks can be imported from hunyuan_video module."""
    from llm_perf.modeling.base.dit_blocks import (
        ShardedMMDoubleStreamBlock,
        ShardedMMSingleStreamBlock,
    )

    assert ShardedMMDoubleStreamBlock is not None
    assert ShardedMMSingleStreamBlock is not None
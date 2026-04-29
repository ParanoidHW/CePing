"""Tests for 3D Rotary Positional Embedding kernel."""

import pytest

from llm_perf.kernels.functional import rotary_embedding_3d
from llm_perf.kernels.op import RotaryPositionalEmbedding3DOp


class TestRotaryEmbedding3DFunctional:
    """Tests for rotary_embedding_3d functional API."""

    def test_output_shape(self):
        """Test output shape matches input."""
        batch = 1
        num_heads = 24
        seq_len = 1024
        head_dim = 128

        result = rotary_embedding_3d(
            query=(batch, num_heads, seq_len, head_dim),
            key=(batch, num_heads, seq_len, head_dim),
            freqs_cis=(seq_len, head_dim // 2),
            dtype="bf16",
        )

        assert result.output == (batch, num_heads, seq_len, head_dim)

    def test_flops_calculation(self):
        """Test FLOPs calculation."""
        batch = 1
        num_heads = 24
        seq_len = 1024
        head_dim = 128

        result = rotary_embedding_3d(
            query=(batch, num_heads, seq_len, head_dim),
            key=(batch, num_heads, seq_len, head_dim),
            freqs_cis=(seq_len, head_dim // 2),
            dtype="bf16",
        )

        q_ops = batch * num_heads * seq_len * (head_dim // 2) * 8
        k_ops = batch * num_heads * seq_len * (head_dim // 2) * 8
        expected_flops = q_ops + k_ops

        assert result.flops == expected_flops

    def test_memory_calculation(self):
        """Test memory bytes calculation."""
        batch = 1
        num_heads = 24
        seq_len = 1024
        head_dim = 128
        dtype = "bf16"
        dtype_size = 2

        result = rotary_embedding_3d(
            query=(batch, num_heads, seq_len, head_dim),
            key=(batch, num_heads, seq_len, head_dim),
            freqs_cis=(seq_len, head_dim // 2),
            dtype=dtype,
        )

        expected_bytes = (
            batch * num_heads * seq_len * head_dim * dtype_size
            + batch * num_heads * seq_len * head_dim * dtype_size
            + seq_len * (head_dim // 2) * dtype_size * 2
            + batch * num_heads * seq_len * head_dim * dtype_size
        )

        assert result.bytes_accessed == expected_bytes

    def test_gqa_support(self):
        """Test with different KV heads (GQA pattern)."""
        batch = 1
        num_heads = 32
        num_kv_heads = 8
        seq_len = 512
        head_dim = 128

        result = rotary_embedding_3d(
            query=(batch, num_heads, seq_len, head_dim),
            key=(batch, num_kv_heads, seq_len, head_dim),
            freqs_cis=(seq_len, head_dim // 2),
            dtype="fp16",
        )

        assert result.output == (batch, num_heads, seq_len, head_dim)

    def test_backward_metrics(self):
        """Test backward FLOPs and memory."""
        batch = 1
        num_heads = 24
        seq_len = 1024
        head_dim = 128

        result = rotary_embedding_3d(
            query=(batch, num_heads, seq_len, head_dim),
            key=(batch, num_heads, seq_len, head_dim),
            freqs_cis=(seq_len, head_dim // 2),
            dtype="bf16",
        )

        assert result.flops_backward > 0
        assert result.bytes_accessed_backward > 0

    def test_no_parameters(self):
        """Test that RoPE has no learnable parameters."""
        result = rotary_embedding_3d(
            query=(1, 24, 1024, 128),
            key=(1, 24, 1024, 128),
            freqs_cis=(1024, 64),
            dtype="bf16",
        )

        assert result.params == 0
        assert result.param_bytes == 0

    def test_saved_inputs_empty(self):
        """Test that RoPE doesn't need saved inputs for backward."""
        result = rotary_embedding_3d(
            query=(1, 24, 1024, 128),
            key=(1, 24, 1024, 128),
            freqs_cis=(1024, 64),
            dtype="bf16",
        )

        assert result.saved_inputs == []


class TestRotaryEmbedding3DOp:
    """Tests for RotaryPositionalEmbedding3DOp."""

    def test_op_creation(self):
        """Test Op can be created."""
        from llm_perf.modeling.tensor import ShardedTensor

        query = ShardedTensor(shape=(1, 24, 1024, 128), shardable={}, dtype="bf16", name="query")
        key = ShardedTensor(shape=(1, 24, 1024, 128), shardable={}, dtype="bf16", name="key")
        freqs_cis = ShardedTensor(shape=(1024, 64), shardable={}, dtype="bf16", name="freqs_cis")
        output = ShardedTensor(shape=(1, 24, 1024, 128), shardable={}, dtype="bf16", name="output")

        op = RotaryPositionalEmbedding3DOp(
            dtype="bf16",
            query=query,
            key=key,
            freqs_cis=freqs_cis,
            output=output,
        )

        assert op.kernel_name == "rotary_embedding_3d"
        assert op.dtype == "bf16"
        assert len(op.inputs) == 3

    def test_saved_tensors_empty(self):
        """Test that Op doesn't save tensors for backward."""
        from llm_perf.modeling.tensor import ShardedTensor

        query = ShardedTensor(shape=(1, 24, 1024, 128), shardable={}, dtype="bf16", name="query")
        key = ShardedTensor(shape=(1, 24, 1024, 128), shardable={}, dtype="bf16", name="key")
        freqs_cis = ShardedTensor(shape=(1024, 64), shardable={}, dtype="bf16", name="freqs_cis")
        output = ShardedTensor(shape=(1, 24, 1024, 128), shardable={}, dtype="bf16", name="output")

        op = RotaryPositionalEmbedding3DOp(
            dtype="bf16",
            query=query,
            key=key,
            freqs_cis=freqs_cis,
            output=output,
        )

        assert op.get_saved_tensors() == []


def test_rotary_embedding_3d_import():
    """Test that rotary_embedding_3d can be imported."""
    from llm_perf.kernels.functional import rotary_embedding_3d

    assert rotary_embedding_3d is not None


def test_rotary_embedding_3d_op_import():
    """Test that RotaryPositionalEmbedding3DOp can be imported."""
    from llm_perf.kernels.op import RotaryPositionalEmbedding3DOp

    assert RotaryPositionalEmbedding3DOp is not None


@pytest.mark.parametrize("batch,num_heads,seq_len,head_dim", [
    (1, 24, 1024, 128),
    (2, 32, 2048, 128),
    (1, 8, 256, 64),
])
def test_rotary_embedding_3d_various_shapes(batch, num_heads, seq_len, head_dim):
    """Test rotary_embedding_3d with various shapes."""
    result = rotary_embedding_3d(
        query=(batch, num_heads, seq_len, head_dim),
        key=(batch, num_heads, seq_len, head_dim),
        freqs_cis=(seq_len, head_dim // 2),
        dtype="bf16",
    )

    assert result.output == (batch, num_heads, seq_len, head_dim)
    assert result.flops > 0
    assert result.bytes_accessed > 0
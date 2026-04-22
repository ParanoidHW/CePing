"""Tests for Phase 2 basic modules."""

import pytest
from llm_perf.modeling import (
    ShardedTensor,
    ShardedModule,
    ParallelContext,
    ShardedEmbedding,
    ShardedAttention,
    ShardedFFN,
    ShardedLMHead,
    silu,
    gelu,
    flash_attention,
)
from llm_perf.modeling.layers import ShardedRMSNorm


class TestShardedEmbedding:
    """Test ShardedEmbedding."""

    def test_embedding_creation(self):
        """Test creating embedding layer."""
        embedding = ShardedEmbedding(
            num_embeddings=32000,
            embedding_dim=4096,
        )

        assert embedding.num_embeddings == 32000
        assert embedding.embedding_dim == 4096
        assert "weight" in embedding._weights
        assert embedding.weight.shape == (32000, 4096)
        assert embedding.weight.shardable == {0: "tp"}

    def test_embedding_forward(self):
        """Test embedding forward."""
        embedding = ShardedEmbedding(32000, 4096)

        input_ids = ShardedTensor(
            shape=(1, 512),
            shardable={1: "sp"},
        )

        output = embedding(input_ids)

        assert output.shape == (1, 512, 4096)
        assert len(output._op_history) == 1

    def test_embedding_params_count(self):
        """Test params count."""
        embedding = ShardedEmbedding(32000, 4096)

        assert embedding.params_count() == 32000 * 4096


class TestShardedRMSNorm:
    """Test ShardedRMSNorm."""

    def test_rmsnorm_creation(self):
        """Test creating RMSNorm layer."""
        norm = ShardedRMSNorm(hidden_size=4096)

        assert norm.hidden_size == 4096
        assert norm.weight.shape == (4096,)
        assert norm.weight.shardable == {}

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward."""
        norm = ShardedRMSNorm(4096)

        input_tensor = ShardedTensor(
            shape=(1, 512, 4096),
            shardable={1: "sp", 2: "tp"},
        )

        output = norm(input_tensor)

        assert output.shape == (1, 512, 4096)
        assert output.shardable == {1: "sp", 2: "tp"}
        assert len(output._op_history) == 1


class TestActivationFunctions:
    """Test activation functions."""

    def test_silu(self):
        """Test SiLU activation."""
        input_tensor = ShardedTensor(
            shape=(1, 512, 11008),
            shardable={2: "tp"},
        )

        output = silu(input_tensor)

        assert output.shape == (1, 512, 11008)
        assert output.shardable == {2: "tp"}
        assert len(output._op_history) == 1

    def test_gelu(self):
        """Test GELU activation."""
        input_tensor = ShardedTensor(shape=(1, 512, 4096))

        output = gelu(input_tensor)

        assert output.shape == (1, 512, 4096)
        assert len(output._op_history) == 1


class TestFlashAttention:
    """Test flash_attention function."""

    def test_attention_basic(self):
        """Test basic attention."""
        query = ShardedTensor(
            shape=(1, 32, 512, 128),
            shardable={1: "tp"},
        )
        key = ShardedTensor(
            shape=(1, 8, 512, 128),
            shardable={1: "tp"},
        )
        value = ShardedTensor(
            shape=(1, 8, 512, 128),
            shardable={1: "tp"},
        )

        output = flash_attention(query, key, value)

        assert output.shape == (1, 32, 512, 128)
        assert 1 in output.shardable
        assert output.shardable[1] == "tp"

    def test_attention_with_sp(self):
        """Test attention with SP."""
        query = ShardedTensor(
            shape=(1, 32, 512, 128),
            shardable={1: "tp", 2: "sp"},
        )
        key = ShardedTensor(
            shape=(1, 8, 512, 128),
            shardable={1: "tp", 2: "sp"},
        )
        value = ShardedTensor(
            shape=(1, 8, 512, 128),
            shardable={1: "tp", 2: "sp"},
        )

        output = flash_attention(query, key, value)

        assert output.shardable == {1: "tp", 2: "sp"}


class TestShardedAttention:
    """Test ShardedAttention."""

    def test_attention_creation(self):
        """Test creating attention layer."""
        attention = ShardedAttention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )

        assert attention.hidden_size == 4096
        assert attention.num_heads == 32
        assert attention.num_kv_heads == 8
        assert attention.head_dim == 128

    def test_attention_weights(self):
        """Test attention weights."""
        attention = ShardedAttention(4096, 32, 8, 128)

        assert attention.q_weight.shape == (4096, 32 * 128)
        assert attention.q_weight.shardable == {1: "tp"}

        assert attention.k_weight.shape == (4096, 8 * 128)
        assert attention.v_weight.shape == (4096, 8 * 128)

        assert attention.o_weight.shape == (32 * 128, 4096)
        assert attention.o_weight.shardable == {0: "tp"}

    def test_attention_params_count(self):
        """Test attention params count."""
        attention = ShardedAttention(4096, 32, 8, 128)

        q_params = 4096 * 32 * 128
        k_params = 4096 * 8 * 128
        v_params = 4096 * 8 * 128
        o_params = 32 * 128 * 4096

        expected = q_params + k_params + v_params + o_params
        assert attention.params_count() == expected

    def test_attention_forward(self):
        """Test attention forward."""
        attention = ShardedAttention(4096, 32, 8, 128)

        hidden = ShardedTensor(
            shape=(1, 512, 4096),
            shardable={1: "sp"},
        )

        output = attention(hidden)

        assert output.shape == (1, 512, 4096)


class TestShardedFFN:
    """Test ShardedFFN."""

    def test_ffn_creation(self):
        """Test creating FFN layer."""
        ffn = ShardedFFN(
            hidden_size=4096,
            intermediate_size=11008,
        )

        assert ffn.hidden_size == 4096
        assert ffn.intermediate_size == 11008

    def test_ffn_weights(self):
        """Test FFN weights."""
        ffn = ShardedFFN(4096, 11008)

        assert ffn.gate_weight.shape == (4096, 11008)
        assert ffn.gate_weight.shardable == {1: "tp"}

        assert ffn.up_weight.shape == (4096, 11008)
        assert ffn.up_weight.shardable == {1: "tp"}

        assert ffn.down_weight.shape == (11008, 4096)
        assert ffn.down_weight.shardable == {0: "tp"}

    def test_ffn_params_count(self):
        """Test FFN params count."""
        ffn = ShardedFFN(4096, 11008)

        gate_params = 4096 * 11008
        up_params = 4096 * 11008
        down_params = 11008 * 4096

        expected = gate_params + up_params + down_params
        assert ffn.params_count() == expected

    def test_ffn_forward(self):
        """Test FFN forward."""
        ffn = ShardedFFN(4096, 11008)

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = ffn(hidden)

        assert output.shape == (1, 512, 4096)


class TestShardedLMHead:
    """Test ShardedLMHead."""

    def test_lmhead_creation(self):
        """Test creating LM head."""
        lm_head = ShardedLMHead(
            hidden_size=4096,
            vocab_size=32000,
        )

        assert lm_head.hidden_size == 4096
        assert lm_head.vocab_size == 32000

    def test_lmhead_weight(self):
        """Test LM head weight."""
        lm_head = ShardedLMHead(4096, 32000)

        assert lm_head.weight.shape == (4096, 32000)
        assert lm_head.weight.shardable == {1: "tp"}

    def test_lmhead_forward(self):
        """Test LM head forward."""
        lm_head = ShardedLMHead(4096, 32000)

        hidden = ShardedTensor(shape=(1, 512, 4096))
        logits = lm_head(hidden)

        assert logits.shape == (1, 512, 32000)


class TestModuleInstanceForLayers:
    """Test ModuleInstance for basic layers."""

    def test_embedding_instance(self):
        """Test embedding module instance."""
        embedding = ShardedEmbedding(32000, 4096)
        ctx = ParallelContext(tp_degree=8)

        instance = embedding.bind(ctx)

        assert instance.params_count_logical == 32000 * 4096
        assert instance.params_count_physical == (32000 // 8) * 4096

    def test_ffn_instance(self):
        """Test FFN module instance."""
        ffn = ShardedFFN(4096, 11008)
        ctx = ParallelContext(tp_degree=8)

        instance = ffn.bind(ctx)

        logical_params = 4096 * 11008 * 3
        physical_params = 4096 * (11008 // 8) * 3

        assert instance.params_count_logical == logical_params
        assert instance.params_count_physical == physical_params

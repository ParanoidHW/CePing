"""Tests for complete models."""

import pytest
from llm_perf.modeling import (
    ShardedTensor,
    ShardedModule,
    ParallelContext,
    ShardedTransformerBlock,
    LlamaModel,
)


class TestShardedTransformerBlock:
    """Test ShardedTransformerBlock."""

    def test_block_creation(self):
        """Test creating transformer block."""
        block = ShardedTransformerBlock(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
        )

        assert block.hidden_size == 4096
        assert block.num_heads == 32
        assert block.num_kv_heads == 8
        assert block.intermediate_size == 11008

    def test_block_submodules(self):
        """Test block has correct submodules."""
        block = ShardedTransformerBlock(4096, 32, 8, 11008)

        assert "input_norm" in block._submodules
        assert "attention" in block._submodules
        assert "post_attn_norm" in block._submodules
        assert "ffn" in block._submodules

    def test_block_params_count(self):
        """Test block params count."""
        block = ShardedTransformerBlock(4096, 32, 8, 128, 11008)

        norm_params = 4096 * 2

        q_params = 4096 * 32 * 128
        k_params = 4096 * 8 * 128
        v_params = 4096 * 8 * 128
        o_params = 32 * 128 * 4096
        attn_params = q_params + k_params + v_params + o_params

        gate_params = 4096 * 11008
        up_params = 4096 * 11008
        down_params = 11008 * 4096
        ffn_params = gate_params + up_params + down_params

        expected = norm_params + attn_params + ffn_params
        assert block.params_count() == expected

    def test_block_forward(self):
        """Test block forward."""
        block = ShardedTransformerBlock(4096, 32, 8, 11008)

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = block(hidden)

        assert output.shape == (1, 512, 4096)
        assert "attn_out" in block._activations
        assert "ffn_out" in block._activations


class TestLlamaModel:
    """Test LlamaModel."""

    def test_llama_creation(self):
        """Test creating Llama model."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
        )

        assert model.vocab_size == 32000
        assert model.hidden_size == 4096
        assert model.num_layers == 32
        assert model.num_heads == 32
        assert model.num_kv_heads == 8
        assert model.intermediate_size == 11008

    def test_llama_layers(self):
        """Test Llama has correct number of layers."""
        model = LlamaModel(32000, 4096, 4, 32, 8, 11008)

        assert len(model.layers) == 4
        assert "embedding" in model._submodules
        assert "final_norm" in model._submodules
        assert "lm_head" in model._submodules

        for i in range(4):
            assert f"layers.{i}" in model._submodules

    def test_llama_params_count(self):
        """Test Llama params count."""
        vocab_size = 32000
        hidden_size = 4096
        num_layers = 2
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        intermediate_size = 11008

        model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
        )

        embedding_params = vocab_size * hidden_size
        lm_head_params = hidden_size * vocab_size

        norm_params = hidden_size

        q_params = hidden_size * num_heads * head_dim
        k_params = hidden_size * num_kv_heads * head_dim
        v_params = hidden_size * num_kv_heads * head_dim
        o_params = num_heads * head_dim * hidden_size
        attn_params = q_params + k_params + v_params + o_params

        gate_params = hidden_size * intermediate_size
        up_params = hidden_size * intermediate_size
        down_params = intermediate_size * hidden_size
        ffn_params = gate_params + up_params + down_params

        layer_params = norm_params + attn_params + norm_params + ffn_params
        total_layer_params = layer_params * num_layers

        final_norm_params = hidden_size

        expected = embedding_params + total_layer_params + final_norm_params + lm_head_params

        assert model.params_count() == expected

    def test_llama_forward(self):
        """Test Llama forward."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
        )

        input_ids = ShardedTensor(shape=(1, 512))
        logits = model(input_ids)

        assert logits.shape == (1, 512, 32000)
        assert "embedding_output" in model._activations
        assert "layer_0_output" in model._activations
        assert "layer_1_output" in model._activations

    def test_llama_module_instance(self):
        """Test Llama module instance."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
        )

        ctx = ParallelContext(tp_degree=8)

        instance = model.bind(ctx)

        assert instance.params_count_logical == model.params_count()
        assert instance.params_count_physical < instance.params_count_logical

    def test_llama_to_dict(self):
        """Test Llama instance to_dict."""
        model = LlamaModel(32000, 4096, 2, 32, 8)
        model._name = "llama_test"

        ctx = ParallelContext(tp_degree=2)

        instance = model.bind(ctx)
        result = instance.to_dict()

        assert "module_name" in result
        assert "params" in result
        assert "submodules" in result
        assert len(result["submodules"]) > 0


class TestEndToEnd:
    """End-to-end tests."""

    def test_simple_llama_pipeline(self):
        """Test simple pipeline with Llama."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
        )

        ctx = ParallelContext(
            tp_degree=8,
            sp_degree=1,
            dtype="fp16",
        )

        input_ids = ShardedTensor(shape=(1, 512))

        logits = model(input_ids)
        instance = model.bind(ctx)

        assert logits.shape == (1, 512, 32000)
        assert instance.params_count_physical > 0

    def test_llama_with_sp(self):
        """Test Llama with SP."""
        model = LlamaModel(32000, 4096, 2, 32, 8)

        ctx = ParallelContext(
            tp_degree=8,
            sp_degree=4,
            dtype="fp16",
        )

        input_ids = ShardedTensor(
            shape=(1, 512),
            shardable={},
        )

        logits = model(input_ids)
        instance = model.bind(ctx)

        assert logits.shape == (1, 512, 32000)

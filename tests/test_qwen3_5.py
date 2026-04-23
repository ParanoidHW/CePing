"""Tests for Qwen3.5 Models.

Qwen3.5 family includes:
- Dense models: Qwen3.5-0.8B/2B/4B/9B/27B (SwiGLU FFN)
- MoE models: Qwen3.5-35B-A3B (256 experts, top-8 routing, shared expert)

All models feature:
- Hybrid attention: 3 linear + 1 full per 4-layer cycle
- Linear attention: O(seq) complexity
"""

import pytest
from llm_perf.modeling import (
    ShardedTensor,
    ParallelContext,
    create_model_from_config,
    get_model_presets,
)
from llm_perf.modeling.qwen3_5 import (
    Qwen3_5MoEModel,
    Qwen3_5Model,
    ShardedQwen3_5MoEBlock,
    ShardedQwen3_5DenseBlock,
    generate_layer_types,
)
from llm_perf.modeling.layers import ShardedLinearAttention, ShardedAttention, ShardedFFN


class TestGenerateLayerTypes:
    """Test layer_types generation."""

    def test_generate_layer_types_40_layers(self):
        """Test generating 40 layers pattern."""
        layer_types = generate_layer_types(40)

        assert len(layer_types) == 40
        assert layer_types.count("linear_attention") == 30
        assert layer_types.count("full_attention") == 10

    def test_generate_layer_types_pattern(self):
        """Test layer_types follows 3+1 pattern."""
        layer_types = generate_layer_types(40)

        for cycle in range(10):
            cycle_layers = layer_types[cycle * 4 : (cycle + 1) * 4]
            assert cycle_layers == [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]

    def test_generate_layer_types_partial_cycle(self):
        """Test generating partial cycle (e.g., 5 layers)."""
        layer_types = generate_layer_types(5)

        assert len(layer_types) == 5
        assert layer_types == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ]


class TestShardedQwen3_5MoEBlock:
    """Test ShardedQwen3_5MoEBlock."""

    def test_linear_attention_block_creation(self):
        """Test creating linear attention block."""
        block = ShardedQwen3_5MoEBlock(
            hidden_size=2048,
            layer_type="linear_attention",
            num_heads=16,
            linear_num_heads=16,
            linear_kernel_dim=4,
            intermediate_size=512,
            num_experts=256,
            num_experts_per_token=8,
            shared_expert_intermediate=512,
        )

        assert block.layer_type == "linear_attention"
        assert block.num_experts == 256
        assert isinstance(block.attention, ShardedLinearAttention)
        assert block.attention.kernel_dim == 4

    def test_full_attention_block_creation(self):
        """Test creating full attention block."""
        block = ShardedQwen3_5MoEBlock(
            hidden_size=2048,
            layer_type="full_attention",
            num_heads=16,
            num_kv_heads=2,
            head_dim=256,
            intermediate_size=512,
            num_experts=256,
            num_experts_per_token=8,
            shared_expert_intermediate=512,
        )

        assert block.layer_type == "full_attention"
        assert isinstance(block.attention, ShardedAttention)
        assert block.attention.num_kv_heads == 2

    def test_block_submodules(self):
        """Test block has correct submodules."""
        block = ShardedQwen3_5MoEBlock(
            hidden_size=2048,
            layer_type="linear_attention",
            num_heads=16,
            num_experts=256,
        )

        assert "attention" in block._submodules
        assert "moe" in block._submodules
        assert "input_norm_weight" in block._weights
        assert "post_attn_norm_weight" in block._weights

    def test_block_forward(self):
        """Test block forward."""
        block = ShardedQwen3_5MoEBlock(
            hidden_size=2048,
            layer_type="linear_attention",
            num_heads=16,
            num_experts=256,
            shared_expert_intermediate=512,
        )

        hidden = ShardedTensor(shape=(1, 512, 2048))
        output = block(hidden)

        assert output.shape == (1, 512, 2048)
        assert "attn_out" in block._activations
        assert "moe_out" in block._activations


class TestQwen3_5MoEConfig:
    """Test Qwen3.5 MoE config validation."""

    def test_qwen3_5_config_validation(self):
        """Test creating Qwen3.5 model with config."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_kv_heads=2,
            head_dim=256,
            linear_num_heads=16,
            linear_num_kv_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_kernel_dim=4,
            intermediate_size=512,
            num_experts=256,
            num_experts_per_token=8,
            shared_expert_intermediate=512,
        )

        assert model.vocab_size == 248320
        assert model.hidden_size == 2048
        assert model.num_layers == 40
        assert model.num_experts == 256
        assert model.linear_kernel_dim == 4

    def test_qwen3_5_layer_types_mixed(self):
        """Test Qwen3.5 has mixed linear/full attention layers."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_experts=256,
        )

        linear_count = sum(1 for lt in model.layer_types if lt == "linear_attention")
        full_count = sum(1 for lt in model.layer_types if lt == "full_attention")

        assert linear_count == 30
        assert full_count == 10

    def test_qwen3_5_linear_vs_full_attention_count(self):
        """Test correct ratio of linear to full attention."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_experts=256,
        )

        linear_blocks = [layer for layer in model.layers if isinstance(layer.attention, ShardedLinearAttention)]
        full_blocks = [layer for layer in model.layers if isinstance(layer.attention, ShardedAttention)]

        assert len(linear_blocks) == 30
        assert len(full_blocks) == 10

    def test_qwen3_5_custom_layer_types(self):
        """Test Qwen3.5 with custom layer_types."""
        custom_types = ["full_attention", "linear_attention", "linear_attention", "linear_attention"]

        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
            layer_types=custom_types,
        )

        assert model.layer_types == custom_types
        assert isinstance(model.layers[0].attention, ShardedAttention)
        assert isinstance(model.layers[1].attention, ShardedLinearAttention)

    def test_qwen3_5_layer_types_length_mismatch(self):
        """Test error when layer_types length != num_layers."""
        with pytest.raises(ValueError):
            Qwen3_5MoEModel(
                vocab_size=248320,
                hidden_size=2048,
                num_layers=40,
                num_heads=16,
                num_experts=256,
                layer_types=["linear_attention"] * 10,
            )


class TestQwen3_5MoEExperts:
    """Test Qwen3.5 MoE expert configuration."""

    def test_qwen3_5_moe_experts(self):
        """Test MoE expert configuration."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_experts=256,
            num_experts_per_token=8,
            shared_expert_intermediate=512,
        )

        assert model.num_experts == 256
        assert model.num_experts_per_token == 8
        assert model.shared_expert_intermediate == 512

        for layer in model.layers:
            assert layer.moe.num_experts == 256
            assert layer.moe.num_experts_per_token == 8
            assert layer.moe.shared_expert_intermediate == 512

    def test_qwen3_5_moe_params_breakdown(self):
        """Test MoE params breakdown per layer."""
        block = ShardedQwen3_5MoEBlock(
            hidden_size=2048,
            layer_type="linear_attention",
            num_heads=16,
            intermediate_size=512,
            num_experts=256,
            num_experts_per_token=8,
            shared_expert_intermediate=512,
        )

        breakdown = block.params_count_breakdown()

        moe_params = sum(v for k, v in breakdown.items() if "moe" in k)
        assert moe_params > 0


class TestQwen3_5MoEParams:
    """Test Qwen3.5 MoE total params."""

    def test_qwen3_5_total_params(self):
        """Test Qwen3.5 total params ~36B.

        Note: Exact count depends on implementation details.
        This test verifies params are in expected range.
        """
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_kv_heads=2,
            head_dim=256,
            linear_num_heads=16,
            linear_num_kv_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_kernel_dim=4,
            intermediate_size=512,
            num_experts=256,
            num_experts_per_token=8,
            shared_expert_intermediate=512,
        )

        params = model.params_count()

        assert params > 1e9, f"Qwen3.5 should have >1B params, got {params / 1e9:.2f}B"

        assert params < 100e9, f"Qwen3.5 should have <100B params, got {params / 1e9:.2f}B"

    def test_qwen3_5_embedding_params(self):
        """Test embedding params."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_experts=256,
        )

        embedding_params = model.embedding.params_count()
        expected = 248320 * 2048
        assert embedding_params == expected


class TestQwen3_5MoEInference:
    """Test Qwen3.5 inference."""

    def test_qwen3_5_inference_eval(self):
        """Test Qwen3.5 forward pass."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
            shared_expert_intermediate=512,
        )

        input_ids = ShardedTensor(shape=(1, 512))
        logits = model(input_ids)

        assert logits.shape == (1, 512, 248320)
        assert "embedding_output" in model._activations
        assert "layer_0_output" in model._activations
        assert "final_norm_output" in model._activations

    def test_qwen3_5_module_instance(self):
        """Test Qwen3.5 module instance with TP."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
        )

        ctx = ParallelContext(tp_degree=8, ep_degree=2)
        instance = model.bind(ctx)

        assert instance.params_count_logical == model.params_count()
        assert instance.params_count_physical < instance.params_count_logical

    def test_qwen3_5_forward_with_op_history(self):
        """Test forward produces op history."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=2,
            num_heads=16,
            num_experts=256,
        )

        input_ids = ShardedTensor(shape=(1, 128))
        logits = model(input_ids)

        assert len(logits._op_history) > 0


class TestQwen3_5MoEFromConfig:
    """Test creating Qwen3.5 from config."""

    def test_create_from_preset(self):
        """Test creating Qwen3.5 from preset."""
        presets = get_model_presets()
        if "qwen3-5-moe" not in presets:
            pytest.skip("qwen3-5-moe preset not yet available")

        model = create_model_from_config({"preset": "qwen3-5-moe"})
        assert isinstance(model, Qwen3_5MoEModel)
        assert model.hidden_size == 2048
        assert model.num_layers == 40

    def test_create_from_type(self):
        """Test creating Qwen3.5 from type field."""
        presets = get_model_presets()
        if "qwen3-5-moe" not in presets:
            pytest.skip("qwen3-5-moe preset not yet available")

        model = create_model_from_config({"type": "qwen3-5-moe"})
        assert isinstance(model, Qwen3_5MoEModel)


class TestQwen3_5MoEBreakdown:
    """Test Qwen3.5 params breakdown."""

    def test_qwen3_5_attention_breakdown(self):
        """Test attention params breakdown."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
        )

        breakdown = model.params_count_breakdown()

        attention_params = sum(
            v for k, v in breakdown.items() if "attention" in k and "moe" not in k
        )
        assert attention_params > 0

    def test_qwen3_5_moe_breakdown(self):
        """Test MoE params breakdown."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
            shared_expert_intermediate=512,
        )

        breakdown = model.params_count_breakdown()

        moe_params = sum(v for k, v in breakdown.items() if "moe" in k)
        assert moe_params > 0


class TestQwen3_5MTP:
    """Test Qwen3.5 MTP (Multi-Token Prediction) layers."""

    def test_qwen3_5_with_mtp_layers(self):
        """Test creating Qwen3.5 with MTP layers."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_experts=256,
            mtp_num_layers=1,
        )

        assert model.mtp_num_layers == 1
        assert len(model.mtp_layers) == 1
        assert model.config.mtp_num_layers == 1

    def test_qwen3_5_mtp_params_count(self):
        """Test MTP params count calculation.

        MTP adds 1 Transformer Layer weights, but shares embedding and lm_head.
        So MTP extra params = 1 Transformer Block weight.
        """
        model_no_mtp = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
            mtp_num_layers=0,
        )

        model_with_mtp = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
            mtp_num_layers=1,
        )

        no_mtp_params = model_no_mtp.params_count()
        with_mtp_params = model_with_mtp.params_count()

        mtp_layer_params = model_with_mtp.mtp_layers[0].params_count()
        expected_diff = mtp_layer_params

        actual_diff = with_mtp_params - no_mtp_params
        assert actual_diff == expected_diff, (
            f"MTP should add {expected_diff} params, but added {actual_diff}"
        )

    def test_qwen3_5_mtp_share_embeddings(self):
        """Test MTP shares embedding and lm_head with main model.

        When mtp_share_embeddings=True, MTP does not create extra embedding/lm_head.
        """
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
            mtp_num_layers=1,
            mtp_share_embeddings=True,
        )

        assert model.mtp_share_embeddings == True

        mtp_layer_params = model.mtp_layers[0].params_count()

        embedding_params = model.embedding.params_count()
        lm_head_params = model.lm_head.params_count()

        assert embedding_params > 0
        assert lm_head_params > 0

        mtp_only_params = mtp_layer_params
        assert mtp_only_params > 0
        assert mtp_only_params < embedding_params + lm_head_params

    def test_qwen3_5_mtp_layer_type(self):
        """Test MTP uses last layer's configuration (full_attention)."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_experts=256,
            mtp_num_layers=1,
        )

        last_layer_type = model.layer_types[-1]
        assert last_layer_type == "full_attention"

        mtp_layer = model.mtp_layers[0]
        assert mtp_layer.layer_type == last_layer_type

    def test_qwen3_5_forward_with_mtp(self):
        """Test forward_with_mtp produces correct outputs."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
            shared_expert_intermediate=512,
            mtp_num_layers=2,
        )

        input_ids = ShardedTensor(shape=(1, 512))
        logits_list = model.forward_with_mtp(input_ids)

        assert len(logits_list) == 3
        assert logits_list[0].shape == (1, 512, 248320)
        assert logits_list[1].shape == (1, 512, 248320)
        assert logits_list[2].shape == (1, 512, 248320)

        assert "mtp_layer_0_output" in model._activations
        assert "mtp_layer_1_output" in model._activations

    def test_qwen3_5_mtp_zero_layers(self):
        """Test model with mtp_num_layers=0 has no MTP layers."""
        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=40,
            num_heads=16,
            num_experts=256,
            mtp_num_layers=0,
        )

        assert model.mtp_num_layers == 0
        assert len(model.mtp_layers) == 0


class TestShardedQwen3_5DenseBlock:
    """Test ShardedQwen3_5DenseBlock."""

    def test_linear_attention_block_creation(self):
        """Test creating linear attention dense block."""
        block = ShardedQwen3_5DenseBlock(
            hidden_size=2048,
            layer_type="linear_attention",
            num_heads=16,
            linear_num_heads=16,
            linear_kernel_dim=4,
            intermediate_size=6144,
        )

        assert block.layer_type == "linear_attention"
        assert isinstance(block.attention, ShardedLinearAttention)
        assert block.attention.kernel_dim == 4
        assert isinstance(block.ffn, ShardedFFN)
        assert block.ffn.ffn_act_type == "swiglu"

    def test_full_attention_block_creation(self):
        """Test creating full attention dense block."""
        block = ShardedQwen3_5DenseBlock(
            hidden_size=4096,
            layer_type="full_attention",
            num_heads=16,
            num_kv_heads=4,
            intermediate_size=12288,
        )

        assert block.layer_type == "full_attention"
        assert isinstance(block.attention, ShardedAttention)
        assert block.attention.num_kv_heads == 4

    def test_block_submodules(self):
        """Test block has correct submodules."""
        block = ShardedQwen3_5DenseBlock(
            hidden_size=2048,
            layer_type="linear_attention",
            num_heads=16,
            intermediate_size=6144,
        )

        assert "attention" in block._submodules
        assert "ffn" in block._submodules
        assert "input_norm_weight" in block._weights
        assert "post_attn_norm_weight" in block._weights

    def test_block_forward(self):
        """Test block forward."""
        block = ShardedQwen3_5DenseBlock(
            hidden_size=2048,
            layer_type="linear_attention",
            num_heads=16,
            intermediate_size=6144,
        )

        hidden = ShardedTensor(shape=(1, 512, 2048))
        output = block(hidden)

        assert output.shape == (1, 512, 2048)
        assert "attn_out" in block._activations
        assert "ffn_out" in block._activations

    def test_swiglu_ffn_params(self):
        """Test SwiGLU FFN params calculation.

        SwiGLU has gate_proj + up_proj + down_proj:
        - gate_proj: hidden_size * intermediate_size
        - up_proj: hidden_size * intermediate_size
        - down_proj: intermediate_size * hidden_size
        Total: 2 * hidden_size * intermediate_size + intermediate_size * hidden_size
        """
        hidden_size = 4096
        intermediate_size = 12288

        block = ShardedQwen3_5DenseBlock(
            hidden_size=hidden_size,
            layer_type="full_attention",
            num_heads=16,
            intermediate_size=intermediate_size,
        )

        ffn_params = block.ffn.params_count()
        expected = 2 * hidden_size * intermediate_size + intermediate_size * hidden_size
        assert ffn_params == expected


class TestQwen3_5DenseConfig:
    """Test Qwen3.5 Dense config validation."""

    def test_qwen3_5_config_validation(self):
        """Test creating Qwen3.5 Dense model with config."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=4096,
            num_layers=32,
            num_heads=16,
            num_kv_heads=4,
            intermediate_size=12288,
            linear_num_heads=16,
            linear_num_kv_heads=32,
            linear_key_head_dim=256,
            linear_value_head_dim=256,
            linear_kernel_dim=4,
        )

        assert model.vocab_size == 248320
        assert model.hidden_size == 4096
        assert model.num_layers == 32
        assert model.linear_kernel_dim == 4

    def test_qwen3_5_layer_types_mixed(self):
        """Test Qwen3.5 Dense has mixed linear/full attention layers."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=4096,
            num_layers=32,
            num_heads=16,
        )

        linear_count = sum(1 for lt in model.layer_types if lt == "linear_attention")
        full_count = sum(1 for lt in model.layer_types if lt == "full_attention")

        assert linear_count == 24
        assert full_count == 8

    def test_qwen3_5_custom_layer_types(self):
        """Test Qwen3.5 Dense with custom layer_types."""
        custom_types = ["full_attention", "linear_attention", "linear_attention", "linear_attention"]

        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            layer_types=custom_types,
        )

        assert model.layer_types == custom_types
        assert isinstance(model.layers[0].attention, ShardedAttention)
        assert isinstance(model.layers[1].attention, ShardedLinearAttention)


class TestQwen3_5DenseTieEmbeddings:
    """Test tie_word_embeddings feature."""

    def test_tie_word_embeddings_true(self):
        """Test model with tie_word_embeddings=True.

        Small models (0.8B, 2B, 4B) share embedding/lm_head weights.
        lm_head is None, params_count accounts for sharing.
        """
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=1024,
            num_layers=24,
            num_heads=8,
            num_kv_heads=2,
            intermediate_size=3584,
            tie_word_embeddings=True,
        )

        assert model.tie_word_embeddings == True
        assert model.lm_head is None

    def test_tie_word_embeddings_false(self):
        """Test model with tie_word_embeddings=False.

        Large models (9B, 27B) have separate embedding/lm_head.
        lm_head is ShardedLMHead instance.
        """
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=4096,
            num_layers=32,
            num_heads=16,
            num_kv_heads=4,
            intermediate_size=12288,
            tie_word_embeddings=False,
        )

        assert model.tie_word_embeddings == False
        assert model.lm_head is not None

    def test_params_count_tie_difference(self):
        """Test params_count difference between tie=True and tie=False.

        When tie=False, model has extra vocab_size * hidden_size params.
        """
        vocab_size = 248320
        hidden_size = 1024

        model_tie = Qwen3_5Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=4,
            num_heads=8,
            tie_word_embeddings=True,
        )

        model_no_tie = Qwen3_5Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=4,
            num_heads=8,
            tie_word_embeddings=False,
        )

        tie_params = model_tie.params_count()
        no_tie_params = model_no_tie.params_count()

        expected_diff = vocab_size * hidden_size
        actual_diff = no_tie_params - tie_params

        assert actual_diff == expected_diff


class TestQwen3_5DenseParams:
    """Test Qwen3.5 Dense total params."""

    def test_qwen3_5_0_8b_params(self):
        """Test Qwen3.5-0.8B params ~0.9B.

        Config from HuggingFace:
        - hidden_size: 1024, num_layers: 24
        - num_heads: 8, num_kv_heads: 2
        - intermediate_size: 3584, head_dim: 256
        - vocab_size: 248320, tie_word_embeddings: True
        - linear_num_value_heads: 16
        """
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=1024,
            num_layers=24,
            num_heads=8,
            num_kv_heads=2,
            head_dim=256,
            intermediate_size=3584,
            linear_num_heads=16,
            linear_num_kv_heads=16,
            linear_key_head_dim=256,
            linear_value_head_dim=256,
            tie_word_embeddings=True,
        )

        params = model.params_count()

        assert params > 0.7e9, f"Qwen3.5-0.8B should have >0.7B params, got {params / 1e9:.2f}B"
        assert params < 1.2e9, f"Qwen3.5-0.8B should have <1.2B params, got {params / 1e9:.2f}B"

    def test_qwen3_5_2b_params(self):
        """Test Qwen3.5-2B params ~2B.

        Config from HuggingFace:
        - hidden_size: 2048, num_layers: 24
        - num_heads: 8, num_kv_heads: 2
        - intermediate_size: 6144
        - vocab_size: 248320, tie_word_embeddings: True
        - linear_num_value_heads: 16
        """
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=24,
            num_heads=8,
            num_kv_heads=2,
            intermediate_size=6144,
            linear_num_heads=16,
            linear_num_kv_heads=16,
            tie_word_embeddings=True,
        )

        params = model.params_count()

        assert params > 1.5e9, f"Qwen3.5-2B should have >1.5B params, got {params / 1e9:.2f}B"
        assert params < 3e9, f"Qwen3.5-2B should have <3B params, got {params / 1e9:.2f}B"

    def test_qwen3_5_4b_params(self):
        """Test Qwen3.5-4B params ~5B.

        Config from HuggingFace:
        - hidden_size: 2560, num_layers: 32
        - num_heads: 16, num_kv_heads: 4
        - intermediate_size: 9216
        - vocab_size: 248320, tie_word_embeddings: True
        - linear_num_value_heads: 32
        """
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2560,
            num_layers=32,
            num_heads=16,
            num_kv_heads=4,
            intermediate_size=9216,
            linear_num_heads=32,
            linear_num_kv_heads=32,
            tie_word_embeddings=True,
        )

        params = model.params_count()

        assert params > 3e9, f"Qwen3.5-4B should have >3B params, got {params / 1e9:.2f}B"
        assert params < 7e9, f"Qwen3.5-4B should have <7B params, got {params / 1e9:.2f}B"

    def test_qwen3_5_9b_params(self):
        """Test Qwen3.5-9B params ~10B.

        Config from HuggingFace:
        - hidden_size: 4096, num_layers: 32
        - num_heads: 16, num_kv_heads: 4
        - intermediate_size: 12288
        - vocab_size: 248320, tie_word_embeddings: False
        - linear_num_value_heads: 32
        """
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=4096,
            num_layers=32,
            num_heads=16,
            num_kv_heads=4,
            intermediate_size=12288,
            linear_num_heads=32,
            linear_num_kv_heads=32,
            tie_word_embeddings=False,
        )

        params = model.params_count()

        assert params > 8e9, f"Qwen3.5-9B should have >8B params, got {params / 1e9:.2f}B"
        assert params < 12e9, f"Qwen3.5-9B should have <12B params, got {params / 1e9:.2f}B"

    def test_qwen3_5_27b_params(self):
        """Test Qwen3.5-27B params ~28B.

        Config from HuggingFace:
        - hidden_size: 5120, num_layers: 64
        - num_heads: 24, num_kv_heads: 4
        - intermediate_size: 17408
        - vocab_size: 248320, tie_word_embeddings: False
        - linear_num_value_heads: 48
        """
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=5120,
            num_layers=64,
            num_heads=24,
            num_kv_heads=4,
            intermediate_size=17408,
            linear_num_heads=48,
            linear_num_kv_heads=48,
            tie_word_embeddings=False,
        )

        params = model.params_count()

        assert params > 25e9, f"Qwen3.5-27B should have >25B params, got {params / 1e9:.2f}B"
        assert params < 35e9, f"Qwen3.5-27B should have <35B params, got {params / 1e9:.2f}B"

    def test_qwen3_5_embedding_params(self):
        """Test embedding params."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=4096,
            num_layers=32,
            num_heads=16,
        )

        embedding_params = model.embedding.params_count()
        expected = 248320 * 4096
        assert embedding_params == expected


class TestQwen3_5DenseInference:
    """Test Qwen3.5 Dense inference."""

    def test_qwen3_5_inference_eval(self):
        """Test Qwen3.5 Dense forward pass."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            intermediate_size=6144,
        )

        input_ids = ShardedTensor(shape=(1, 512))
        logits = model(input_ids)

        assert logits.shape == (1, 512, 248320)
        assert "embedding_output" in model._activations
        assert "layer_0_output" in model._activations
        assert "final_norm_output" in model._activations

    def test_qwen3_5_module_instance(self):
        """Test Qwen3.5 Dense module instance with TP."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
        )

        ctx = ParallelContext(tp_degree=8)
        instance = model.bind(ctx)

        assert instance.params_count_logical == model.params_count()
        assert instance.params_count_physical < instance.params_count_logical

    def test_qwen3_5_forward_with_op_history(self):
        """Test forward produces op history."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=2,
            num_heads=16,
        )

        input_ids = ShardedTensor(shape=(1, 128))
        logits = model(input_ids)

        assert len(logits._op_history) > 0

    def test_qwen3_5_tie_forward(self):
        """Test forward with tie_word_embeddings=True."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=1024,
            num_layers=4,
            num_heads=8,
            tie_word_embeddings=True,
        )

        input_ids = ShardedTensor(shape=(1, 512))
        logits = model(input_ids)

        assert logits.shape == (1, 512, 248320)


class TestQwen3_5DenseFromConfig:
    """Test creating Qwen3.5 Dense from config."""

    def test_create_from_preset(self):
        """Test creating Qwen3.5 Dense from preset."""
        presets = get_model_presets()
        if "qwen3_5" not in presets:
            pytest.skip("qwen3_5 preset not yet available")

        model = create_model_from_config({"preset": "qwen3_5"})
        assert isinstance(model, Qwen3_5Model)

    def test_create_from_type(self):
        """Test creating Qwen3.5 Dense from type field."""
        presets = get_model_presets()
        if "qwen3_5" not in presets:
            pytest.skip("qwen3_5 preset not yet available")

        model = create_model_from_config({"type": "qwen3_5"})
        assert isinstance(model, Qwen3_5Model)


class TestQwen3_5DenseBreakdown:
    """Test Qwen3.5 Dense params breakdown."""

    def test_qwen3_5_attention_breakdown(self):
        """Test attention params breakdown."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
        )

        breakdown = model.params_count_breakdown()

        attention_params = sum(
            v for k, v in breakdown.items() if "attention" in k and "ffn" not in k
        )
        assert attention_params > 0

    def test_qwen3_5_ffn_breakdown(self):
        """Test FFN params breakdown."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            intermediate_size=6144,
        )

        breakdown = model.params_count_breakdown()

        ffn_params = sum(v for k, v in breakdown.items() if "ffn" in k)
        assert ffn_params > 0

    def test_qwen3_5_tie_breakdown(self):
        """Test params breakdown with tie_word_embeddings."""
        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=1024,
            num_layers=4,
            num_heads=8,
            tie_word_embeddings=True,
        )

        breakdown = model.params_count_breakdown()

        assert "lm_head (shared with embedding)" in breakdown
        assert breakdown["lm_head (shared with embedding)"] == 0
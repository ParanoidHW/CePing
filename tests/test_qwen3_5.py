"""Tests for Qwen3.5 MoE Model.

Qwen3.5-35B-A3B features:
- Hybrid attention: 3 linear + 1 full per 4-layer cycle
- MoE: 256 experts, top-8 routing, shared expert
- 40 layers, ~36B params
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
    ShardedQwen3_5MoEBlock,
    generate_layer_types,
)
from llm_perf.modeling.layers import ShardedLinearAttention, ShardedAttention


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

        linear_blocks = [l for l in model.layers if isinstance(l.attention, ShardedLinearAttention)]
        full_blocks = [l for l in model.layers if isinstance(l.attention, ShardedAttention)]

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
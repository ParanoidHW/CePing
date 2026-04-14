"""Tests for model classes."""

import unittest
from llm_perf.models.base import ModelConfig, LayerConfig, BaseModel
from llm_perf.models.llama import LlamaConfig, LlamaModel
from llm_perf.models.moe import MoEConfig, MoEModel


class TestLayerConfig(unittest.TestCase):
    """Test LayerConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a LayerConfig."""
        layer = LayerConfig(
            name="test_layer",
            input_shape=(1, 10),
            output_shape=(1, 20),
            params_count=100,
            flops=200,
            activation_bytes=40,
        )
        self.assertEqual(layer.name, "test_layer")
        self.assertEqual(layer.params_count, 100)
        self.assertEqual(layer.flops, 200)
        self.assertFalse(layer.is_moe)

    def test_moe_layer(self):
        """Test creating an MoE layer config."""
        layer = LayerConfig(
            name="moe_layer",
            input_shape=(1, 10),
            output_shape=(1, 20),
            params_count=1000,
            flops=2000,
            activation_bytes=80,
            is_moe=True,
        )
        self.assertTrue(layer.is_moe)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass."""

    def test_required_fields(self):
        """Test that required fields must be provided."""
        config = ModelConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
        )
        self.assertEqual(config.name, "test")
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_attention_heads, 4)

    def test_optional_defaults(self):
        """Test default values for optional fields."""
        config = ModelConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
        )
        self.assertIsNone(config.num_key_value_heads)
        self.assertEqual(config.intermediate_size, 0)
        self.assertEqual(config.max_seq_len, 4096)
        self.assertEqual(config.dtype, "fp16")


class TestLlamaConfig(unittest.TestCase):
    """Test LlamaConfig post-init behavior."""

    def test_default_intermediate_size(self):
        """Test that intermediate size defaults to hidden_size * 8 / 3."""
        config = LlamaConfig(
            name="test",
            vocab_size=1000,
            hidden_size=300,
            num_layers=2,
            num_attention_heads=4,
        )
        self.assertEqual(config.intermediate_size, 800)

    def test_default_num_key_value_heads(self):
        """Test that num_key_value_heads defaults to num_attention_heads."""
        config = LlamaConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=8,
        )
        self.assertEqual(config.num_key_value_heads, 8)

    def test_custom_values_preserved(self):
        """Test that explicitly provided values are preserved."""
        config = LlamaConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=512,
        )
        self.assertEqual(config.num_key_value_heads, 2)
        self.assertEqual(config.intermediate_size, 512)


class TestLlamaModel(unittest.TestCase):
    """Test LlamaModel layer building."""

    def setUp(self):
        self.config = LlamaConfig(
            name="tiny-llama",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_seq_len=128,
            dtype="fp16",
        )

    def test_total_params_positive(self):
        """Test that total parameters is positive."""
        model = LlamaModel(self.config)
        self.assertGreater(model.total_params, 0)

    def test_layers_count(self):
        """Test correct number of layers is built."""
        model = LlamaModel(self.config)
        # embedding + 2 transformer layers (12 sublayers each: 6 attention + 6 ffn) + final_norm + lm_head
        expected = 1 + 2 * 12 + 2
        self.assertEqual(len(model.layers), expected)

    def test_layer_names(self):
        """Test that expected layer names exist."""
        model = LlamaModel(self.config)
        names = [layer.name for layer in model.layers]
        self.assertIn("embedding", names)
        self.assertIn("final_norm", names)
        self.assertIn("lm_head", names)
        self.assertIn("layer_0_q_proj", names)
        self.assertIn("layer_0_attention", names)
        self.assertIn("layer_1_down_proj", names)

    def test_get_layer_by_name(self):
        """Test retrieving a layer by name."""
        model = LlamaModel(self.config)
        layer = model.get_layer_by_name("embedding")
        self.assertIsNotNone(layer)
        self.assertEqual(layer.name, "embedding")
        self.assertIsNone(model.get_layer_by_name("nonexistent"))

    def test_to_dict_structure(self):
        """Test that to_dict produces expected structure."""
        model = LlamaModel(self.config)
        data = model.to_dict()
        self.assertEqual(data["name"], "tiny-llama")
        self.assertIn("config", data)
        self.assertIn("total_params", data)
        self.assertIn("layers", data)
        self.assertEqual(len(data["layers"]), len(model.layers))

    def test_flops_forward_backward(self):
        """Test forward and backward FLOPs relationship."""
        model = LlamaModel(self.config)
        self.assertEqual(model.total_flops_backward, model.total_flops_forward * 2)

    def test_activation_memory(self):
        """Test activation memory is positive."""
        model = LlamaModel(self.config)
        self.assertGreater(model.activation_memory, 0)


class TestMoEConfig(unittest.TestCase):
    """Test MoEConfig post-init behavior."""

    def test_default_expert_intermediate_size(self):
        """Test that expert_intermediate_size defaults to intermediate_size."""
        config = MoEConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
        )
        self.assertEqual(config.expert_intermediate_size, 512)

    def test_custom_expert_intermediate_size(self):
        """Test that custom expert_intermediate_size is preserved."""
        config = MoEConfig(
            name="test",
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            expert_intermediate_size=1024,
        )
        self.assertEqual(config.expert_intermediate_size, 1024)


class TestMoEModel(unittest.TestCase):
    """Test MoEModel layer building."""

    def setUp(self):
        self.config = MoEConfig(
            name="tiny-moe",
            vocab_size=1000,
            hidden_size=256,
            num_layers=8,
            num_attention_heads=4,
            intermediate_size=512,
            max_seq_len=128,
            dtype="fp16",
            num_experts=8,
            num_experts_per_token=2,
        )

    def test_total_params_positive(self):
        """Test that total parameters is positive."""
        model = MoEModel(self.config)
        self.assertGreater(model.total_params, 0)

    def test_moe_layers_present(self):
        """Test that MoE layers exist at expected positions."""
        model = MoEModel(self.config)
        moe_layers = [layer for layer in model.layers if layer.is_moe]
        self.assertGreater(len(moe_layers), 0)
        # Every 4th layer (index 3, 7) should be MoE
        names = [layer.name for layer in model.layers]
        self.assertIn("layer_3_router", names)
        self.assertIn("layer_7_router", names)

    def test_dense_layers_present(self):
        """Test that dense layers exist at non-MoE positions."""
        model = MoEModel(self.config)
        names = [layer.name for layer in model.layers]
        self.assertIn("layer_0_q_proj", names)
        self.assertIn("layer_1_up_proj", names)

    def test_get_layer_by_name(self):
        """Test retrieving an MoE layer by name."""
        model = MoEModel(self.config)
        layer = model.get_layer_by_name("layer_3_router")
        self.assertIsNotNone(layer)
        self.assertTrue(layer.is_moe)

    def test_to_dict_structure(self):
        """Test that to_dict produces expected structure."""
        model = MoEModel(self.config)
        data = model.to_dict()
        self.assertEqual(data["name"], "tiny-moe")
        self.assertIn("config", data)
        self.assertIn("total_params", data)


if __name__ == "__main__":
    unittest.main()

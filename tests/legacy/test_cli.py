"""Tests for CLI module."""

import unittest
from llm_perf.cli.main import create_model


class TestCreateModel(unittest.TestCase):
    """Test create_model function with Registry integration."""

    def test_create_model_by_type_llama(self):
        """Test creating LLaMA model by type field."""
        config = {
            "type": "llama",
            "name": "test-llama",
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 2,
            "num_attention_heads": 4,
            "max_seq_len": 128,
            "dtype": "fp16",
        }
        model = create_model(config)
        self.assertEqual(model.config.name, "test-llama")
        self.assertEqual(model.config.hidden_size, 256)
        self.assertEqual(model.config.num_layers, 2)
        self.assertGreater(model.total_params, 0)

    def test_create_model_by_type_moe(self):
        """Test creating MoE model by type field."""
        config = {
            "type": "moe",
            "name": "test-moe",
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 8,
            "num_attention_heads": 4,
            "max_seq_len": 128,
            "dtype": "fp16",
            "num_experts": 8,
            "num_experts_per_token": 2,
        }
        model = create_model(config)
        self.assertEqual(model.config.name, "test-moe")
        self.assertEqual(model.config.num_experts, 8)
        self.assertGreater(model.total_params, 0)

    def test_create_model_by_preset_llama7b(self):
        """Test creating model using preset configuration."""
        config = {
            "preset": "llama-7b",
        }
        model = create_model(config)
        # llama-7b preset has hidden_size=4096, num_layers=32
        self.assertEqual(model.config.hidden_size, 4096)
        self.assertEqual(model.config.num_layers, 32)

    def test_create_model_by_preset_with_override(self):
        """Test creating model with preset and custom overrides."""
        config = {
            "preset": "llama-7b",
            "hidden_size": 512,  # Override default 4096
            "num_layers": 4,     # Override default 32
        }
        model = create_model(config)
        self.assertEqual(model.config.hidden_size, 512)
        self.assertEqual(model.config.num_layers, 4)

    def test_create_model_by_preset_mixtral(self):
        """Test creating Mixtral model using preset."""
        config = {
            "preset": "mixtral-8x7b",
        }
        model = create_model(config)
        self.assertEqual(model.config.num_experts, 8)
        self.assertEqual(model.config.num_experts_per_token, 2)

    def test_create_model_by_registered_name(self):
        """Test creating model by registered name field."""
        config = {
            "name": "llama",
            "vocab_size": 1000,
            "hidden_size": 512,
            "num_layers": 4,
            "num_attention_heads": 8,
        }
        model = create_model(config)
        self.assertEqual(model.config.hidden_size, 512)
        self.assertEqual(model.config.num_layers, 4)

    def test_create_model_unknown_type_raises_error(self):
        """Test that unknown model type raises ValueError."""
        config = {
            "type": "unknown-model",
            "hidden_size": 256,
        }
        with self.assertRaises(ValueError) as ctx:
            create_model(config)
        self.assertIn("Unknown model", str(ctx.exception))
        self.assertIn("available models", str(ctx.exception).lower())

    def test_create_model_deepseek(self):
        """Test creating DeepSeek model."""
        config = {
            "type": "deepseek",
            "name": "test-deepseek",
            "vocab_size": 1000,
            "hidden_size": 512,
            "num_layers": 4,
            "num_attention_heads": 8,
        }
        model = create_model(config)
        self.assertEqual(model.config.name, "test-deepseek")
        self.assertGreater(model.total_params, 0)


if __name__ == "__main__":
    unittest.main()
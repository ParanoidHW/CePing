"""Tests for ModelRegistry and PipelineRegistry."""

import unittest

from llm_perf.core.registry import ModelRegistry, PipelineRegistry
from llm_perf.models.base import BaseModel, ModelConfig
from llm_perf.pipelines.register import register_all_pipelines


class TestModelRegistry(unittest.TestCase):
    """Test ModelRegistry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()

    def tearDown(self):
        """Clean up after tests."""
        pass  # Don't clear singleton registry

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        self.assertIs(registry1, registry2)

    def test_register_and_create(self):
        """Test registering and creating a model."""
        # Register a test model
        self.registry.register(
            name="test-model",
            config_class=ModelConfig,
            model_class=BaseModel,
            description="Test model",
            category="test",
            default_config={"vocab_size": 1000, "hidden_size": 512, "num_layers": 2},
        )

        # Check registration
        self.assertTrue(self.registry.is_registered("test-model"))
        self.assertIn("test-model", self.registry.list_models())

        # Get info
        info = self.registry.get_info("test-model")
        self.assertEqual(info.name, "test-model")
        self.assertEqual(info.category, "test")

    def test_list_by_category(self):
        """Test listing models by category."""
        # Register models in different categories
        self.registry.register(
            name="test-llm",
            config_class=ModelConfig,
            model_class=BaseModel,
            description="Test LLM",
            category="llm",
            default_config={"vocab_size": 1000, "hidden_size": 512, "num_layers": 2},
        )
        self.registry.register(
            name="test-vae",
            config_class=ModelConfig,
            model_class=BaseModel,
            description="Test VAE",
            category="vae",
            default_config={"vocab_size": 1, "hidden_size": 256, "num_layers": 4},
        )

        by_category = self.registry.list_by_category()
        self.assertIn("llm", by_category)
        self.assertIn("vae", by_category)
        self.assertIn("test-llm", by_category["llm"])
        self.assertIn("test-vae", by_category["vae"])

    def test_unregister(self):
        """Test unregistering a model."""
        self.registry.register(
            name="temp-model",
            config_class=ModelConfig,
            model_class=BaseModel,
            description="Temporary model",
            category="test",
        )

        self.assertTrue(self.registry.is_registered("temp-model"))
        self.registry.unregister("temp-model")
        self.assertFalse(self.registry.is_registered("temp-model"))

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        self.registry.register(
            name="unique-model",
            config_class=ModelConfig,
            model_class=BaseModel,
            description="Unique model",
            category="test",
        )

        with self.assertRaises(ValueError):
            self.registry.register(
                name="unique-model",
                config_class=ModelConfig,
                model_class=BaseModel,
                description="Duplicate",
                category="test",
            )

    def test_get_nonexistent_raises(self):
        """Test that getting nonexistent model raises KeyError."""
        with self.assertRaises(KeyError):
            self.registry.get_info("nonexistent-model")


class TestPipelineRegistry(unittest.TestCase):
    """Test PipelineRegistry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = PipelineRegistry()

    def tearDown(self):
        """Clean up after tests."""
        pass  # Don't clear singleton registry

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = PipelineRegistry()
        registry2 = PipelineRegistry()
        self.assertIs(registry1, registry2)

    def test_register_builtin_pipelines(self):
        """Test registering built-in pipelines."""
        register_all_pipelines()

        # Check inference pipeline
        self.assertTrue(self.registry.is_registered("inference"))
        info = self.registry.get_info("inference")
        self.assertIn("llm", info.supported_models)

        # Check diffusion-video pipeline
        self.assertTrue(self.registry.is_registered("diffusion-video"))
        info = self.registry.get_info("diffusion-video")
        self.assertIn("dit", info.supported_models)

    def test_list_pipelines(self):
        """Test listing registered pipelines."""
        register_all_pipelines()
        pipelines = self.registry.list_pipelines()
        self.assertIn("inference", pipelines)
        self.assertIn("diffusion-video", pipelines)

    def test_get_for_model_category(self):
        """Test getting pipelines for model category."""
        register_all_pipelines()

        llm_pipelines = self.registry.get_for_model_category("llm")
        self.assertIn("inference", llm_pipelines)

        dit_pipelines = self.registry.get_for_model_category("dit")
        self.assertIn("diffusion-video", dit_pipelines)


class TestModelRegistryIntegration(unittest.TestCase):
    """Integration tests for ModelRegistry with real models."""

    def test_llama_registration(self):
        """Test that Llama models are registered."""
        registry = ModelRegistry()

        # Llama should be registered
        self.assertTrue(registry.is_registered("llama"))

        # Get info
        info = registry.get_info("llama")
        self.assertEqual(info.category, "llm")

    def test_moe_registration(self):
        """Test that MoE models are registered."""
        registry = ModelRegistry()

        self.assertTrue(registry.is_registered("moe"))
        info = registry.get_info("moe")
        self.assertEqual(info.category, "moe")

    def test_wan_models_registration(self):
        """Test that Wan2.1 models are registered."""
        registry = ModelRegistry()

        # Check all Wan models
        self.assertTrue(registry.is_registered("wan-text-encoder"))
        self.assertTrue(registry.is_registered("wan-dit"))
        self.assertTrue(registry.is_registered("wan-vae"))

        # Check categories
        self.assertEqual(registry.get_info("wan-text-encoder").category, "text_encoder")
        self.assertEqual(registry.get_info("wan-dit").category, "dit")
        self.assertEqual(registry.get_info("wan-vae").category, "vae")

    def test_create_llama_model(self):
        """Test creating a Llama model through registry."""
        registry = ModelRegistry()

        model = registry.create(
            "llama",
            name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
        )

        self.assertIsNotNone(model)
        self.assertEqual(model.config.name, "llama-7b")
        self.assertEqual(model.config.hidden_size, 4096)

    def test_registry_to_dict(self):
        """Test converting registry to dictionary."""
        registry = ModelRegistry()
        data = registry.to_dict()

        self.assertIn("models", data)
        self.assertIn("categories", data)


if __name__ == "__main__":
    unittest.main()

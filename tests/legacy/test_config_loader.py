"""Tests for ConfigLoader."""

import unittest
import tempfile
import json
from pathlib import Path

from llm_perf.utils.config_loader import (
    ConfigLoader,
    ModelConfigDict,
    HardwareConfigDict,
    StrategyConfigDict,
)
from llm_perf.models.base import ModelConfig
from llm_perf.strategy.base import StrategyConfig
from llm_perf.hardware.device import Device


class TestConfigLoaderBasics(unittest.TestCase):
    """Test basic ConfigLoader functionality."""

    def test_list_available_presets(self):
        """Test listing all available presets."""
        presets = ConfigLoader.list_available_presets()

        self.assertIn("models", presets)
        self.assertIn("hardware", presets)
        self.assertIn("strategies", presets)
        self.assertIn("devices", presets)

        self.assertIn("llama-7b", presets["models"])
        self.assertIn("tp8", presets["strategies"])
        self.assertIn("h100_8gpu", presets["hardware"])

    def test_list_config_files(self):
        """Test listing configuration files."""
        files = ConfigLoader.list_config_files()

        self.assertIn("models", files)
        self.assertIn("hardware", files)
        self.assertIn("strategies", files)

        self.assertTrue(len(files["models"]) >= 0)
        self.assertTrue(len(files["hardware"]) >= 0)
        self.assertTrue(len(files["strategies"]) >= 0)

    def test_get_device_preset_names(self):
        """Test getting device preset names."""
        names = ConfigLoader.get_device_preset_names()

        self.assertIn("A100-SXM-80GB", names)
        self.assertIn("H100-SXM-80GB", names)
        self.assertIn("Ascend-910B2", names)

    def test_get_device_config(self):
        """Test getting device config by preset."""
        config = ConfigLoader.get_device_config("H100-SXM-80GB")

        self.assertEqual(config.name, "H100-SXM-80GB")
        self.assertEqual(config.memory_gb, 80.0)
        self.assertTrue(config.fp16_tflops_cube > 0)


class TestModelConfigLoader(unittest.TestCase):
    """Test model configuration loading."""

    def test_load_from_preset(self):
        """Test loading model config from preset name."""
        config = ConfigLoader.load_model_config("llama-7b")

        self.assertEqual(config.name, "llama-7b")
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)

    def test_load_from_preset_llama70b(self):
        """Test loading llama-70b preset."""
        config = ConfigLoader.load_model_config("llama-70b")

        self.assertEqual(config.name, "llama-70b")
        self.assertEqual(config.hidden_size, 8192)
        self.assertEqual(config.num_key_value_heads, 8)

    def test_load_from_preset_mixtral(self):
        """Test loading mixtral-8x7b preset."""
        config = ConfigLoader.load_model_config("mixtral-8x7b")

        self.assertEqual(config.name, "mixtral-8x7b")
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_experts_per_token, 2)

    def test_load_from_dict(self):
        """Test loading model config from dictionary."""
        config_data = {
            "type": "llama",
            "name": "custom-model",
            "vocab_size": 50000,
            "hidden_size": 2048,
            "num_layers": 12,
            "num_attention_heads": 16,
        }

        config = ConfigLoader.load_model_config(config_data)

        self.assertEqual(config.name, "custom-model")
        self.assertEqual(config.vocab_size, 50000)
        self.assertEqual(config.hidden_size, 2048)

    def test_load_from_json_file(self):
        """Test loading model config from JSON file."""
        config = ConfigLoader.load_model_config("model_llama7b.json")

        self.assertEqual(config.name, "llama-7b")
        self.assertEqual(config.vocab_size, 32000)

    def test_override_with_kwargs(self):
        """Test overriding config with kwargs."""
        config = ConfigLoader.load_model_config("llama-7b", max_seq_len=8192, dtype="bf16")

        self.assertEqual(config.max_seq_len, 8192)
        self.assertEqual(config.dtype, "bf16")

    def test_invalid_preset_raises_error(self):
        """Test that invalid preset name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ConfigLoader.load_model_config("invalid-model")

        self.assertIn("Unknown model preset", str(ctx.exception))

    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ConfigLoader.load_model_config({"name": "test"})

        self.assertIn("missing required fields", str(ctx.exception))


class TestHardwareConfigLoader(unittest.TestCase):
    """Test hardware configuration loading."""

    def test_load_from_preset(self):
        """Test loading hardware config from preset."""
        config = ConfigLoader.load_hardware_config("h100_8gpu")

        self.assertEqual(config.device_preset, "H100-SXM-80GB")
        self.assertEqual(config.num_devices, 8)
        self.assertEqual(config.devices_per_node, 8)

    def test_load_from_dict(self):
        """Test loading hardware config from dictionary."""
        config_data = {
            "device_preset": "A100-SXM-80GB",
            "num_devices": 16,
            "devices_per_node": 8,
        }

        config = ConfigLoader.load_hardware_config(config_data)

        self.assertEqual(config.device_preset, "A100-SXM-80GB")
        self.assertEqual(config.num_devices, 16)

    def test_load_from_json_file(self):
        """Test loading hardware config from JSON file."""
        config = ConfigLoader.load_hardware_config("hardware_h100_8gpu.json")

        self.assertEqual(config.device_preset, "H100-SXM-80GB")
        self.assertEqual(config.num_devices, 8)

    def test_invalid_device_preset_raises_error(self):
        """Test that invalid device preset raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ConfigLoader.load_hardware_config({"device_preset": "InvalidDevice", "num_devices": 8})

        self.assertIn("Unknown device preset", str(ctx.exception))

    def test_ascend_hardware_preset(self):
        """Test loading Ascend hardware preset."""
        config = ConfigLoader.load_hardware_config("ascend_910c_8npu")

        self.assertEqual(config.device_preset, "Ascend-910C")
        self.assertEqual(config.num_devices, 8)


class TestStrategyConfigLoader(unittest.TestCase):
    """Test strategy configuration loading."""

    def test_load_from_preset(self):
        """Test loading strategy config from preset."""
        config = ConfigLoader.load_strategy_config("tp8")

        self.assertEqual(config.tp_degree, 8)
        self.assertEqual(config.pp_degree, 1)
        self.assertEqual(config.dp_degree, 1)
        self.assertEqual(config.world_size, 8)

    def test_load_from_preset_megatron_sp(self):
        """Test loading megatron_sp preset."""
        config = ConfigLoader.load_strategy_config("megatron_sp")

        self.assertEqual(config.tp_degree, 8)
        self.assertEqual(config.sp_degree, 8)
        self.assertTrue(config.sequence_parallel)
        self.assertTrue(config.use_megatron)

    def test_load_from_dict(self):
        """Test loading strategy config from dictionary."""
        config_data = {
            "tp": 4,
            "pp": 2,
            "dp": 2,
        }

        config = ConfigLoader.load_strategy_config(config_data)

        self.assertEqual(config.tp_degree, 4)
        self.assertEqual(config.pp_degree, 2)
        self.assertEqual(config.dp_degree, 2)
        self.assertEqual(config.world_size, 16)

    def test_load_from_json_file(self):
        """Test loading strategy config from JSON file."""
        config = ConfigLoader.load_strategy_config("strategy_tp8.json")

        self.assertEqual(config.tp_degree, 8)
        self.assertTrue(config.sequence_parallel)

    def test_load_moe_strategy(self):
        """Test loading MoE strategy."""
        config = ConfigLoader.load_strategy_config("moe_ep4")

        self.assertEqual(config.ep_degree, 4)

    def test_override_with_kwargs(self):
        """Test overriding config with kwargs."""
        config = ConfigLoader.load_strategy_config("tp8", activation_checkpointing=True, zero_stage=1)

        self.assertTrue(config.activation_checkpointing)
        self.assertEqual(config.zero_stage, 1)


class TestClusterCreation(unittest.TestCase):
    """Test cluster creation from hardware config."""

    def test_create_cluster_from_hardware_config(self):
        """Test creating cluster from hardware config dict."""
        hw_config = ConfigLoader.load_hardware_config("h100_8gpu")
        cluster = ConfigLoader.create_cluster_from_hardware_config(hw_config)

        self.assertEqual(cluster.num_devices, 8)
        self.assertEqual(cluster.devices_per_node, 8)
        self.assertEqual(cluster.devices[0].config.name, "H100-SXM-80GB")

    def test_create_cluster_from_dict(self):
        """Test creating cluster from raw dictionary."""
        cluster = ConfigLoader.create_cluster_from_hardware_config(
            {
                "device_preset": "A100-SXM-80GB",
                "num_devices": 16,
                "devices_per_node": 8,
            }
        )

        self.assertEqual(cluster.num_devices, 16)
        self.assertEqual(cluster.num_nodes, 2)


class TestModelCreation(unittest.TestCase):
    """Test model creation from config."""

    def test_create_llama_model(self):
        """Test creating LlamaModel from config."""
        model_config = ConfigLoader.load_model_config("llama-7b")
        model = ConfigLoader.create_model_from_config(model_config)

        model.build_layers()

        self.assertEqual(model.config.name, "llama-7b")
        self.assertTrue(len(model.layers) > 0)
        self.assertTrue(model.total_params > 0)

    def test_create_moe_model(self):
        """Test creating MoEModel from config."""
        model_config = ConfigLoader.load_model_config("mixtral-8x7b")
        model = ConfigLoader.create_model_from_config(model_config)

        model.build_layers()

        self.assertEqual(model.config.num_experts, 8)


class TestValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_validate_strategy_for_hardware_valid(self):
        """Test validation of valid strategy for hardware."""
        strategy = ConfigLoader.load_strategy_config("tp8")
        hw_config = ConfigLoader.load_hardware_config("h100_8gpu")

        result = ConfigLoader.validate_strategy_for_hardware(strategy, hw_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_strategy_for_hardware_invalid(self):
        """Test validation of invalid strategy (too many devices)."""
        strategy = ConfigLoader.load_strategy_config("tp8")
        hw_config = ConfigLoader.load_hardware_config(
            {
                "device_preset": "H100-SXM-80GB",
                "num_devices": 4,
            }
        )

        result = ConfigLoader.validate_strategy_for_hardware(strategy, hw_config)

        self.assertFalse(result["valid"])
        self.assertTrue(len(result["errors"]) > 0)
        self.assertIn("requires 8 devices", result["errors"][0])

    def test_validate_strategy_tp_crosses_nodes_warning(self):
        """Test warning when TP crosses nodes."""
        strategy = ConfigLoader.load_strategy_config(tp=16)
        hw_config = ConfigLoader.load_hardware_config("h100_8gpu")

        result = ConfigLoader.validate_strategy_for_hardware(strategy, hw_config)

        self.assertTrue(len(result["warnings"]) > 0)
        self.assertIn("exceeds devices per node", result["warnings"][0])


class TestJSONLoading(unittest.TestCase):
    """Test JSON file loading."""

    def test_load_json_file_exists(self):
        """Test loading existing JSON file."""
        data = ConfigLoader.load_json("model_llama7b.json")

        self.assertIn("name", data)
        self.assertEqual(data["name"], "llama-7b")

    def test_load_json_file_not_found(self):
        """Test error when file not found."""
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load_json("nonexistent.json")

    def test_load_json_with_absolute_path(self):
        """Test loading JSON with absolute path."""
        config_path = ConfigLoader.CONFIG_DIR / "model_llama7b.json"
        data = ConfigLoader.load_json(config_path)

        self.assertEqual(data["name"], "llama-7b")


class TestLoadAllConfigs(unittest.TestCase):
    """Test loading all configurations."""

    def test_load_all_configs_from_dir(self):
        """Test loading all config files from directory."""
        all_configs = ConfigLoader.load_all_configs_from_dir()

        self.assertIn("models", all_configs)
        self.assertIn("hardware", all_configs)
        self.assertIn("strategies", all_configs)


if __name__ == "__main__":
    unittest.main()

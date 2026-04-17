"""Tests for new CLI module."""

import unittest
import json
import tempfile
from pathlib import Path

from llm_perf.cli.main import create_model, create_hardware, create_strategy, main
from llm_perf.modeling import LlamaModel


class TestCreateModel(unittest.TestCase):
    """Test create_model function."""

    def test_create_model_by_preset(self):
        """Test creating model using preset."""
        config = {"preset": "llama-7b"}
        model = create_model(config)
        self.assertIsNotNone(model)
        self.assertEqual(model.hidden_size, 4096)
        self.assertEqual(model.num_layers, 32)

    def test_create_model_by_preset_with_override(self):
        """Test creating model with preset and overrides."""
        config = {
            "preset": "llama-7b",
            "hidden_size": 512,
            "num_layers": 4,
        }
        model = create_model(config)
        self.assertEqual(model.hidden_size, 512)
        self.assertEqual(model.num_layers, 4)

    def test_create_model_by_type(self):
        """Test creating model with type field."""
        config = {
            "type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 4,
            "num_heads": 32,
        }
        model = create_model(config)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, LlamaModel)
        self.assertEqual(model.hidden_size, 4096)
        self.assertEqual(model.num_layers, 4)


class TestCreateHardware(unittest.TestCase):
    """Test create_hardware function."""

    def test_create_hardware_with_preset(self):
        """Test creating hardware with preset."""
        config = {"device_preset": "H100-SXM-80GB", "num_devices": 8}
        device, cluster = create_hardware(config)
        self.assertIsNotNone(device)
        self.assertIsNotNone(cluster)
        self.assertEqual(device.config.name, "H100-SXM-80GB")
        self.assertEqual(cluster.num_devices, 8)

    def test_create_hardware_default(self):
        """Test creating hardware with defaults."""
        config = {"device_preset": "H100-SXM-80GB"}
        device, cluster = create_hardware(config)
        self.assertIsNotNone(device)
        self.assertIsNotNone(cluster)


class TestCreateStrategy(unittest.TestCase):
    """Test create_strategy function."""

    def test_create_strategy_default(self):
        """Test creating strategy with defaults."""
        config = {}
        strategy = create_strategy(config)
        self.assertEqual(strategy.tp_degree, 1)
        self.assertEqual(strategy.pp_degree, 1)
        self.assertEqual(strategy.dp_degree, 1)

    def test_create_strategy_with_tp(self):
        """Test creating strategy with TP."""
        config = {"tp": 8}
        strategy = create_strategy(config)
        self.assertEqual(strategy.tp_degree, 8)

    def test_create_strategy_with_all(self):
        """Test creating strategy with all degrees."""
        config = {"tp": 4, "pp": 2, "dp": 2}
        strategy = create_strategy(config)
        self.assertEqual(strategy.tp_degree, 4)
        self.assertEqual(strategy.pp_degree, 2)
        self.assertEqual(strategy.dp_degree, 2)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        self.model_config = {
            "preset": "llama-7b",
        }
        self.model_config_path = Path(self.temp_dir) / "model.json"
        with open(self.model_config_path, "w") as f:
            json.dump(self.model_config, f)

        self.hardware_config = {
            "device_preset": "H100-SXM-80GB",
            "num_devices": 8,
        }
        self.hardware_config_path = Path(self.temp_dir) / "hardware.json"
        with open(self.hardware_config_path, "w") as f:
            json.dump(self.hardware_config, f)

        self.strategy_config = {
            "tp": 8,
        }
        self.strategy_config_path = Path(self.temp_dir) / "strategy.json"
        with open(self.strategy_config_path, "w") as f:
            json.dump(self.strategy_config, f)

    def tearDown(self):
        """Clean up temp files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_cli_training_basic(self):
        """Test CLI training mode basic."""
        result = main(
            [
                "evaluate",
                "--model-config",
                str(self.model_config_path),
                "--hardware-config",
                str(self.hardware_config_path),
                "--strategy-config",
                str(self.strategy_config_path),
                "--mode",
                "training",
                "--batch-size",
                "32",
                "--seq-len",
                "2048",
            ]
        )
        self.assertIsNone(result)

    def test_cli_inference_basic(self):
        """Test CLI inference mode basic."""
        result = main(
            [
                "evaluate",
                "--model-config",
                str(self.model_config_path),
                "--hardware-config",
                str(self.hardware_config_path),
                "--strategy-config",
                str(self.strategy_config_path),
                "--mode",
                "inference",
                "--batch-size",
                "1",
                "--prompt-len",
                "512",
                "--generation-len",
                "128",
            ]
        )
        self.assertIsNone(result)

    def test_cli_json_output(self):
        """Test CLI with JSON output."""
        output_path = Path(self.temp_dir) / "output.json"
        result = main(
            [
                "evaluate",
                "--model-config",
                str(self.model_config_path),
                "--hardware-config",
                str(self.hardware_config_path),
                "--strategy-config",
                str(self.strategy_config_path),
                "--mode",
                "training",
                "--batch-size",
                "32",
                "--seq-len",
                "2048",
                "--json",
                "--output",
                str(output_path),
            ]
        )
        self.assertIsNone(result)
        self.assertTrue(output_path.exists())

    def test_cli_no_command(self):
        """Test CLI without command."""
        result = main([])
        self.assertEqual(result, 1)

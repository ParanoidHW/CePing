"""Tests for CLI module.

Tests cover:
- Argument parsing
- Config building
- Output formatting
- Interactive mode (basic)
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llm_perf.workload.cli.main import (
    parse_args,
    build_config,
    run_evaluation,
    _parse_hardware,
    _parse_strategy,
    _parse_params,
    main,
)
from llm_perf.workload.cli.output import CLIReporter, YamlReporter
from llm_perf.workload.cli.interactive import InteractiveSession
from llm_perf.workload.loader import WorkloadLoader, get_loader
from llm_perf.workload.schema import HardwareSchema, StrategySchema


class TestArgumentParsing:
    """Test argument parsing."""

    def test_parse_args_basic(self):
        """Test basic argument parsing."""
        args = parse_args(["--workload", "training/training", "--model", "llama-7b"])
        assert args.workload == "training/training"
        assert args.model == "llama-7b"
        assert args.output == "json"

    def test_parse_args_output_format(self):
        """Test output format parsing."""
        args = parse_args(["--output", "yaml"])
        assert args.output == "yaml"

        args = parse_args(["--output", "table"])
        assert args.output == "table"

    def test_parse_args_hardware(self):
        """Test hardware parsing."""
        args = parse_args(["--hardware", "910B:8"])
        assert args.hardware == "910B:8"

        args = parse_args(["--hardware", "device=H100,num_devices=8"])
        assert args.hardware == "device=H100,num_devices=8"

    def test_parse_args_strategy(self):
        """Test strategy parsing."""
        args = parse_args(["--strategy", "tp=8,pp=1"])
        assert args.strategy == "tp=8,pp=1"

    def test_parse_args_config_file(self):
        """Test config file parsing."""
        args = parse_args(["--config", "my_config.yaml"])
        assert args.config == Path("my_config.yaml")

    def test_parse_args_interactive(self):
        """Test interactive flag parsing."""
        args = parse_args(["--interactive"])
        assert args.interactive is True

    def test_parse_args_list_workloads(self):
        """Test list-workloads flag."""
        args = parse_args(["--list-workloads"])
        assert args.list_workloads is True

    def test_parse_args_list_models(self):
        """Test list-models flag."""
        args = parse_args(["--list-models"])
        assert args.list_models is True

    def test_parse_args_verbose(self):
        """Test verbose flag."""
        args = parse_args(["--verbose"])
        assert args.verbose is True

    def test_parse_args_dry_run(self):
        """Test dry-run flag."""
        args = parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_parse_args_breakdown_level(self):
        """Test breakdown-level parsing."""
        args = parse_args(["--breakdown-level", "stage", "phase"])
        assert args.breakdown_level == ["stage", "phase"]

    def test_parse_args_params(self):
        """Test params parsing."""
        args = parse_args(["--params", "batch_size=32,seq_len=4096"])
        assert args.params == "batch_size=32,seq_len=4096"


class TestHardwareParsing:
    """Test hardware spec parsing."""

    def test_parse_simple_format(self):
        """Test simple format: '910B:8'."""
        hardware = _parse_hardware("910B:8")
        assert hardware.device_preset == "910B"
        assert hardware.num_devices == 8

    def test_parse_explicit_format(self):
        """Test explicit format: 'device=H100,num_devices=16'."""
        hardware = _parse_hardware("device=H100,num_devices=16")
        assert hardware.device_preset == "H100"
        assert hardware.num_devices == 16

    def test_parse_default_values(self):
        """Test default values."""
        hardware = _parse_hardware("910B")
        assert hardware.device_preset == "910B"
        assert hardware.num_devices == 8


class TestStrategyParsing:
    """Test strategy spec parsing."""

    def test_parse_simple_format(self):
        """Test simple format: 'tp=8,pp=1'."""
        strategy = _parse_strategy("tp=8,pp=1")
        assert strategy.tp_degree == 8
        assert strategy.pp_degree == 1

    def test_parse_explicit_format(self):
        """Test explicit format: 'tp_degree=8,pp_degree=2'."""
        strategy = _parse_strategy("tp_degree=8,pp_degree=2")
        assert strategy.tp_degree == 8
        assert strategy.pp_degree == 2

    def test_parse_dp_ep(self):
        """Test DP and EP parsing."""
        strategy = _parse_strategy("tp=8,dp=4,sp=2")
        assert strategy.tp_degree == 8
        assert strategy.dp_degree == 4
        assert strategy.sp_degree == 2


class TestParamsParsing:
    """Test params spec parsing."""

    def test_parse_int_params(self):
        """Test integer params."""
        params = _parse_params("batch_size=32,seq_len=4096")
        assert params["batch_size"] == 32
        assert params["seq_len"] == 4096

    def test_parse_float_params(self):
        """Test float params."""
        params = _parse_params("dropout=0.1")
        assert params["dropout"] == 0.1

    def test_parse_string_params(self):
        """Test string params."""
        params = _parse_params("mode=inference")
        assert params["mode"] == "inference"


class TestConfigBuilding:
    """Test config building."""

    def test_build_config_from_args(self):
        """Test building config from arguments."""
        loader = get_loader()

        args = parse_args([
            "--workload", "training/training",
            "--model", "llama-7b",
            "--hardware", "910B:8",
            "--strategy", "tp=8,pp=1",
            "--params", "batch_size=32,seq_len=4096",
        ])

        config = build_config(args, loader)

        assert config["workload_name"] == "training/training"
        assert config["model_name"] == "llama-7b"
        assert config["hardware"]["device_preset"] == "910B"
        assert config["hardware"]["num_devices"] == 8
        assert config["strategy"]["tp_degree"] == 8
        assert config["params"]["batch_size"] == 32

    def test_build_config_missing_workload(self):
        """Test config with missing workload."""
        loader = get_loader()

        args = parse_args(["--model", "llama-7b"])
        config = build_config(args, loader)

        assert "workload_name" not in config


class TestOutputFormatting:
    """Test output formatting."""

    def test_yaml_reporter_format_dict(self):
        """Test YAML reporter formatting for dict."""
        import yaml

        reporter = YamlReporter()

        result_dict = {
            "summary": {"total_time_sec": 1.5, "throughput": 1000.0},
            "stages": [{"name": "forward", "total_time_sec": 1.0}],
        }

        yaml_output = yaml.dump(result_dict, default_flow_style=False)

        assert "total_time_sec: 1.5" in yaml_output
        assert "throughput: 1000.0" in yaml_output

    def test_cli_reporter_format_json(self):
        """Test CLI reporter JSON format."""
        reporter = CLIReporter()

        result_dict = {"success": True, "result": {"total_time_sec": 1.5}}

        output = reporter.format(result_dict, "json")
        parsed = json.loads(output)

        assert parsed["success"] is True
        assert parsed["result"]["total_time_sec"] == 1.5

    def test_cli_reporter_format_yaml(self):
        """Test CLI reporter YAML format."""
        import yaml

        reporter = CLIReporter()

        result_dict = {"success": True, "result": {"total_time_sec": 1.5}}

        output = reporter.format(result_dict, "yaml")
        parsed = yaml.safe_load(output)

        assert parsed["success"] is True

    def test_cli_reporter_format_table(self):
        """Test CLI reporter table format."""
        reporter = CLIReporter()

        result_dict = {
            "success": True,
            "result": {
                "summary": {"throughput": 1000.0, "mfu": 0.5},
                "stages": [{"name": "forward", "total_time_sec": 1.0}],
            },
        }

        output = reporter.format(result_dict, "table")

        assert "Results Summary" in output
        assert "Throughput:" in output


class TestCLIIntegration:
    """Test CLI integration."""

    def test_list_workloads(self):
        """Test list-workloads command."""
        loader = get_loader()

        args = parse_args(["--list-workloads"])
        result = main(["--list-workloads"])

        assert result == 0

    def test_list_models(self):
        """Test list-models command."""
        result = main(["--list-models"])

        assert result == 0

    def test_dry_run(self):
        """Test dry-run mode."""
        with patch("llm_perf.workload.cli.main._print_config_summary") as mock_print:
            result = main([
                "--workload", "training/training",
                "--model", "llama-7b",
                "--dry-run",
            ])

            mock_print.assert_called_once()
            assert result == 0


class TestInteractiveSession:
    """Test interactive session (basic)."""

    def test_interactive_session_init(self):
        """Test InteractiveSession initialization."""
        loader = get_loader()
        session = InteractiveSession(loader)

        assert session.loader is loader
        assert session.config == {}
        assert session.output_format == "json"

    def test_interactive_get_output_format(self):
        """Test get_output_format."""
        loader = get_loader()
        session = InteractiveSession(loader)
        session.output_format = "yaml"

        assert session.get_output_format() == "yaml"

    def test_interactive_should_run(self):
        """Test should_run."""
        loader = get_loader()
        session = InteractiveSession(loader)
        session._should_run = True

        assert session.should_run() is True

    def test_interactive_should_save(self):
        """Test should_save."""
        loader = get_loader()
        session = InteractiveSession(loader)
        session._should_save = True

        assert session.should_save() is True

    def test_interactive_get_save_path(self):
        """Test get_save_path."""
        loader = get_loader()
        session = InteractiveSession(loader)
        session.config = {"workload_name": "training/training", "model_name": "llama-7b"}
        session.output_format = "json"

        path = session.get_save_path()
        assert "training" in str(path)
        assert "llama-7b" in str(path)


class TestRunEvaluation:
    """Test run_evaluation."""

    def test_run_evaluation_missing_workload(self):
        """Test evaluation with missing workload."""
        config = {"model_name": "llama-7b"}
        result = run_evaluation(config)

        assert result["success"] is False
        assert "Workload not specified" in result["error"]

    def test_run_evaluation_missing_model(self):
        """Test evaluation with missing model."""
        config = {"workload_name": "training/training"}
        result = run_evaluation(config)

        assert result["success"] is False
        assert "Model not specified" in result["error"]


class TestCLIReporterEdgeCases:
    """Test CLIReporter edge cases."""

    def test_format_unknown_type(self):
        """Test unknown format type."""
        reporter = CLIReporter()

        with pytest.raises(ValueError, match="Unknown format type"):
            reporter.format({"success": True}, "unknown")

    def test_dict_to_table_empty(self):
        """Test dict_to_table with empty result."""
        reporter = CLIReporter()

        output = reporter._dict_to_table({})
        assert output == ""

    def test_dict_to_table_with_validation_errors(self):
        """Test dict_to_table with validation errors."""
        reporter = CLIReporter()

        result = {
            "validation": {
                "is_valid": False,
                "errors": ["Memory exceeds limit"],
            }
        }

        output = reporter._dict_to_table(result)
        assert "Validation Warnings" in output
        assert "Memory exceeds limit" in output
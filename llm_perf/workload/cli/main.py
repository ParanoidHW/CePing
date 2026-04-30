"""CLI main entry point.

Provides command-line interface for workload evaluation:
- Command-line parameter mode: --workload, --model, --hardware, --strategy, --config
- Configuration file mode: JSON/YAML config file
- Interactive mode: no arguments enters interactive configuration
- Output format: JSON/YAML/Table
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from llm_perf.modeling import create_model_from_config
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer import UnifiedAnalyzer, get_workload
from llm_perf.analyzer.base import WorkloadConfig

from llm_perf.workload.loader import WorkloadLoader, get_loader
from llm_perf.workload.schema import HardwareSchema, StrategySchema
from llm_perf.workload.engine import EvaluationEngine, EvaluationRequest
from llm_perf.workload.validator import WorkloadValidator
from llm_perf.workload.breakdown import calculate_breakdown, WorkloadBreakdown

from .output import CLIReporter, YamlReporter
from .interactive import InteractiveSession


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (default: sys.argv)

    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="eval-cli",
        description="LLM Performance Evaluator CLI - Unified evaluation entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Command-line parameter mode
  eval-cli --workload inference --model llama-7b --hardware 910B:8 --output json
  eval-cli --workload training --model deepseek-v3 --hardware H100-SXM-80GB:8 --strategy tp=8,pp=1 --output table

  # Configuration file mode
  eval-cli --config my_config.yaml --output json
  eval-cli --config my_config.json --output table

  # Interactive mode
  eval-cli --interactive
  eval-cli

  # List available options
  eval-cli --list-workloads
  eval-cli --list-models
        """,
    )

    parser.add_argument(
        "--workload", "-w",
        help="Workload name (e.g., 'inference/autoregressive', 'training/training')"
    )

    parser.add_argument(
        "--model", "-m",
        help="Model name (e.g., 'llama-7b', 'deepseek-v3')"
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Evaluation config file (JSON/YAML)"
    )

    parser.add_argument(
        "--hardware",
        help="Hardware spec (e.g., '910B:8', 'H100-SXM-80GB:8', 'device=H100,num_devices=8')"
    )

    parser.add_argument(
        "--strategy",
        help="Parallel strategy (e.g., 'tp=8,pp=1', 'tp_degree=8,pp_degree=1')"
    )

    parser.add_argument(
        "--output", "-o",
        choices=["json", "yaml", "table"],
        default="json",
        help="Output format (default: json)"
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file path"
    )

    parser.add_argument(
        "--breakdown-level",
        choices=["stage", "phase", "submodule", "memory"],
        nargs="+",
        default=["stage"],
        help="Breakdown levels to show (default: stage)"
    )

    parser.add_argument(
        "--params",
        help="Workload parameters (e.g., 'batch_size=32,seq_len=4096')"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )

    parser.add_argument(
        "--list-workloads",
        action="store_true",
        help="List all available workloads"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running evaluation"
    )

    return parser.parse_args(argv)


def run_cli_mode(args: argparse.Namespace) -> int:
    """Run CLI mode (non-interactive).

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    loader = get_loader()

    if args.list_workloads:
        return _list_workloads(loader)

    if args.list_models:
        return _list_models(loader)

    config = build_config(args, loader)

    if args.dry_run:
        _print_config_summary(config)
        return 0

    if args.verbose:
        _print_config_summary(config)

    result = run_evaluation(config)

    reporter = CLIReporter()
    output = reporter.format(result, args.output)

    if args.output_file:
        reporter.save(result, args.output_file, args.output)
        print(f"Results saved to: {args.output_file}")
    else:
        print(output)

    return 0 if result.get("success", False) else 1


def run_interactive_mode() -> int:
    """Run interactive mode.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    loader = get_loader()
    session = InteractiveSession(loader)

    config = session.run()

    if config is None:
        print("Configuration cancelled.")
        return 1

    if session.should_run():
        result = run_evaluation(config)

        output_format = session.get_output_format()
        reporter = CLIReporter()
        output = reporter.format(result, output_format)

        print(output)

        if session.should_save():
            save_path = session.get_save_path()
            reporter.save(result, save_path, output_format)
            print(f"Results saved to: {save_path}")

        return 0 if result.get("success", False) else 1

    return 0


def build_config(args: argparse.Namespace, loader: WorkloadLoader) -> Dict[str, Any]:
    """Build evaluation config from arguments.

    Args:
        args: Parsed arguments
        loader: WorkloadLoader instance

    Returns:
        Config dict
    """
    config: Dict[str, Any] = {}

    if args.config:
        file_config = _load_config_file(args.config)
        config.update(file_config)

    if args.workload:
        workload_schema = loader.get_workload_schema(args.workload)
        config["workload"] = workload_schema.to_dict()
        config["workload_name"] = args.workload

    if args.model:
        model_schema = loader.get_model_schema(args.model)
        config["model"] = model_schema.to_dict()
        config["model_name"] = args.model

    if args.hardware:
        hardware = _parse_hardware(args.hardware)
        config["hardware"] = hardware.to_dict()

    if args.strategy:
        strategy = _parse_strategy(args.strategy)
        config["strategy"] = strategy.to_dict()

    if args.params:
        params = _parse_params(args.params)
        config["params"] = params

    return config


def run_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluation with config.

    Args:
        config: Evaluation config

    Returns:
        Result dict
    """
    loader = get_loader()

    workload_name = config.get("workload_name")
    model_name = config.get("model_name")

    if not workload_name:
        return {"success": False, "error": "Workload not specified"}

    if not model_name:
        return {"success": False, "error": "Model not specified"}

    hardware_config = config.get("hardware", {})
    strategy_config = config.get("strategy", {})
    params = config.get("params", {})

    try:
        model_yaml = loader.load_model_yaml(model_name)
        model = create_model_from_config(model_yaml)

        hardware_schema = HardwareSchema(
            device_preset=hardware_config.get("device_preset", "910B"),
            num_devices=hardware_config.get("num_devices", 8),
        )

        strategy_schema = StrategySchema(
            tp_degree=strategy_config.get("tp_degree", 1),
            pp_degree=strategy_config.get("pp_degree", 1),
            dp_degree=strategy_config.get("dp_degree", 1),
            ep_degree=strategy_config.get("ep_degree", 1),
            sp_degree=strategy_config.get("sp_degree", 1),
        )

        request = EvaluationRequest(
            workload_name=workload_name,
            model_name=model_name,
            hardware=hardware_schema,
            strategy=strategy_schema,
            params=params,
        )

        engine = EvaluationEngine(loader)
        result = engine.evaluate(request, model)

        return result.to_dict()

    except Exception as e:
        return {"success": False, "error": str(e)}


def _list_workloads(loader: WorkloadLoader) -> int:
    """List available workloads."""
    categories = loader.list_workload_categories()

    print("Available Workloads:")
    print()

    for category, workloads in sorted(categories.items()):
        print(f"  [{category}]")
        for w in workloads:
            print(f"    - {category}/{w}")

    print()
    print(f"Total: {sum(len(v) for v in categories.values())} workloads")

    return 0


def _list_models(loader: WorkloadLoader) -> int:
    """List available models."""
    models = loader.list_models()

    print("Available Models:")
    print()

    for m in models:
        print(f"  - {m}")

    print()
    print(f"Total: {len(models)} models")

    return 0


def _load_config_file(path: Path) -> Dict[str, Any]:
    """Load config from JSON or YAML file."""
    content = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        return json.loads(content)
    elif path.suffix in [".yaml", ".yml"]:
        return yaml.safe_load(content)
    else:
        raise ValueError(f"Unknown config file format: {path.suffix}")


def _parse_hardware(spec: str) -> HardwareSchema:
    """Parse hardware spec string.

    Formats:
    - '910B:8' -> device_preset=910B, num_devices=8
    - 'device=H100-SXM-80GB,num_devices=8'
    """
    if ":" in spec and "=" not in spec:
        parts = spec.split(":")
        return HardwareSchema(
            device_preset=parts[0],
            num_devices=int(parts[1]) if len(parts) > 1 else 8,
        )

    params = {}
    for part in spec.split(","):
        if "=" in part:
            key, value = part.split("=")
            params[key.strip()] = value.strip()

    return HardwareSchema(
        device_preset=params.get("device", "910B"),
        num_devices=int(params.get("num_devices", 8)),
    )


def _parse_strategy(spec: str) -> StrategySchema:
    """Parse strategy spec string.

    Formats:
    - 'tp=8,pp=1' -> tp_degree=8, pp_degree=1
    - 'tp_degree=8,pp_degree=1'
    """
    params = {}
    for part in spec.split(","):
        if "=" in part:
            key, value = part.split("=")
            key = key.strip()
            value = value.strip()

            if key in ["tp", "tp_degree"]:
                params["tp_degree"] = int(value)
            elif key in ["pp", "pp_degree"]:
                params["pp_degree"] = int(value)
            elif key in ["dp", "dp_degree"]:
                params["dp_degree"] = int(value)
            elif key in ["ep", "ep_degree"]:
                params["ep_degree"] = int(value)
            elif key in ["sp", "sp_degree"]:
                params["sp_degree"] = int(value)

    return StrategySchema(
        tp_degree=params.get("tp_degree", 1),
        pp_degree=params.get("pp_degree", 1),
        dp_degree=params.get("dp_degree", 1),
        ep_degree=params.get("ep_degree", 1),
        sp_degree=params.get("sp_degree", 1),
    )


def _parse_params(spec: str) -> Dict[str, Any]:
    """Parse params spec string.

    Format: 'batch_size=32,seq_len=4096'
    """
    params = {}
    for part in spec.split(","):
        if "=" in part:
            key, value = part.split("=")
            key = key.strip()
            value = value.strip()

            try:
                params[key] = int(value)
            except ValueError:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value

    return params


def _print_config_summary(config: Dict[str, Any]) -> None:
    """Print configuration summary."""
    print("─" * 50)
    print("Configuration Summary:")
    print("─" * 50)

    if "workload_name" in config:
        print(f"  Workload: {config['workload_name']}")

    if "model_name" in config:
        print(f"  Model: {config['model_name']}")

    hardware = config.get("hardware", {})
    if hardware:
        device = hardware.get("device_preset", "unknown")
        num = hardware.get("num_devices", 1)
        print(f"  Hardware: {num}x {device}")

    strategy = config.get("strategy", {})
    if strategy:
        tp = strategy.get("tp_degree", 1)
        pp = strategy.get("pp_degree", 1)
        dp = strategy.get("dp_degree", 1)
        print(f"  Strategy: TP={tp}, PP={pp}, DP={dp}")

    params = config.get("params", {})
    if params:
        print("  Parameters:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    print("─" * 50)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point.

    Args:
        argv: Optional argument list (default: sys.argv)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args(argv)

    if args.interactive or (len(sys.argv) == 1 and argv is None):
        return run_interactive_mode()
    else:
        return run_cli_mode(args)


if __name__ == "__main__":
    sys.exit(main())
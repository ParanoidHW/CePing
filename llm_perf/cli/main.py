"""Main CLI entry point."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from llm_perf.modeling import create_model_from_config
from llm_perf.analyzer import infer_workload
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer import UnifiedAnalyzer, get_workload, list_workloads
from llm_perf.reporter import TableReporter, JSONReporter, HTMLReporter


def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def create_model(config: dict):
    """Create model from configuration."""
    return create_model_from_config(config)


def create_hardware(config: dict):
    """Create hardware configuration."""
    device_preset = config.get("device_preset")

    if device_preset:
        device = Device.from_preset(device_preset)
    else:
        device_config = config.get("device", {})
        device = Device.from_dict(device_config)

    topology = NetworkTopology(
        name="default",
        intra_node_bandwidth_gbps=config.get("intra_node_bw_gbps", 200.0),
        intra_node_latency_us=config.get("intra_node_latency_us", 1.0),
        inter_node_bandwidth_gbps=config.get("inter_node_bw_gbps", 25.0),
        inter_node_latency_us=config.get("inter_node_latency_us", 10.0),
    )

    num_devices = config.get("num_devices", 8)

    cluster = Cluster.create_homogeneous(device.config, num_devices, topology)

    return device, cluster


def create_strategy(config: dict) -> StrategyConfig:
    """Create strategy configuration."""
    return StrategyConfig(
        tp_degree=config.get("tp", 1),
        pp_degree=config.get("pp", 1),
        dp_degree=config.get("dp", 1),
        ep_degree=config.get("ep", 1),
    )


def cmd_evaluate(args):
    """Execute evaluate command."""
    model_config = load_config(args.model_config)
    hardware_config = load_config(args.hardware_config)
    strategy_config = load_config(args.strategy_config)

    model = create_model(model_config)
    device, cluster = create_hardware(hardware_config)
    strategy = create_strategy(strategy_config)

    model_name = model.name if hasattr(model, "name") else model._name if hasattr(model, "_name") else "unknown"
    hidden_size = model.hidden_size if hasattr(model, "hidden_size") else 4096
    num_layers = model.num_layers if hasattr(model, "num_layers") else 32

    print(f"Model: {model_name}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {num_layers}")

    print(f"\nHardware: {device.config.name}")
    print(f"  Devices: {cluster.num_devices}")

    print(f"\nStrategy:")
    print(f"  TP: {strategy.tp_degree}, PP: {strategy.pp_degree}, DP: {strategy.dp_degree}")
    print(f"  World size: {strategy.world_size}")

    workload = args.workload if args.workload else infer_workload(model_name, args.mode)

    analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

    if args.mode == "training":
        result = analyzer.analyze(
            workload,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
    else:
        result = analyzer.analyze(
            workload,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            generation_len=args.generation_len,
        )

    reporter = TableReporter()
    print("\n" + reporter.report(result, generation_len=args.generation_len if args.mode == "inference" else 0))

    if args.json:
        json_reporter = JSONReporter()
        output_path = Path(args.output) if args.output else Path(f"{args.mode}_result.json")
        json_reporter.save(result, output_path)
        print(f"\nSaved JSON report to {output_path}")

    if args.html:
        html_reporter = HTMLReporter()
        output_path = Path(args.output.replace(".json", ".html")) if args.output else Path(f"{args.mode}_result.html")
        html_reporter.save(result, output_path, generation_len=args.generation_len if args.mode == "inference" else 0)
        print(f"Saved HTML report to {output_path}")


def cmd_compare(args):
    """Execute compare command."""
    results = {}

    model_config = load_config(args.model_config)
    hardware_config = load_config(args.hardware_config)

    model = create_model(model_config)
    device, cluster = create_hardware(hardware_config)

    workload = (
        args.workload
        if args.workload
        else infer_workload(model._name if hasattr(model, "_name") else "llama", args.mode)
    )

    for strategy_path in args.strategy_configs:
        strategy_config = load_config(strategy_path)
        strategy = create_strategy(strategy_config)

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            workload,
            batch_size=args.batch_size,
            seq_len=args.seq_len if args.mode == "training" else None,
            prompt_len=args.prompt_len if args.mode == "inference" else None,
            generation_len=args.generation_len if args.mode == "inference" else None,
        )

        strategy_name = strategy_path.stem if isinstance(strategy_path, Path) else strategy_path
        results[strategy_name] = result

    reporter = TableReporter()
    print(reporter.report_comparison(results, args.metric))


def cmd_list_workloads(args):
    """List available workload presets."""
    workloads = list_workloads()
    print("Available workload presets:")
    for name, info in workloads.items():
        print(f"  {name}: {info['description']} ({info['type']})")


def main(argv: Optional[list] = None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="llm-perf",
        description="LLM Training and Inference Performance Evaluator",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single configuration")
    eval_parser.add_argument("--model-config", required=True, help="Model configuration JSON")
    eval_parser.add_argument("--hardware-config", required=True, help="Hardware configuration JSON")
    eval_parser.add_argument("--strategy-config", required=True, help="Strategy configuration JSON")
    eval_parser.add_argument("--mode", choices=["training", "inference"], default="inference")
    eval_parser.add_argument("--workload", help="Workload preset name (auto-inferred if not specified)")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--seq-len", type=int, default=4096)
    eval_parser.add_argument("--prompt-len", type=int, default=1024)
    eval_parser.add_argument("--generation-len", type=int, default=128)
    eval_parser.add_argument("--json", action="store_true", help="Output JSON report")
    eval_parser.add_argument("--html", action="store_true", help="Output HTML report")
    eval_parser.add_argument("--output", help="Output file path")
    eval_parser.set_defaults(func=cmd_evaluate)

    compare_parser = subparsers.add_parser("compare", help="Compare multiple strategies")
    compare_parser.add_argument("--model-config", required=True)
    compare_parser.add_argument("--hardware-config", required=True)
    compare_parser.add_argument("--strategy-configs", nargs="+", required=True)
    compare_parser.add_argument("--mode", choices=["training", "inference"], default="inference")
    compare_parser.add_argument("--workload", help="Workload preset name")
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument("--seq-len", type=int, default=4096)
    compare_parser.add_argument("--prompt-len", type=int, default=1024)
    compare_parser.add_argument("--generation-len", type=int, default=128)
    compare_parser.add_argument("--metric", default="throughput", help="Metric to compare")
    compare_parser.set_defaults(func=cmd_compare)

    list_parser = subparsers.add_parser("list-workloads", help="List available workload presets")
    list_parser.set_defaults(func=cmd_list_workloads)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

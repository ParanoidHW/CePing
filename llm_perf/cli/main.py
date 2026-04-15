"""Main CLI entry point."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ..core.registry import ModelRegistry
from ..models.registry import get_model_presets  # Models auto-registered on import
from ..hardware.device import Device
from ..hardware.cluster import Cluster, NetworkConfig
from ..strategy.base import StrategyConfig
from ..analyzer.training import TrainingAnalyzer
from ..analyzer.inference import InferenceAnalyzer
from ..reporter.table import TableReporter
from ..reporter.json_reporter import JSONReporter
from ..reporter.html_reporter import HTMLReporter


def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_model(config: dict):
    """Create model from configuration using ModelRegistry.

    Supports two modes:
    1. Preset mode: config contains 'preset' field (e.g., 'llama-7b', 'mixtral-8x7b')
    2. Type mode: config contains 'type' field (e.g., 'llama', 'moe')
    3. Name mode: config contains 'name' field matching a registered model

    Args:
        config: Model configuration dictionary

    Returns:
        Instantiated model from registry
    """
    registry = ModelRegistry()
    presets = get_model_presets()

    # Meta fields that should not be passed to model config
    meta_fields = {"type", "preset", "description"}

    # Check for preset first (e.g., "llama-7b", "mixtral-8x7b")
    preset_name = config.get("preset")
    if preset_name and preset_name in presets:
        preset_config = presets[preset_name]
        # Merge preset with user overrides
        merged_config = dict(preset_config)
        merged_config.update(config)
        # Remove meta fields before passing to model
        model_type = merged_config.get("type", preset_name)
        for field in meta_fields:
            merged_config.pop(field, None)
        return registry.create(model_type, **merged_config)

    # Use 'type' field as model identifier (e.g., "llama", "moe")
    model_type = config.get("type")
    if model_type and registry.is_registered(model_type):
        model_config = dict(config)
        for field in meta_fields:
            model_config.pop(field, None)
        return registry.create(model_type, **model_config)

    # Use 'name' field as model identifier
    model_name = config.get("name")
    if model_name and registry.is_registered(model_name):
        model_config = dict(config)
        for field in meta_fields:
            model_config.pop(field, None)
        return registry.create(model_name, **model_config)

    # Fallback: try to find by category or raise error
    available_models = registry.list_models()
    raise ValueError(
        f"Unknown model type/name: '{model_type or model_name}'. "
        f"Available models: {available_models}. "
        f"Available presets: {list(presets.keys())}"
    )


def create_hardware(config: dict):
    """Create hardware configuration."""
    device_preset = config.get("device_preset")
    
    if device_preset:
        device = Device.from_preset(device_preset)
    else:
        device_config = config.get("device", {})
        device = Device.from_dict(device_config)
    
    network_config = NetworkConfig(
        intra_node_bandwidth_gbps=config.get("intra_node_bw_gbps", 900),
        intra_node_latency_us=config.get("intra_node_latency_us", 1.0),
        inter_node_bandwidth_gbps=config.get("inter_node_bw_gbps", 400),
        inter_node_latency_us=config.get("inter_node_latency_us", 2.0),
        topology=config.get("topology", "full_mesh"),
        oversubscription_ratio=config.get("oversubscription_ratio", 1.0),
    )
    
    num_devices = config.get("num_devices", 8)
    devices_per_node = config.get("devices_per_node", 8)
    
    cluster = Cluster.create_homogeneous(
        device.config, num_devices, network_config, devices_per_node
    )
    
    return device, cluster


def create_strategy(config: dict) -> StrategyConfig:
    """Create strategy configuration."""
    return StrategyConfig.from_dict({
        "model_name": config.get("model_name", ""),
        "parallelism": {
            "tp": config.get("tp", 1),
            "pp": config.get("pp", 1),
            "dp": config.get("dp", 1),
            "ep": config.get("ep", 1),
            "sp": config.get("sp", 1),
            "cp": config.get("cp", 1),
        },
        "scheduling": {
            "pipeline_schedule": config.get("pipeline_schedule", "1f1b"),
            "micro_batch_size": config.get("micro_batch_size", 1),
        },
        "optimization": {
            "activation_checkpointing": config.get("activation_checkpointing", False),
            "sequence_parallel": config.get("sequence_parallel", False),
            "use_megatron": config.get("use_megatron", True),
            "zero_stage": config.get("zero_stage", 0),
        }
    })


def cmd_evaluate(args):
    """Execute evaluate command."""
    # Load configurations
    model_config = load_config(args.model_config)
    hardware_config = load_config(args.hardware_config)
    strategy_config = load_config(args.strategy_config)
    
    # Create components
    model = create_model(model_config)
    device, cluster = create_hardware(hardware_config)
    strategy = create_strategy(strategy_config)
    
    print(f"Model: {model.config.name}")
    print(f"  Parameters: {model.total_params / 1e9:.2f}B")
    print(f"  Layers: {model.config.num_layers}")
    print(f"  Hidden size: {model.config.hidden_size}")
    
    print(f"\nHardware: {device.config.name}")
    print(f"  Devices: {cluster.num_devices}")
    print(f"  Nodes: {cluster.num_nodes}")
    
    print(f"\nStrategy:")
    print(f"  TP: {strategy.tp_degree}, PP: {strategy.pp_degree}, "
          f"DP: {strategy.dp_degree}, EP: {strategy.ep_degree}")
    print(f"  World size: {strategy.world_size}")
    
    # Analyze
    if args.mode == "training":
        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
        )
        
        # Report
        reporter = TableReporter()
        print("\n" + reporter.report_training(result))
        
        if args.json:
            json_reporter = JSONReporter()
            output_path = Path(args.output) if args.output else Path("training_result.json")
            json_reporter.save(result, output_path, {
                "model": model_config,
                "hardware": hardware_config,
                "strategy": strategy_config,
            })
            print(f"\nSaved JSON report to {output_path}")
        
        if args.html:
            html_reporter = HTMLReporter()
            output_path = Path(args.output.replace('.json', '.html')) if args.output else Path("training_result.html")
            html_reporter.save(result, output_path, "Training Performance Report", {
                "model": model_config,
                "hardware": hardware_config,
                "strategy": strategy_config,
            })
            print(f"Saved HTML report to {output_path}")
    
    else:  # inference
        analyzer = InferenceAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            generation_len=args.generation_len,
        )
        
        # Report
        reporter = TableReporter()
        print("\n" + reporter.report_inference(result))
        
        if args.json:
            json_reporter = JSONReporter()
            output_path = Path(args.output) if args.output else Path("inference_result.json")
            json_reporter.save(result, output_path, {
                "model": model_config,
                "hardware": hardware_config,
                "strategy": strategy_config,
            })
            print(f"\nSaved JSON report to {output_path}")
        
        if args.html:
            html_reporter = HTMLReporter()
            output_path = Path(args.output.replace('.json', '.html')) if args.output else Path("inference_result.html")
            html_reporter.save(result, output_path, "Inference Performance Report", {
                "model": model_config,
                "hardware": hardware_config,
                "strategy": strategy_config,
            })
            print(f"Saved HTML report to {output_path}")


def cmd_compare(args):
    """Execute compare command."""
    # Load multiple strategy configs and compare
    results = {}
    
    model_config = load_config(args.model_config)
    hardware_config = load_config(args.hardware_config)
    
    model = create_model(model_config)
    device, cluster = create_hardware(hardware_config)
    
    for strategy_path in args.strategy_configs:
        strategy_config = load_config(strategy_path)
        strategy = create_strategy(strategy_config)
        
        if args.mode == "training":
            analyzer = TrainingAnalyzer(model, device, cluster, strategy)
            result = analyzer.analyze(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )
        else:
            analyzer = InferenceAnalyzer(model, device, cluster, strategy)
            result = analyzer.analyze(
                batch_size=args.batch_size,
                prompt_len=args.prompt_len,
                generation_len=args.generation_len,
            )
        
        strategy_name = strategy_path.stem if isinstance(strategy_path, Path) else strategy_path
        results[strategy_name] = result
    
    # Generate comparison report
    reporter = TableReporter()
    print(reporter.report_comparison(results, args.metric))


def main(argv: Optional[list] = None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="llm-perf",
        description="LLM Training and Inference Performance Evaluator"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single configuration")
    eval_parser.add_argument("--model-config", required=True, help="Model configuration JSON")
    eval_parser.add_argument("--hardware-config", required=True, help="Hardware configuration JSON")
    eval_parser.add_argument("--strategy-config", required=True, help="Strategy configuration JSON")
    eval_parser.add_argument("--mode", choices=["training", "inference"], required=True)
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--seq-len", type=int, default=4096)
    eval_parser.add_argument("--prompt-len", type=int, default=1024)
    eval_parser.add_argument("--generation-len", type=int, default=128)
    eval_parser.add_argument("--num-steps", type=int, default=1000)
    eval_parser.add_argument("--json", action="store_true", help="Output JSON report")
    eval_parser.add_argument("--html", action="store_true", help="Output HTML report")
    eval_parser.add_argument("--output", help="Output file path")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple strategies")
    compare_parser.add_argument("--model-config", required=True)
    compare_parser.add_argument("--hardware-config", required=True)
    compare_parser.add_argument("--strategy-configs", nargs="+", required=True)
    compare_parser.add_argument("--mode", choices=["training", "inference"], required=True)
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument("--seq-len", type=int, default=4096)
    compare_parser.add_argument("--prompt-len", type=int, default=1024)
    compare_parser.add_argument("--generation-len", type=int, default=128)
    compare_parser.add_argument("--metric", default="throughput", 
                               help="Metric to compare (throughput, memory, ttft)")
    compare_parser.set_defaults(func=cmd_compare)
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

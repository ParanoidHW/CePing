#!/usr/bin/env python3
"""
LLM Performance Evaluator - Quick Start Script

This script provides a simple way to run performance evaluations.
"""

import sys
import json
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_perf.models.llama import LlamaConfig, LlamaModel
from llm_perf.models.moe import MoEConfig, MoEModel
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster, NetworkConfig
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer.training import TrainingAnalyzer
from llm_perf.analyzer.inference import InferenceAnalyzer
from llm_perf.reporter.table import TableReporter
from llm_perf.reporter.json_reporter import JSONReporter


def example_training_eval():
    """Run a simple training evaluation example."""
    print("=" * 80)
    print("EXAMPLE: Training Performance Evaluation")
    print("=" * 80)
    
    # Create Llama-7B model
    model_config = LlamaConfig(
        name="llama-7b",
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        max_seq_len=4096,
        dtype="fp16",
    )
    model = LlamaModel(model_config)
    
    print(f"\nModel: {model.config.name}")
    print(f"  Parameters: {model.total_params / 1e9:.2f}B")
    print(f"  Layers: {model.config.num_layers}")
    
    # Create H100 cluster (8 GPUs)
    device = Device.from_preset("H100-SXM-80GB")
    network = NetworkConfig(
        intra_node_bandwidth_gbps=900,
        inter_node_bandwidth_gbps=400,
    )
    cluster = Cluster.create_homogeneous(device.config, 8, network, 8)
    
    print(f"\nHardware: {device.config.name}")
    print(f"  FP16 Compute: {device.config.fp16_tflops:.0f} TFLOPS")
    print(f"  Memory BW: {device.config.memory_bandwidth_gbps:.0f} GB/s")
    print(f"  Cluster: {cluster.num_devices} GPUs across {cluster.num_nodes} nodes")
    
    # Create strategy: TP=8 (single node)
    strategy = StrategyConfig(
        tp_degree=8,
        pp_degree=1,
        dp_degree=1,
        ep_degree=1,
        sequence_parallel=True,
    )
    
    print(f"\nStrategy: TP={strategy.tp_degree}, PP={strategy.pp_degree}, DP={strategy.dp_degree}")
    
    # Analyze
    analyzer = TrainingAnalyzer(model, device, cluster, strategy)
    result = analyzer.analyze(batch_size=32, seq_len=4096)
    
    # Report
    reporter = TableReporter()
    print("\n" + reporter.report_training(result))
    
    return result


def example_inference_eval():
    """Run a simple inference evaluation example."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Inference Performance Evaluation")
    print("=" * 80)
    
    # Create Llama-7B model
    model_config = LlamaConfig(
        name="llama-7b",
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        max_seq_len=4096,
        dtype="fp16",
    )
    model = LlamaModel(model_config)
    
    print(f"\nModel: {model.config.name}")
    
    # Create H100 cluster
    device = Device.from_preset("H100-SXM-80GB")
    network = NetworkConfig(
        intra_node_bandwidth_gbps=900,
        inter_node_bandwidth_gbps=400,
    )
    cluster = Cluster.create_homogeneous(device.config, 8, network, 8)
    
    # Create strategy: TP=4, PP=2
    strategy = StrategyConfig(
        tp_degree=4,
        pp_degree=2,
        dp_degree=1,
        ep_degree=1,
    )
    
    print(f"\nStrategy: TP={strategy.tp_degree}, PP={strategy.pp_degree}")
    
    # Analyze inference
    analyzer = InferenceAnalyzer(model, device, cluster, strategy)
    result = analyzer.analyze(
        batch_size=8,
        prompt_len=1024,
        generation_len=128,
    )
    
    # Report
    reporter = TableReporter()
    print("\n" + reporter.report_inference(result))
    
    return result


def example_moe_eval():
    """Run a MoE model evaluation example."""
    print("\n" + "=" * 80)
    print("EXAMPLE: MoE Model Evaluation")
    print("=" * 80)
    
    # Create Mixtral-style MoE model
    model_config = MoEConfig(
        name="mixtral-8x7b",
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        max_seq_len=32768,
        dtype="fp16",
        num_experts=8,
        num_experts_per_token=2,
    )
    model = MoEModel(model_config)
    
    print(f"\nModel: {model.config.name}")
    print(f"  Total Parameters: {model.total_params / 1e9:.2f}B")
    print(f"  Active Parameters: {model.total_params * 2 / 8 / 1e9:.2f}B")
    print(f"  Experts: {model.config.num_experts}, Top-K: {model.config.num_experts_per_token}")
    
    # Create hardware
    device = Device.from_preset("H100-SXM-80GB")
    network = NetworkConfig(
        intra_node_bandwidth_gbps=900,
        inter_node_bandwidth_gbps=400,
    )
    cluster = Cluster.create_homogeneous(device.config, 8, network, 8)
    
    # Create EP strategy
    strategy = StrategyConfig(
        tp_degree=2,
        pp_degree=1,
        dp_degree=1,
        ep_degree=4,  # Expert Parallelism
        sequence_parallel=True,
    )
    
    print(f"\nStrategy: TP={strategy.tp_degree}, EP={strategy.ep_degree}")
    
    # Analyze training
    analyzer = TrainingAnalyzer(model, device, cluster, strategy)
    result = analyzer.analyze(batch_size=16, seq_len=8192)
    
    reporter = TableReporter()
    print("\n" + reporter.report_training(result))
    
    return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Performance Evaluator Examples")
    parser.add_argument(
        "example",
        choices=["training", "inference", "moe", "all"],
        default="all",
        nargs="?",
        help="Which example to run"
    )
    parser.add_argument("--json", action="store_true", help="Save results as JSON")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.example in ["training", "all"]:
        results["training"] = example_training_eval()
    
    if args.example in ["inference", "all"]:
        results["inference"] = example_inference_eval()
    
    if args.example in ["moe", "all"]:
        results["moe"] = example_moe_eval()
    
    if args.json:
        json_reporter = JSONReporter()
        json_reporter.save_batch(results, "results.json")
        print("\n✅ Results saved to results.json")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

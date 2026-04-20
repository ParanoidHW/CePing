#!/usr/bin/env python3
"""Run model tests via HTTP API and output detailed memory values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from web.app import app


def format_table(headers: list, rows: list) -> str:
    """Format a simple table."""
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-".join("-" * (w + 2) for w in col_widths).strip()
    row_lines = []
    for row in rows:
        row_line = "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        row_lines.append(row_line)
    return f"| {header_line} |\n|{separator}|\n" + "\n".join(f"| {line} |" for line in row_lines)


def run_model_test(model_type: str, batch_size: int = 32, tp: int = 8) -> dict:
    """Run HTTP API test for a model."""
    client = app.test_client()

    response = client.post(
        "/api/evaluate",
        json={
            "model": {"type": model_type},
            "device": "H100-SXM-80GB",
            "num_devices": 8,
            "devices_per_node": 8,
            "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
            "strategy": {"tp": tp, "pp": 1, "dp": 1},
            "workload": "llm-training",
            "batch_size": batch_size,
        },
    )

    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}"}

    data = response.get_json()
    if not data.get("success"):
        return {"error": data.get("error", "Unknown error")}

    return data["result"]


def print_model_report(model_name: str, result: dict, batch_size: int, tp: int):
    """Print detailed model report."""
    print(f"\n{'=' * 60}")
    print(f"=== Model: {model_name} ===")
    print(f"Configuration: TP={tp}, batch_size={batch_size}")
    print("=" * 60)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    detailed = result.get("detailed_breakdown", {})

    mem = detailed.get("memory", {})
    by_type = mem.get("by_type", {})

    print("\nMemory by_type:")
    by_type_rows = [
        ("weight", f"{by_type.get('weight', 0):.2f}"),
        ("gradient", f"{by_type.get('gradient', 0):.2f}"),
        ("optimizer", f"{by_type.get('optimizer', 0):.2f}"),
        ("activation", f"{by_type.get('activation', 0):.2f}"),
        ("total", f"{by_type.get('total', 0):.2f}"),
    ]
    print(format_table(["Type", "Size (GB)"], by_type_rows))

    by_submodule_type = mem.get("by_submodule_type", {})
    if by_submodule_type:
        print("\nMemory by_submodule_type:")
        submodule_rows = []
        for stype, data in sorted(by_submodule_type.items()):
            weight_gb = data.get("weight_gb", 0)
            activation_gb = data.get("activation_gb", data.get("activations_gb", 0))
            submodule_rows.append((stype, f"{weight_gb:.2f}", f"{activation_gb:.2f}"))
        print(format_table(["Submodule Type", "Weight (GB)", "Activation (GB)"], submodule_rows))

    breakdown = result.get("breakdown", {})
    time_breakdown = breakdown.get("time_breakdown", {})
    compute_sec = time_breakdown.get("compute_sec", 0)

    phases = result.get("phases", [])
    forward_phases = [p for p in phases if "forward" in p.get("name", "").lower()]
    forward_time = sum(p.get("total_time_sec", 0) for p in forward_phases)

    print("\nTime:")
    time_rows = [
        ("compute_sec", f"{compute_sec:.2f} s"),
        ("forward phase", f"{forward_time:.2f} s"),
    ]
    print(format_table(["Metric", "Value"], time_rows))

    comm_breakdown = result.get("communication_breakdown", {})
    by_parallelism = comm_breakdown.get("by_parallelism", {})

    print("\nCommunication:")
    comm_rows = []
    if by_parallelism:
        for ptype, pdata in sorted(by_parallelism.items()):
            total_gb = pdata.get("total_bytes", 0) / 1e9
            ops_str = ""
            if "operations" in pdata:
                ops_parts = []
                for op_name, op_data in pdata["operations"].items():
                    op_gb = op_data.get("total_bytes", 0) / 1e9
                    ops_parts.append(f"{op_name}: {op_gb:.2f} GB")
                ops_str = ", ".join(ops_parts)
            comm_rows.append((ptype, f"{total_gb:.2f}", ops_str or "N/A"))
    if comm_rows:
        print(format_table(["Parallel Type", "Total (GB)", "Operations"], comm_rows))

    metadata = result.get("metadata", {})
    topology = metadata.get("topology", {})

    print("\nTopology Levels:")
    topo_rows = []
    if topology and "levels" in topology:
        for level in topology["levels"]:
            name = level.get("name", "unknown")
            bw = level.get("bandwidth_gbps", 0)
            topo_rows.append((name, f"{bw:.0f}"))
    if topo_rows:
        print(format_table(["Level", "Bandwidth (GB/s)"], topo_rows))

    print("\nValidation:")
    by_submodule_sum = sum(v.get("weight_gb", 0) for v in by_submodule_type.values())
    by_type_weight = by_type.get("weight", 0)
    weight_diff = abs(by_type_weight - by_submodule_sum)
    weight_ok = weight_diff < 0.1

    compute_diff = abs(compute_sec - forward_time)
    compute_ok = compute_diff < 0.01

    print(f"  by_type.weight == sum(by_submodule_type.weight_gb): {weight_ok}")
    print(f"    by_type.weight = {by_type_weight:.2f} GB")
    print(f"    sum(by_submodule_type) = {by_submodule_sum:.2f} GB")
    print(f"    diff = {weight_diff:.4f} GB")

    print(f"  compute_sec == forward phase time: {compute_ok}")
    print(f"    compute_sec = {compute_sec:.2f} s")
    print(f"    forward_time = {forward_time:.2f} s")
    print(f"    diff = {compute_diff:.4f} s")


def main():
    """Run tests for all models."""
    models = [
        ("llama-7b", 32, 8),
        ("llama-13b", 32, 8),
        ("llama-70b", 8, 8),
        ("deepseek-v3", 1, 8),
        ("wan-dit", 1, 8),
    ]

    all_results = {}
    for model_type, batch_size, tp in models:
        print(f"\nTesting {model_type}...")
        result = run_model_test(model_type, batch_size, tp)
        all_results[model_type] = (result, batch_size, tp)

    print("\n" + "=" * 80)
    print("DETAILED MEMORY VALUES REPORT")
    print("=" * 80)

    for model_type, (result, batch_size, tp) in all_results.items():
        print_model_report(model_type, result, batch_size, tp)

    print("\n" + "=" * 80)
    print("SUMMARY VALIDATION RESULTS")
    print("=" * 80)

    for model_type, (result, batch_size, tp) in all_results.items():
        if "error" in result:
            print(f"{model_type}: FAILED - {result['error']}")
            continue

        detailed = result.get("detailed_breakdown", {})
        mem = detailed.get("memory", {})
        by_type = mem.get("by_type", {})
        by_submodule_type = mem.get("by_submodule_type", {})

        breakdown = result.get("breakdown", {})
        compute_sec = breakdown.get("time_breakdown", {}).get("compute_sec", 0)
        phases = result.get("phases", [])
        forward_time = sum(p.get("total_time_sec", 0) for p in phases if "forward" in p.get("name", "").lower())

        by_type_weight = by_type.get("weight", 0)
        by_submodule_sum = sum(v.get("weight_gb", 0) for v in by_submodule_type.values())

        weight_ok = abs(by_type_weight - by_submodule_sum) < 0.1
        compute_ok = abs(compute_sec - forward_time) < 0.01

        status = "PASS" if weight_ok and compute_ok else "FAIL"
        print(f"{model_type}: {status}")
        if not weight_ok:
            print(f"  Weight mismatch: {by_type_weight:.2f} vs {by_submodule_sum:.2f}")
        if not compute_ok:
            print(f"  Compute mismatch: {compute_sec:.2f} vs {forward_time:.2f}")


if __name__ == "__main__":
    main()

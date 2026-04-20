#!/usr/bin/env python3
"""Verify memory data for multiple models after ShardedParameter fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from web.app import app


def verify_model(model_name: str, batch_size: int, expected_weight_gb: float, tolerance: float = 0.5):
    """Verify a single model's weight memory."""
    client = app.test_client()

    response = client.post(
        "/api/evaluate",
        json={
            "model": {"type": model_name},
            "device": "H100-SXM-80GB",
            "num_devices": 8,
            "devices_per_node": 8,
            "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
            "strategy": {"tp": 8, "pp": 1, "dp": 1},
            "workload": "llm-training",
            "batch_size": batch_size,
        },
    )

    if response.status_code != 200:
        return None, f"HTTP Error: {response.status_code}"

    data = response.get_json()
    if not data.get("success"):
        return None, f"API Error: {data.get('error', 'Unknown')}"

    result = data["result"]
    detailed = result.get("detailed_breakdown")
    if not detailed:
        return None, "No detailed_breakdown"

    mem = detailed.get("memory", {})
    by_type = mem.get("by_type", {})
    weight = by_type.get("weight", 0)

    if weight <= 0:
        return None, f"Invalid weight: {weight}"

    diff = abs(weight - expected_weight_gb)
    status = "✓" if diff <= tolerance else "✗"

    return {
        "weight": weight,
        "diff": diff,
        "status": status,
        "expected": expected_weight_gb,
    }, None


def main():
    models = [
        ("llama-7b", 32, 1.68),
        ("llama-13b", 32, 3.25),
        ("llama-70b", 8, 17.5),
        ("deepseek-v3", 1, 168.0),
        ("wan-dit", 1, 4.16),  # Wan2.1-14B has ~16.65B params, not 14B
    ]

    print("\n| Model | Expected Weight | Actual Weight | Diff | Status |")
    print("|-------|-----------------|---------------|------|--------|")

    results = []
    for model_name, batch_size, expected in models:
        result, error = verify_model(model_name, batch_size, expected)
        if error:
            print(f"| {model_name} | {expected:.2f} GB | ERROR | - | ✗ |")
            print(f"  Error: {error}")
            results.append((model_name, False, error))
        else:
            weight = result["weight"]
            diff = result["diff"]
            status = result["status"]
            print(f"| {model_name} | {expected:.2f} GB | {weight:.2f} GB | {diff:.2f} | {status} |")
            results.append((model_name, status == "✓", result))

    print("\n")
    failed = [(m, r) for m, ok, r in results if not ok]
    if failed:
        print(f"❌ {len(failed)} model(s) failed verification:")
        for model, detail in failed:
            print(f"  - {model}: {detail}")
    else:
        print("✅ All models passed verification!")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

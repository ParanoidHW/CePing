"""Helper functions."""

import json
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str):
    """Save data to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def format_bytes(bytes_val: float) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} EB"


def format_time(seconds: float) -> str:
    """Format time to human readable string."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def format_throughput(tokens_per_sec: float) -> str:
    """Format throughput to human readable string."""
    if tokens_per_sec >= 1e9:
        return f"{tokens_per_sec / 1e9:.2f}B tokens/s"
    elif tokens_per_sec >= 1e6:
        return f"{tokens_per_sec / 1e6:.2f}M tokens/s"
    elif tokens_per_sec >= 1e3:
        return f"{tokens_per_sec / 1e3:.2f}K tokens/s"
    else:
        return f"{tokens_per_sec:.2f} tokens/s"

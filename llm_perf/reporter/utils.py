"""Data transformation utilities for reporters."""

from typing import Dict, Any, List


def flatten_submodules(phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten all submodules from phases, including nested submodules.

    Args:
        phases: List of phase dicts containing submodules

    Returns:
        List of flattened submodule dicts with phase context
    """
    flat = []
    for phase in phases:
        phase_name = phase.get("name", "")
        component = phase.get("component", "")
        compute_type = phase.get("compute_type", "")

        for sm in phase.get("submodules", []):
            flat_sm = {
                "phase_name": phase_name,
                "component": component,
                "compute_type": compute_type,
                "name": sm.get("name", ""),
                "submodule_type": sm.get("submodule_type", ""),
                "time_sec": sm.get("time_sec", 0),
                "flops": sm.get("flops", 0),
                "count": sm.get("count", 1),
                "params_count": sm.get("params_count", 0),
                "weight_memory_gb": sm.get("memory", {}).get("weight_gb", 0),
                "gradient_memory_gb": sm.get("memory", {}).get("gradient_gb", 0),
                "optimizer_memory_gb": sm.get("memory", {}).get("optimizer_gb", 0),
                "activation_memory_gb": sm.get("memory", {}).get("activation_gb", 0),
                "communication_gb": sm.get("communication", {}).get("gb", 0),
                "communication_time_sec": sm.get("communication", {}).get("time_sec", 0),
            }
            flat.append(flat_sm)

            for nested in sm.get("nested_submodules", []):
                nested_sm = {
                    "phase_name": phase_name,
                    "component": component,
                    "compute_type": compute_type,
                    "parent_type": sm.get("submodule_type", ""),
                    "name": nested.get("name", ""),
                    "submodule_type": nested.get("submodule_type", ""),
                    "time_sec": nested.get("time_sec", 0),
                    "flops": nested.get("flops", 0),
                    "count": nested.get("count", 1),
                    "params_count": nested.get("params_count", 0),
                    "weight_memory_gb": nested.get("memory", {}).get("weight_gb", 0),
                    "gradient_memory_gb": nested.get("memory", {}).get("gradient_gb", 0),
                    "optimizer_memory_gb": nested.get("memory", {}).get("optimizer_gb", 0),
                    "activation_memory_gb": nested.get("memory", {}).get("activation_gb", 0),
                    "communication_gb": nested.get("communication", {}).get("gb", 0),
                    "communication_time_sec": nested.get("communication", {}).get("time_sec", 0),
                }
                flat.append(nested_sm)

    return flat


def format_bytes_gb(bytes_val: float) -> str:
    """Format bytes value to GB string with 2 decimal places.

    Args:
        bytes_val: Bytes value (can be float or int)

    Returns:
        Formatted GB string like "1.23 GB"
    """
    if bytes_val is None or bytes_val == 0:
        return "0.00 GB"
    gb = bytes_val / 1e9
    return f"{gb:.2f} GB"


def format_time_ms(sec_val: float) -> str:
    """Format seconds value to milliseconds string with 2 decimal places.

    Args:
        sec_val: Seconds value

    Returns:
        Formatted ms string like "123.45 ms"
    """
    if sec_val is None or sec_val == 0:
        return "0.00 ms"
    ms = sec_val * 1000
    return f"{ms:.2f} ms"


def format_percentage(value: float, total: float) -> str:
    """Format percentage value with 1 decimal place.

    Args:
        value: Part value
        total: Total value

    Returns:
        Formatted percentage string like "12.3%"
    """
    if total is None or total == 0:
        return "0.0%"
    pct = (value / total) * 100
    return f"{pct:.1f}%"


def group_by_submodule_type(flat_submodules: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Group flattened submodules by submodule_type and aggregate metrics.

    Args:
        flat_submodules: List of flattened submodule dicts

    Returns:
        Dict mapping submodule_type to aggregated metrics
        
    Note:
        Nested submodules (with parent_type) are aggregated separately with '_nested' suffix
        to avoid double-counting in total time calculation.
    """
    grouped = {}
    nested_grouped = {}
    
    for sm in flat_submodules:
        sm_type = sm.get("submodule_type", "unknown")
        parent_type = sm.get("parent_type")
        
        if parent_type:
            nested_type = f"{sm_type}_nested"
            if nested_type not in nested_grouped:
                nested_grouped[nested_type] = {
                    "time_sec": 0,
                    "flops": 0,
                    "weight_memory_gb": 0,
                    "gradient_memory_gb": 0,
                    "optimizer_memory_gb": 0,
                    "activation_memory_gb": 0,
                    "communication_gb": 0,
                    "communication_time_sec": 0,
                    "count": 0,
                    "parent_type": parent_type,
                }
            nested_grouped[nested_type]["time_sec"] += sm.get("time_sec", 0)
            nested_grouped[nested_type]["flops"] += sm.get("flops", 0)
            nested_grouped[nested_type]["weight_memory_gb"] += sm.get("weight_memory_gb", 0)
            nested_grouped[nested_type]["gradient_memory_gb"] += sm.get("gradient_memory_gb", 0)
            nested_grouped[nested_type]["optimizer_memory_gb"] += sm.get("optimizer_memory_gb", 0)
            nested_grouped[nested_type]["activation_memory_gb"] += sm.get("activation_memory_gb", 0)
            nested_grouped[nested_type]["communication_gb"] += sm.get("communication_gb", 0)
            nested_grouped[nested_type]["communication_time_sec"] += sm.get("communication_time_sec", 0)
            nested_grouped[nested_type]["count"] += sm.get("count", 1)
        else:
            if sm_type not in grouped:
                grouped[sm_type] = {
                    "time_sec": 0,
                    "flops": 0,
                    "weight_memory_gb": 0,
                    "gradient_memory_gb": 0,
                    "optimizer_memory_gb": 0,
                    "activation_memory_gb": 0,
                    "communication_gb": 0,
                    "communication_time_sec": 0,
                    "count": 0,
                }
            grouped[sm_type]["time_sec"] += sm.get("time_sec", 0)
            grouped[sm_type]["flops"] += sm.get("flops", 0)
            grouped[sm_type]["weight_memory_gb"] += sm.get("weight_memory_gb", 0)
            grouped[sm_type]["gradient_memory_gb"] += sm.get("gradient_memory_gb", 0)
            grouped[sm_type]["optimizer_memory_gb"] += sm.get("optimizer_memory_gb", 0)
            grouped[sm_type]["activation_memory_gb"] += sm.get("activation_memory_gb", 0)
            grouped[sm_type]["communication_gb"] += sm.get("communication_gb", 0)
            grouped[sm_type]["communication_time_sec"] += sm.get("communication_time_sec", 0)
            grouped[sm_type]["count"] += sm.get("count", 1)

    return {**grouped, **nested_grouped}


def group_by_component(flat_submodules: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Group flattened submodules by component and aggregate metrics.

    Args:
        flat_submodules: List of flattened submodule dicts

    Returns:
        Dict mapping component to aggregated metrics
    """
    grouped = {}
    for sm in flat_submodules:
        component = sm.get("component", "unknown")
        if component not in grouped:
            grouped[component] = {
                "time_sec": 0,
                "flops": 0,
                "weight_memory_gb": 0,
                "gradient_memory_gb": 0,
                "optimizer_memory_gb": 0,
                "activation_memory_gb": 0,
                "communication_gb": 0,
                "communication_time_sec": 0,
            }
        grouped[component]["time_sec"] += sm.get("time_sec", 0)
        grouped[component]["flops"] += sm.get("flops", 0)
        grouped[component]["weight_memory_gb"] += sm.get("weight_memory_gb", 0)
        grouped[component]["gradient_memory_gb"] += sm.get("gradient_memory_gb", 0)
        grouped[component]["optimizer_memory_gb"] += sm.get("optimizer_memory_gb", 0)
        grouped[component]["activation_memory_gb"] += sm.get("activation_memory_gb", 0)
        grouped[component]["communication_gb"] += sm.get("communication_gb", 0)
        grouped[component]["communication_time_sec"] += sm.get("communication_time_sec", 0)

    return grouped
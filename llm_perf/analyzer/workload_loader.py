"""Workload configuration loader.

Loads workload configurations from YAML files.
Workload describes computation characteristics, NOT model types.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml

from .base import (
    Phase,
    WorkloadConfig,
    WorkloadType,
    ComputeType,
    ThroughputMetric,
    ComputePattern,
)


CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "workloads"

_workload_cache: Dict[str, WorkloadConfig] = {}


BACKWARD_COMPAT_MAP: Dict[str, str] = {
    "llm-training": "training",
    "llm-inference": "autoregressive-inference",
    "llm-speculative-decoding": "speculative-decoding",
    "llm-rl-ppo": "rl-ppo",
    "llm-rl-grpo": "rl-grpo",
    "diffusion-training": "denoise-training",
    "diffusion-inference": "diffusion-pipeline",
    "diffusion-video-inference": "diffusion-pipeline",
    "moe-training": "training",
    "moe-inference": "autoregressive-inference",
    "resnet-training": "training",
    "resnet-inference": "resnet",
    "vae-encode": "conv-encoder",
    "vae-decode": "conv-decoder",
}


def load_workload_from_yaml(path: Union[str, Path]) -> WorkloadConfig:
    """Load workload configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        WorkloadConfig instance
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Workload config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return _parse_workload_dict(data, config_file=str(path))


def _parse_workload_dict(data: Dict[str, Any], config_file: Optional[str] = None) -> WorkloadConfig:
    """Parse workload config from dictionary.

    Args:
        data: Dictionary from YAML
        config_file: Optional source file path

    Returns:
        WorkloadConfig instance
    """
    workload_type_str = data.get("workload_type", "inference")
    workload_type = WorkloadType(workload_type_str)

    throughput_metric_str = data.get("throughput_metric", "tokens_per_sec")
    throughput_metric = ThroughputMetric(throughput_metric_str)

    phases = []
    for phase_data in data.get("phases", []):
        compute_type = ComputeType(phase_data.get("compute_type", "forward"))

        compute_pattern = None
        if "compute_pattern" in phase_data:
            compute_pattern = ComputePattern(phase_data["compute_pattern"])

        phases.append(
            Phase(
                name=phase_data.get("name", "phase"),
                compute_type=compute_type,
                component=phase_data.get("component", "main"),
                repeat=phase_data.get("repeat", 1),
                seq_len_factor=phase_data.get("seq_len_factor", 1.0),
                compute_pattern=compute_pattern,
                extra_params=phase_data.get("extra_params", {}),
            )
        )

    return WorkloadConfig(
        name=data.get("name", "unknown"),
        description=data.get("description", ""),
        workload_type=workload_type,
        phases=phases,
        default_params=data.get("default_params", {}),
        optimizer_factor=data.get("optimizer_factor", 1.5),
        gradient_accumulation_steps=data.get("gradient_accumulation_steps", 1),
        throughput_metric=throughput_metric,
        throughput_formula=data.get("throughput_formula"),
        component_mapping=data.get("component_mapping", {}),
        config_file=config_file,
    )


def get_workload(name: str) -> WorkloadConfig:
    """Get workload configuration by name.

    Supports:
    - New workload names: "training", "autoregressive-inference", etc.
    - Old workload names (backward compat): "llm-training", "diffusion-inference", etc.
    - YAML file paths: "configs/workloads/custom/my-workload.yaml"

    Args:
        name: Workload name or file path

    Returns:
        WorkloadConfig instance
    """
    if name in _workload_cache:
        return _workload_cache[name]

    if name.endswith(".yaml") or "/" in name:
        config = load_workload_from_yaml(name)
        _workload_cache[name] = config
        return config

    resolved_name = BACKWARD_COMPAT_MAP.get(name, name)

    yaml_path = _find_workload_yaml(resolved_name)

    if yaml_path:
        config = load_workload_from_yaml(yaml_path)
        _workload_cache[name] = config
        _workload_cache[resolved_name] = config
        return config

    raise KeyError(f"Workload '{name}' not found. Available: {list_workloads()}. Or provide a YAML file path.")


def _find_workload_yaml(name: str) -> Optional[Path]:
    """Find workload YAML file by name.

    Searches in:
    - configs/workloads/base/
    - configs/workloads/autoregressive/
    - configs/workloads/iterative/
    - configs/workloads/conv/
    - configs/workloads/custom/
    """
    search_dirs = [
        CONFIG_DIR / "base",
        CONFIG_DIR / "autoregressive",
        CONFIG_DIR / "iterative",
        CONFIG_DIR / "conv",
        CONFIG_DIR / "custom",
    ]

    for dir_path in search_dirs:
        yaml_path = dir_path / f"{name}.yaml"
        if yaml_path.exists():
            return yaml_path

    return None


def list_workloads() -> Dict[str, Dict[str, str]]:
    """List all available workload configurations.

    Returns:
        Dict mapping workload name to description and type
    """
    result = {}

    for old_name, resolved in BACKWARD_COMPAT_MAP.items():
        yaml_path = _find_workload_yaml(resolved)
        if yaml_path:
            config = load_workload_from_yaml(yaml_path)
            result[old_name] = {
                "description": config.description,
                "type": config.workload_type.value,
                "resolved_to": resolved,
            }

    for search_dir in [
        CONFIG_DIR / "base",
        CONFIG_DIR / "autoregressive",
        CONFIG_DIR / "iterative",
        CONFIG_DIR / "conv",
    ]:
        if search_dir.exists():
            for yaml_file in search_dir.glob("*.yaml"):
                config = load_workload_from_yaml(yaml_file)
                if config.name not in result and config.name not in BACKWARD_COMPAT_MAP.values():
                    result[config.name] = {
                        "description": config.description,
                        "type": config.workload_type.value,
                    }

    return result


def register_workload(config: WorkloadConfig) -> None:
    """Register a workload configuration manually.

    Args:
        config: WorkloadConfig to register
    """
    _workload_cache[config.name] = config


def load_all_builtin_workloads() -> None:
    """Pre-load all built-in workload configurations."""
    for search_dir in [
        CONFIG_DIR / "base",
        CONFIG_DIR / "autoregressive",
        CONFIG_DIR / "iterative",
        CONFIG_DIR / "conv",
    ]:
        if search_dir.exists():
            for yaml_file in search_dir.glob("*.yaml"):
                try:
                    config = load_workload_from_yaml(yaml_file)
                    _workload_cache[config.name] = config
                except Exception:
                    pass


def clear_cache() -> None:
    """Clear workload cache."""
    _workload_cache.clear()


def is_builtin_workload(name: str) -> bool:
    """Check if workload name is a built-in configuration.

    Args:
        name: Workload name

    Returns:
        True if built-in workload
    """
    resolved = BACKWARD_COMPAT_MAP.get(name, name)
    return _find_workload_yaml(resolved) is not None

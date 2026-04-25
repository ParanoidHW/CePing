"""Workload type definitions and validation schemas.

Each workload type has:
1. Required parameters
2. Optional parameters
3. Default values
"""

from enum import Enum
from typing import Dict, List, Any


class WorkloadType(Enum):
    """Workload type enumeration."""
    
    TRAINING = "training"
    INFERENCE = "inference"
    DIFFUSION = "diffusion"
    RL_TRAINING = "rl_training"
    PD_DISAGG = "pd_disagg"


WORKLOAD_PARAMS: Dict[WorkloadType, Dict[str, Any]] = {
    WorkloadType.TRAINING: {
        "required": ["batch_size", "seq_len"],
        "optional": ["gradient_accumulation_steps", "micro_batch_size"],
        "defaults": {
            "batch_size": 32,
            "seq_len": 4096,
            "micro_batch_size": 1,
        },
    },
    WorkloadType.INFERENCE: {
        "required": ["batch_size"],
        "optional": ["prompt_len", "generation_len", "kv_cache_len"],
        "defaults": {
            "batch_size": 8,
            "prompt_len": 1024,
            "generation_len": 128,
        },
    },
    WorkloadType.DIFFUSION: {
        "required": ["batch_size"],
        "optional": ["image_height", "image_width", "diffusion_steps", "num_frames"],
        "defaults": {
            "batch_size": 1,
            "image_height": 64,
            "image_width": 64,
            "diffusion_steps": 50,
            "num_frames": 81,
        },
    },
    WorkloadType.RL_TRAINING: {
        "required": ["batch_size", "seq_len"],
        "optional": ["num_rollouts", "ppo_epochs"],
        "defaults": {
            "batch_size": 32,
            "seq_len": 4096,
            "num_rollouts": 100,
            "ppo_epochs": 4,
        },
    },
    WorkloadType.PD_DISAGG: {
        "required": ["batch_size"],
        "optional": ["input_tokens", "output_tokens", "prefill_devices", "decode_devices"],
        "defaults": {
            "batch_size": 1,
            "input_tokens": 1000,
            "output_tokens": 100,
            "prefill_devices": 32,
            "decode_devices": 32,
        },
    },
}


DEFAULT_SUPPORTED_WORKLOADS: Dict[str, List[WorkloadType]] = {
    "llama": [WorkloadType.TRAINING, WorkloadType.INFERENCE],
    "deepseek": [WorkloadType.TRAINING, WorkloadType.INFERENCE],
    "mixtral": [WorkloadType.TRAINING, WorkloadType.INFERENCE],
    "qwen3_5": [WorkloadType.TRAINING, WorkloadType.INFERENCE],
    "qwen3_5_moe": [WorkloadType.TRAINING, WorkloadType.INFERENCE],
    "hunyuan_image_3": [WorkloadType.INFERENCE, WorkloadType.DIFFUSION],
    "wan_dit": [WorkloadType.DIFFUSION],
    "wan_text_encoder": [WorkloadType.INFERENCE],
    "wan_vae": [WorkloadType.DIFFUSION],
    "vae": [WorkloadType.DIFFUSION],
    "resnet": [WorkloadType.TRAINING, WorkloadType.INFERENCE],
}


def get_supported_workloads(architecture: str) -> List[WorkloadType]:
    """Get supported workloads for a given architecture.
    
    Args:
        architecture: Model architecture name
        
    Returns:
        List of supported workload types
    """
    return DEFAULT_SUPPORTED_WORKLOADS.get(architecture, [WorkloadType.TRAINING, WorkloadType.INFERENCE])


def is_workload_supported(architecture: str, workload_type: WorkloadType) -> bool:
    """Check if a workload type is supported for an architecture.
    
    Args:
        architecture: Model architecture name
        workload_type: Workload type to check
        
    Returns:
        True if supported, False otherwise
    """
    supported = get_supported_workloads(architecture)
    return workload_type in supported


def get_workload_defaults(workload_type: WorkloadType) -> Dict[str, Any]:
    """Get default parameters for a workload type.
    
    Args:
        workload_type: Workload type
        
    Returns:
        Dict of default parameters
    """
    params = WORKLOAD_PARAMS.get(workload_type, {})
    return params.get("defaults", {})
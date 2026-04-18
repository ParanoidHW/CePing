"""Built-in workload presets for UnifiedAnalyzer."""

from typing import Dict

from .base import (
    Phase,
    WorkloadConfig,
    WorkloadType,
    ComputeType,
    ThroughputMetric,
)


WORKLOAD_PRESETS: Dict[str, WorkloadConfig] = {}


def register_workload(config: WorkloadConfig) -> None:
    """Register a workload configuration."""
    WORKLOAD_PRESETS[config.name] = config


def get_workload(name: str) -> WorkloadConfig:
    """Get a workload configuration by name."""
    if name not in WORKLOAD_PRESETS:
        raise KeyError(f"Workload '{name}' not found. Available: {list(WORKLOAD_PRESETS.keys())}")
    return WORKLOAD_PRESETS[name]


def list_workloads() -> Dict[str, Dict[str, str]]:
    """List all workload presets with descriptions."""
    return {
        name: {"description": cfg.description, "type": cfg.workload_type.value}
        for name, cfg in WORKLOAD_PRESETS.items()
    }


def infer_workload(model_type: str, mode: str) -> str:
    """Infer workload preset from model type and mode.

    Args:
        model_type: Model type (llama, deepseek, wan-dit, vae, etc.)
        mode: Mode (training, inference)

    Returns:
        Workload preset name
    """
    model_lower = model_type.lower()

    if "wan" in model_lower or "dit" in model_lower:
        base = "diffusion"
    elif "vae" in model_lower:
        base = "diffusion"
    elif "moe" in model_lower or "mixtral" in model_lower or "deepseek" in model_lower:
        base = "moe"
    else:
        base = "llm"

    workload_name = f"{base}-{mode}"

    if workload_name in WORKLOAD_PRESETS:
        return workload_name

    return f"llm-{mode}"


# ========== LLM Workloads ==========

register_workload(
    WorkloadConfig(
        name="llm-training",
        description="Standard LLM training (forward + backward + optimizer)",
        workload_type=WorkloadType.TRAINING,
        phases=[
            Phase("forward", ComputeType.FORWARD, component="main", repeat=1),
            Phase("backward", ComputeType.BACKWARD, component="main", repeat=1),
            Phase("optimizer", ComputeType.OPTIMIZER, component="main", repeat=1),
        ],
        default_params={"batch_size": 32, "seq_len": 2048},
        optimizer_factor=1.5,
        throughput_metric=ThroughputMetric.TOKENS_PER_SEC,
    )
)

register_workload(
    WorkloadConfig(
        name="llm-inference",
        description="Standard LLM inference (prefill + decode)",
        workload_type=WorkloadType.INFERENCE,
        phases=[
            Phase("prefill", ComputeType.FORWARD, component="main", repeat=1, seq_len_factor=1.0),
            Phase("decode", ComputeType.FORWARD, component="main", repeat="generation_len", seq_len_factor=1.0),
        ],
        default_params={"batch_size": 1, "prompt_len": 512, "generation_len": 128},
        throughput_metric=ThroughputMetric.TOKENS_PER_SEC,
    )
)

# ========== LLM Mixed Workloads ==========

register_workload(
    WorkloadConfig(
        name="llm-speculative-decoding",
        description="Speculative decoding with draft and target models",
        workload_type=WorkloadType.MIXED,
        phases=[
            Phase("draft_prefill", ComputeType.FORWARD, component="draft", repeat=1),
            Phase("draft_decode", ComputeType.FORWARD, component="draft", repeat="draft_len"),
            Phase("target_verify", ComputeType.FORWARD, component="target", repeat="verify_count"),
        ],
        default_params={"batch_size": 1, "prompt_len": 512, "generation_len": 128, "draft_len": 5, "verify_count": 1},
        throughput_metric=ThroughputMetric.TOKENS_PER_SEC,
    )
)

register_workload(
    WorkloadConfig(
        name="llm-rl-ppo",
        description="RL PPO training with policy, value, and reference models",
        workload_type=WorkloadType.MIXED,
        phases=[
            Phase("rollout_generate", ComputeType.FORWARD, component="policy", repeat="num_rollout_steps"),
            Phase("policy_forward", ComputeType.FORWARD, component="policy", repeat=1),
            Phase("policy_backward", ComputeType.BACKWARD, component="policy", repeat=1),
            Phase("policy_optimizer", ComputeType.OPTIMIZER, component="policy", repeat=1),
            Phase("value_forward", ComputeType.FORWARD, component="value", repeat=1),
            Phase("value_backward", ComputeType.BACKWARD, component="value", repeat=1),
            Phase("value_optimizer", ComputeType.OPTIMIZER, component="value", repeat=1),
            Phase("reference_forward", ComputeType.FORWARD, component="reference", repeat=1),
        ],
        default_params={"batch_size": 16, "seq_len": 512, "num_rollout_steps": 100},
        optimizer_factor=1.5,
        throughput_metric=ThroughputMetric.SAMPLES_PER_SEC,
    )
)

register_workload(
    WorkloadConfig(
        name="llm-rl-grpo",
        description="GRPO (Group Relative Policy Optimization) training",
        workload_type=WorkloadType.MIXED,
        phases=[
            Phase("rollout_generate", ComputeType.FORWARD, component="policy", repeat="num_rollout_steps"),
            Phase("policy_forward", ComputeType.FORWARD, component="policy", repeat=1),
            Phase("policy_backward", ComputeType.BACKWARD, component="policy", repeat=1),
            Phase("policy_optimizer", ComputeType.OPTIMIZER, component="policy", repeat=1),
            Phase("reference_forward", ComputeType.FORWARD, component="reference", repeat=1),
        ],
        default_params={"batch_size": 16, "seq_len": 512, "num_rollout_steps": 100, "group_size": 4},
        optimizer_factor=1.5,
        throughput_metric=ThroughputMetric.SAMPLES_PER_SEC,
    )
)

# ========== Diffusion Workloads ==========

register_workload(
    WorkloadConfig(
        name="diffusion-training",
        description="Diffusion model training (single denoising step)",
        workload_type=WorkloadType.TRAINING,
        phases=[
            Phase("forward", ComputeType.FORWARD, component="dit", repeat=1),
            Phase("backward", ComputeType.BACKWARD, component="dit", repeat=1),
            Phase("optimizer", ComputeType.OPTIMIZER, component="dit", repeat=1),
        ],
        default_params={"batch_size": 1, "num_frames": 81, "height": 720, "width": 1280},
        optimizer_factor=1.2,
        throughput_metric=ThroughputMetric.PIXELS_PER_SEC,
    )
)

register_workload(
    WorkloadConfig(
        name="diffusion-inference",
        description="Diffusion inference (text_encoder + dit denoising + vae decoder)",
        workload_type=WorkloadType.INFERENCE,
        phases=[
            Phase("text_encoder", ComputeType.FORWARD, component="text_encoder", repeat=1),
            Phase("dit_denoise", ComputeType.FORWARD, component="dit", repeat="num_inference_steps"),
            Phase("vae_decoder", ComputeType.FORWARD, component="vae", repeat=1),
        ],
        default_params={
            "batch_size": 1,
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 50,
            "use_cfg": True,
        },
        throughput_metric=ThroughputMetric.PIXELS_PER_SEC,
    )
)

register_workload(
    WorkloadConfig(
        name="diffusion-video-inference",
        description="Full video generation pipeline (text_encoder + dit + vae)",
        workload_type=WorkloadType.INFERENCE,
        phases=[
            Phase("text_encoder", ComputeType.FORWARD, component="text_encoder", repeat=1),
            Phase("dit_denoise", ComputeType.FORWARD, component="dit", repeat="num_inference_steps"),
            Phase("vae_decoder", ComputeType.FORWARD, component="vae", repeat=1),
        ],
        default_params={"num_frames": 81, "height": 720, "width": 1280, "num_inference_steps": 50, "use_cfg": True},
        throughput_metric=ThroughputMetric.VIDEOS_PER_SEC,
    )
)

# ========== MoE Workloads ==========

register_workload(
    WorkloadConfig(
        name="moe-training",
        description="MoE model training with expert routing",
        workload_type=WorkloadType.TRAINING,
        phases=[
            Phase("forward", ComputeType.FORWARD, component="main", repeat=1),
            Phase("backward", ComputeType.BACKWARD, component="main", repeat=1),
            Phase("optimizer", ComputeType.OPTIMIZER, component="main", repeat=1),
        ],
        default_params={"batch_size": 32, "seq_len": 2048},
        optimizer_factor=2.0,
        throughput_metric=ThroughputMetric.TOKENS_PER_SEC,
    )
)

register_workload(
    WorkloadConfig(
        name="moe-inference",
        description="MoE model inference with expert routing",
        workload_type=WorkloadType.INFERENCE,
        phases=[
            Phase("prefill", ComputeType.FORWARD, component="main", repeat=1),
            Phase("decode", ComputeType.FORWARD, component="main", repeat="generation_len"),
        ],
        default_params={"batch_size": 1, "prompt_len": 512, "generation_len": 128},
        throughput_metric=ThroughputMetric.TOKENS_PER_SEC,
    )
)

# ========== VAE Workloads ==========

register_workload(
    WorkloadConfig(
        name="vae-encode",
        description="VAE encoder only",
        workload_type=WorkloadType.INFERENCE,
        phases=[
            Phase("encode", ComputeType.FORWARD, component="vae", repeat=1),
        ],
        default_params={"batch_size": 1, "num_frames": 81, "height": 720, "width": 1280},
        throughput_metric=ThroughputMetric.PIXELS_PER_SEC,
    )
)

register_workload(
    WorkloadConfig(
        name="vae-decode",
        description="VAE decoder only",
        workload_type=WorkloadType.INFERENCE,
        phases=[
            Phase("decode", ComputeType.FORWARD, component="vae", repeat=1),
        ],
        default_params={"batch_size": 1, "num_frames": 81, "height": 720, "width": 1280},
        throughput_metric=ThroughputMetric.PIXELS_PER_SEC,
    )
)

# ========== ResNet Workloads ==========

register_workload(
    WorkloadConfig(
        name="resnet-training",
        description="ResNet image classification training",
        workload_type=WorkloadType.TRAINING,
        phases=[
            Phase("forward", ComputeType.FORWARD, component="main", repeat=1),
            Phase("backward", ComputeType.BACKWARD, component="main", repeat=1),
            Phase("optimizer", ComputeType.OPTIMIZER, component="main", repeat=1),
        ],
        default_params={"batch_size": 256, "image_size": 224},
        optimizer_factor=1.5,
        throughput_metric=ThroughputMetric.IMAGES_PER_SEC,
    )
)

register_workload(
    WorkloadConfig(
        name="resnet-inference",
        description="ResNet image classification inference",
        workload_type=WorkloadType.INFERENCE,
        phases=[
            Phase("forward", ComputeType.FORWARD, component="main", repeat=1),
        ],
        default_params={"batch_size": 1, "image_size": 224},
        throughput_metric=ThroughputMetric.IMAGES_PER_SEC,
    )
)

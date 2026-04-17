"""LLM Performance Evaluator.

Main package for LLM training and inference performance evaluation.
"""

from .modeling import (
    ShardedTensor,
    ShardedModule,
    LlamaModel,
    DeepSeekModel,
    ShardedVAE,
    ShardedResNet,
    create_model_from_config,
    get_model_presets,
    ModelingRegistry,
)

from .hardware.device import Device
from .hardware.cluster import Cluster
from .strategy.base import StrategyConfig

__all__ = [
    "ShardedTensor",
    "ShardedModule",
    "LlamaModel",
    "DeepSeekModel",
    "ShardedVAE",
    "ShardedResNet",
    "create_model_from_config",
    "get_model_presets",
    "ModelingRegistry",
    "Device",
    "Cluster",
    "StrategyConfig",
]

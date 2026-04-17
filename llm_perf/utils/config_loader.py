"""Unified configuration loader for LLM performance estimation.

This module provides a centralized way to load and validate configurations
from JSON files or preset names, consolidating scattered configuration logic.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, TypeVar, TYPE_CHECKING
from dataclasses import dataclass, asdict

from ..hardware.device import Device, DeviceConfig
from ..hardware.cluster import Cluster, NetworkConfig
from ..hardware.topology import NetworkTopology, TopologyType
from ..legacy.models.base import ModelConfig, BaseModel
from ..strategy.base import StrategyConfig, SPType

if TYPE_CHECKING:
    from ..legacy.models.llama import LlamaModel, LlamaConfig
    from ..legacy.models.moe import MoEModel, MoEConfig

T = TypeVar("T")


@dataclass
class ModelConfigDict:
    """Model configuration from JSON file."""

    type: str
    name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    intermediate_size: int = 0
    max_seq_len: int = 4096
    dtype: str = "fp16"
    num_experts: Optional[int] = None
    num_experts_per_token: Optional[int] = None
    expert_intermediate_size: Optional[int] = None


@dataclass
class HardwareConfigDict:
    """Hardware configuration from JSON file."""

    device_preset: str
    num_devices: int
    devices_per_node: int = 8
    intra_node_bw_gbps: Optional[float] = None
    intra_node_latency_us: Optional[float] = None
    inter_node_bw_gbps: Optional[float] = None
    inter_node_latency_us: Optional[float] = None
    topology: Optional[Union[str, Dict[str, Any]]] = None
    oversubscription_ratio: float = 1.0


@dataclass
class StrategyConfigDict:
    """Strategy configuration from JSON file."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
    sp: int = 1
    cp: int = 1
    activation_checkpointing: bool = False
    sequence_parallel: bool = False
    use_megatron: bool = True
    zero_stage: int = 0
    sp_type: Optional[str] = None
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None


class ConfigLoader:
    """Unified configuration loader for model, hardware, and strategy configs.

    This class consolidates configuration loading from:
    - JSON files in configs/ directory
    - Preset names (for devices, models, strategies)
    - Direct dictionary inputs

    Features:
    - Single entry point for all configuration types
    - Validation of configuration parameters
    - Support for both file paths and preset names
    - Automatic type conversion and error handling
    """

    CONFIG_DIR = Path(__file__).parent.parent / "configs"

    MODEL_PRESETS = {
        "llama-7b": {
            "type": "llama",
            "name": "llama-7b",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
        "llama-13b": {
            "type": "llama",
            "name": "llama-13b",
            "vocab_size": 32000,
            "hidden_size": 5120,
            "num_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 40,
            "intermediate_size": 13824,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
        "llama-70b": {
            "type": "llama",
            "name": "llama-70b",
            "vocab_size": 32000,
            "hidden_size": 8192,
            "num_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 28672,
            "max_seq_len": 4096,
            "dtype": "fp16",
        },
        "mixtral-8x7b": {
            "type": "moe",
            "name": "mixtral-8x7b",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 8,
            "num_experts_per_token": 2,
            "expert_intermediate_size": 14336,
        },
    }

    STRATEGY_PRESETS = {
        "tp1": {"tp": 1, "pp": 1, "dp": 1, "ep": 1, "sp": 1, "cp": 1},
        "tp2": {"tp": 2, "pp": 1, "dp": 1, "ep": 1, "sp": 1, "cp": 1},
        "tp4": {"tp": 4, "pp": 1, "dp": 1, "ep": 1, "sp": 1, "cp": 1},
        "tp8": {"tp": 8, "pp": 1, "dp": 1, "ep": 1, "sp": 1, "cp": 1},
        "tp4_dp2": {"tp": 4, "pp": 1, "dp": 2, "ep": 1, "sp": 1, "cp": 1},
        "tp2_pp4": {"tp": 2, "pp": 4, "dp": 1, "ep": 1, "sp": 1, "cp": 1},
        "moe_ep4": {"tp": 1, "pp": 1, "dp": 1, "ep": 4, "sp": 1, "cp": 1},
        "megatron_sp": {
            "tp": 8,
            "pp": 1,
            "dp": 1,
            "ep": 1,
            "sp": 8,
            "cp": 1,
            "sequence_parallel": True,
            "use_megatron": True,
        },
    }

    HARDWARE_PRESETS = {
        "a100_8gpu": {
            "device_preset": "A100-SXM-80GB",
            "num_devices": 8,
            "devices_per_node": 8,
            "intra_node_bw_gbps": 600.0,
            "inter_node_bw_gbps": 200.0,
        },
        "h100_8gpu": {
            "device_preset": "H100-SXM-80GB",
            "num_devices": 8,
            "devices_per_node": 8,
            "intra_node_bw_gbps": 900.0,
            "inter_node_bw_gbps": 400.0,
        },
        "h200_8gpu": {
            "device_preset": "H200-SXM-141GB",
            "num_devices": 8,
            "devices_per_node": 8,
            "intra_node_bw_gbps": 900.0,
            "inter_node_bw_gbps": 400.0,
        },
        "mi300x_8gpu": {
            "device_preset": "MI300X",
            "num_devices": 8,
            "devices_per_node": 8,
            "intra_node_bw_gbps": 896.0,
            "inter_node_bw_gbps": 400.0,
        },
        "ascend_910b_8npu": {
            "device_preset": "Ascend-910B2",
            "num_devices": 8,
            "devices_per_node": 8,
            "intra_node_bw_gbps": 200.0,
            "inter_node_bw_gbps": 100.0,
        },
        "ascend_910c_8npu": {
            "device_preset": "Ascend-910C",
            "num_devices": 8,
            "devices_per_node": 8,
            "intra_node_bw_gbps": 400.0,
            "inter_node_bw_gbps": 200.0,
        },
    }

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration file.

        Args:
            path: Path to JSON file (absolute or relative to configs/)

        Returns:
            Dictionary with configuration data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON parsing fails
        """
        path = Path(path)

        if not path.is_absolute():
            path = cls.CONFIG_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON from {path}: {e}")

        return data

    @classmethod
    def validate_config(cls, config: Dict[str, Any], required_fields: List[str], config_type: str = "config") -> None:
        """Validate configuration has required fields.

        Args:
            config: Configuration dictionary
            required_fields: List of required field names
            config_type: Type name for error messages

        Raises:
            ValueError: If required fields are missing or invalid
        """
        missing = []
        invalid = []

        for field in required_fields:
            if field not in config:
                missing.append(field)
            elif config[field] is None and field in required_fields:
                invalid.append(field)

        if missing:
            raise ValueError(f"{config_type} missing required fields: {missing}")

        if invalid:
            raise ValueError(f"{config_type} has invalid (null) required fields: {invalid}")

    @classmethod
    def load_model_config(
        cls, path_or_name: Optional[Union[str, Path, Dict[str, Any]]] = None, **kwargs
    ) -> ModelConfig:
        """Load model configuration from file, preset name, or dictionary.

        Args:
            path_or_name: Can be:
                - Path to JSON file (absolute or relative to configs/)
                - Preset name (e.g., "llama-7b", "llama-70b", "mixtral-8x7b")
                - Dictionary with model config
                - None (use kwargs)
            **kwargs: Additional model parameters (override loaded config)

        Returns:
            ModelConfig instance

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If file doesn't exist

        Example:
            >>> config = ConfigLoader.load_model_config("llama-7b")
            >>> config = ConfigLoader.load_model_config("model_llama7b.json")
            >>> config = ConfigLoader.load_model_config({"type": "llama", "name": "custom", ...})
        """
        config_data = {}

        if path_or_name is None:
            config_data = kwargs
        elif isinstance(path_or_name, dict):
            config_data = path_or_name.copy()
        elif isinstance(path_or_name, (str, Path)):
            path_str = str(path_or_name)

            if path_str.endswith(".json"):
                config_data = cls.load_json(path_or_name)
            elif path_str in cls.MODEL_PRESETS:
                config_data = cls.MODEL_PRESETS[path_str].copy()
            else:
                preset_names = list(cls.MODEL_PRESETS.keys())
                raise ValueError(
                    f"Unknown model preset or file: '{path_str}'. "
                    f"Available presets: {preset_names}. "
                    f"Or provide a .json file path."
                )

        config_data.update(kwargs)

        cls.validate_config(
            config_data, ["name", "vocab_size", "hidden_size", "num_layers", "num_attention_heads"], "model config"
        )

        model_type = config_data.get("type", "llama")

        model_config = ModelConfig(
            name=config_data.get("name", "unknown"),
            vocab_size=config_data["vocab_size"],
            hidden_size=config_data["hidden_size"],
            num_layers=config_data["num_layers"],
            num_attention_heads=config_data["num_attention_heads"],
            num_key_value_heads=config_data.get("num_key_value_heads"),
            intermediate_size=config_data.get("intermediate_size", 0),
            max_seq_len=config_data.get("max_seq_len", 4096),
            dtype=config_data.get("dtype", "fp16"),
            num_experts=config_data.get("num_experts"),
            num_experts_per_token=config_data.get("num_experts_per_token"),
        )

        return model_config

    @classmethod
    def load_hardware_config(
        cls, path_or_name: Optional[Union[str, Path, Dict[str, Any]]] = None, **kwargs
    ) -> HardwareConfigDict:
        """Load hardware configuration from file, preset name, or dictionary.

        Args:
            path_or_name: Can be:
                - Path to JSON file (absolute or relative to configs/)
                - Preset name (e.g., "a100_8gpu", "h100_8gpu")
                - Dictionary with hardware config
                - None (use kwargs)
            **kwargs: Additional hardware parameters (override loaded config)

        Returns:
            HardwareConfigDict instance

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If file doesn't exist
        """
        config_data = {}

        if path_or_name is None:
            config_data = kwargs
        elif isinstance(path_or_name, dict):
            config_data = path_or_name.copy()
        elif isinstance(path_or_name, (str, Path)):
            path_str = str(path_or_name)

            if path_str.endswith(".json"):
                config_data = cls.load_json(path_or_name)
            elif path_str in cls.HARDWARE_PRESETS:
                config_data = cls.HARDWARE_PRESETS[path_str].copy()
            else:
                preset_names = list(cls.HARDWARE_PRESETS.keys())
                raise ValueError(
                    f"Unknown hardware preset or file: '{path_str}'. "
                    f"Available presets: {preset_names}. "
                    f"Or provide a .json file path."
                )

        config_data.update(kwargs)

        cls.validate_config(config_data, ["device_preset", "num_devices"], "hardware config")

        device_preset = config_data["device_preset"]
        if device_preset not in Device.PRESETS:
            available = list(Device.PRESETS.keys())
            raise ValueError(f"Unknown device preset: '{device_preset}'. Available: {available}")

        hw_config = HardwareConfigDict(
            device_preset=device_preset,
            num_devices=config_data["num_devices"],
            devices_per_node=config_data.get("devices_per_node", 8),
            intra_node_bw_gbps=config_data.get("intra_node_bw_gbps"),
            intra_node_latency_us=config_data.get("intra_node_latency_us"),
            inter_node_bw_gbps=config_data.get("inter_node_bw_gbps"),
            inter_node_latency_us=config_data.get("inter_node_latency_us"),
            topology=config_data.get("topology"),
            oversubscription_ratio=config_data.get("oversubscription_ratio", 1.0),
        )

        return hw_config

    @classmethod
    def load_strategy_config(
        cls, path_or_name: Optional[Union[str, Path, Dict[str, Any]]] = None, **kwargs
    ) -> StrategyConfig:
        """Load strategy configuration from file, preset name, or dictionary.

        Args:
            path_or_name: Can be:
                - Path to JSON file (absolute or relative to configs/)
                - Preset name (e.g., "tp8", "tp4_dp2", "megatron_sp")
                - Dictionary with strategy config
                - None (use kwargs)
            **kwargs: Additional strategy parameters (override loaded config)

        Returns:
            StrategyConfig instance

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If file doesn't exist
        """
        config_data = {}

        if path_or_name is None:
            config_data = kwargs
        elif isinstance(path_or_name, dict):
            config_data = path_or_name.copy()
        elif isinstance(path_or_name, (str, Path)):
            path_str = str(path_or_name)

            if path_str.endswith(".json"):
                config_data = cls.load_json(path_or_name)
            elif path_str in cls.STRATEGY_PRESETS:
                config_data = cls.STRATEGY_PRESETS[path_str].copy()
            else:
                preset_names = list(cls.STRATEGY_PRESETS.keys())
                raise ValueError(
                    f"Unknown strategy preset or file: '{path_str}'. "
                    f"Available presets: {preset_names}. "
                    f"Or provide a .json file path."
                )

        config_data.update(kwargs)

        strategy_config = StrategyConfig(
            model_name=config_data.get("model_name", ""),
            tp_degree=config_data.get("tp", 1),
            pp_degree=config_data.get("pp", 1),
            dp_degree=config_data.get("dp", 1),
            ep_degree=config_data.get("ep", 1),
            sp_degree=config_data.get("sp", 1),
            cp_degree=config_data.get("cp", 1),
            activation_checkpointing=config_data.get("activation_checkpointing", False),
            sequence_parallel=config_data.get("sequence_parallel", False),
            use_megatron=config_data.get("use_megatron", True),
            zero_stage=config_data.get("zero_stage", 0),
        )

        if config_data.get("sp_type"):
            try:
                strategy_config.sp_type = SPType(config_data["sp_type"])
            except ValueError:
                pass

        if config_data.get("ulysses_degree"):
            strategy_config.ulysses_degree = config_data["ulysses_degree"]

        if config_data.get("ring_degree"):
            strategy_config.ring_degree = config_data["ring_degree"]

        return strategy_config

    @classmethod
    def create_cluster_from_hardware_config(cls, hw_config: Union[HardwareConfigDict, Dict[str, Any]]) -> Cluster:
        """Create Cluster instance from hardware configuration.

        Args:
            hw_config: HardwareConfigDict or dictionary with hardware config

        Returns:
            Cluster instance

        Raises:
            ValueError: If configuration is invalid
        """
        if isinstance(hw_config, dict):
            hw_config = cls.load_hardware_config(hw_config)

        device = Device.from_preset(hw_config.device_preset)

        topology = cls._build_topology_from_config(hw_config)

        cluster = Cluster.create_homogeneous(
            device_config=device.config,
            num_devices=hw_config.num_devices,
            topology=topology,
            devices_per_node=hw_config.devices_per_node,
        )

        return cluster

    @classmethod
    def _build_topology_from_config(cls, hw_config: HardwareConfigDict) -> NetworkTopology:
        """Build NetworkTopology from hardware configuration.

        Args:
            hw_config: HardwareConfigDict instance

        Returns:
            NetworkTopology instance
        """
        topology_data = hw_config.topology

        if topology_data is None:
            intra_bw = hw_config.intra_node_bw_gbps or 600.0
            inter_bw = hw_config.inter_node_bw_gbps or 200.0

            return NetworkTopology.create_2tier_simple(
                intra_node_bw_gbps=intra_bw,
                inter_node_bw_gbps=inter_bw,
                devices_per_node=hw_config.devices_per_node,
            )

        if isinstance(topology_data, str):
            return NetworkTopology.create_2tier_simple(
                intra_node_bw_gbps=hw_config.intra_node_bw_gbps or 600.0,
                inter_node_bw_gbps=hw_config.inter_node_bw_gbps or 200.0,
            )

        if isinstance(topology_data, dict):
            return cls._parse_topology_dict(topology_data)

        return NetworkTopology.create_2tier_simple()

    @classmethod
    def _parse_topology_dict(cls, topo_data: Dict[str, Any]) -> NetworkTopology:
        """Parse topology dictionary into NetworkTopology.

        Args:
            topo_data: Dictionary with topology configuration

        Returns:
            NetworkTopology instance
        """
        topology_type = topo_data.get("topology_type", "hierarchical")

        if topology_type == "supernode" or "levels" in topo_data:
            levels = []
            for level_data in topo_data.get("levels", []):
                from ..hardware.topology import TopologyLevel

                level = TopologyLevel(
                    level=level_data.get("level", len(levels)),
                    name=level_data.get("name", f"level_{len(levels)}"),
                    bandwidth_gbps=level_data.get("bandwidth_gbps", 100.0),
                    latency_us=level_data.get("latency_us", 1.0),
                    devices_per_group=level_data.get("devices_per_group", 8),
                    oversubscription_ratio=level_data.get("oversubscription_ratio", 1.0),
                )
                levels.append(level)

            topo_type = TopologyType.HIERARCHICAL
            if topology_type == "supernode":
                topo_type = TopologyType.SUPERNODE

            return NetworkTopology(
                topology_type=topo_type,
                name=topo_data.get("name", "custom_topology"),
                levels=levels,
            )

        return NetworkTopology.create_2tier_simple(
            intra_node_bw_gbps=topo_data.get("intra_node_bw_gbps", 600.0),
            inter_node_bw_gbps=topo_data.get("inter_node_bw_gbps", 200.0),
        )

    @classmethod
    def create_model_from_config(cls, model_config: Union[ModelConfig, Dict[str, Any]]) -> BaseModel:
        """Create model instance from configuration.

        Args:
            model_config: ModelConfig or dictionary with model config

        Returns:
            BaseModel instance (LlamaModel or MoEModel)

        Raises:
            ValueError: If model type is unknown
        """
        from llm_perf.legacy.models.llama import LlamaModel, LlamaConfig
        from llm_perf.legacy.models.moe import MoEModel, MoEConfig

        if isinstance(model_config, dict):
            model_config = cls.load_model_config(model_config)

        model_type = getattr(model_config, "type", None)
        if model_type is None and isinstance(model_config, dict):
            model_type = model_config.get("type", "llama")
        elif model_type is None:
            model_type = "llama"

        num_experts = getattr(model_config, "num_experts", None)
        if num_experts is None and isinstance(model_config, dict):
            num_experts = model_config.get("num_experts")

        if num_experts and num_experts > 1:
            moe_config = MoEConfig(
                name=model_config.name,
                vocab_size=model_config.vocab_size,
                hidden_size=model_config.hidden_size,
                num_layers=model_config.num_layers,
                num_attention_heads=model_config.num_attention_heads,
                num_key_value_heads=model_config.num_key_value_heads,
                intermediate_size=model_config.intermediate_size,
                max_seq_len=model_config.max_seq_len,
                dtype=model_config.dtype,
                num_experts=num_experts,
                num_experts_per_token=getattr(model_config, "num_experts_per_token", 2),
                expert_intermediate_size=getattr(
                    model_config, "expert_intermediate_size", model_config.intermediate_size
                ),
            )
            model = MoEModel(moe_config)
        else:
            llama_config = LlamaConfig(
                name=model_config.name,
                vocab_size=model_config.vocab_size,
                hidden_size=model_config.hidden_size,
                num_layers=model_config.num_layers,
                num_attention_heads=model_config.num_attention_heads,
                num_key_value_heads=model_config.num_key_value_heads,
                intermediate_size=model_config.intermediate_size,
                max_seq_len=model_config.max_seq_len,
                dtype=model_config.dtype,
            )
            model = LlamaModel(llama_config)

        model.build_layers()

        return model

    @classmethod
    def list_available_presets(cls) -> Dict[str, List[str]]:
        """List all available preset names.

        Returns:
            Dictionary with preset categories and their names
        """
        return {
            "models": list(cls.MODEL_PRESETS.keys()),
            "hardware": list(cls.HARDWARE_PRESETS.keys()),
            "strategies": list(cls.STRATEGY_PRESETS.keys()),
            "devices": list(Device.PRESETS.keys()),
        }

    @classmethod
    def list_config_files(cls) -> Dict[str, List[str]]:
        """List all configuration files in configs/ directory.

        Returns:
            Dictionary with config categories and their file names
        """
        config_files = {
            "models": [],
            "hardware": [],
            "strategies": [],
        }

        if cls.CONFIG_DIR.exists():
            for file in cls.CONFIG_DIR.glob("*.json"):
                name = file.name
                if name.startswith("model_"):
                    config_files["models"].append(name)
                elif name.startswith("hardware_"):
                    config_files["hardware"].append(name)
                elif name.startswith("strategy_"):
                    config_files["strategies"].append(name)

        return config_files

    @classmethod
    def load_all_configs_from_dir(cls) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files from configs/ directory.

        Returns:
            Dictionary with all loaded configurations by category
        """
        all_configs = {
            "models": {},
            "hardware": {},
            "strategies": {},
        }

        config_files = cls.list_config_files()

        for category, files in config_files.items():
            for filename in files:
                try:
                    config_data = cls.load_json(filename)
                    name = filename.replace(f"{category}_", "").replace(".json", "")
                    all_configs[category][name] = config_data
                except (FileNotFoundError, ValueError):
                    pass

        return all_configs

    @classmethod
    def get_device_preset_names(cls) -> List[str]:
        """Get list of available device preset names.

        Returns:
            List of device preset names
        """
        return list(Device.PRESETS.keys())

    @classmethod
    def get_device_config(cls, preset_name: str) -> DeviceConfig:
        """Get device configuration by preset name.

        Args:
            preset_name: Device preset name

        Returns:
            DeviceConfig instance

        Raises:
            ValueError: If preset name is unknown
        """
        device = Device.from_preset(preset_name)
        return device.config

    @classmethod
    def validate_strategy_for_hardware(cls, strategy: StrategyConfig, hw_config: HardwareConfigDict) -> Dict[str, Any]:
        """Validate that strategy fits hardware constraints.

        Args:
            strategy: StrategyConfig to validate
            hw_config: HardwareConfigDict for constraints

        Returns:
            Dictionary with validation results:
                - valid: bool
                - errors: List[str]
                - warnings: List[str]
        """
        errors = []
        warnings = []

        total_devices_needed = strategy.world_size
        total_devices_available = hw_config.num_devices

        if total_devices_needed > total_devices_available:
            errors.append(
                f"Strategy requires {total_devices_needed} devices, but hardware has only {total_devices_available}"
            )

        if strategy.tp_degree > hw_config.devices_per_node:
            warnings.append(
                f"TP degree ({strategy.tp_degree}) exceeds devices per node "
                f"({hw_config.devices_per_node}). Communication will cross nodes "
                f"with lower bandwidth."
            )

        if strategy.pp_degree > 1 and strategy.pp_degree % hw_config.devices_per_node != 0:
            warnings.append(
                f"PP degree ({strategy.pp_degree}) may lead to uneven stage distribution "
                f"across nodes ({hw_config.devices_per_node} devices/node)."
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

"""Evaluator: Convenient performance evaluation API.

Provides high-level methods to quickly evaluate training and inference
performance with support for various input formats and workloads.
"""

from typing import Any, Dict, Optional, Union, List

from llm_perf.modeling import ShardedModule, create_model_from_config, get_model_presets
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer import (
    UnifiedAnalyzer,
    UnifiedResult,
    WorkloadConfig,
    get_workload,
    infer_workload,
    list_workloads,
)


class Evaluator:
    """Convenient evaluator for quick performance estimation.

    Supports multiple input formats:
    - Preset names: "llama-7b", "H100-SXM-80GB"
    - Config dicts: {"hidden_size": 4096, ...}
    - Instance objects: ShardedModule, Cluster, StrategyConfig

    Supports multiple workloads:
    - Built-in presets: "training", "autoregressive-inference", "diffusion-pipeline"
    - Custom workloads via WorkloadConfig

    Example:
        >>> evaluator = Evaluator()
        >>> result = evaluator.evaluate("llama-7b", "H100", "training", batch_size=32)
        >>> print(f"Throughput: {result.throughput['tokens_per_sec']:.2f}")
    """

    def __init__(self):
        self._model_cache: Dict[str, ShardedModule] = {}
        self._cluster_cache: Dict[str, Cluster] = {}

    def evaluate(
        self,
        model: Union[str, Dict, ShardedModule, Dict[str, ShardedModule]],
        hardware: Union[str, Dict, Cluster],
        workload: Union[str, WorkloadConfig],
        strategy: Union[str, Dict, StrategyConfig] = None,
        **kwargs,
    ) -> UnifiedResult:
        """Evaluate performance for any workload.

        Args:
            model: Model specification (preset name, config dict, ShardedModule, or multi-component dict)
            hardware: Hardware specification (preset name or Cluster)
            workload: Workload preset name or WorkloadConfig
            strategy: Strategy specification (config dict or StrategyConfig)
            **kwargs: Additional parameters (batch_size, seq_len, generation_len, etc.)

        Returns:
            UnifiedResult with performance metrics
        """
        model_obj = self._resolve_model(model, **kwargs)
        cluster_obj = self._resolve_hardware(hardware, **kwargs)
        strategy_obj = self._resolve_strategy(strategy, **kwargs)

        device = cluster_obj.devices[0] if cluster_obj.devices else Device.from_preset("H100-SXM-80GB")

        analyzer = UnifiedAnalyzer(model_obj, device, cluster_obj, strategy_obj)
        return analyzer.analyze(workload, **kwargs)

    def evaluate_training(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategy: Union[str, Dict, StrategyConfig] = None,
        batch_size: int = 32,
        seq_len: int = 2048,
        **kwargs,
    ) -> UnifiedResult:
        """Evaluate training performance (legacy convenience method).

        Args:
            model: Model specification
            hardware: Hardware specification
            strategy: Strategy specification
            batch_size: Global batch size
            seq_len: Sequence length

        Returns:
            UnifiedResult with training metrics
        """
        workload = infer_workload(self._get_model_type(model), "training")
        return self.evaluate(
            model,
            hardware,
            workload,
            strategy,
            batch_size=batch_size,
            seq_len=seq_len,
            **kwargs,
        )

    def evaluate_inference(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategy: Union[str, Dict, StrategyConfig] = None,
        batch_size: int = 1,
        prompt_len: int = 512,
        generation_len: int = 128,
        **kwargs,
    ) -> UnifiedResult:
        """Evaluate inference performance (legacy convenience method).

        Args:
            model: Model specification
            hardware: Hardware specification
            strategy: Strategy specification
            batch_size: Batch size
            prompt_len: Prompt length
            generation_len: Generation length

        Returns:
            UnifiedResult with inference metrics
        """
        workload = infer_workload(self._get_model_type(model), "inference")
        return self.evaluate(
            model,
            hardware,
            workload,
            strategy,
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=generation_len,
            **kwargs,
        )

    def evaluate_diffusion(
        self,
        models: Dict[str, Union[str, Dict, ShardedModule]],
        hardware: Union[str, Dict, Cluster],
        strategy: Union[str, Dict, StrategyConfig] = None,
        workload: str = "diffusion-pipeline",
        num_frames: int = 81,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 50,
        **kwargs,
    ) -> UnifiedResult:
        """Evaluate diffusion model performance.

        Args:
            models: Dict of components {"text_encoder": ..., "dit": ..., "vae": ...}
            hardware: Hardware specification
            strategy: Strategy specification
            workload: Workload preset name
            num_frames: Number of video frames
            height: Video height
            width: Video width
            num_inference_steps: Number of denoising steps

        Returns:
            UnifiedResult with diffusion metrics
        """
        model_dict = {}
        for component_name, model_spec in models.items():
            model_dict[component_name] = self._resolve_model(model_spec, **kwargs)

        cluster_obj = self._resolve_hardware(hardware, **kwargs)
        strategy_obj = self._resolve_strategy(strategy, **kwargs)

        device = cluster_obj.devices[0] if cluster_obj.devices else Device.from_preset("H100-SXM-80GB")

        analyzer = UnifiedAnalyzer(model_dict, device, cluster_obj, strategy_obj)
        return analyzer.analyze(
            workload,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            **kwargs,
        )

    def compare_workloads(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        workloads: List[str],
        strategy: Union[str, Dict, StrategyConfig] = None,
        **kwargs,
    ) -> Dict[str, UnifiedResult]:
        """Compare different workloads for the same model.

        Args:
            model: Model specification
            hardware: Hardware specification
            workloads: List of workload preset names to compare
            strategy: Strategy specification
            **kwargs: Additional parameters

        Returns:
            Dict mapping workload name to UnifiedResult
        """
        results = {}
        for workload in workloads:
            results[workload] = self.evaluate(model, hardware, workload, strategy, **kwargs)
        return results

    def compare_strategies(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategies: List[Union[str, Dict, StrategyConfig]],
        workload: Union[str, WorkloadConfig],
        **kwargs,
    ) -> Dict[str, UnifiedResult]:
        """Compare different strategies for the same workload.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategies: List of strategy configurations
            workload: Workload preset name or WorkloadConfig
            **kwargs: Additional parameters

        Returns:
            Dict mapping strategy name to UnifiedResult
        """
        results = {}
        for i, strategy in enumerate(strategies):
            strategy_obj = self._resolve_strategy(strategy, **kwargs)
            strategy_name = f"strategy_{i}"
            if isinstance(strategy, str):
                strategy_name = strategy
            elif isinstance(strategy, dict) and "name" in strategy:
                strategy_name = strategy["name"]

            results[strategy_name] = self.evaluate(model, hardware, workload, strategy_obj, **kwargs)

        return results

    def list_available_presets(self) -> Dict[str, List]:
        """List available presets for models, hardware, workloads."""
        return {
            "models": list(get_model_presets().keys()),
            "hardware": list(Device.PRESETS.keys()),
            "workloads": list(list_workloads().keys()),
        }

    def list_workloads(self) -> Dict[str, Dict[str, str]]:
        """List all available workload presets."""
        return list_workloads()

    def _get_model_type(self, model: Union[str, Dict, ShardedModule]) -> str:
        """Get model type for workload inference."""
        if isinstance(model, str):
            if model in get_model_presets():
                preset = get_model_presets()[model]
                return preset.get("architecture", "llama")
            return "llama"
        elif isinstance(model, dict):
            return model.get("type", model.get("architecture", "llama"))
        elif isinstance(model, ShardedModule):
            name = getattr(model, "_name", "")
            class_name = type(model).__name__.lower()
            if "dit" in class_name or "dit" in name:
                return "dit"
            elif "vae" in class_name or "vae" in name:
                return "vae"
            return "llama"
        return "llama"

    def _resolve_model(
        self,
        model: Union[str, Dict, ShardedModule, Dict[str, Union[str, Dict, ShardedModule]]],
        **kwargs,
    ) -> Union[ShardedModule, Dict[str, ShardedModule]]:
        """Resolve model specification to ShardedModule or dict of ShardedModules."""
        if isinstance(model, dict):
            if all(isinstance(v, (str, dict, ShardedModule)) for v in model.values()):
                if any(isinstance(v, ShardedModule) for v in model.values()):
                    resolved = {}
                    for k, v in model.items():
                        if isinstance(v, ShardedModule):
                            resolved[k] = v
                        else:
                            resolved[k] = self._resolve_single_model(v, **kwargs)
                    return resolved
                else:
                    return self._resolve_single_model(model, **kwargs)
            return self._resolve_single_model(model, **kwargs)
        elif isinstance(model, ShardedModule):
            return model
        elif isinstance(model, str):
            return self._resolve_single_model(model, **kwargs)
        else:
            raise ValueError(f"Invalid model specification: {model}")

    def _resolve_single_model(
        self,
        model: Union[str, Dict],
        **kwargs,
    ) -> ShardedModule:
        """Resolve single model specification."""
        if isinstance(model, str):
            if model in self._model_cache:
                return self._model_cache[model]
            model_obj = create_model_from_config({"preset": model})
            self._model_cache[model] = model_obj
            return model_obj
        elif isinstance(model, dict):
            model_obj = create_model_from_config(model)
            return model_obj
        else:
            raise ValueError(f"Invalid model specification: {model}")

    def _resolve_hardware(
        self,
        hardware: Union[str, Dict, Cluster],
        **kwargs,
    ) -> Cluster:
        """Resolve hardware specification to Cluster."""
        if isinstance(hardware, Cluster):
            return hardware

        num_devices = kwargs.get("num_devices", 8)

        if isinstance(hardware, str):
            cache_key = f"{hardware}_{num_devices}"
            if cache_key in self._cluster_cache:
                return self._cluster_cache[cache_key]

            device = Device.from_preset(hardware)
            topology = NetworkTopology(
                name="default",
                intra_node_bandwidth_gbps=200.0,
                intra_node_latency_us=1.0,
                inter_node_bandwidth_gbps=25.0,
                inter_node_latency_us=10.0,
            )
            cluster = Cluster.create_homogeneous(device.config, num_devices, topology)
            self._cluster_cache[cache_key] = cluster
            return cluster

        elif isinstance(hardware, dict):
            device_preset = hardware.get("device_preset", "H100-SXM-80GB")
            device = Device.from_preset(device_preset)
            topology = NetworkTopology(
                name="default",
                intra_node_bandwidth_gbps=hardware.get("intra_node_bw_gbps", 200.0),
                intra_node_latency_us=hardware.get("intra_node_latency_us", 1.0),
                inter_node_bandwidth_gbps=hardware.get("inter_node_bw_gbps", 25.0),
                inter_node_latency_us=hardware.get("inter_node_latency_us", 10.0),
            )
            num_devices = hardware.get("num_devices", num_devices)
            cluster = Cluster.create_homogeneous(device.config, num_devices, topology)
            return cluster

        else:
            raise ValueError(f"Invalid hardware specification: {hardware}")

    def _resolve_strategy(
        self,
        strategy: Union[str, Dict, StrategyConfig, None],
        **kwargs,
    ) -> StrategyConfig:
        """Resolve strategy specification to StrategyConfig."""
        if strategy is None:
            return StrategyConfig()

        if isinstance(strategy, StrategyConfig):
            return strategy

        if isinstance(strategy, str):
            return self._parse_strategy_name(strategy)

        if isinstance(strategy, dict):
            return StrategyConfig(
                tp_degree=strategy.get("tp", strategy.get("tp_degree", 1)),
                pp_degree=strategy.get("pp", strategy.get("pp_degree", 1)),
                dp_degree=strategy.get("dp", strategy.get("dp_degree", 1)),
                ep_degree=strategy.get("ep", strategy.get("ep_degree", 1)),
            )

        return StrategyConfig()

    def _parse_strategy_name(self, name: str) -> StrategyConfig:
        """Parse strategy name like 'tp8' or 'tp4_dp2'."""
        parts = name.lower().replace("_", "").replace("-", "")

        tp = 1
        pp = 1
        dp = 1

        import re

        tp_match = re.search(r"tp(\d+)", parts)
        if tp_match:
            tp = int(tp_match.group(1))

        pp_match = re.search(r"pp(\d+)", parts)
        if pp_match:
            pp = int(pp_match.group(1))

        dp_match = re.search(r"dp(\d+)", parts)
        if dp_match:
            dp = int(dp_match.group(1))

        return StrategyConfig(tp_degree=tp, pp_degree=pp, dp_degree=dp)

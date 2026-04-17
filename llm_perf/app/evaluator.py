"""Evaluator: Convenient performance evaluation API.

Provides high-level methods to quickly evaluate training and inference
performance with support for various input formats.
"""

from typing import Any, Dict, Optional, Union

from llm_perf.modeling import ShardedModule, create_model_from_config
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer import TrainingAnalyzer, TrainingResult
from llm_perf.analyzer import InferenceAnalyzer, InferenceResult


class Evaluator:
    """Convenient evaluator for quick performance estimation.

    Supports multiple input formats:
    - Preset names: "llama-7b", "H100-SXM-80GB"
    - Config dicts: {"hidden_size": 4096, ...}
    - Instance objects: ShardedModule, Cluster, StrategyConfig

    Example:
        >>> evaluator = Evaluator()
        >>> result = evaluator.evaluate_training("llama-7b", "H100", "tp8", batch_size=32)
        >>> print(f"Throughput: {result.samples_per_sec:.2f} samples/sec")
    """

    def __init__(self):
        self._model_cache: Dict[str, ShardedModule] = {}
        self._cluster_cache: Dict[str, Cluster] = {}

    def evaluate_training(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategy: Union[str, Dict, StrategyConfig] = None,
        batch_size: int = 32,
        seq_len: int = 2048,
        **kwargs,
    ) -> TrainingResult:
        """Evaluate training performance.

        Args:
            model: Model specification (preset name, config dict, or ShardedModule)
            hardware: Hardware specification (preset name or Cluster)
            strategy: Strategy specification (config dict or StrategyConfig)
            batch_size: Global batch size
            seq_len: Sequence length

        Returns:
            TrainingResult with performance metrics
        """
        model_obj = self._resolve_model(model, **kwargs)
        cluster_obj = self._resolve_hardware(hardware, **kwargs)
        strategy_obj = self._resolve_strategy(strategy, **kwargs)

        device = cluster_obj.devices[0] if cluster_obj.devices else Device.from_preset("H100-SXM-80GB")

        analyzer = TrainingAnalyzer(model_obj, device, cluster_obj, strategy_obj)
        return analyzer.analyze(batch_size=batch_size, seq_len=seq_len)

    def evaluate_inference(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategy: Union[str, Dict, StrategyConfig] = None,
        batch_size: int = 1,
        prompt_len: int = 512,
        generation_len: int = 128,
        **kwargs,
    ) -> InferenceResult:
        """Evaluate inference performance.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategy: Strategy specification
            batch_size: Batch size
            prompt_len: Prompt length
            generation_len: Generation length

        Returns:
            InferenceResult with performance metrics
        """
        model_obj = self._resolve_model(model, **kwargs)
        cluster_obj = self._resolve_hardware(hardware, **kwargs)
        strategy_obj = self._resolve_strategy(strategy, **kwargs)

        device = cluster_obj.devices[0] if cluster_obj.devices else Device.from_preset("H100-SXM-80GB")

        analyzer = InferenceAnalyzer(model_obj, device, cluster_obj, strategy_obj)
        return analyzer.analyze(
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=generation_len,
        )

    def compare_strategies(
        self,
        model: Union[str, Dict, ShardedModule],
        hardware: Union[str, Dict, Cluster],
        strategies: list,
        mode: str = "training",
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare performance across different strategies.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategies: List of strategy configs or names
            mode: "training" or "inference"

        Returns:
            Comparison results
        """
        model_obj = self._resolve_model(model, **kwargs)
        cluster_obj = self._resolve_hardware(hardware, **kwargs)

        results = []
        for strategy_spec in strategies:
            strategy_obj = self._resolve_strategy(strategy_spec, **kwargs)

            if mode == "training":
                result = self.evaluate_training(model_obj, cluster_obj, strategy_obj, **kwargs)
                results.append(
                    {
                        "strategy": strategy_obj,
                        "samples_per_sec": result.samples_per_sec,
                        "tokens_per_sec": result.tokens_per_sec,
                        "memory_gb": result.memory_per_gpu_gb,
                    }
                )
            else:
                result = self.evaluate_inference(model_obj, cluster_obj, strategy_obj, **kwargs)
                results.append(
                    {
                        "strategy": strategy_obj,
                        "prefill_tps": result.prefill_tokens_per_sec,
                        "decode_tps": result.decode_tokens_per_sec,
                        "memory_gb": result.memory_per_gpu_gb,
                    }
                )

        if results:
            best = max(results, key=lambda x: x.get("samples_per_sec") or x.get("prefill_tps", 0))
        else:
            best = None

        return {
            "best_strategy": best,
            "all_results": results,
        }

    def list_available_presets(self) -> Dict[str, list]:
        """List all available presets."""
        from llm_perf.modeling import get_model_presets

        return {
            "models": list(get_model_presets().keys()),
            "hardware": list(Device.PRESETS.keys()),
            "strategies": ["tp1", "tp2", "tp4", "tp8", "tp4_dp2", "tp2_pp4"],
        }

    def _resolve_model(
        self,
        model: Union[str, Dict, ShardedModule],
        **kwargs,
    ) -> ShardedModule:
        """Resolve model specification to ShardedModule instance."""
        if isinstance(model, ShardedModule):
            return model

        if isinstance(model, str):
            if model in self._model_cache:
                return self._model_cache[model]
            model_obj = create_model_from_config({"preset": model})
            self._model_cache[model] = model_obj
            return model_obj

        if isinstance(model, dict):
            if "preset" in model:
                return create_model_from_config(model)
            return create_model_from_config(model)

        raise ValueError(f"Unknown model specification: {model}")

    def _resolve_hardware(
        self,
        hardware: Union[str, Dict, Cluster],
        **kwargs,
    ) -> Cluster:
        """Resolve hardware specification to Cluster instance."""
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

        raise ValueError(f"Unknown hardware specification: {hardware}")

    def _resolve_strategy(
        self,
        strategy: Union[str, Dict, StrategyConfig],
        **kwargs,
    ) -> StrategyConfig:
        """Resolve strategy specification to StrategyConfig instance."""
        if strategy is None:
            return StrategyConfig(tp_degree=1)

        if isinstance(strategy, StrategyConfig):
            return strategy

        if isinstance(strategy, str):
            return self._parse_strategy_name(strategy)

        if isinstance(strategy, dict):
            return StrategyConfig(**strategy)

        raise ValueError(f"Unknown strategy specification: {strategy}")

    def _parse_strategy_name(self, name: str) -> StrategyConfig:
        """Parse strategy name like 'tp8', 'tp4_dp2'."""
        parts = name.split("_")
        tp = 1
        pp = 1
        dp = 1

        for part in parts:
            if part.startswith("tp"):
                tp = int(part[2:])
            elif part.startswith("pp"):
                pp = int(part[2:])
            elif part.startswith("dp"):
                dp = int(part[2:])

        return StrategyConfig(tp_degree=tp, pp_degree=pp, dp_degree=dp)

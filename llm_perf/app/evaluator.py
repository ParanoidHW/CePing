"""Evaluator: Convenient performance evaluation API.

Provides high-level methods to quickly evaluate training, inference, and pipeline
performance with support for various input formats (preset names, configs, objects).
"""

from typing import Any, Dict, Optional, Union

from ..utils.config_loader import ConfigLoader, HardwareConfigDict
from ..models.base import BaseModel, ModelConfig
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig
from ..analyzer.training import TrainingAnalyzer, TrainingResult
from ..analyzer.inference import InferenceAnalyzer, InferenceResult
from ..core.pipeline import Pipeline, PipelineResult


class Evaluator:
    """Convenient evaluator for quick performance estimation.

    Supports multiple input formats:
    - Preset names: "llama-7b", "h100_8gpu", "tp8"
    - Config dicts: {"hidden_size": 4096, ...}
    - Config objects: ModelConfig, StrategyConfig
    - Instance objects: BaseModel, Cluster, etc.

    Example:
        >>> evaluator = Evaluator()
        >>> # Using presets
        >>> result = evaluator.evaluate_training("llama-7b", "h100_8gpu", "tp8", batch_size=32)
        >>> # Using objects
        >>> result = evaluator.evaluate_training(model, cluster, strategy, batch_size=32)
    """

    def __init__(self):
        self._model_cache: Dict[str, BaseModel] = {}
        self._cluster_cache: Dict[str, Cluster] = {}

    def evaluate_training(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
        hardware: Union[str, Dict, HardwareConfigDict, Cluster],
        strategy: Union[str, Dict, StrategyConfig] = "tp1",
        batch_size: int = 32,
        seq_len: int = 2048,
        detailed: bool = True,
        **kwargs,
    ) -> TrainingResult:
        """Evaluate training performance.

        Args:
            model: Model specification (preset name, config dict, ModelConfig, or BaseModel)
            hardware: Hardware specification (preset name, config dict, or Cluster)
            strategy: Strategy specification (preset name, config dict, or StrategyConfig)
            batch_size: Global batch size
            seq_len: Sequence length
            detailed: Include detailed breakdown
            **kwargs: Additional parameters for config loading

        Returns:
            TrainingResult with performance metrics

        Example:
            >>> result = evaluator.evaluate_training("llama-7b", "h100_8gpu", "tp8", batch_size=32)
            >>> print(f"Throughput: {result.samples_per_sec:.2f} samples/sec")
        """
        model_obj = self._resolve_model(model, **kwargs)
        cluster_obj = self._resolve_hardware(hardware, **kwargs)
        strategy_obj = self._resolve_strategy(strategy, model_obj.config.name, **kwargs)

        self._validate_strategy_hardware(strategy_obj, cluster_obj)

        device = cluster_obj.devices[0] if cluster_obj.devices else cluster_obj.master_device

        analyzer = TrainingAnalyzer(model_obj, device, cluster_obj, strategy_obj)
        result = analyzer.analyze(batch_size=batch_size, seq_len=seq_len)

        return result

    def evaluate_inference(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
        hardware: Union[str, Dict, HardwareConfigDict, Cluster],
        strategy: Union[str, Dict, StrategyConfig] = "tp1",
        batch_size: int = 1,
        prompt_len: int = 512,
        generation_len: int = 128,
        detailed: bool = True,
        **kwargs,
    ) -> InferenceResult:
        """Evaluate inference performance.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategy: Strategy specification
            batch_size: Batch size for inference
            prompt_len: Prompt sequence length
            generation_len: Number of tokens to generate
            detailed: Include detailed breakdown
            **kwargs: Additional parameters

        Returns:
            InferenceResult with prefill and decode metrics

        Example:
            >>> result = evaluator.evaluate_inference("llama-7b", "h100_8gpu", "tp4", batch_size=1)
            >>> print(f"TTFT: {result.prefill_time_sec*1000:.2f} ms")
            >>> print(f"TPS: {result.decode_tokens_per_sec:.2f}")
        """
        model_obj = self._resolve_model(model, **kwargs)
        cluster_obj = self._resolve_hardware(hardware, **kwargs)
        strategy_obj = self._resolve_strategy(strategy, model_obj.config.name, **kwargs)

        self._validate_strategy_hardware(strategy_obj, cluster_obj)

        device = cluster_obj.devices[0] if cluster_obj.devices else cluster_obj.master_device

        analyzer = InferenceAnalyzer(model_obj, device, cluster_obj, strategy_obj)
        result = analyzer.analyze(
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=generation_len,
        )

        return result

    def evaluate_pipeline(
        self,
        pipeline: Union[Pipeline, str],
        hardware: Union[str, Dict, HardwareConfigDict, Cluster],
        strategy: Union[str, Dict, StrategyConfig] = "tp1",
        **kwargs,
    ) -> PipelineResult:
        """Evaluate pipeline performance.

        Args:
            pipeline: Pipeline instance or preset name
            hardware: Hardware specification
            strategy: Strategy specification
            **kwargs: Additional parameters (batch_size, prompt_len, etc.)

        Returns:
            PipelineResult with execution metrics

        Example:
            >>> pipeline = create_wan_t2v_pipeline(...)
            >>> result = evaluator.evaluate_pipeline(pipeline, "h100_8gpu", "tp8")
        """
        if isinstance(pipeline, str):
            raise ValueError(f"Pipeline preset '{pipeline}' not supported yet. Please provide Pipeline instance.")

        if not isinstance(pipeline, Pipeline):
            raise ValueError(f"pipeline must be Pipeline instance, got {type(pipeline)}")

        cluster_obj = self._resolve_hardware(hardware, **kwargs)
        strategy_obj = self._resolve_strategy(strategy, "", **kwargs)

        self._validate_strategy_hardware(strategy_obj, cluster_obj)

        result = pipeline.run(kwargs)

        return result

    def compare_strategies(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
        hardware: Union[str, Dict, HardwareConfigDict, Cluster],
        strategies: list,
        mode: str = "training",
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare multiple strategies for the same model and hardware.

        Args:
            model: Model specification
            hardware: Hardware specification
            strategies: List of strategy specifications
            mode: "training" or "inference"
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary with comparison results

        Example:
            >>> results = evaluator.compare_strategies(
            ...     "llama-7b", "h100_8gpu",
            ...     ["tp1", "tp2", "tp4", "tp8"],
            ...     batch_size=32
            ... )
        """
        results = []
        model_obj = self._resolve_model(model, **kwargs)
        cluster_obj = self._resolve_hardware(hardware, **kwargs)

        hw_name = getattr(cluster_obj, "name", f"{cluster_obj.num_devices}-gpu cluster")

        for strategy_spec in strategies:
            strategy_obj = self._resolve_strategy(strategy_spec, model_obj.config.name)

            try:
                if mode == "training":
                    result = self.evaluate_training(model_obj, cluster_obj, strategy_obj, **kwargs)
                    metric = {
                        "strategy": strategy_spec,
                        "samples_per_sec": result.samples_per_sec,
                        "tokens_per_sec": result.tokens_per_sec,
                        "memory_gb": result.memory_per_gpu_gb,
                        "time_per_step_ms": result.time_per_step_sec * 1000,
                    }
                else:
                    result = self.evaluate_inference(model_obj, cluster_obj, strategy_obj, **kwargs)
                    metric = {
                        "strategy": strategy_spec,
                        "prefill_tps": result.prefill_tokens_per_sec,
                        "decode_tps": result.decode_tokens_per_sec,
                        "ttft_ms": result.prefill_time_sec * 1000,
                        "memory_gb": result.memory_per_gpu_gb,
                    }

                metric["valid"] = True
                results.append(metric)
            except Exception as e:
                results.append(
                    {
                        "strategy": strategy_spec,
                        "valid": False,
                        "error": str(e),
                    }
                )

        return {
            "model": model_obj.config.name,
            "hardware": hw_name,
            "mode": mode,
            "results": results,
            "best_strategy": self._find_best_strategy(results, mode),
        }

    def _find_best_strategy(self, results: list, mode: str) -> Optional[str]:
        """Find best strategy based on mode.

        Args:
            results: List of strategy comparison results
            mode: "training" or "inference"

        Returns:
            Best strategy name or None
        """
        valid_results = [r for r in results if r.get("valid", False)]

        if not valid_results:
            return None

        if mode == "training":
            return max(valid_results, key=lambda x: x.get("samples_per_sec", 0))["strategy"]
        else:
            return max(valid_results, key=lambda x: x.get("decode_tps", 0))["strategy"]

    def _resolve_model(
        self,
        model: Union[str, Dict, ModelConfig, BaseModel],
        **kwargs,
    ) -> BaseModel:
        """Resolve model specification to BaseModel instance.

        Args:
            model: Model specification
            **kwargs: Additional parameters

        Returns:
            BaseModel instance
        """
        if isinstance(model, BaseModel):
            return model

        if isinstance(model, ModelConfig):
            return ConfigLoader.create_model_from_config(model)

        cache_key = None
        if isinstance(model, str):
            cache_key = model
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]

        model_config = ConfigLoader.load_model_config(model, **kwargs)
        model_obj = ConfigLoader.create_model_from_config(model_config)

        if cache_key:
            self._model_cache[cache_key] = model_obj

        return model_obj

    def _resolve_hardware(
        self,
        hardware: Union[str, Dict, HardwareConfigDict, Cluster],
        **kwargs,
    ) -> Cluster:
        """Resolve hardware specification to Cluster instance.

        Args:
            hardware: Hardware specification
            **kwargs: Additional parameters

        Returns:
            Cluster instance
        """
        if isinstance(hardware, Cluster):
            return hardware

        cache_key = None
        if isinstance(hardware, str):
            cache_key = hardware
            if cache_key in self._cluster_cache:
                return self._cluster_cache[cache_key]

        hw_config = ConfigLoader.load_hardware_config(hardware, **kwargs)
        cluster_obj = ConfigLoader.create_cluster_from_hardware_config(hw_config)

        if cache_key:
            self._cluster_cache[cache_key] = cluster_obj

        return cluster_obj

    def _resolve_strategy(
        self,
        strategy: Union[str, Dict, StrategyConfig],
        model_name: str = "",
        **kwargs,
    ) -> StrategyConfig:
        """Resolve strategy specification to StrategyConfig instance.

        Args:
            strategy: Strategy specification
            model_name: Model name for strategy config
            **kwargs: Additional parameters

        Returns:
            StrategyConfig instance
        """
        if isinstance(strategy, StrategyConfig):
            return strategy

        strategy_config = ConfigLoader.load_strategy_config(strategy, **kwargs)
        strategy_config.model_name = model_name

        return strategy_config

    def _validate_strategy_hardware(
        self,
        strategy: StrategyConfig,
        cluster: Cluster,
    ) -> None:
        """Validate strategy fits hardware constraints.

        Args:
            strategy: Strategy configuration
            cluster: Cluster configuration

        Raises:
            ValueError: If strategy requires more devices than available
        """
        if strategy.world_size > cluster.num_devices:
            raise ValueError(
                f"Strategy requires {strategy.world_size} devices but cluster has only {cluster.num_devices}"
            )

    def list_available_presets(self) -> Dict[str, list]:
        """List all available presets.

        Returns:
            Dictionary with preset categories
        """
        return ConfigLoader.list_available_presets()

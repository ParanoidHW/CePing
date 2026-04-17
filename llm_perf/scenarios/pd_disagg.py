"""PD Disaggregation Scenario implementation.

Prefill-Decode disaggregated inference scenario where:
- Prefill nodes: Handle prompt processing (compute-heavy)
- Decode nodes: Handle token generation (memory-bound)

This separation allows:
- Different parallelism configurations for each phase
- Better resource utilization
- Scalable inference serving
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Scenario, ScenarioConfig, ScenarioResult, ScenarioType, ParallelismType
from llm_perf.legacy.models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig


@dataclass
class PDNodeConfig:
    """Configuration for a PD node (Prefill or Decode)."""

    role: str
    tp_degree: int = 1
    pp_degree: int = 1
    sp_degree: int = 1
    num_nodes: int = 1
    devices_per_node: int = 8

    @property
    def total_devices(self) -> int:
        """Total devices for this node."""
        return self.num_nodes * self.devices_per_node

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "tp_degree": self.tp_degree,
            "pp_degree": self.pp_degree,
            "sp_degree": self.sp_degree,
            "num_nodes": self.num_nodes,
            "devices_per_node": self.devices_per_node,
            "total_devices": self.total_devices,
        }


@dataclass
class PDDisaggConfig(ScenarioConfig):
    """Configuration for PD disaggregated inference scenario.

    Attributes:
        prefill_config: Prefill node configuration
        decode_config: Decode node configuration
        batch_size: Inference batch size
        prompt_len: Prompt sequence length
        generation_len: Generation sequence length
        transfer_bandwidth_gbps: Bandwidth for KV transfer between nodes
    """

    prefill_config: PDNodeConfig = None
    decode_config: PDNodeConfig = None
    batch_size: int = 1
    prompt_len: int = 512
    generation_len: int = 128
    transfer_bandwidth_gbps: float = 400.0

    def __post_init__(self):
        """Set scenario type and required models."""
        self.scenario_type = ScenarioType.PD_DISAGG
        if not self.required_models:
            self.required_models = ["prefill", "decode"]
        if not self.supported_parallelisms:
            self.supported_parallelisms = [
                ParallelismType.TP,
                ParallelismType.PP,
                ParallelismType.SP,
            ]
        if self.prefill_config is None:
            self.prefill_config = PDNodeConfig(role="prefill", tp_degree=4)
        if self.decode_config is None:
            self.decode_config = PDNodeConfig(role="decode", tp_degree=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "prefill_config": self.prefill_config.to_dict(),
                "decode_config": self.decode_config.to_dict(),
                "batch_size": self.batch_size,
                "prompt_len": self.prompt_len,
                "generation_len": self.generation_len,
                "transfer_bandwidth_gbps": self.transfer_bandwidth_gbps,
            }
        )
        return base


@dataclass
class PDNodeResult:
    """Result for a single PD node."""

    role: str
    total_time_sec: float = 0.0
    throughput: float = 0.0
    memory_gb: float = 0.0
    devices_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "total_time_sec": self.total_time_sec,
            "throughput": self.throughput,
            "memory_gb": self.memory_gb,
            "devices_used": self.devices_used,
        }


@dataclass
class PDDisaggResult(ScenarioResult):
    """Result of PD disaggregated inference evaluation.

    Attributes:
        prefill_result: Prefill node performance
        decode_result: Decode node performance
        kv_transfer_time_sec: Time to transfer KV cache
        total_latency_sec: End-to-end latency
        overall_throughput: Overall throughput (tokens/sec)
    """

    prefill_result: PDNodeResult = None
    decode_result: PDNodeResult = None
    kv_transfer_time_sec: float = 0.0
    total_latency_sec: float = 0.0
    overall_throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "prefill": self.prefill_result.to_dict() if self.prefill_result else {},
                "decode": self.decode_result.to_dict() if self.decode_result else {},
                "kv_transfer": {
                    "time_sec": self.kv_transfer_time_sec,
                    "time_ms": self.kv_transfer_time_sec * 1000,
                },
                "total_latency_sec": self.total_latency_sec,
                "overall_throughput": self.overall_throughput,
            }
        )
        return base


class PDDisaggScenario(Scenario):
    """PD disaggregated inference performance scenario.

    Models prefill and decode phases running on separate nodes with
    KV cache transfer between them.

    Architecture:
    - Prefill Node(s): High compute, process entire prompt
    - KV Transfer: Network transfer of KV cache
    - Decode Node(s): Memory-bound, generate tokens one by one

    Example:
        >>> from llm_perf.scenarios.registry import ScenarioRegistry
        >>> from llm_perf.modeling import LlamaModel, LlamaConfig
        >>> from llm_perf.hardware.device import Device
        >>>
        >>> registry = ScenarioRegistry()
        >>> model = LlamaModel(LlamaConfig(...))
        >>> device = Device(name="H100", memory_gb=80)
        >>>
        >>> scenario = registry.create_scenario(
        ...     "pd-disagg",
        ...     models={"prefill": model, "decode": model},
        ...     device=device,
        ...     cluster=cluster,
        ...     strategy=strategy,
        ...     prefill_tp=8,
        ...     decode_tp=2,
        ... )
        >>> result = scenario.analyze(prompt_len=512, generation_len=128)
    """

    def __init__(
        self,
        config: PDDisaggConfig,
        models: Dict[str, BaseModel],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """Initialize PD disaggregated scenario.

        Args:
            config: PD disaggregated configuration
            models: Dictionary with "prefill" and "decode" models
            device: Device configuration
            cluster: Cluster configuration
            strategy: Base strategy (used as template)
        """
        super().__init__(config, models, device, cluster, strategy)
        self._prefill_strategy: StrategyConfig = None
        self._decode_strategy: StrategyConfig = None

    def _create_node_strategy(
        self,
        node_config: PDNodeConfig,
    ) -> StrategyConfig:
        """Create strategy for a specific node.

        Args:
            node_config: Node configuration

        Returns:
            StrategyConfig for the node
        """
        return StrategyConfig(
            model_name=self.strategy.model_name,
            tp_degree=node_config.tp_degree,
            pp_degree=node_config.pp_degree,
            sp_degree=node_config.sp_degree,
            dp_degree=1,
            ep_degree=1,
            sp_type=self.strategy.sp_type,
        )

    def _estimate_prefill_time(
        self,
        batch_size: int,
        prompt_len: int,
    ) -> float:
        """Estimate prefill phase time.

        Args:
            batch_size: Batch size
            prompt_len: Prompt length

        Returns:
            Prefill time in seconds
        """
        prefill_model = self.models.get("prefill")
        prefill_strategy = self._create_node_strategy(self.config.prefill_config)

        from ..analyzer.inference import InferenceAnalyzer

        analyzer = InferenceAnalyzer(
            model=prefill_model,
            device=self.device,
            cluster=self.cluster,
            strategy=prefill_strategy,
        )

        result = analyzer.analyze(
            batch_size=batch_size,
            prompt_len=prompt_len,
            generation_len=0,
        )

        return result.prefill_time_sec

    def _estimate_decode_time(
        self,
        batch_size: int,
        generation_len: int,
    ) -> float:
        """Estimate decode phase time.

        Args:
            batch_size: Batch size
            generation_len: Number of tokens to generate

        Returns:
            Total decode time in seconds
        """
        decode_model = self.models.get("decode")
        decode_strategy = self._create_node_strategy(self.config.decode_config)

        from ..analyzer.inference import InferenceAnalyzer

        analyzer = InferenceAnalyzer(
            model=decode_model,
            device=self.device,
            cluster=self.cluster,
            strategy=decode_strategy,
        )

        result = analyzer.analyze(
            batch_size=batch_size,
            prompt_len=0,
            generation_len=generation_len,
        )

        return result.decode_time_per_step_sec * generation_len

    def _estimate_kv_transfer_time(
        self,
        batch_size: int,
        prompt_len: int,
    ) -> float:
        """Estimate KV cache transfer time between nodes.

        Args:
            batch_size: Batch size
            prompt_len: Prompt length

        Returns:
            Transfer time in seconds
        """
        prefill_model = self.models.get("prefill")
        dtype_size = 2 if prefill_model.config.dtype == "fp16" else 2

        hidden_size = prefill_model.config.hidden_size
        num_layers = prefill_model.config.num_layers
        num_kv_heads = prefill_model.config.num_key_value_heads or prefill_model.config.num_attention_heads
        head_dim = hidden_size // prefill_model.config.num_attention_heads

        kv_bytes_per_token = 2 * num_kv_heads * head_dim * dtype_size
        total_kv_bytes = batch_size * prompt_len * num_layers * kv_bytes_per_token

        transfer_bandwidth_bps = self.config.transfer_bandwidth_gbps * 1e9

        return total_kv_bytes / transfer_bandwidth_bps

    def analyze(
        self,
        batch_size: int = None,
        prompt_len: int = None,
        generation_len: int = None,
        **kwargs,
    ) -> PDDisaggResult:
        """Run PD disaggregated analysis.

        Args:
            batch_size: Batch size
            prompt_len: Prompt length
            generation_len: Generation length
            **kwargs: Additional parameters

        Returns:
            PDDisaggResult with performance metrics
        """
        batch_size = batch_size or self.config.batch_size
        prompt_len = prompt_len or self.config.prompt_len
        generation_len = generation_len or self.config.generation_len

        prefill_time = self._estimate_prefill_time(batch_size, prompt_len)
        kv_transfer_time = self._estimate_kv_transfer_time(batch_size, prompt_len)
        decode_time = self._estimate_decode_time(batch_size, generation_len)

        total_latency = prefill_time + kv_transfer_time + decode_time
        total_tokens = batch_size * (prompt_len + generation_len)
        overall_throughput = total_tokens / total_latency if total_latency > 0 else 0

        prefill_result = PDNodeResult(
            role="prefill",
            total_time_sec=prefill_time,
            throughput=batch_size * prompt_len / prefill_time if prefill_time > 0 else 0,
            devices_used=self.config.prefill_config.total_devices,
        )

        decode_result = PDNodeResult(
            role="decode",
            total_time_sec=decode_time,
            throughput=batch_size / (decode_time / generation_len) if decode_time > 0 else 0,
            devices_used=self.config.decode_config.total_devices,
        )

        return PDDisaggResult(
            scenario_name=self.config.name,
            total_time_sec=total_latency,
            throughput=overall_throughput,
            memory_peak_gb=0.0,
            prefill_result=prefill_result,
            decode_result=decode_result,
            kv_transfer_time_sec=kv_transfer_time,
            total_latency_sec=total_latency,
            overall_throughput=overall_throughput,
            breakdown={
                "prefill_time_sec": prefill_time,
                "kv_transfer_time_sec": kv_transfer_time,
                "decode_time_sec": decode_time,
                "prefill_percent": prefill_time / total_latency * 100 if total_latency > 0 else 0,
                "kv_transfer_percent": kv_transfer_time / total_latency * 100 if total_latency > 0 else 0,
                "decode_percent": decode_time / total_latency * 100 if total_latency > 0 else 0,
            },
            metadata={
                "batch_size": batch_size,
                "prompt_len": prompt_len,
                "generation_len": generation_len,
                "prefill_config": self.config.prefill_config.to_dict(),
                "decode_config": self.config.decode_config.to_dict(),
            },
        )

    def get_analyzer(self) -> Any:
        """PD disaggregation uses multiple analyzers.

        Returns:
            None (this scenario doesn't have a single analyzer)
        """
        return None

    def get_prefill_analyzer(self) -> Any:
        """Get analyzer for prefill node.

        Returns:
            InferenceAnalyzer for prefill
        """
        from ..analyzer.inference import InferenceAnalyzer

        prefill_model = self.models.get("prefill")
        prefill_strategy = self._create_node_strategy(self.config.prefill_config)

        return InferenceAnalyzer(
            model=prefill_model,
            device=self.device,
            cluster=self.cluster,
            strategy=prefill_strategy,
        )

    def get_decode_analyzer(self) -> Any:
        """Get analyzer for decode node.

        Returns:
            InferenceAnalyzer for decode
        """
        from ..analyzer.inference import InferenceAnalyzer

        decode_model = self.models.get("decode")
        decode_strategy = self._create_node_strategy(self.config.decode_config)

        return InferenceAnalyzer(
            model=decode_model,
            device=self.device,
            cluster=self.cluster,
            strategy=decode_strategy,
        )

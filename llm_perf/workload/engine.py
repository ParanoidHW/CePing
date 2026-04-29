"""Evaluation engine for workload analysis.

This engine:
1. Receives EvaluationRequest from API/CLI
2. Validates the request
3. Calls existing UnifiedAnalyzer
4. Returns EvaluationResult

Design principle: Engine is a thin wrapper around UnifiedAnalyzer.
All core analysis logic remains in UnifiedAnalyzer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

from llm_perf.analyzer.unified import UnifiedAnalyzer
from llm_perf.analyzer.base import WorkloadConfig, WorkloadType
from llm_perf.analyzer.workload_loader import get_workload
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.device import Device
from llm_perf.strategy.base import StrategyConfig
from llm_perf.modeling import ShardedModule

from .loader import WorkloadLoader, get_loader
from .schema import HardwareSchema, StrategySchema
from .validator import WorkloadValidator, ValidationResult


@dataclass
class EvaluationRequest:
    """Evaluation request from API/CLI.

    Attributes:
        workload_name: Workload name (e.g., "inference/autoregressive")
        model_name: Model name (e.g., "llama-7b")
        hardware: Hardware configuration
        strategy: Strategy configuration
        params: Workload parameters (batch_size, seq_len, etc.)
    """

    workload_name: str
    model_name: str
    hardware: HardwareSchema
    strategy: StrategySchema
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_name": self.workload_name,
            "model_name": self.model_name,
            "hardware": self.hardware.to_dict(),
            "strategy": self.strategy.to_dict(),
            "params": self.params,
        }


@dataclass
class EvaluationResult:
    """Evaluation result for API/CLI.

    Attributes:
        success: Whether evaluation succeeded
        validation: Validation result
        result: UnifiedResult dict (if success)
        error: Error message (if failed)
    """

    success: bool
    validation: Optional[ValidationResult] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"success": self.success}
        if self.validation:
            d["validation"] = self.validation.to_dict()
        if self.result:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d


class EvaluationEngine:
    """Evaluation engine for workload analysis.

    This engine wraps UnifiedAnalyzer and provides:
    1. Request validation
    2. Model instantiation
    3. Hardware/Strategy configuration
    4. Result formatting
    """

    def __init__(self, loader: Optional[WorkloadLoader] = None):
        self.loader = loader or get_loader()
        self.validator = WorkloadValidator()

    def evaluate(
        self,
        request: EvaluationRequest,
        model: ShardedModule,
    ) -> EvaluationResult:
        """Evaluate workload performance.

        Args:
            request: EvaluationRequest
            model: ShardedModule instance (instantiated by caller)

        Returns:
            EvaluationResult
        """
        validation = self.validator.validate_request(request, model)
        if not validation.is_valid:
            return EvaluationResult(
                success=False,
                validation=validation,
                error="Validation failed",
            )

        try:
            device = self._create_device(request.hardware)
            cluster = self._create_cluster(request.hardware, device)
            strategy = self._create_strategy(request.strategy)
            workload_config = self._create_workload_config(request.workload_name)

            analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
            unified_result = analyzer.analyze(workload_config, **request.params)

            return EvaluationResult(
                success=True,
                validation=validation,
                result=unified_result.to_dict(),
            )

        except Exception as e:
            return EvaluationResult(
                success=False,
                validation=validation,
                error=str(e),
            )

    def evaluate_raw(
        self,
        workload_name: str,
        model: ShardedModule,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate with raw parameters (for CLI).

        Args:
            workload_name: Workload name
            model: ShardedModule instance
            device: Device instance
            cluster: Cluster instance
            strategy: StrategyConfig instance
            params: Workload parameters

        Returns:
            UnifiedResult dict
        """
        workload_config = self._create_workload_config(workload_name)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        unified_result = analyzer.analyze(workload_config, **params)
        return unified_result.to_dict()

    def _create_device(self, hardware: HardwareSchema) -> Device:
        """Create Device from HardwareSchema."""
        return Device.from_preset(hardware.device_preset)

    def _create_cluster(self, hardware: HardwareSchema, device: Device) -> Cluster:
        """Create Cluster from HardwareSchema."""
        return Cluster.create_homogeneous(device, hardware.num_devices)

    def _create_strategy(self, strategy: StrategySchema) -> StrategyConfig:
        """Create StrategyConfig from StrategySchema."""
        return StrategyConfig(
            tp_degree=strategy.tp_degree,
            pp_degree=strategy.pp_degree,
            dp_degree=strategy.dp_degree,
            ep_degree=strategy.ep_degree,
            sp_degree=strategy.sp_degree,
            activation_checkpointing=strategy.activation_checkpointing,
            zero_stage=strategy.zero_stage,
        )

    def _create_workload_config(self, workload_name: str) -> WorkloadConfig:
        """Create WorkloadConfig from workload name."""
        return get_workload(workload_name)
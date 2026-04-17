"""Pipeline Parallelism Model wrapper.

Includes:
- PPStageModule: Represents layers in a PP stage
- PPModel: Model wrapper with PP stage division
"""

from typing import Dict, Optional, List, Any, TYPE_CHECKING

from llm_perf.modeling.base import ShardedModule, ModuleInstance, ShardedTensor
from .pp_strategy import PPStrategy
from llm_perf.utils.constants import DTYPE_SIZES

if TYPE_CHECKING:
    from .parallel_context import ParallelContext


class PPStageModule(ShardedModule):
    """PP Stage Module - represents layers in one PP stage.

    Attributes:
        stage_idx: Stage index
        layers: Layers in this stage
        pp_strategy: PP strategy reference
    """

    def __init__(
        self,
        stage_idx: int,
        layers: List[ShardedModule],
        pp_strategy: PPStrategy,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.pp_strategy = pp_strategy

        for i, layer in enumerate(layers):
            setattr(self, f"layer_{i}", layer)

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """Stage forward - process through all layers."""
        for name in sorted(self._submodules.keys()):
            layer = self._submodules[name]
            hidden = layer(hidden)

        return hidden

    def get_layers(self) -> List[ShardedModule]:
        """Get all layers in this stage."""
        sorted_names = sorted(self._submodules.keys())
        return [self._submodules[name] for name in sorted_names]

    def params_count_breakdown(self) -> Dict[str, int]:
        """Parameters breakdown by layer."""
        breakdown = {}
        for name, layer in self._submodules.items():
            breakdown[name] = layer.params_count()
        return breakdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_idx": self.stage_idx,
            "num_layers": len(self._submodules),
            "layers": sorted(self._submodules.keys()),
            "params_count": self.params_count(),
            "params_count_breakdown": self.params_count_breakdown(),
        }


class PPModel(ShardedModule):
    """PP Model - wraps a model with PP stage division.

    Divides a model (e.g., LlamaModel) into multiple PPStageModules
    based on PP strategy.
    """

    def __init__(
        self,
        model: ShardedModule,
        pp_strategy: PPStrategy,
    ):
        super().__init__()
        self.original_model = model
        self.pp_strategy = pp_strategy

        self.hidden_size = getattr(model, "hidden_size", 4096)
        self.max_seq_len = getattr(model, "max_seq_len", 4096)
        self.dtype = getattr(model, "dtype", "fp16")

        assignment = pp_strategy.assign_layers(model, method="custom" if pp_strategy.stage_assignment else "balanced")
        self._stage_assignment = assignment

        stage_layers: Dict[int, List[ShardedModule]] = {}
        for name, submodule in model._submodules.items():
            if name in assignment:
                stage_idx = assignment[name]
                if stage_idx not in stage_layers:
                    stage_layers[stage_idx] = []
                stage_layers[stage_idx].append(submodule)

        self._stage_layers = stage_layers

        for stage_idx, layers in stage_layers.items():
            setattr(self, f"stage_{stage_idx}", PPStageModule(stage_idx, layers, pp_strategy))

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        """PP forward - process through all stages."""
        for i in range(self.pp_strategy.num_stages):
            stage = self.get_stage(i)
            if stage:
                hidden = stage(hidden)

        return hidden

    def get_stage(self, stage_idx: int) -> Optional[PPStageModule]:
        """Get a specific stage."""
        name = f"stage_{stage_idx}"
        return self._submodules.get(name)

    def get_all_stages(self) -> List[Optional[PPStageModule]]:
        """Get all stages."""
        return [self.get_stage(i) for i in range(self.pp_strategy.num_stages)]

    def bind_stage(
        self,
        stage_idx: int,
        ctx: "ParallelContext",
        mode: str = "forward_backward",
    ) -> Optional[ModuleInstance]:
        """Bind a single stage to ParallelContext."""
        stage = self.get_stage(stage_idx)
        if stage:
            return stage.bind(ctx, pp_stage=stage_idx, mode=mode)
        return None

    def bind_all_stages(
        self,
        ctx: "ParallelContext",
        mode: str = "forward_backward",
    ) -> List[Optional[ModuleInstance]]:
        """Bind all stages."""
        return [self.bind_stage(i, ctx, mode=mode) for i in range(self.pp_strategy.num_stages)]

    def estimate_pp_time(
        self,
        ctx: "ParallelContext",
        backend: Any,
        mode: str = "forward_backward",
    ) -> Dict[str, float]:
        """Estimate PP total time including bubble.

        Args:
            ctx: ParallelContext
            backend: KernelBackend for time estimation
            mode: "forward" or "forward_backward"

        Returns:
            Dict with time breakdown and throughput
        """
        stage_times = []
        stage_instances = self.bind_all_stages(ctx, mode=mode)

        for stage_instance in stage_instances:
            if stage_instance:
                stage_time = stage_instance.estimate_time(backend)
                stage_times.append(stage_time)
            else:
                stage_times.append(0.0)

        max_stage_time = max(stage_times) if stage_times else 0.0

        if max_stage_time <= 0:
            return {
                "ideal_time_sec": 0.0,
                "bubble_time_sec": 0.0,
                "bubble_ratio": self.pp_strategy.get_bubble_ratio(),
                "stage_comm_time_sec": 0.0,
                "total_time_sec": 0.0,
                "throughput_tokens_per_sec": 0.0,
            }

        ideal_time = max_stage_time * self.pp_strategy.num_micro_batches

        bubble_time = ideal_time * self.pp_strategy.get_bubble_ratio()

        activation_bytes = self._estimate_activation_bytes_between_stages(ctx)

        comm_time_per_transfer = self._estimate_p2p_time(ctx, activation_bytes)
        num_transfers = self.pp_strategy.num_stages - 1

        if self.pp_strategy.schedule in ["interleaved", "vpp"]:
            vpp_factor = self.pp_strategy.num_virtual_stages
            stage_comm_time = comm_time_per_transfer * num_transfers * vpp_factor
        else:
            stage_comm_time = comm_time_per_transfer * num_transfers

        total_time = ideal_time + bubble_time + stage_comm_time

        total_tokens = self.pp_strategy.micro_batch_size * self.pp_strategy.num_micro_batches
        throughput = total_tokens / total_time if total_time > 0 else 0.0

        return {
            "ideal_time_sec": ideal_time,
            "bubble_time_sec": bubble_time,
            "bubble_ratio": self.pp_strategy.get_bubble_ratio(),
            "stage_comm_time_sec": stage_comm_time,
            "total_time_sec": total_time,
            "throughput_tokens_per_sec": throughput,
            "max_stage_time_sec": max_stage_time,
            "stage_times_sec": stage_times,
        }

    def _estimate_activation_bytes_between_stages(self, ctx: "ParallelContext") -> int:
        """Estimate activation size between stages."""
        dtype_size = DTYPE_SIZES.get(ctx.dtype if hasattr(ctx, "dtype") else self.dtype, 2)

        hidden_size = self.hidden_size
        seq_len = self.max_seq_len
        batch_size = self.pp_strategy.micro_batch_size

        tp_degree = getattr(ctx, "tp_degree", 1)
        sp_degree = getattr(ctx, "sp_degree", 1)

        if tp_degree > 1:
            hidden_size = hidden_size // tp_degree
        if sp_degree > 1:
            seq_len = seq_len // sp_degree

        physical_bytes = batch_size * seq_len * hidden_size * dtype_size

        return physical_bytes

    def _estimate_p2p_time(
        self,
        ctx: "ParallelContext",
        data_bytes: int,
    ) -> float:
        """Estimate P2P communication time."""
        bandwidth_gbps = 100.0

        if hasattr(ctx, "cluster") and ctx.cluster:
            cluster = ctx.cluster
            if hasattr(cluster, "p2p_bandwidth_gbps"):
                bandwidth_gbps = cluster.p2p_bandwidth_gbps
            elif hasattr(cluster, "inter_node_bandwidth_gbps"):
                bandwidth_gbps = cluster.inter_node_bandwidth_gbps

        data_gb = data_bytes / 1e9

        transfer_time = data_gb / bandwidth_gbps

        latency_ms = 0.1
        latency_sec = latency_ms / 1000.0

        return transfer_time + latency_sec

    def get_stage_assignment(self) -> Dict[str, int]:
        """Get layer-to-stage assignment."""
        return self._stage_assignment

    def get_stage_layers(self) -> Dict[int, List[ShardedModule]]:
        """Get layers for each stage."""
        return self._stage_layers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_stages": self.pp_strategy.num_stages,
            "num_virtual_stages": self.pp_strategy.num_virtual_stages,
            "schedule": self.pp_strategy.schedule,
            "num_micro_batches": self.pp_strategy.num_micro_batches,
            "micro_batch_size": self.pp_strategy.micro_batch_size,
            "bubble_ratio": self.pp_strategy.get_bubble_ratio(),
            "stage_assignment": self._stage_assignment,
            "stages": {
                f"stage_{i}": stage.to_dict() if stage else None for i, stage in enumerate(self.get_all_stages())
            },
        }

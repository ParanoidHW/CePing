"""ParallelContext - Parallel strategy context.

Integrates StrategyConfig and Cluster for unified parallel strategy management.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from llm_perf.strategy.base import StrategyConfig
    from llm_perf.hardware.cluster import Cluster


class SPType(Enum):
    """Sequence Parallelism type."""

    NONE = "none"
    ULYSSES = "ulysses"
    RING_P2P = "ring_p2p"
    RING_ALLGATHER = "ring_allgather"
    MEGATRON = "megatron"
    UNIFIED_2D = "unified_2d"


class CommDomain:
    """Communication domain for a parallel strategy.

    Attributes:
        ptype: Parallel type (tp, sp, ep, dp, pp)
        degree: Parallel degree
        ranks: List of ranks in this domain
        bandwidth_gbps: Network bandwidth in GB/s
    """

    def __init__(
        self,
        ptype: str,
        degree: int,
        ranks: List[int],
        bandwidth_gbps: float = 100.0,
    ):
        self.ptype = ptype
        self.degree = degree
        self.ranks = ranks
        self.bandwidth_gbps = bandwidth_gbps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ptype": self.ptype,
            "degree": self.degree,
            "ranks": self.ranks,
            "bandwidth_gbps": self.bandwidth_gbps,
        }


@dataclass
class ParallelContext:
    """Parallel strategy context - unified model modeling and cost modeling.
    
    Layered Parallelism Support:
    - Attention layers: use tp_degree for tensor parallelism
    - MoE/FFN layers: use expert_tp_degree + ep_degree for independent parallelism
    
    Attributes:
        tp_degree: Tensor parallelism degree (Attention layers)
        pp_degree: Pipeline parallelism degree
        ep_degree: Expert parallelism degree (MoE)
        sp_degree: Sequence parallelism degree
        dp_degree: Data parallelism degree
        expert_tp_degree: Expert tensor parallelism degree (MoE/FFN layers)
            - If not set or equals tp_degree: uniform TP across all layers
            - If different from tp_degree: layered parallelism
        sp_type: Sequence parallelism type
        ulysses_degree: Ulysses SP degree (for unified 2D)
        ring_degree: Ring SP degree (for unified 2D)
        dtype: Data type
        device: Target device
        activation_checkpointing: Whether to use activation checkpointing
        activation_checkpointing_ratio: Checkpointing ratio
        zero_stage: ZeRO optimization stage
        comm_domains: Communication domain mapping
    """

    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    sp_degree: int = 1
    dp_degree: int = 1
    expert_tp_degree: Optional[int] = None
    sp_type: SPType = SPType.NONE
    ulysses_degree: int = 1
    ring_degree: int = 1
    dtype: str = "fp16"
    device: Optional[Any] = None
    activation_checkpointing: bool = False
    activation_checkpointing_ratio: int = 1
    zero_stage: int = 0
    hidden_size: int = 4096

    comm_domains: Dict[str, CommDomain] = field(default_factory=dict)

    def __post_init__(self):
        """Set expert_tp_degree default to tp_degree if not specified."""
        if self.expert_tp_degree is None:
            self.expert_tp_degree = self.tp_degree

    def get_degree(self, ptype: str) -> int:
        """Get degree for a parallel strategy."""
        degrees = {
            "tp": self.tp_degree,
            "pp": self.pp_degree,
            "ep": self.ep_degree,
            "sp": self.sp_degree,
            "dp": self.dp_degree,
        }
        return degrees.get(ptype, 1)

    def get_comm_domain(self, ptype: str) -> Optional[CommDomain]:
        """Get communication domain for a parallel strategy."""
        return self.comm_domains.get(ptype)

    def get_total_gpus(self) -> int:
        """Total number of GPUs for Attention part.
        
        Uses tp_degree for Attention layers.
        Formula: tp_degree × pp_degree × sp_degree × dp_degree
        """
        return self.tp_degree * self.pp_degree * self.sp_degree * self.dp_degree

    def get_moe_total_gpus(self) -> int:
        """Total number of GPUs for MoE part.
        
        Uses expert_tp_degree for MoE/FFN layers.
        Formula: expert_tp_degree × ep_degree × pp_degree × sp_degree × dp_degree
        
        If expert_tp_degree == tp_degree (uniform TP), this equals get_total_gpus().
        """
        effective_expert_tp = self.expert_tp_degree or self.tp_degree
        return effective_expert_tp * self.ep_degree * self.pp_degree * self.sp_degree * self.dp_degree

    def build_from_strategy(
        self,
        strategy: "StrategyConfig",
        cluster: Optional["Cluster"] = None,
    ) -> "ParallelContext":
        """Build from StrategyConfig and Cluster."""
        self.tp_degree = getattr(strategy, "tp_degree", 1)
        self.pp_degree = getattr(strategy, "pp_degree", 1)
        self.ep_degree = getattr(strategy, "ep_degree", 1)
        self.sp_degree = getattr(strategy, "sp_degree", 1)
        self.dp_degree = getattr(strategy, "dp_degree", 1)
        self.expert_tp_degree = getattr(strategy, "expert_tp_degree", None)
        if self.expert_tp_degree is None:
            self.expert_tp_degree = self.tp_degree

        sp_type_str = getattr(strategy, "sp_type", None)
        if sp_type_str:
            try:
                self.sp_type = SPType(sp_type_str.lower())
            except ValueError:
                self.sp_type = SPType.NONE

        self.ulysses_degree = getattr(strategy, "ulysses_degree", 1)
        self.ring_degree = getattr(strategy, "ring_degree", 1)
        self.dtype = getattr(strategy, "dtype", "fp16")

        if cluster:
            self._build_comm_domains(strategy, cluster)

        return self

    def _build_comm_domains(
        self,
        strategy: "StrategyConfig",
        cluster: "Cluster",
    ):
        """Build communication domains."""
        comm_mapping = getattr(strategy, "get_communication_domain_mapping", None)
        if comm_mapping:
            try:
                mapping = comm_mapping(
                    devices_per_node=cluster.devices_per_node,
                    num_nodes=cluster.num_nodes,
                )
                for ptype, info in mapping.items():
                    self.comm_domains[ptype] = CommDomain(
                        ptype=ptype,
                        degree=info.get("degree", 1),
                        ranks=info.get("groups", [[0]])[0],
                        bandwidth_gbps=self._get_bandwidth_for_domain(cluster, ptype),
                    )
            except Exception:
                self._build_default_comm_domains(cluster)
        else:
            self._build_default_comm_domains(cluster)

    def _build_default_comm_domains(self, cluster: "Cluster"):
        """Build default communication domains."""
        devices_per_node = getattr(cluster, "devices_per_node", 8)

        intra_node_bw = getattr(cluster, "intra_node_bw_gbps", 400.0)
        inter_node_bw = getattr(cluster, "inter_node_bw_gbps", 100.0)

        if self.tp_degree > 1:
            tp_ranks = list(range(self.tp_degree))
            bw = intra_node_bw if self.tp_degree <= devices_per_node else inter_node_bw
            self.comm_domains["tp"] = CommDomain("tp", self.tp_degree, tp_ranks, bw)

        if self.sp_degree > 1:
            sp_ranks = list(range(self.sp_degree))
            bw = intra_node_bw if self.sp_degree <= devices_per_node else inter_node_bw
            self.comm_domains["sp"] = CommDomain("sp", self.sp_degree, sp_ranks, bw)

        if self.ep_degree > 1:
            ep_ranks = list(range(self.ep_degree))
            bw = intra_node_bw if self.ep_degree <= devices_per_node else inter_node_bw
            self.comm_domains["ep"] = CommDomain("ep", self.ep_degree, ep_ranks, bw)

        if self.dp_degree > 1:
            dp_ranks = list(range(self.dp_degree))
            bw = inter_node_bw if self.dp_degree > devices_per_node else intra_node_bw
            self.comm_domains["dp"] = CommDomain("dp", self.dp_degree, dp_ranks, bw)

    def _get_bandwidth_for_domain(self, cluster: "Cluster", ptype: str) -> float:
        """Get bandwidth for a communication domain."""
        degree = self.get_degree(ptype)
        devices_per_node = getattr(cluster, "devices_per_node", 8)

        intra_node_bw = getattr(cluster, "intra_node_bw_gbps", 400.0)
        inter_node_bw = getattr(cluster, "inter_node_bw_gbps", 100.0)

        if degree <= devices_per_node:
            return intra_node_bw
        else:
            return inter_node_bw

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tp_degree": self.tp_degree,
            "pp_degree": self.pp_degree,
            "ep_degree": self.ep_degree,
            "sp_degree": self.sp_degree,
            "dp_degree": self.dp_degree,
            "expert_tp_degree": self.expert_tp_degree,
            "sp_type": self.sp_type.value,
            "ulysses_degree": self.ulysses_degree,
            "ring_degree": self.ring_degree,
            "dtype": self.dtype,
            "total_gpus": self.get_total_gpus(),
            "moe_total_gpus": self.get_moe_total_gpus(),
            "activation_checkpointing": self.activation_checkpointing,
            "zero_stage": self.zero_stage,
            "comm_domains": {ptype: domain.to_dict() for ptype, domain in self.comm_domains.items()},
        }

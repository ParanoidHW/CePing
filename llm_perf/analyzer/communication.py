"""Communication estimation utilities."""

from typing import Dict, Tuple

from llm_perf.legacy.models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig, SPType
from ..utils.constants import DTYPE_SIZES, PHASE_PREFILL, PHASE_DECODE


class CommunicationEstimator:
    """Estimates communication time for various parallelism strategies."""

    def __init__(
        self,
        model: BaseModel,
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        self.model = model
        self.device = device
        self.cluster = cluster
        self.strategy = strategy

    def estimate_training_communication(self, seq_len: int, micro_batch: int) -> Tuple[float, Dict[str, float]]:
        """Estimate communication time for one training step.

        Returns:
            Tuple of (communication_time, breakdown_dict)
        """
        comm_time = 0.0
        breakdown = {}

        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        hidden_size = self.model.config.hidden_size
        effective_num_layers = self._get_effective_num_layers()

        comm_mapping = self.strategy.get_communication_domain_mapping(
            devices_per_node=self.cluster.devices_per_node,
            total_devices=self.cluster.num_devices,
        )

        if self.strategy.tp_degree > 1:
            tp_time = self._estimate_tp_communication(
                micro_batch, seq_len, hidden_size, dtype_size, effective_num_layers, comm_mapping
            )
            comm_time += tp_time
            breakdown["tensor_parallel"] = tp_time

        if self.strategy.pp_degree > 1:
            pp_time = self._estimate_pp_communication(micro_batch, seq_len, hidden_size, dtype_size, comm_mapping)
            comm_time += pp_time
            breakdown["pipeline_parallel"] = pp_time

        if self.strategy.dp_degree > 1:
            dp_time = self._estimate_dp_communication(dtype_size, comm_mapping)
            comm_time += dp_time
            breakdown["data_parallel"] = dp_time

        if self.strategy.ep_degree > 1:
            ep_time = self._estimate_ep_communication(micro_batch, seq_len, hidden_size, dtype_size, comm_mapping)
            comm_time += ep_time
            breakdown["expert_parallel"] = ep_time

        if self.strategy.sp_degree > 1:
            sp_time = self._estimate_sp_communication(micro_batch, seq_len, hidden_size, dtype_size, comm_mapping)
            comm_time += sp_time
            breakdown["sequence_parallel"] = sp_time

        return comm_time, breakdown

    def estimate_inference_communication(self, phase: str) -> Tuple[float, Dict[str, float]]:
        """Estimate communication time for inference phase.

        Returns:
            Tuple of (communication_time, breakdown_dict)
        """
        comm_time = 0.0
        breakdown = {}

        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        hidden_size = self.model.config.hidden_size

        comm_mapping = self.strategy.get_communication_domain_mapping(
            devices_per_node=self.cluster.devices_per_node,
            total_devices=self.cluster.num_devices,
        )

        if self.strategy.tp_degree > 1:
            tp_time = self._estimate_tp_communication_inference(phase, hidden_size, dtype_size, comm_mapping)
            comm_time += tp_time
            breakdown["tensor_parallel"] = tp_time

        if self.strategy.pp_degree > 1:
            pp_time = self._estimate_pp_communication_inference(hidden_size, dtype_size, comm_mapping)
            comm_time += pp_time
            breakdown["pipeline_parallel"] = pp_time

        if self.strategy.ep_degree > 1:
            ep_time = self._estimate_ep_communication_inference(phase, hidden_size, dtype_size, comm_mapping)
            comm_time += ep_time
            breakdown["expert_parallel"] = ep_time

        if self.strategy.sp_degree > 1:
            sp_time = self._estimate_sp_communication_inference(phase, hidden_size, dtype_size, comm_mapping)
            comm_time += sp_time
            breakdown["sequence_parallel"] = sp_time

        return comm_time, breakdown

    def _estimate_tp_communication(
        self,
        micro_batch: int,
        seq_len: int,
        hidden_size: int,
        dtype_size: int,
        effective_num_layers: int,
        comm_mapping: Dict,
    ) -> float:
        """Estimate TP communication time for training."""
        activation_bytes = micro_batch * seq_len * hidden_size * dtype_size
        tp_bytes = activation_bytes * 4 * effective_num_layers

        tp_info = comm_mapping.get("tp", {})
        tp_ranks = tp_info.get("ranks", list(range(self.strategy.tp_degree)))

        return self.cluster.estimate_allreduce_time(tp_bytes, tp_ranks)

    def _estimate_tp_communication_inference(
        self,
        phase: str,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: Dict,
    ) -> float:
        """Estimate TP communication time for inference."""
        batch_size = 1

        if phase == PHASE_PREFILL:
            seq_len = self.model.config.max_seq_len
        else:
            seq_len = 1

        activation_bytes = batch_size * seq_len * hidden_size * dtype_size
        num_layers = self.model.config.num_layers
        tp_bytes = activation_bytes * 2 * num_layers

        tp_info = comm_mapping.get("tp", {})
        tp_ranks = tp_info.get("ranks", list(range(self.strategy.tp_degree)))

        return self.cluster.estimate_allreduce_time(tp_bytes, tp_ranks)

    def _estimate_pp_communication(
        self,
        micro_batch: int,
        seq_len: int,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: Dict,
    ) -> float:
        """Estimate PP communication time for training."""
        pp_bytes = micro_batch * seq_len * hidden_size * dtype_size
        num_micro_batches = self.strategy.dp_degree

        pp_info = comm_mapping.get("pp", {})
        bandwidth_domain = pp_info.get("bandwidth_domain", "inter_rack")
        devices_per_group = pp_info.get("devices_per_group", self.cluster.num_devices)

        pp_bw = self.cluster.get_bandwidth_for_topology_level(bandwidth_domain, devices_per_group)

        pp_time = pp_bytes / (pp_bw * 1e9)
        pp_time *= 2 * num_micro_batches

        return pp_time

    def _estimate_pp_communication_inference(
        self,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: Dict,
    ) -> float:
        """Estimate PP communication time for inference."""
        batch_size = 1
        pp_bytes = batch_size * hidden_size * dtype_size

        pp_info = comm_mapping.get("pp", {})
        topology_level = pp_info.get("topology_level", "inter_node")
        devices_per_group = pp_info.get("devices_per_group", self.cluster.num_devices)

        pp_bw = self.cluster.get_bandwidth_for_communication_domain(topology_level, devices_per_group)

        return pp_bytes / (pp_bw * 1e9)

    def _estimate_dp_communication(self, dtype_size: int, comm_mapping: Dict) -> float:
        """Estimate DP communication time for training."""
        effective_params = self.model.total_params // self.strategy.tp_degree
        if self.strategy.pp_degree > 1:
            effective_params = effective_params // self.strategy.pp_degree
        grad_bytes = effective_params * dtype_size

        zero_factor = {0: 1.0, 1: 1.0, 2: 1.0 / self.strategy.dp_degree, 3: 0}
        grad_bytes *= zero_factor.get(self.strategy.zero_stage, 1.0)

        dp_info = comm_mapping.get("dp", {})
        dp_ranks = dp_info.get("ranks", list(range(self.strategy.dp_degree)))

        return self.cluster.estimate_allreduce_time(grad_bytes, dp_ranks)

    def _estimate_ep_communication(
        self,
        micro_batch: int,
        seq_len: int,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: Dict,
    ) -> float:
        """Estimate EP communication time for training."""
        effective_hidden_size = hidden_size // self.strategy.tp_degree
        token_bytes = micro_batch * seq_len * effective_hidden_size * dtype_size

        ep_info = comm_mapping.get("ep", {})
        ep_ranks = ep_info.get("ranks", list(range(self.strategy.ep_degree)))

        dispatch_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
        combine_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)

        return (dispatch_time + combine_time) * 2

    def _estimate_ep_communication_inference(
        self,
        phase: str,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: Dict,
    ) -> float:
        """Estimate EP communication time for inference."""
        batch_size = 1
        seq_len = 1 if phase == PHASE_DECODE else self.model.config.max_seq_len
        token_bytes = batch_size * seq_len * hidden_size * dtype_size

        ep_info = comm_mapping.get("ep", {})
        ep_ranks = ep_info.get("ranks", list(range(self.strategy.ep_degree)))

        dispatch_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
        combine_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)

        return dispatch_time + combine_time

    def _estimate_sp_communication(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: Dict,
    ) -> float:
        """Estimate SP communication time for training."""
        sp_degree = self.strategy.sp_degree
        sp_type = self.strategy.sp_type
        num_layers = self.model.config.num_layers
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        head_dim = hidden_size // self.model.config.num_attention_heads

        activation_bytes = batch_size * seq_len * hidden_size * dtype_size
        kv_bytes_per_step = batch_size * (seq_len // sp_degree) * num_kv_heads * head_dim * 2 * dtype_size

        sp_info = comm_mapping.get("sp", {})
        sp_ranks = sp_info.get("ranks", list(range(sp_degree)))
        sp_topology_level = sp_info.get("topology_level", "node")
        sp_devices_per_group = sp_info.get("devices_per_group", self.cluster.devices_per_node)

        return self._compute_sp_time(
            sp_type,
            sp_degree,
            num_layers,
            activation_bytes,
            kv_bytes_per_step,
            sp_ranks,
            sp_topology_level,
            sp_devices_per_group,
            is_training=True,
        )

    def _estimate_sp_communication_inference(
        self,
        phase: str,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: Dict,
    ) -> float:
        """Estimate SP communication time for inference."""
        sp_degree = self.strategy.sp_degree
        sp_type = self.strategy.sp_type
        num_layers = self.model.config.num_layers
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        head_dim = hidden_size // self.model.config.num_attention_heads

        batch_size = 1
        seq_len = 1 if phase == PHASE_DECODE else self.model.config.max_seq_len

        activation_bytes = batch_size * seq_len * hidden_size * dtype_size
        kv_bytes_per_step = batch_size * (seq_len // sp_degree) * num_kv_heads * head_dim * 2 * dtype_size

        sp_info = comm_mapping.get("sp", {})
        sp_ranks = sp_info.get("ranks", list(range(sp_degree)))
        sp_topology_level = sp_info.get("topology_level", "node")
        sp_devices_per_group = sp_info.get("devices_per_group", self.cluster.devices_per_node)

        return self._compute_sp_time(
            sp_type,
            sp_degree,
            num_layers,
            activation_bytes,
            kv_bytes_per_step,
            sp_ranks,
            sp_topology_level,
            sp_devices_per_group,
            is_training=False,
            phase=phase,
        )

    def _compute_sp_time(
        self,
        sp_type: SPType,
        sp_degree: int,
        num_layers: int,
        activation_bytes: int,
        kv_bytes_per_step: int,
        sp_ranks: list,
        sp_topology_level: str,
        sp_devices_per_group: int,
        is_training: bool,
        phase: str = None,
    ) -> float:
        """Compute SP communication time based on SP type."""
        total_time = 0.0

        if sp_type == SPType.ULYSSES:
            alltoall_time = self.cluster.estimate_alltoall_time(activation_bytes, sp_ranks)
            if is_training:
                total_time = alltoall_time * 8 * num_layers
            else:
                total_time = alltoall_time * 4 * num_layers

        elif sp_type == SPType.RING_P2P:
            if sp_degree > 1:
                sp_bw = self.cluster.get_bandwidth_for_topology_level(sp_topology_level, sp_devices_per_group)
                step_time = kv_bytes_per_step / (sp_bw * 1e9)
                if is_training:
                    total_time = step_time * (sp_degree - 1) * 2 * num_layers
                else:
                    total_time = step_time * (sp_degree - 1) * num_layers

        elif sp_type == SPType.RING_ALLGATHER:
            kv_bytes_per_block = kv_bytes_per_step
            allgather_bytes = kv_bytes_per_block * sp_degree

            allgather_time = self.cluster.estimate_allgather_time(allgather_bytes, sp_ranks)

            num_forward_ag = 2 if self.strategy.kv_separate_allgather else 1

            if is_training:
                reducescatter_time = self.cluster.estimate_reducescatter_time(allgather_bytes, sp_ranks)
                num_backward_rs = num_forward_ag
                total_time = (allgather_time * num_forward_ag + reducescatter_time * num_backward_rs) * num_layers
            else:
                total_time = allgather_time * num_forward_ag * num_layers

        elif sp_type == SPType.UNIFIED_2D:
            ulysses_degree, ring_degree = self._resolve_2d_sp_config(sp_degree)

            ulysses_ranks = list(range(ulysses_degree))

            ulysses_time = self.cluster.estimate_alltoall_time(activation_bytes, ulysses_ranks)
            if is_training:
                ulysses_time *= 8 * num_layers
            else:
                ulysses_time *= 4 * num_layers

            ring_time = 0.0
            if ring_degree > 1:
                ring_kv_bytes = kv_bytes_per_step
                ring_bw = self.cluster.get_bandwidth_for_topology_level("rack", self.cluster.devices_per_node * 16)
                ring_step_time = ring_kv_bytes / (ring_bw * 1e9)
                if is_training:
                    ring_time = ring_step_time * (ring_degree - 1) * 2 * num_layers
                else:
                    ring_time = ring_step_time * (ring_degree - 1) * num_layers

            total_time = ulysses_time + ring_time

        elif sp_type == SPType.MEGATRON:
            rs_time = self.cluster.estimate_reducescatter_time(activation_bytes, sp_ranks)
            ag_time = self.cluster.estimate_allgather_time(activation_bytes, sp_ranks)
            if is_training:
                total_time = (rs_time + ag_time) * 4 * num_layers
            else:
                total_time = (rs_time + ag_time) * 2 * num_layers

        return total_time

    def _resolve_2d_sp_config(self, sp_degree: int) -> Tuple[int, int]:
        """Resolve 2D SP configuration."""
        ulysses_degree = self.strategy.ulysses_degree or 1
        ring_degree = self.strategy.ring_degree or 1

        if ulysses_degree * ring_degree != sp_degree:
            ulysses_degree = min(sp_degree, 8)
            ring_degree = sp_degree // ulysses_degree
            while ulysses_degree * ring_degree != sp_degree and ring_degree > 1:
                ulysses_degree -= 1
                ring_degree = sp_degree // ulysses_degree

        return ulysses_degree, ring_degree

    def _get_effective_num_layers(self) -> int:
        """Get effective number of layers after PP sharding."""
        num_layers = self.model.config.num_layers
        if self.strategy.pp_degree > 1:
            return num_layers // self.strategy.pp_degree
        return num_layers

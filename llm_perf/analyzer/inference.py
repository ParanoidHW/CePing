"""Inference performance analyzer."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig, SPType
from ..kernels.compute import ComputeKernelRegistry
from ..kernels.communication import CommKernelRegistry
from ..utils.constants import DTYPE_SIZES, PHASE_PREFILL, PHASE_DECODE
from .breakdown import PerformanceBreakdown, LayerBreakdown, KernelBreakdown
from .detailed_breakdown import DetailedPerformanceResult
from .breakdown_generator import BreakdownGenerator
from .result_base import BaseResult


@dataclass
class InferenceResult(BaseResult):
    """Result of inference performance analysis."""

    # Phase-specific metrics
    prefill_time_sec: float  # TTFT (Time To First Token)
    decode_time_per_step_sec: float  # TPOT (Time Per Output Token)

    # Throughput
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float  # TPS (Tokens Per Second)

    # End-to-end metrics
    total_time_sec: float  # For full generation
    total_tokens: int

    # Memory
    memory_per_gpu_gb: float
    kv_cache_memory_gb: float

    # Per-GPU throughput (must come after non-default fields)
    prefill_tokens_per_sec_per_gpu: float = 0.0
    decode_tokens_per_sec_per_gpu: float = 0.0
    total_gpus: int = 1

    # Detailed breakdown
    prefill_breakdown: PerformanceBreakdown = None
    decode_breakdown: PerformanceBreakdown = None

    # Detailed performance breakdown with memory/communication breakdown
    detailed_breakdown: Optional[DetailedPerformanceResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "prefill": {
                "ttft_sec": self.prefill_time_sec,
                "ttft_ms": self.prefill_time_sec * 1000,
                "tokens_per_sec": self.prefill_tokens_per_sec,
                "tokens_per_sec_per_gpu": self.prefill_tokens_per_sec_per_gpu,
            },
            "decode": {
                "tpot_sec": self.decode_time_per_step_sec,
                "tpot_ms": self.decode_time_per_step_sec * 1000,
                "tps": self.decode_tokens_per_sec,
                "tps_per_gpu": self.decode_tokens_per_sec_per_gpu,
            },
            "end_to_end": {
                "total_time_sec": self.total_time_sec,
                "total_tokens": self.total_tokens,
                "overall_tps": self.total_tokens / self.total_time_sec,
            },
            "memory": {
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
                "kv_cache_gb": self.kv_cache_memory_gb,
            },
            "parallelism": {
                "total_gpus": self.total_gpus,
            },
            "prefill_breakdown": self.prefill_breakdown.to_dict() if self.prefill_breakdown else None,
            "decode_breakdown": self.decode_breakdown.to_dict() if self.decode_breakdown else None,
        }
        if self.detailed_breakdown is not None:
            result["detailed_breakdown"] = self.detailed_breakdown.to_dict()
        return result


class InferenceAnalyzer:
    """Analyzes inference performance for prefill and decode phases."""

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

        self.compute_registry = ComputeKernelRegistry(device)
        self.comm_registry = CommKernelRegistry(cluster)

    def _get_effective_seq_len(self, seq_len: int) -> int:
        """Get sequence length after SP/CP sharding."""
        # SP cuts along sequence dimension
        sp_degree = self.strategy.sp_degree
        cp_degree = self.strategy.cp_degree
        # Take the maximum of SP and CP since they both partition sequence
        seq_parallel_degree = max(sp_degree, cp_degree)
        if seq_parallel_degree > 1:
            return max(1, seq_len // seq_parallel_degree)
        return seq_len

    def _get_effective_num_heads(self) -> tuple:
        """Get attention heads after TP sharding.

        Returns:
            tuple: (effective_num_heads, effective_num_kv_heads)
        """
        tp_degree = self.strategy.tp_degree
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = self.model.config.num_key_value_heads or num_heads

        if tp_degree > 1:
            effective_num_heads = max(1, num_heads // tp_degree)
            effective_num_kv_heads = max(1, num_kv_heads // tp_degree)
            return effective_num_heads, effective_num_kv_heads
        return num_heads, num_kv_heads

    def _get_effective_num_layers(self) -> int:
        """Get number of layers for current PP stage."""
        pp_degree = self.strategy.pp_degree
        num_layers = self.model.config.num_layers

        if pp_degree > 1:
            # Each PP stage handles a subset of layers
            return max(1, num_layers // pp_degree)
        return num_layers

    def _get_effective_intermediate_size(self) -> int:
        """Get FFN intermediate size after TP sharding."""
        tp_degree = self.strategy.tp_degree
        intermediate_size = self.model.config.intermediate_size

        if intermediate_size > 0 and tp_degree > 1:
            return max(1, intermediate_size // tp_degree)
        return intermediate_size

    def _get_total_gpus(self) -> int:
        """Get total number of GPUs used."""
        return self.strategy.world_size
    
    def analyze(
        self,
        batch_size: int,
        prompt_len: int,
        generation_len: int,
    ) -> InferenceResult:
        """
        Analyze inference performance.

        Args:
            batch_size: Number of concurrent requests
            prompt_len: Input prompt length
            generation_len: Number of tokens to generate

        Returns:
            InferenceResult with performance metrics
        """
        # Prefill phase (process prompt)
        prefill_time = self._estimate_prefill_time(batch_size, prompt_len)

        # Decode phase (generate tokens one by one)
        decode_time_per_step = self._estimate_decode_time(batch_size)

        # Memory estimation
        memory_bytes, kv_cache_bytes = self._estimate_memory(batch_size, prompt_len + generation_len)

        # Calculate throughput metrics
        prefill_tokens = batch_size * prompt_len
        decode_tokens = batch_size * generation_len

        prefill_tps = prefill_tokens / prefill_time if prefill_time > 0 else 0
        decode_tps = batch_size / decode_time_per_step if decode_time_per_step > 0 else 0

        # Calculate per-GPU throughput
        total_gpus = self._get_total_gpus()
        prefill_tps_per_gpu = prefill_tps / total_gpus if total_gpus > 0 else prefill_tps
        decode_tps_per_gpu = decode_tps / total_gpus if total_gpus > 0 else decode_tps

        # End-to-end time
        total_time = prefill_time + decode_time_per_step * generation_len

        # Build breakdowns
        prefill_breakdown = self._build_prefill_breakdown(batch_size, prompt_len, prefill_time)
        decode_breakdown = self._build_decode_breakdown(batch_size, decode_time_per_step)

        # Generate detailed breakdown
        detailed_breakdown = self._generate_detailed_breakdown(
            prefill_time, decode_time_per_step, generation_len, memory_bytes
        )

        return InferenceResult(
            prefill_time_sec=prefill_time,
            decode_time_per_step_sec=decode_time_per_step,
            prefill_tokens_per_sec=prefill_tps,
            decode_tokens_per_sec=decode_tps,
            total_time_sec=total_time,
            total_tokens=prefill_tokens + decode_tokens,
            memory_per_gpu_gb=memory_bytes / 1024 / 1024 / 1024,
            kv_cache_memory_gb=kv_cache_bytes / 1024 / 1024 / 1024,
            prefill_breakdown=prefill_breakdown,
            decode_breakdown=decode_breakdown,
            detailed_breakdown=detailed_breakdown,
            # Add per-GPU metrics
            prefill_tokens_per_sec_per_gpu=prefill_tps_per_gpu,
            decode_tokens_per_sec_per_gpu=decode_tps_per_gpu,
            total_gpus=total_gpus,
        )
    
    def _generate_detailed_breakdown(
        self, prefill_time: float, decode_time_per_step: float, 
        generation_len: int, memory_bytes: int
    ) -> DetailedPerformanceResult:
        """Generate detailed performance breakdown for inference."""
        # Create generator with is_training=False for inference mode
        generator = BreakdownGenerator(
            self.model, self.device, self.cluster, self.strategy, is_training=False
        )
        
        submodel = generator.generate_submodel_breakdown(
            model_name=self.model.config.name or "model",
            model_type=getattr(self.model.config, 'model_type', None) or "transformer",
            compute_time_sec=prefill_time + decode_time_per_step * generation_len,
            num_iterations=1,
        )
        
        # Build memory breakdown from submodel, including by_block_type
        from .detailed_breakdown import MemoryBreakdown, CommunicationBreakdown

        # Aggregate by_block_type from blocks
        by_block_type: Dict[str, Dict[Any, int]] = {}
        for block in submodel.blocks:
            if block.block_type not in by_block_type:
                by_block_type[block.block_type] = {}
            for mem_type, bytes_val in block.memory_by_type.items():
                if mem_type not in by_block_type[block.block_type]:
                    by_block_type[block.block_type][mem_type] = 0
                by_block_type[block.block_type][mem_type] += bytes_val

        memory_breakdown = MemoryBreakdown(
            by_type=submodel.memory_by_type,
            by_submodel={submodel.model_name: submodel.memory_by_type},
            by_block_type=by_block_type,
        )
        
        # Build communication breakdown from submodel
        comm_breakdown = CommunicationBreakdown(
            by_type=submodel.comm_by_parallelism,
            by_submodel={submodel.model_name: [
                op for ops in submodel.comm_by_parallelism.values() for op in ops
            ]},
        )
        
        total_time = prefill_time + decode_time_per_step * generation_len
        return DetailedPerformanceResult(
            total_time_sec=total_time,
            throughput=generation_len / total_time if total_time > 0 else 0,
            submodels=[submodel],
            memory=memory_breakdown,
            communication=comm_breakdown,
        )
    
    def _estimate_prefill_time(self, batch_size: int, seq_len: int) -> float:
        """Estimate prefill phase time (prompt processing).

        Takes into account parallel sharding strategies:
        - TP: Each GPU handles sharded heads
        - SP: Each GPU handles sharded sequence
        - PP: Each GPU handles subset of layers
        """
        dtype = self.model.config.dtype
        total_time = 0.0

        # Get effective parameters considering parallelism
        effective_seq_len = self._get_effective_seq_len(seq_len)
        effective_num_layers = self._get_effective_num_layers()

        # Prefill processes the full sequence at once
        # Attention is O(batch * effective_seq^2 * effective_heads * dim)

        for layer in self.model.layers:
            kernel = self._get_compute_kernel_for_phase(
                layer, batch_size, effective_seq_len, dtype, PHASE_PREFILL
            )
            if kernel:
                time = kernel.estimate_time(
                    layer.input_shape,
                    layer.output_shape,
                    dtype
                )
                total_time += time

        # Scale compute time by effective layers (for PP)
        # We've iterated through all layers, but PP only runs on a subset
        original_num_layers = self.model.config.num_layers
        if original_num_layers > 0:
            total_time = total_time * effective_num_layers / original_num_layers

        # Add communication overhead
        comm_time, _ = self._estimate_communication_time_for_phase(PHASE_PREFILL)

        # Apply overlap factor
        overlap_factor = 0.8
        total_time = max(total_time, comm_time * (1 - overlap_factor) + max(total_time, comm_time * overlap_factor))

        return total_time
    
    def _estimate_decode_time(self, batch_size: int) -> float:
        """Estimate decode phase time (single token generation).

        Takes into account parallel sharding strategies:
        - TP: Each GPU handles sharded heads
        - SP: For decode, seq_len=1, SP communication still applies
        - PP: Each GPU handles subset of layers
        """
        dtype = self.model.config.dtype
        total_time = 0.0

        # Get effective parameters considering parallelism
        # For decode, seq_len=1, but we need effective heads and layers
        effective_num_layers = self._get_effective_num_layers()

        # Decode processes one token at a time (seq_len=1)
        # But attention still needs to access full KV cache
        seq_len = 1

        for layer in self.model.layers:
            kernel = self._get_compute_kernel_for_phase(
                layer, batch_size, seq_len, dtype, PHASE_DECODE
            )
            if kernel:
                time = kernel.estimate_time(
                    layer.input_shape,
                    layer.output_shape,
                    dtype
                )
                total_time += time

        # Scale compute time by effective layers (for PP)
        original_num_layers = self.model.config.num_layers
        if original_num_layers > 0:
            total_time = total_time * effective_num_layers / original_num_layers

        # KV cache read overhead (memory bandwidth bound)
        # Uses effective heads for TP sharding
        kv_read_time = self._estimate_kv_cache_read_time(batch_size)
        total_time += kv_read_time

        # Communication overhead
        comm_time, _ = self._estimate_communication_time_for_phase(PHASE_DECODE)

        # Apply overlap factor
        overlap_factor = 0.7
        total_time = max(total_time, comm_time * (1 - overlap_factor) + max(total_time, comm_time * overlap_factor))

        return total_time
    
    def _estimate_kv_cache_read_time(self, batch_size: int) -> float:
        """Estimate time to read KV cache during decode.

        Takes into account TP sharding (heads are distributed).
        """
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        # Get effective parameters for TP sharding
        _, effective_num_kv_heads = self._get_effective_num_heads()
        effective_num_layers = self._get_effective_num_layers()

        # KV cache size per token per layer
        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // self.model.config.num_attention_heads

        # K and V per token per layer (using effective heads for TP)
        kv_per_token = 2 * effective_num_kv_heads * head_dim * dtype_size

        # Read all previous tokens (average case)
        avg_seq_len = self.model.config.max_seq_len // 2
        total_kv_bytes = batch_size * avg_seq_len * effective_num_layers * kv_per_token

        # Memory bandwidth bound
        mem_bw = self.device.get_memory_bw_gbps() * 1e9
        return total_kv_bytes / mem_bw
    
    def _estimate_communication_time_for_phase(self, phase: str) -> tuple:
        """Estimate communication time for a phase using topology-aware bandwidth."""
        comm_time = 0.0
        breakdown = {}

        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        hidden_size = self.model.config.hidden_size

        # Get communication domain mapping
        comm_mapping = self.strategy.get_communication_domain_mapping(
            devices_per_node=self.cluster.devices_per_node,
            total_devices=self.cluster.num_devices,
        )

        # TP communication (same as training, less data in decode)
        if self.strategy.tp_degree > 1:
            batch_size = 1  # Per device for decode

            if phase == PHASE_PREFILL:
                seq_len = self.model.config.max_seq_len
                activation_bytes = batch_size * seq_len * hidden_size * dtype_size
            else:
                seq_len = 1
                activation_bytes = batch_size * seq_len * hidden_size * dtype_size

            num_layers = self.model.config.num_layers
            tp_bytes = activation_bytes * 2 * num_layers

            # Use communication domain mapping for topology-aware ranks
            tp_info = comm_mapping.get("tp", {})
            tp_ranks = tp_info.get("ranks", list(range(self.strategy.tp_degree)))

            tp_time = self.cluster.estimate_allreduce_time(tp_bytes, tp_ranks)

            comm_time += tp_time
            breakdown["tensor_parallel"] = tp_time

        # PP communication (P2P between pipeline stages)
        if self.strategy.pp_degree > 1:
            batch_size = 1

            pp_bytes = batch_size * hidden_size * dtype_size

            # Use communication domain mapping for topology-aware bandwidth
            pp_info = comm_mapping.get("pp", {})
            topology_level = pp_info.get("topology_level", "inter_node")
            devices_per_group = pp_info.get("devices_per_group", self.cluster.num_devices)

            pp_bw = self.cluster.get_bandwidth_for_communication_domain(
                topology_level, devices_per_group
            )

            # P2P communication time
            pp_time = pp_bytes / (pp_bw * 1e9)

            comm_time += pp_time
            breakdown["pipeline_parallel"] = pp_time

        # EP communication for MoE
        if self.strategy.ep_degree > 1:
            batch_size = 1
            seq_len = 1 if phase == PHASE_DECODE else self.model.config.max_seq_len

            token_bytes = batch_size * seq_len * hidden_size * dtype_size

            # Use communication domain mapping for topology-aware ranks
            ep_info = comm_mapping.get("ep", {})
            ep_ranks = ep_info.get("ranks", list(range(self.strategy.ep_degree)))

            dispatch_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
            combine_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)

            ep_time = dispatch_time + combine_time
            comm_time += ep_time
            breakdown["expert_parallel"] = ep_time

        # SP communication (Sequence Parallelism)
        if self.strategy.sp_degree > 1:
            sp_time = self._estimate_sp_communication_time_for_phase(
                phase, hidden_size, dtype_size, comm_mapping
            )
            comm_time += sp_time
            breakdown["sequence_parallel"] = sp_time

        return comm_time, breakdown
    
    def _estimate_sp_communication_time_for_phase(
        self,
        phase: str,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: dict = None,
    ) -> float:
        """
        Estimate sequence parallelism communication time for inference.

        Supports ulysses-SP, ring-SP (p2p/allgather), and unified-2D-SP.
        Uses topology-aware bandwidth from communication domain mapping.
        """
        sp_degree = self.strategy.sp_degree
        sp_type = self.strategy.sp_type
        num_layers = self.model.config.num_layers
        num_kv_heads = getattr(
            self.model.config, "num_key_value_heads",
            self.model.config.num_attention_heads
        )
        head_dim = hidden_size // self.model.config.num_attention_heads
        batch_size = 1
        seq_len = 1 if phase == PHASE_DECODE else self.model.config.max_seq_len

        activation_bytes = batch_size * seq_len * hidden_size * dtype_size
        kv_bytes_per_step = (
            batch_size * (seq_len // sp_degree) *
            num_kv_heads * head_dim * 2 * dtype_size
        )

        # Get SP info from communication domain mapping
        if comm_mapping is None:
            comm_mapping = self.strategy.get_communication_domain_mapping(
                devices_per_node=self.cluster.devices_per_node,
                total_devices=self.cluster.num_devices,
            )

        sp_info = comm_mapping.get("sp", {})
        sp_ranks = sp_info.get("ranks", list(range(sp_degree)))
        sp_topology_level = sp_info.get("topology_level", "node")
        sp_devices_per_group = sp_info.get("devices_per_group", self.cluster.devices_per_node)

        total_time = 0.0

        if sp_type == SPType.ULYSSES:
            # 4 all-to-all per attention layer:
            # pre-attention: Q, K, V each needs one all-to-all
            # post-attention: O needs one all-to-all
            alltoall_time = self.cluster.estimate_alltoall_time(
                activation_bytes, sp_ranks
            )
            total_time = alltoall_time * 4 * num_layers

        elif sp_type == SPType.RING_P2P:
            # (sp_degree - 1) P2P steps per layer
            # Each step moves kv_bytes_per_step
            if sp_degree > 1:
                # Use topology-aware bandwidth
                sp_bw = self.cluster.get_bandwidth_for_communication_domain(
                    sp_topology_level, sp_devices_per_group
                )
                step_time = kv_bytes_per_step / (sp_bw * 1e9)
                total_time = step_time * (sp_degree - 1) * num_layers

        elif sp_type == SPType.RING_ALLGATHER:
            # 1 allgather per layer for KV aggregation
            allgather_time = self.cluster.estimate_allgather_time(
                kv_bytes_per_step * sp_degree, sp_ranks
            )
            total_time = allgather_time * num_layers

        elif sp_type == SPType.UNIFIED_2D:
            ulysses_degree = self.strategy.ulysses_degree or 1
            ring_degree = self.strategy.ring_degree or 1
            if ulysses_degree * ring_degree != sp_degree:
                ulysses_degree = min(sp_degree, 8)
                ring_degree = sp_degree // ulysses_degree
                while ulysses_degree * ring_degree != sp_degree and ring_degree > 1:
                    ulysses_degree -= 1
                    ring_degree = sp_degree // ulysses_degree

            # Get ulysses and ring info from communication domain mapping
            ulysses_info = comm_mapping.get("ulysses", {})
            ring_info = comm_mapping.get("ring", {})

            ulysses_ranks = ulysses_info.get("ranks", list(range(ulysses_degree)))
            ulysses_level = ulysses_info.get("topology_level", "node")
            ulysses_dpg = ulysses_info.get("devices_per_group", self.cluster.devices_per_node)

            ring_ranks = ring_info.get("ranks", list(range(ring_degree)))
            ring_level = ring_info.get("topology_level", "rack")
            ring_dpg = ring_info.get("devices_per_group", self.cluster.devices_per_node * 16)

            # Ulysses part: 4 all-to-all (Q/K/V pre + O post)
            ulysses_time = self.cluster.estimate_alltoall_time(
                activation_bytes, ulysses_ranks
            ) * 4 * num_layers

            ring_kv_bytes = (
                batch_size * (seq_len // sp_degree) *
                num_kv_heads * head_dim * 2 * dtype_size
            )
            if ring_degree > 1:
                # Use topology-aware bandwidth for ring
                ring_bw = self.cluster.get_bandwidth_for_communication_domain(
                    ring_level, ring_dpg
                )
                ring_step_time = ring_kv_bytes / (ring_bw * 1e9)
                ring_time = ring_step_time * (ring_degree - 1) * num_layers
            else:
                ring_time = 0.0

            total_time = ulysses_time + ring_time

        elif sp_type == SPType.MEGATRON:
            # Megatron-SP: ReduceScatter + AllGather per layer
            # Communication volume: 2 * activation_bytes per layer (same as TP AllReduce)
            rs_time = self.cluster.estimate_reducescatter_time(
                activation_bytes, sp_ranks
            )
            ag_time = self.cluster.estimate_allgather_time(
                activation_bytes, sp_ranks
            )
            # 2 communication ops per layer (forward pass)
            total_time = (rs_time + ag_time) * 2 * num_layers

        return total_time
    
    def _estimate_memory(self, batch_size: int, max_seq_len: int) -> tuple:
        """Estimate memory per GPU. Returns (total_memory, kv_cache_memory).

        Takes into account parallel sharding strategies:
        - TP: Parameters and KV cache are sharded across TP ranks
        - PP: Parameters and layers are distributed, only own stage's KV cache
        - SP: Activations are sharded along sequence dimension
        """
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        # Get effective parameters
        effective_num_layers = self._get_effective_num_layers()
        _, effective_num_kv_heads = self._get_effective_num_heads()
        effective_seq_len = self._get_effective_seq_len(max_seq_len)
        tp_degree = self.strategy.tp_degree
        pp_degree = self.strategy.pp_degree
        sp_degree = self.strategy.sp_degree
        cp_degree = self.strategy.cp_degree

        # Model parameters - sharded by TP and PP
        param_memory = self.model.total_params * dtype_size
        param_memory //= tp_degree  # TP sharding
        if pp_degree > 1:
            param_memory //= pp_degree  # PP sharding

        # KV cache - sharded by TP (heads) and PP (layers)
        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // self.model.config.num_attention_heads

        # K and V per token per layer using effective heads
        kv_per_token = 2 * effective_num_kv_heads * head_dim * dtype_size

        # KV cache for all tokens and effective layers (PP stage only)
        kv_cache_memory = batch_size * max_seq_len * effective_num_layers * kv_per_token

        # Activations - use inference mode estimate
        is_distributed = (
            self.strategy.tp_degree > 1
            or self.strategy.dp_degree > 1
            or self.strategy.pp_degree > 1
            or self.strategy.sp_degree > 1
        )
        base_memory = self.model.estimate_memory(
            inference_mode=True,  # Inference mode
            batch_size=batch_size,
            is_distributed=is_distributed,
            apply_calibration=False,  # Handle manually for fine-grained control
        )
        param_memory_total = self.model.total_params * dtype_size
        activation_memory = base_memory - param_memory_total

        # Apply parallelism sharding to activation memory
        if tp_degree > 1:
            activation_memory //= tp_degree  # TP sharding
        if sp_degree > 1 or cp_degree > 1:
            # SP/CP sharding along sequence dimension
            seq_parallel_degree = max(sp_degree, cp_degree)
            activation_memory //= seq_parallel_degree
        if pp_degree > 1:
            activation_memory //= pp_degree  # PP sharding

        total_memory = param_memory + kv_cache_memory + activation_memory

        # Apply calibration factors from model config
        calib = self.model.config.memory_calibration
        total_memory = calib.apply(total_memory, is_distributed)

        return total_memory, kv_cache_memory
    
    def _get_compute_kernel_for_phase(
        self,
        layer,
        batch_size: int,
        seq_len: int,
        dtype: str,
        phase: str
    ):
        """Get appropriate compute kernel for a layer and phase.

        Uses effective parameters considering parallelism:
        - m = batch × effective_seq_len (for prefill) or batch × 1 (for decode)
        - n = effective_heads × head_dim (attention) or effective_intermediate_size (ffn)
        - k = hidden_size
        """
        name = layer.name
        hidden_size = self.model.config.hidden_size

        # Get effective parameters
        effective_num_heads, _ = self._get_effective_num_heads()
        head_dim = hidden_size // self.model.config.num_attention_heads
        effective_intermediate_size = self._get_effective_intermediate_size()

        if "proj" in name or "up" in name or "gate" in name or "down" in name:
            # FFN layers - m × k @ k × n
            m = batch_size * seq_len
            k = layer.input_shape[-1] if layer.input_shape else hidden_size
            # For FFN, use effective intermediate size (TP sharded)
            n = layer.output_shape[-1] if layer.output_shape else effective_intermediate_size
            if effective_intermediate_size > 0:
                n = effective_intermediate_size
            return self.compute_registry.get_or_create_matmul(m, n, k, dtype)
        elif "attention" in name:
            # Different attention pattern for prefill vs decode
            # Use effective heads for TP sharding
            effective_head_dim = head_dim
            effective_n = effective_num_heads * effective_head_dim
            if phase == PHASE_PREFILL:
                return self.compute_registry.get(f"flash_attn_{batch_size}_{seq_len}_{effective_num_heads}_{effective_head_dim}_{dtype}")
            else:
                # Decode: seq_len=1 but accesses full KV
                return self.compute_registry.get(f"flash_attn_{batch_size}_1_{effective_num_heads}_{effective_head_dim}_{dtype}")
        elif "norm" in name:
            return self.compute_registry.get(f"rmsnorm_{layer.input_shape[-1]}_{dtype}")
        elif "swiglu" in name or "activation" in name:
            return self.compute_registry.get(f"swiglu_{layer.input_shape[-1]}_{dtype}")

        return None
    
    def _build_prefill_breakdown(
        self,
        batch_size: int,
        seq_len: int,
        total_time: float
    ) -> PerformanceBreakdown:
        """Build detailed breakdown for prefill phase."""
        return self._build_phase_breakdown(batch_size, seq_len, total_time, PHASE_PREFILL)
    
    def _build_decode_breakdown(
        self,
        batch_size: int,
        total_time: float
    ) -> PerformanceBreakdown:
        """Build detailed breakdown for decode phase."""
        return self._build_phase_breakdown(batch_size, 1, total_time, PHASE_DECODE)
    
    def _build_phase_breakdown(
        self,
        batch_size: int,
        seq_len: int,
        total_time: float,
        phase: str
    ) -> PerformanceBreakdown:
        """Build detailed breakdown for a phase."""
        
        layer_breakdowns = []
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        compute_time = 0.0
        
        for layer in self.model.layers:
            kernels = []
            
            kernel = self._get_compute_kernel_for_phase(
                layer, batch_size, seq_len, dtype, phase
            )
            if kernel:
                ktime = kernel.estimate_time(
                    layer.input_shape, layer.output_shape, dtype
                )
                kernels.append(KernelBreakdown(
                    name=layer.name + f"_{phase}",
                    kernel_type="compute",
                    time_sec=ktime,
                    memory_bytes=kernel.estimate_memory(
                        layer.input_shape, layer.output_shape, dtype
                    ),
                    flops=layer.flops,
                ))
                compute_time += ktime
            
            layer_breakdowns.append(LayerBreakdown(
                name=layer.name,
                kernels=kernels
            ))
        
        comm_time, comm_breakdown = self._estimate_communication_time_for_phase(phase)
        
        # Calculate throughput
        tokens = batch_size * seq_len
        throughput = tokens / total_time if total_time > 0 else 0
        
        return PerformanceBreakdown(
            total_time_sec=total_time,
            throughput=throughput,
            compute_time_sec=compute_time,
            communication_time_sec=comm_time,
            memory_time_sec=0.0,
            layers=layer_breakdowns,
            peak_memory_bytes=0,  # Will be filled by caller
            activation_memory_bytes=self.model.activation_memory * batch_size,
            parameter_memory_bytes=self.model.total_params * dtype_size,
            comm_breakdown=comm_breakdown,
        )

"""Training performance analyzer."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..models.base import BaseModel, SubmoduleType
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig, SPType
from ..kernels.compute import ComputeKernelRegistry
from ..kernels.communication import CommKernelRegistry
from ..utils.constants import DTYPE_SIZES
from .breakdown import PerformanceBreakdown, LayerBreakdown, KernelBreakdown
from .detailed_breakdown import DetailedPerformanceResult
from .breakdown_generator import BreakdownGenerator
from .result_base import BaseResult


@dataclass
class TrainingResult(BaseResult):
    """Result of training performance analysis."""

    # Throughput metrics (global)
    samples_per_sec: float
    tokens_per_sec: float

    # Time metrics (required fields must come before optional)
    time_per_step_sec: float
    time_to_solution_sec: float  # For given dataset size

    # Memory metrics
    memory_per_gpu_gb: float

    # Per-GPU throughput metrics (optional)
    samples_per_sec_per_gpu: float = 0.0
    tokens_per_sec_per_gpu: float = 0.0

    # Detailed breakdown
    breakdown: PerformanceBreakdown = None

    # Detailed performance breakdown with memory/communication breakdown
    detailed_breakdown: Optional[DetailedPerformanceResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "throughput": {
                "samples_per_sec": self.samples_per_sec,
                "tokens_per_sec": self.tokens_per_sec,
                "samples_per_sec_per_gpu": self.samples_per_sec_per_gpu,
                "tokens_per_sec_per_gpu": self.tokens_per_sec_per_gpu,
            },
            "time": {
                "time_per_step_sec": self.time_per_step_sec,
                "time_per_step_ms": self.time_per_step_sec * 1000,
                "time_to_solution_sec": self.time_to_solution_sec,
            },
            "memory": {
                "memory_per_gpu_gb": self.memory_per_gpu_gb,
            },
            "breakdown": self.breakdown.to_dict() if self.breakdown else None,
        }
        if self.detailed_breakdown is not None:
            result["detailed_breakdown"] = self.detailed_breakdown.to_dict()
        return result


class TrainingAnalyzer:
    """Analyzes training performance."""

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

    # ============================================================
    # Helper methods for getting effective parallelism-sharded sizes
    # ============================================================

    def _get_effective_seq_len(self, seq_len: int) -> int:
        """Get effective sequence length after SP sharding.

        Sequence Parallelism (SP) shards the sequence dimension across GPUs,
        so each GPU only processes seq_len / sp_degree tokens.
        """
        if self.strategy.sp_degree > 1:
            return seq_len // self.strategy.sp_degree
        return seq_len

    def _get_effective_num_heads(self) -> int:
        """Get effective number of attention heads after TP sharding.

        Tensor Parallelism (TP) shards attention heads across GPUs,
        so each GPU handles num_heads / tp_degree heads.
        """
        num_heads = self.model.config.num_attention_heads
        if self.strategy.tp_degree > 1:
            return num_heads // self.strategy.tp_degree
        return num_heads

    def _get_effective_num_kv_heads(self) -> int:
        """Get effective number of KV heads after TP sharding.

        For GQA models, KV heads are also sharded by TP.
        """
        num_kv_heads = getattr(
            self.model.config, "num_key_value_heads",
            self.model.config.num_attention_heads
        )
        if self.strategy.tp_degree > 1:
            return max(1, num_kv_heads // self.strategy.tp_degree)
        return num_kv_heads

    def _get_effective_intermediate_size(self) -> int:
        """Get effective intermediate size after TP sharding.

        Tensor Parallelism (TP) shards the FFN intermediate dimension,
        so each GPU handles intermediate_size / tp_degree.
        """
        intermediate_size = self.model.config.intermediate_size
        if intermediate_size == 0:
            # Default: 4x hidden size
            intermediate_size = self.model.config.hidden_size * 4
        if self.strategy.tp_degree > 1:
            return intermediate_size // self.strategy.tp_degree
        return intermediate_size

    def _get_effective_hidden_size(self) -> int:
        """Get effective hidden size after TP sharding for column-parallel layers.

        For column-parallel layers (QKV output, first FFN), the output dimension
        is sharded, so effective hidden size per GPU is hidden_size / tp_degree.
        """
        hidden_size = self.model.config.hidden_size
        if self.strategy.tp_degree > 1:
            return hidden_size // self.strategy.tp_degree
        return hidden_size

    def _get_effective_num_layers(self) -> int:
        """Get effective number of layers after PP sharding.

        Pipeline Parallelism (PP) distributes layers across pipeline stages,
        so each GPU handles num_layers / pp_degree layers.
        """
        num_layers = self.model.config.num_layers
        if self.strategy.pp_degree > 1:
            return num_layers // self.strategy.pp_degree
        return num_layers

    def _get_total_gpus(self) -> int:
        """Get total number of GPUs used in training."""
        return (
            self.strategy.tp_degree *
            self.strategy.pp_degree *
            self.strategy.dp_degree *
            self.strategy.sp_degree *
            self.strategy.ep_degree
        )
    
    def analyze(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int = 1000,
    ) -> TrainingResult:
        """
        Analyze training performance.

        Args:
            batch_size: Global batch size
            seq_len: Sequence length
            num_steps: Number of training steps (for time-to-solution)

        Returns:
            TrainingResult with performance metrics
        """
        # Calculate per-device batch size (DP sharding)
        local_batch_size = batch_size // self.strategy.dp_degree

        # Get effective dimensions after parallelism sharding
        effective_seq_len = self._get_effective_seq_len(seq_len)
        effective_num_layers = self._get_effective_num_layers()

        # Estimate compute time for one step (per-GPU compute time)
        compute_time = self._estimate_compute_time(local_batch_size, seq_len)

        # Estimate communication time
        comm_time, comm_breakdown = self._estimate_communication_time(seq_len)

        # Memory estimation (per-GPU memory)
        memory_bytes = self._estimate_memory(local_batch_size, seq_len)

        # Total time per step (with overlap assumption)
        # Assume 70% overlap between compute and communication
        overlap_factor = 0.7
        effective_comm_time = comm_time * (1 - overlap_factor)
        time_per_step = compute_time + effective_comm_time

        # Throughput calculations (global throughput)
        samples_per_sec = batch_size / time_per_step
        tokens_per_sec = samples_per_sec * seq_len

        # Per-GPU throughput
        total_gpus = self._get_total_gpus()
        samples_per_sec_per_gpu = samples_per_sec / total_gpus
        tokens_per_sec_per_gpu = tokens_per_sec / total_gpus

        # Build breakdown
        breakdown = self._build_breakdown(
            compute_time, comm_time, comm_breakdown, memory_bytes
        )

        # Generate detailed breakdown
        detailed_breakdown = self._generate_detailed_breakdown(
            compute_time, comm_time, memory_bytes, batch_size
        )

        return TrainingResult(
            samples_per_sec=samples_per_sec,
            tokens_per_sec=tokens_per_sec,
            time_per_step_sec=time_per_step,
            time_to_solution_sec=time_per_step * num_steps,
            memory_per_gpu_gb=memory_bytes / 1024 / 1024 / 1024,
            breakdown=breakdown,
            detailed_breakdown=detailed_breakdown,
            samples_per_sec_per_gpu=samples_per_sec_per_gpu,
            tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
        )
    
    def _generate_detailed_breakdown(
        self, compute_time: float, comm_time: float, memory_bytes: int, batch_size: int = 1
    ) -> DetailedPerformanceResult:
        """Generate detailed performance breakdown."""
        generator = BreakdownGenerator(
            self.model, self.device, self.cluster, self.strategy
        )
        
        submodel = generator.generate_submodel_breakdown(
            model_name=self.model.config.name or "model",
            model_type=getattr(self.model.config, 'model_type', None) or "transformer",
            compute_time_sec=compute_time,
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
        
        return DetailedPerformanceResult(
            total_time_sec=compute_time + comm_time,
            throughput=batch_size / (compute_time + comm_time) if (compute_time + comm_time) > 0 else 0,
            submodels=[submodel],
            memory=memory_breakdown,
            communication=comm_breakdown,
        )
    
    def _estimate_compute_time(self, batch_size: int, seq_len: int) -> float:
        """Estimate compute time for one training step.

        This method correctly accounts for parallelism sharding:
        - TP: shards attention heads and FFN intermediate size
        - PP: distributes layers across stages
        - SP: shards sequence length

        The compute time returned is per-GPU compute time (forward + backward).
        """
        dtype = self.model.config.dtype

        # Get effective dimensions after parallelism sharding
        effective_seq_len = self._get_effective_seq_len(seq_len)
        effective_num_heads = self._get_effective_num_heads()
        effective_intermediate_size = self._get_effective_intermediate_size()
        effective_num_layers = self._get_effective_num_layers()
        effective_hidden_size = self._get_effective_hidden_size()

        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // self.model.config.num_attention_heads

        forward_time = 0.0
        backward_time = 0.0

        # Count layers by type to compute compute time correctly
        # We process only effective_num_layers (PP-sharded)
        attention_layers = 0
        ffn_layers = 0

        for layer in self.model.layers[:effective_num_layers]:
            layer_forward_time = 0.0
            layer_backward_time = 0.0

            # Forward pass compute time
            # For attention layers: QKV projection + attention + O projection
            if layer.submodule_type == SubmoduleType.ATTENTION or "attention" in layer.name:
                attention_layers += 1

                # QKV projection: batch * seq * hidden -> batch * seq * (3 * hidden)
                # With TP, output is sharded: batch * seq * (3 * effective_hidden_size)
                # m = batch_size * effective_seq_len (SP-sharded)
                # n = 3 * effective_hidden_size (TP-sharded)
                # k = hidden_size (input dimension, not sharded)
                m = batch_size * effective_seq_len
                n = 3 * effective_hidden_size  # Q, K, V each have effective_hidden_size
                k = hidden_size
                qkv_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
                layer_forward_time += qkv_kernel.estimate_time((m, k), (m, n), dtype)
                # Backward for linear/matmul: 2x forward FLOPs
                layer_backward_time += qkv_kernel.estimate_backward_time((m, k), (m, n), dtype)

                # Attention computation: batch * heads * seq * seq * head_dim
                # With TP and SP, effective_heads and effective_seq_len are used
                # Flash attention handles this efficiently
                attn_kernel = self.compute_registry.get(
                    f"flash_attn_{batch_size}_{effective_seq_len}_{effective_num_heads}_{head_dim}_{dtype}"
                )
                if attn_kernel:
                    layer_forward_time += attn_kernel.estimate_time(
                        (batch_size, effective_seq_len, effective_num_heads * head_dim),
                        (batch_size, effective_seq_len, hidden_size),
                        dtype
                    )
                    # Backward for attention: 2x forward FLOPs
                    layer_backward_time += attn_kernel.estimate_backward_time(
                        (batch_size, effective_seq_len, effective_num_heads * head_dim),
                        (batch_size, effective_seq_len, hidden_size),
                        dtype
                    )
                else:
                    # Fallback: manual attention FLOPs estimate
                    # Attention FLOPs = 2 * batch * seq * heads * seq * head_dim
                    # With SP: seq -> effective_seq_len, with TP: heads -> effective_num_heads
                    attention_flops = 2 * batch_size * effective_seq_len * effective_num_heads * effective_seq_len * head_dim
                    # Assume device can achieve 50% of peak FLOPs for attention
                    effective_tflops = self.device.get_compute_tflops(dtype) * 0.5
                    layer_forward_time += attention_flops / (effective_tflops * 1e12)
                    # Backward: 2x forward
                    layer_backward_time += attention_flops * 2 / (effective_tflops * 1e12)

                # O projection: batch * seq * effective_hidden -> batch * seq * hidden
                # With TP, input is sharded (effective_hidden_size), output is full (hidden_size)
                # This requires all-reduce after, but compute is local
                m = batch_size * effective_seq_len
                n = hidden_size  # Output dimension (not sharded)
                k = effective_hidden_size  # Input dimension (TP-sharded)
                o_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
                layer_forward_time += o_kernel.estimate_time((m, k), (m, n), dtype)
                layer_backward_time += o_kernel.estimate_backward_time((m, k), (m, n), dtype)

            # For FFN layers: up/gate projection + activation + down projection
            elif layer.submodule_type == SubmoduleType.FFN or "ffn" in layer.name or "proj" in layer.name:
                ffn_layers += 1

                # Up/Gate projection (SwiGLU has two): hidden -> intermediate
                # With TP, intermediate_size is sharded -> effective_intermediate_size
                m = batch_size * effective_seq_len
                n = effective_intermediate_size  # TP-sharded
                k = hidden_size
                up_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
                # SwiGLU has 2 projections (up and gate)
                layer_forward_time += up_kernel.estimate_time((m, k), (m, n), dtype) * 2
                # Backward for each matmul
                layer_backward_time += up_kernel.estimate_backward_time((m, k), (m, n), dtype) * 2

                # Activation (SwiGLU): element-wise
                # Forward: memory bandwidth bound
                activation_bytes = batch_size * effective_seq_len * effective_intermediate_size * DTYPE_SIZES.get(dtype, 2)
                layer_forward_time += activation_bytes / (self.device.get_memory_bw_gbps() * 1e9)
                # Backward for activation: ~2x forward (need to recompute)
                layer_backward_time += activation_bytes * 1.5 / (self.device.get_memory_bw_gbps() * 1e9)

                # Down projection: intermediate -> hidden
                # With TP, input is sharded (effective_intermediate_size), output is full (hidden_size)
                m = batch_size * effective_seq_len
                n = hidden_size
                k = effective_intermediate_size
                down_kernel = self.compute_registry.get_or_create_matmul(m, n, k, dtype)
                layer_forward_time += down_kernel.estimate_time((m, k), (m, n), dtype)
                layer_backward_time += down_kernel.estimate_backward_time((m, k), (m, n), dtype)

            forward_time += layer_forward_time
            backward_time += layer_backward_time

        # Total compute time = forward + backward
        total_time = forward_time + backward_time

        # Optimizer step (for DP only, or ZeRO-1+)
        # Parameter memory per GPU is affected by TP/PP sharding and ZeRO
        if self.strategy.dp_degree > 1 or self.strategy.zero_stage > 0:
            # Effective parameters per GPU
            effective_params = self.model.total_params // self.strategy.tp_degree
            if self.strategy.pp_degree > 1:
                effective_params = effective_params // self.strategy.pp_degree

            # ZeRO stage affects optimizer state sharding
            if self.strategy.zero_stage >= 1:
                # ZeRO-1: optimizer states sharded by DP
                effective_params = effective_params // self.strategy.dp_degree

            # Rough estimate: 3x parameter size operations (read grad, update, write)
            opt_ops = effective_params * 3
            opt_time = opt_ops / (self.device.get_memory_bw_gbps() * 1e9)
            total_time += opt_time

        return total_time
    
    def _estimate_communication_time(self, seq_len: int) -> tuple:
        """Estimate communication time for one step using topology-aware bandwidth.

        Args:
            seq_len: Actual sequence length used in training

        Returns:
            Tuple of (communication_time, breakdown_dict)
        """
        comm_time = 0.0
        breakdown = {}

        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        hidden_size = self.model.config.hidden_size
        effective_hidden_size = self._get_effective_hidden_size()
        effective_num_layers = self._get_effective_num_layers()
        micro_batch = self.strategy.micro_batch_size

        # Get communication domain mapping
        comm_mapping = self.strategy.get_communication_domain_mapping(
            devices_per_node=self.cluster.devices_per_node,
            total_devices=self.cluster.num_devices,
        )

        # TP communication
        if self.strategy.tp_degree > 1:
            # TP all-reduce communication volume
            # Each GPU computes with sharded hidden_size, then all-reduce to get full output
            # Communication volume per layer: batch * seq * hidden (full hidden, not sharded)
            # This is because all-reduce aggregates the sharded results to full hidden
            activation_bytes = micro_batch * seq_len * hidden_size * dtype_size

            # TP AllReduce communication:
            # Forward: 2 all-reduces per transformer layer (after QKV/O and after FFN)
            # Backward: 2 all-reduces per transformer layer (gradients for QKV/O and FFN)
            # Total: 4 all-reduces per layer
            tp_bytes = activation_bytes * 4 * effective_num_layers

            # Use communication domain mapping for topology-aware ranks
            tp_info = comm_mapping.get("tp", {})
            tp_ranks = tp_info.get("ranks", list(range(self.strategy.tp_degree)))

            tp_time = self.cluster.estimate_allreduce_time(tp_bytes, tp_ranks)

            comm_time += tp_time
            breakdown["tensor_parallel"] = tp_time

        # PP communication
        if self.strategy.pp_degree > 1:
            # Activation size between stages
            # PP sends activations from one stage to the next
            # The activation size is: batch * seq * hidden (full hidden, not TP-sharded)
            # But if TP is also used, each PP stage only has TP-sharded activation
            pp_bytes = micro_batch * seq_len * hidden_size * dtype_size

            # For each micro-batch, send to next stage
            # Number of micro-batches in a pipeline schedule
            num_micro_batches = self.strategy.dp_degree  # Simplified: assume 1 micro-batch per DP replica

            # Use communication domain mapping for topology-aware bandwidth
            pp_info = comm_mapping.get("pp", {})
            topology_level = pp_info.get("topology_level", "inter_node")
            bandwidth_domain = pp_info.get("bandwidth_domain", "inter_rack")
            devices_per_group = pp_info.get("devices_per_group", self.cluster.num_devices)

            # Use topology level for bandwidth lookup
            pp_bw = self.cluster.get_bandwidth_for_topology_level(
                bandwidth_domain, devices_per_group
            )

            # P2P communication time
            # Forward: send activations to next stage
            # Backward: send gradients to previous stage
            # Total: 2 transfers per micro-batch (forward + backward)
            pp_time = pp_bytes / (pp_bw * 1e9)
            pp_time *= 2 * num_micro_batches

            comm_time += pp_time
            breakdown["pipeline_parallel"] = pp_time

        # DP communication
        if self.strategy.dp_degree > 1:
            # Gradient all-reduce
            # Gradient size is model parameters, already TP/PP sharded per GPU
            # For DP all-reduce, we need to communicate the per-GPU gradients
            # which are already sharded by TP/PP
            effective_params = self.model.total_params // self.strategy.tp_degree
            if self.strategy.pp_degree > 1:
                effective_params = effective_params // self.strategy.pp_degree
            grad_bytes = effective_params * dtype_size

            # ZeRO reduces communication
            zero_factor = {0: 1.0, 1: 1.0, 2: 1.0 / self.strategy.dp_degree, 3: 0}
            grad_bytes *= zero_factor.get(self.strategy.zero_stage, 1.0)

            # Use communication domain mapping for topology-aware ranks
            dp_info = comm_mapping.get("dp", {})
            dp_ranks = dp_info.get("ranks", list(range(self.strategy.dp_degree)))

            dp_time = self.cluster.estimate_allreduce_time(grad_bytes, dp_ranks)

            comm_time += dp_time
            breakdown["data_parallel"] = dp_time

        # EP communication (if MoE)
        if self.strategy.ep_degree > 1:
            # All-to-all for MoE
            # Dispatch: send tokens to their assigned experts
            # Combine: gather results from experts
            # Token size: batch * seq * hidden (but each GPU only has TP-sharded hidden)
            token_bytes = micro_batch * seq_len * effective_hidden_size * dtype_size

            # Use communication domain mapping for topology-aware ranks
            ep_info = comm_mapping.get("ep", {})
            ep_ranks = ep_info.get("ranks", list(range(self.strategy.ep_degree)))

            # Dispatch and combine
            # Forward: dispatch tokens to experts + combine results
            # Backward: dispatch gradients + combine gradients
            # Total: 2 * (dispatch + combine) = 4 all-to-alls
            dispatch_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
            combine_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)

            ep_time = (dispatch_time + combine_time) * 2
            comm_time += ep_time
            breakdown["expert_parallel"] = ep_time

        # SP communication (Sequence Parallelism)
        if self.strategy.sp_degree > 1:
            sp_time = self._estimate_sp_communication_time(
                micro_batch, seq_len, hidden_size, dtype_size, comm_mapping
            )
            comm_time += sp_time
            breakdown["sequence_parallel"] = sp_time

        return comm_time, breakdown
    
    def _estimate_sp_communication_time(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        dtype_size: int,
        comm_mapping: dict = None,
    ) -> float:
        """
        Estimate sequence parallelism communication time.

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

        # Activation bytes for all-to-all (Ulysses)
        activation_bytes = batch_size * seq_len * hidden_size * dtype_size

        # KV bytes per step for Ring
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
            # Ulysses-SP: All-to-all for sequence parallelism
            # Forward: 4 all-to-all per attention layer (Q, K, V pre-attention + O post-attention)
            # Backward: 4 all-to-all per attention layer (gradients flow in reverse)
            # Total: 8 all-to-all per layer
            # Communication volume is constant w.r.t. sp_degree
            alltoall_time = self.cluster.estimate_alltoall_time(
                activation_bytes, sp_ranks
            )
            total_time = alltoall_time * 8 * num_layers

        elif sp_type == SPType.RING_P2P:
            # Ring-P2P: (sp_degree - 1) P2P steps per layer
            # Each step moves kv_bytes_per_step
            # Forward: (sp_degree - 1) steps for KV transmission
            # Backward: (sp_degree - 1) steps for gradient transmission
            # Total: 2 * (sp_degree - 1) steps per layer
            if sp_degree > 1:
                # Use topology-aware bandwidth
                sp_bw = self.cluster.get_bandwidth_for_topology_level(
                    sp_topology_level, sp_devices_per_group
                )
                step_time = kv_bytes_per_step / (sp_bw * 1e9)
                total_time = step_time * (sp_degree - 1) * 2 * num_layers

        elif sp_type == SPType.RING_ALLGATHER:
            # Ring-AllGather: AllGather for KV aggregation
            # Forward: 1 or 2 allgather depending on kv_separate_allgather config
            #   - kv_separate_allgather=False: K+V一起传输，1个 AllGather
            #   - kv_separate_allgather=True: K、V分开传输，2个 AllGather
            # Backward: ReduceScatter (AllGather 的逆操作)
            # Total per layer:
            #   - kv_separate_allgather=False: 1 AG (forward) + 1 RS (backward) = 2 ops
            #   - kv_separate_allgather=True: 2 AG (forward) + 2 RS (backward) = 4 ops

            # 每个 KV block 的字节数（分片后的）
            kv_bytes_per_block = kv_bytes_per_step

            # 根据 kv_separate_allgather 配置确定前向 AllGather 数量
            num_forward_ag = 2 if self.strategy.kv_separate_allgather else 1
            num_backward_rs = num_forward_ag  # 反向使用 ReduceScatter，数量与前向 AllGather 相同

            # AllGather 通信量: kv_bytes_per_block * sp_degree（收集所有 ranks 的 KV）
            allgather_bytes = kv_bytes_per_block * sp_degree
            allgather_time = self.cluster.estimate_allgather_time(
                allgather_bytes, sp_ranks
            )

            # ReduceScatter 通信量: kv_bytes_per_block * sp_degree（与 AllGather 相同）
            # ReduceScatter 时间约等于 AllReduce / 2，与 AllGather 时间接近
            reducescatter_time = self.cluster.estimate_reducescatter_time(
                allgather_bytes, sp_ranks
            )

            # 总通信时间
            total_time = (
                allgather_time * num_forward_ag +
                reducescatter_time * num_backward_rs
            ) * num_layers

        elif sp_type == SPType.UNIFIED_2D:
            # 2D-SP: ulysses + ring combined
            ulysses_degree = self.strategy.ulysses_degree or 1
            ring_degree = self.strategy.ring_degree or 1
            if ulysses_degree * ring_degree != sp_degree:
                # Auto-partition: prefer larger ulysses if possible
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

            # Ulysses part: 8 all-to-all per layer (4 forward + 4 backward)
            ulysses_time = self.cluster.estimate_alltoall_time(
                activation_bytes, ulysses_ranks
            ) * 8 * num_layers

            # Ring part
            ring_kv_bytes = (
                batch_size * (seq_len // sp_degree) *
                num_kv_heads * head_dim * 2 * dtype_size
            )
            if ring_degree > 1:
                # Use topology-aware bandwidth for ring
                # Forward: (ring_degree - 1) steps
                # Backward: (ring_degree - 1) steps
                # Total: 2 * (ring_degree - 1) steps per layer
                ring_bw = self.cluster.get_bandwidth_for_topology_level(
                    ring_level, ring_dpg
                )
                ring_step_time = ring_kv_bytes / (ring_bw * 1e9)
                ring_time = ring_step_time * (ring_degree - 1) * 2 * num_layers
            else:
                ring_time = 0.0

            total_time = ulysses_time + ring_time

        elif sp_type == SPType.MEGATRON:
            # Megatron-SP: ReduceScatter + AllGather per layer
            # Communication volume: 2 * activation_bytes per layer (same as TP AllReduce)
            # But memory is sharded by sp_degree
            rs_time = self.cluster.estimate_reducescatter_time(
                activation_bytes, sp_ranks
            )
            ag_time = self.cluster.estimate_allgather_time(
                activation_bytes, sp_ranks
            )
            # Forward: 2 ops (1 rs + 1 ag)
            # Backward: 2 ops (1 rs + 1 ag, reverse direction)
            # Total: 4 ops per layer (2 rs + 2 ag)
            total_time = (rs_time + ag_time) * 4 * num_layers

        return total_time
    
    def _estimate_memory(self, batch_size: int, seq_len: int) -> int:
        """Estimate memory per GPU.

        This method correctly accounts for parallelism sharding:
        - TP: shards QKV/FFN weights and activations
        - PP: distributes layers, reduces per-GPU parameters and activations
        - SP: shards sequence dimension, reduces activation memory
        - EP: shards expert parameters (for MoE)
        - ZeRO: shards optimizer states and/or gradients

        Returns:
            Estimated memory per GPU in bytes
        """
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        # Get effective dimensions for memory calculation
        effective_seq_len = self._get_effective_seq_len(seq_len)
        effective_num_layers = self._get_effective_num_layers()

        # ============================================================
        # 1. Parameter memory (model weights)
        # ============================================================
        # Total model parameters
        total_params = self.model.total_params

        # TP sharding: QKV and FFN weights are sharded across TP ranks
        # For non-MoE models, all parameters except embeddings are TP-sharded
        # For simplicity, assume all parameters are TP-sharded
        # (Embeddings typically not TP-sharded, but this is a simplification)
        param_memory = total_params * dtype_size

        # Divide by TP degree (parameters are sharded by TP)
        if self.strategy.tp_degree > 1:
            param_memory = param_memory // self.strategy.tp_degree

        # PP sharding: parameters are distributed across pipeline stages
        if self.strategy.pp_degree > 1:
            param_memory = param_memory // self.strategy.pp_degree

        # EP sharding (for MoE): expert parameters are additionally sharded by EP
        # Note: EP and TP typically share the same sharding for experts
        # This is a simplification - actual EP sharding depends on MoE implementation
        if self.strategy.ep_degree > 1 and hasattr(self.model.config, 'num_experts'):
            # Expert parameters are sharded by EP
            # For simplicity, assume expert parameters are already accounted in TP sharding
            # and EP provides additional sharding for expert-specific layers
            # This is conservative (may overestimate)
            pass  # Keep as is, as TP sharding already handles this

        # ============================================================
        # 2. Gradient memory
        # ============================================================
        # Gradients have the same size as parameters per GPU
        grad_memory = param_memory

        # ZeRO stage 2: gradients are sharded by DP
        if self.strategy.zero_stage >= 2:
            grad_memory = grad_memory // self.strategy.dp_degree

        # ============================================================
        # 3. Optimizer state memory (Adam: momentum + variance in fp32)
        # ============================================================
        # Optimizer states: 2x parameter count (momentum + variance) in fp32
        optimizer_memory = param_memory * 2 * 4  # fp32 = 4 bytes

        # ZeRO stage 1: optimizer states are sharded by DP
        if self.strategy.zero_stage >= 1:
            optimizer_memory = optimizer_memory // self.strategy.dp_degree

        # ZeRO stage 3: parameters are also sharded by DP during training
        # This affects optimizer state sharding (already sharded in stage 1)
        # Stage 3 further reduces optimizer memory

        # ============================================================
        # 4. Activation memory (for backward pass)
        # ============================================================
        # Activations are saved for backward pass
        # Size depends on batch_size, seq_len, hidden_size, and number of layers

        # Base activation memory (without parallelism sharding)
        # Use model's estimate_memory for base activation calculation
        is_distributed = (
            self.strategy.tp_degree > 1
            or self.strategy.dp_degree > 1
            or self.strategy.pp_degree > 1
            or self.strategy.sp_degree > 1
        )

        # Get base memory and extract activation component
        base_memory = self.model.estimate_memory(
            inference_mode=False,  # Training mode (all activations saved)
            batch_size=batch_size,
            is_distributed=is_distributed,
            apply_calibration=False,  # We'll handle scaling manually
        )

        # Extract activation memory from base memory
        # base_memory = param_memory_total + activation_memory
        param_memory_total = self.model.total_params * dtype_size
        activation_memory = base_memory - param_memory_total

        # Scale by sequence length ratio
        seq_ratio = seq_len // max(self.model.config.max_seq_len, 1)
        if seq_ratio > 0:
            activation_memory *= seq_ratio

        # TP sharding: QKV and FFN activations are sharded
        # Each GPU only stores effective_hidden_size activations for QKV/FFN layers
        # This applies to attention outputs and FFN intermediate activations
        if self.strategy.tp_degree > 1:
            activation_memory = activation_memory // self.strategy.tp_degree

        # SP sharding: sequence dimension is sharded
        # Each GPU only stores seq_len / sp_len tokens
        if self.strategy.sp_degree > 1:
            activation_memory = activation_memory // self.strategy.sp_degree

        # PP sharding: only activations for current stage are stored
        # Each GPU only handles effective_num_layers layers
        if self.strategy.pp_degree > 1:
            activation_memory = activation_memory // self.strategy.pp_degree

        # ============================================================
        # 5. KV Cache memory (during forward pass for attention)
        # ============================================================
        # KV cache is needed for training (for gradient computation)
        # Size: batch_size * seq_len * num_kv_heads * head_dim * 2 (K and V)
        num_kv_heads = getattr(
            self.model.config, "num_key_value_heads",
            self.model.config.num_attention_heads
        )
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads

        # KV cache per GPU
        kv_cache_memory = batch_size * effective_seq_len * num_kv_heads * head_dim * 2 * dtype_size

        # TP sharding for KV cache
        if self.strategy.tp_degree > 1:
            # KV heads are sharded by TP (for GQA, this may be different)
            effective_kv_heads = self._get_effective_num_kv_heads()
            kv_cache_memory = batch_size * effective_seq_len * effective_kv_heads * head_dim * 2 * dtype_size

        # ============================================================
        # 6. Activation checkpointing: reduces activation memory
        # ============================================================
        if self.strategy.activation_checkpointing:
            # With activation checkpointing, only one layer's activation needs to be stored
            # at a time (other layers are recomputed during backward)
            # Find max single layer activation
            max_layer_activation = max(
                layer.activation_bytes for layer in self.model.layers[:effective_num_layers]
            ) if self.model.layers else 0

            # Scale by batch and effective sequence length
            max_layer_activation *= batch_size * effective_seq_len // max(self.model.config.max_seq_len, 1)

            # Apply TP and PP sharding
            if self.strategy.tp_degree > 1:
                max_layer_activation = max_layer_activation // self.strategy.tp_degree

            activation_memory = max_layer_activation

        # ============================================================
        # 7. Total memory per GPU
        # ============================================================
        total_memory = (
            param_memory +
            activation_memory +
            grad_memory +
            optimizer_memory +
            kv_cache_memory
        )

        # Apply calibration factors from model config
        calib = self.model.config.memory_calibration
        total_memory = calib.apply(total_memory, is_distributed)

        return total_memory
    
    def _get_compute_kernel(
        self,
        layer,
        batch_size: int,
        seq_len: int,
        dtype: str
    ):
        """Get appropriate compute kernel for a layer."""
        name = layer.name
        
        # Try to match by name pattern
        if "proj" in name or "up" in name or "gate" in name or "down" in name:
            # Matmul
            m = batch_size * seq_len
            n = layer.output_shape[-1]
            k = layer.input_shape[-1]
            return self.compute_registry.get_or_create_matmul(m, n, k, dtype)
        elif "attention" in name:
            # Attention - use registered kernel
            return self.compute_registry.get(f"flash_attn_{batch_size}_{seq_len}_32_128_{dtype}")
        elif "norm" in name:
            # Normalization
            return self.compute_registry.get(f"rmsnorm_{layer.input_shape[-1]}_{dtype}")
        elif "swiglu" in name or "activation" in name:
            # Activation
            return self.compute_registry.get(f"swiglu_{layer.input_shape[-1]}_{dtype}")
        
        return None
    
    def _build_breakdown(
        self,
        compute_time: float,
        comm_time: float,
        comm_breakdown: Dict[str, float],
        memory_bytes: int,
    ) -> PerformanceBreakdown:
        """Build detailed performance breakdown."""
        
        layer_breakdowns = []
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        for layer in self.model.layers:
            kernels = []
            
            # Add compute kernel
            kernel = self._get_compute_kernel(layer, 1, 4096, dtype)
            if kernel:
                kernels.append(KernelBreakdown(
                    name=layer.name + "_compute",
                    kernel_type="compute",
                    time_sec=kernel.estimate_time(
                        layer.input_shape, layer.output_shape, dtype
                    ),
                    memory_bytes=kernel.estimate_memory(
                        layer.input_shape, layer.output_shape, dtype
                    ),
                    flops=layer.flops,
                ))
            
            layer_breakdowns.append(LayerBreakdown(
                name=layer.name,
                kernels=kernels
            ))
        
        return PerformanceBreakdown(
            total_time_sec=compute_time + comm_time,
            throughput=0.0,  # Will be set by caller
            compute_time_sec=compute_time,
            communication_time_sec=comm_time,
            memory_time_sec=0.0,
            layers=layer_breakdowns,
            peak_memory_bytes=memory_bytes,
            activation_memory_bytes=self.model.activation_memory,
            parameter_memory_bytes=self.model.total_params * dtype_size,
            comm_breakdown=comm_breakdown,
        )

"""Generator for detailed performance breakdowns.

This module provides utilities to generate detailed breakdowns for models
and pipelines, including memory by type, communication by parallelism, etc.
"""

from typing import List, Dict, Any, Optional, Tuple

from ..models.base import BaseModel
from ..strategy.base import StrategyConfig
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..utils.constants import DTYPE_SIZES

from .detailed_breakdown import (
    SubModelBreakdown,
    BlockBreakdown,
    MemoryBreakdown,
    CommunicationBreakdown,
    CommunicationDetail,
    ParallelismType,
    MemoryType,
    DetailedPerformanceResult,
)


class BreakdownGenerator:
    """Generator for detailed performance breakdowns."""

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
        self.dtype_size = DTYPE_SIZES.get(model.config.dtype, 2)

    def generate_submodel_breakdown(
        self,
        model_name: str,
        model_type: str,
        compute_time_sec: float,
        num_iterations: int = 1,
    ) -> SubModelBreakdown:
        """Generate detailed breakdown for a sub-model."""
        submodel = SubModelBreakdown(
            model_name=model_name,
            model_type=model_type,
            total_time_sec=compute_time_sec * num_iterations,
            num_iterations=num_iterations,
            total_flops=self.model.total_flops_forward * num_iterations,
            compute_time_sec=compute_time_sec * num_iterations,
        )

        # Generate memory breakdown by type
        submodel.memory_by_type = self._estimate_memory_by_type()

        # Generate block-level breakdown
        submodel.blocks = self._generate_block_breakdowns()

        # Generate communication breakdown
        submodel.comm_by_parallelism = self._generate_communication_breakdown()

        return submodel

    def _estimate_memory_by_type(self) -> Dict[MemoryType, int]:
        """Estimate memory usage by type."""
        memory_by_type: Dict[MemoryType, int] = {}

        # Parameters
        param_memory = self.model.total_params * self.dtype_size
        # Apply TP sharding
        param_memory //= max(self.strategy.tp_degree, 1)
        memory_by_type[MemoryType.PARAMETER] = param_memory

        # Activations (use model's estimate)
        activation_memory = self._estimate_activation_memory()
        memory_by_type[MemoryType.ACTIVATION] = activation_memory

        # Gradients (same as parameters for training)
        if self._is_training():
            grad_memory = param_memory
            # Apply ZeRO sharding if applicable
            if self.strategy.zero_stage >= 2:
                grad_memory //= max(self.strategy.dp_degree, 1)
            memory_by_type[MemoryType.GRADIENT] = grad_memory

            # Optimizer states (Adam: 2x parameters in fp32)
            optimizer_memory = param_memory * 2 * 4 // self.dtype_size  # fp32
            if self.strategy.zero_stage >= 3:
                optimizer_memory //= max(self.strategy.dp_degree, 1)
            memory_by_type[MemoryType.OPTIMIZER] = optimizer_memory

        # KV Cache (for inference with LLMs)
        kv_cache = self._estimate_kv_cache_memory()
        if kv_cache > 0:
            memory_by_type[MemoryType.KV_CACHE] = kv_cache

        # Communication buffers (for distributed)
        if self._is_distributed():
            comm_buffer = self._estimate_comm_buffer_memory()
            memory_by_type[MemoryType.COMM_BUFFER] = comm_buffer

        return memory_by_type

    def _estimate_activation_memory(self) -> int:
        """Estimate activation memory."""
        # Use inference mode for activation (layer-wise reuse)
        # This gives the max single-layer activation
        if not self.model.layers:
            return 0

        max_activation = max(
            layer.activation_bytes for layer in self.model.layers
        )

        # Apply SP reduction
        if self.strategy.sp_degree > 1:
            max_activation //= self.strategy.sp_degree

        return max_activation

    def _estimate_kv_cache_memory(self) -> int:
        """Estimate KV cache memory for transformer models."""
        cfg = self.model.config

        # Check if this is a transformer model with KV cache
        if not hasattr(cfg, 'num_key_value_heads') or cfg.num_key_value_heads is None:
            return 0
        if cfg.num_attention_heads == 0:
            return 0

        # KV cache: 2 (K and V) * num_heads * head_dim * seq_len * num_layers * batch
        num_kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        seq_len = cfg.max_seq_len
        batch_size = 1  # Default batch

        kv_per_token = 2 * num_kv_heads * head_dim * self.dtype_size
        kv_cache = batch_size * seq_len * cfg.num_layers * kv_per_token

        # Apply TP sharding
        kv_cache //= max(self.strategy.tp_degree, 1)

        return kv_cache

    def _estimate_comm_buffer_memory(self) -> int:
        """Estimate communication buffer memory."""
        # Rough estimate: proportional to parameter size and parallelism degree
        param_memory = self.model.total_params * self.dtype_size

        # Different parallelism types need different buffer sizes
        buffer_size = 0

        if self.strategy.tp_degree > 1:
            # TP needs buffers for allreduce
            buffer_size += param_memory // self.strategy.tp_degree * 0.1

        if self.strategy.dp_degree > 1:
            # DP needs buffers for gradient allreduce
            buffer_size += param_memory // self.strategy.dp_degree * 0.2

        return int(buffer_size)

    def _generate_block_breakdowns(self) -> List[BlockBreakdown]:
        """Generate block-level breakdowns by grouping layers."""
        blocks: List[BlockBreakdown] = []

        # Group layers by type
        layer_groups = self._group_layers_by_type()

        for block_type, layer_indices in layer_groups.items():
            block = BlockBreakdown(
                block_type=block_type,
                block_name=f"{block_type}_block",
                layer_indices=layer_indices,
            )

            # Compute FLOPs
            block.flops = sum(
                self.model.layers[i].flops for i in layer_indices
                if i < len(self.model.layers)
            )

            # Memory by type for this block
            block.memory_by_type = self._estimate_block_memory(block_type, layer_indices)

            # Communication for this block
            block.comm_operations = self._estimate_block_communication(
                block_type, layer_indices
            )

            blocks.append(block)

        return blocks

    def _group_layers_by_type(self) -> Dict[str, List[int]]:
        """Group layer indices by block type."""
        groups: Dict[str, List[int]] = {}

        for idx, layer in enumerate(self.model.layers):
            layer_name = layer.name.lower()

            # Classify layer by name
            if "embed" in layer_name:
                block_type = "embedding"
            elif "attn" in layer_name or "attention" in layer_name:
                block_type = "attention"
            elif "mlp" in layer_name or "ffn" in layer_name or "proj" in layer_name:
                block_type = "ffn"
            elif "norm" in layer_name:
                block_type = "norm"
            elif "router" in layer_name or "expert" in layer_name:
                block_type = "moe"
            else:
                block_type = "other"

            if block_type not in groups:
                groups[block_type] = []
            groups[block_type].append(idx)

        return groups

    def _estimate_block_memory(
        self, block_type: str, layer_indices: List[int]
    ) -> Dict[MemoryType, int]:
        """Estimate memory for a block."""
        memory: Dict[MemoryType, int] = {}

        # Parameters
        param_memory = sum(
            self.model.layers[i].params_count * self.dtype_size
            for i in layer_indices if i < len(self.model.layers)
        )
        param_memory //= max(self.strategy.tp_degree, 1)
        memory[MemoryType.PARAMETER] = param_memory

        # Activations
        max_activation = max(
            self.model.layers[i].activation_bytes
            for i in layer_indices if i < len(self.model.layers)
        ) if layer_indices else 0
        memory[MemoryType.ACTIVATION] = max_activation

        return memory

    def _estimate_block_communication(
        self, block_type: str, layer_indices: List[int]
    ) -> List[CommunicationDetail]:
        """Estimate communication operations for a block."""
        comm_ops: List[CommunicationDetail] = []

        # TP communication for attention and FFN
        if self.strategy.tp_degree > 1 and block_type in ["attention", "ffn"]:
            # All-reduce for output projection
            param_size = sum(
                self.model.layers[i].params_count * self.dtype_size
                for i in layer_indices if i < len(self.model.layers)
            )

            comm_ops.append(CommunicationDetail(
                parallelism_type=ParallelismType.TP,
                operation="allreduce",
                volume_bytes=param_size * 2,  # Forward + backward
                time_sec=0.0,  # Will be computed later
                description=f"TP all-reduce for {block_type}",
            ))

        # SP communication for attention
        if self.strategy.sp_degree > 1 and block_type == "attention":
            seq_len = self.model.config.max_seq_len
            hidden_size = self.model.config.hidden_size
            batch_size = 1

            # All-to-all for sequence parallelism
            activation_size = batch_size * seq_len * hidden_size * self.dtype_size

            comm_ops.append(CommunicationDetail(
                parallelism_type=ParallelismType.SP,
                operation="alltoall",
                volume_bytes=activation_size * 4,  # Q, K, V, O
                time_sec=0.0,
                description="SP all-to-all for attention",
            ))

        return comm_ops

    def _generate_communication_breakdown(self) -> Dict[ParallelismType, List[CommunicationDetail]]:
        """Generate communication breakdown by parallelism type."""
        comm_by_type: Dict[ParallelismType, List[CommunicationDetail]] = {}

        # Aggregate from all blocks
        for block in self._generate_block_breakdowns():
            for op in block.comm_operations:
                if op.parallelism_type not in comm_by_type:
                    comm_by_type[op.parallelism_type] = []
                comm_by_type[op.parallelism_type].append(op)

        # Add DP communication if applicable
        if self.strategy.dp_degree > 1:
            comm_by_type[ParallelismType.DP] = self._generate_dp_communication()

        # Add PP communication if applicable
        if self.strategy.pp_degree > 1:
            comm_by_type[ParallelismType.PP] = self._generate_pp_communication()

        return comm_by_type

    def _generate_dp_communication(self) -> List[CommunicationDetail]:
        """Generate data parallelism communication."""
        if self.strategy.dp_degree <= 1:
            return []

        param_memory = self.model.total_params * self.dtype_size

        # Gradient all-reduce
        return [CommunicationDetail(
            parallelism_type=ParallelismType.DP,
            operation="allreduce",
            volume_bytes=param_memory * 2,  # Gradients
            time_sec=0.0,
            description="DP gradient all-reduce",
        )]

    def _generate_pp_communication(self) -> List[CommunicationDetail]:
        """Generate pipeline parallelism communication."""
        if self.strategy.pp_degree <= 1:
            return []

        # Activation transfer between stages
        batch_size = 1
        seq_len = self.model.config.max_seq_len
        hidden_size = self.model.config.hidden_size

        activation_size = batch_size * seq_len * hidden_size * self.dtype_size

        return [CommunicationDetail(
            parallelism_type=ParallelismType.PP,
            operation="p2p",
            volume_bytes=activation_size * 2,  # Forward + backward
            time_sec=0.0,
            description="PP activation transfer",
        )]

    def _is_training(self) -> bool:
        """Check if in training mode."""
        # This is a heuristic - in practice would be passed explicitly
        return hasattr(self.strategy, 'zero_stage') and self.strategy.zero_stage >= 0

    def _is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return (
            self.strategy.tp_degree > 1
            or self.strategy.dp_degree > 1
            or self.strategy.pp_degree > 1
            or self.strategy.ep_degree > 1
            or self.strategy.sp_degree > 1
        )


class PipelineBreakdownGenerator:
    """Generator for pipeline-level detailed breakdowns."""

    def __init__(
        self,
        submodel_generators: List[Tuple[str, str, BreakdownGenerator]],
        device: Device,
        cluster: Cluster,
        strategy: StrategyConfig,
    ):
        """
        Args:
            submodel_generators: List of (model_name, model_type, generator) tuples
        """
        self.submodel_generators = submodel_generators
        self.device = device
        self.cluster = cluster
        self.strategy = strategy

    def generate(
        self,
        total_time_sec: float,
        throughput: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DetailedPerformanceResult:
        """Generate complete pipeline breakdown."""
        result = DetailedPerformanceResult(
            total_time_sec=total_time_sec,
            throughput=throughput,
            metadata=metadata or {},
        )

        # Generate sub-model breakdowns
        for model_name, model_type, generator in self.submodel_generators:
            submodel = generator.generate_submodel_breakdown(
                model_name=model_name,
                model_type=model_type,
                compute_time_sec=0.0,  # Will be computed
            )
            result.submodels.append(submodel)

        # Aggregate memory breakdown
        result.memory = self._aggregate_memory(result.submodels)

        # Aggregate communication breakdown
        result.communication = self._aggregate_communication(result.submodels)

        return result

    def _aggregate_memory(self, submodels: List[SubModelBreakdown]) -> MemoryBreakdown:
        """Aggregate memory across all sub-models."""
        memory = MemoryBreakdown()

        for submodel in submodels:
            # By sub-model
            memory.by_submodel[submodel.model_name] = submodel.memory_by_type.copy()

            # By type (aggregate)
            for mem_type, bytes_val in submodel.memory_by_type.items():
                if mem_type not in memory.by_type:
                    memory.by_type[mem_type] = 0
                memory.by_type[mem_type] += bytes_val

            # By block type
            for block in submodel.blocks:
                if block.block_type not in memory.by_block_type:
                    memory.by_block_type[block.block_type] = {}
                block_mem = memory.by_block_type[block.block_type]
                for mem_type, bytes_val in block.memory_by_type.items():
                    if mem_type not in block_mem:
                        block_mem[mem_type] = 0
                    block_mem[mem_type] += bytes_val

        return memory

    def _aggregate_communication(
        self, submodels: List[SubModelBreakdown]
    ) -> CommunicationBreakdown:
        """Aggregate communication across all sub-models."""
        comm = CommunicationBreakdown()

        for submodel in submodels:
            # By sub-model
            submodel_ops: List[CommunicationDetail] = []
            for ops in submodel.comm_by_parallelism.values():
                submodel_ops.extend(ops)
            comm.by_submodel[submodel.model_name] = submodel_ops

            # By type (aggregate)
            for para_type, ops in submodel.comm_by_parallelism.items():
                if para_type not in comm.by_type:
                    comm.by_type[para_type] = []
                comm.by_type[para_type].extend(ops)

        return comm

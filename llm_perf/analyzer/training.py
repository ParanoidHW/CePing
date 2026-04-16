"""Training performance analyzer."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..models.base import BaseModel
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
    
    # Throughput metrics
    samples_per_sec: float
    tokens_per_sec: float
    
    # Time metrics
    time_per_step_sec: float
    time_to_solution_sec: float  # For given dataset size
    
    # Memory metrics
    memory_per_gpu_gb: float
    
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
        # Calculate per-device batch size
        local_batch_size = batch_size // self.strategy.dp_degree
        
        # Estimate compute time for one step
        compute_time = self._estimate_compute_time(local_batch_size, seq_len)
        
        # Estimate communication time
        comm_time, comm_breakdown = self._estimate_communication_time()
        
        # Memory estimation
        memory_bytes = self._estimate_memory(local_batch_size, seq_len)
        
        # Total time per step (with overlap assumption)
        # Assume 70% overlap between compute and communication
        overlap_factor = 0.7
        effective_comm_time = comm_time * (1 - overlap_factor)
        time_per_step = compute_time + effective_comm_time
        
        # Throughput calculations
        samples_per_sec = batch_size / time_per_step
        tokens_per_sec = samples_per_sec * seq_len
        
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
        
        # Build memory breakdown from submodel
        from .detailed_breakdown import MemoryBreakdown, CommunicationBreakdown
        memory_breakdown = MemoryBreakdown(
            by_type=submodel.memory_by_type,
            by_submodel={submodel.model_name: submodel.memory_by_type},
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
        """Estimate compute time for one training step."""
        dtype = self.model.config.dtype
        
        total_time = 0.0
        
        for layer in self.model.layers:
            # Forward pass
            kernel = self._get_compute_kernel(layer, batch_size, seq_len, dtype)
            if kernel:
                total_time += kernel.estimate_time(
                    layer.input_shape,
                    layer.output_shape,
                    dtype
                )
            
            # Backward pass (typically 2x forward FLOPs)
            if kernel:
                total_time += kernel.estimate_time(
                    layer.input_shape,
                    layer.output_shape,
                    dtype
                ) * 2
            
            # Optimizer step (for DP only, or ZeRO-1+)
            if self.strategy.dp_degree > 1 or self.strategy.zero_stage > 0:
                # Rough estimate: 3x parameter size operations (read grad, update, write)
                opt_ops = layer.params_count * 3
                opt_time = opt_ops / (self.device.get_memory_bw_gbps() * 1e9)
                total_time += opt_time
        
        return total_time
    
    def _estimate_communication_time(self) -> tuple:
        """Estimate communication time for one step."""
        comm_time = 0.0
        breakdown = {}
        
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        hidden_size = self.model.config.hidden_size
        seq_len = self.model.config.max_seq_len
        micro_batch = self.strategy.micro_batch_size
        
        # TP communication
        if self.strategy.tp_degree > 1:
            # Estimate based on activation size
            activation_bytes = micro_batch * seq_len * hidden_size * dtype_size
            # 2 all-reduces per layer
            num_layers = self.model.config.num_layers
            tp_bytes = activation_bytes * 2 * num_layers
            
            tp_ranks = list(range(self.strategy.tp_degree))
            tp_time = self.cluster.estimate_allreduce_time(tp_bytes, tp_ranks)
            
            comm_time += tp_time
            breakdown["tensor_parallel"] = tp_time
        
        # PP communication
        if self.strategy.pp_degree > 1:
            # Activation size between stages
            pp_bytes = micro_batch * seq_len * hidden_size * dtype_size
            # For each micro-batch, send to next stage
            num_micro_batches = 1  # Simplified
            
            # P2P is relatively fast within node
            pp_time = pp_bytes / (self.cluster.network.intra_node_bandwidth_gbps * 1e9)
            pp_time *= num_micro_batches
            
            comm_time += pp_time
            breakdown["pipeline_parallel"] = pp_time
        
        # DP communication
        if self.strategy.dp_degree > 1:
            # Gradient all-reduce
            grad_bytes = self.model.total_params * dtype_size
            
            # ZeRO reduces communication
            zero_factor = {0: 1.0, 1: 1.0, 2: 1.0 / self.strategy.dp_degree, 3: 0}
            grad_bytes *= zero_factor.get(self.strategy.zero_stage, 1.0)
            
            dp_ranks = list(range(self.strategy.dp_degree))
            dp_time = self.cluster.estimate_allreduce_time(grad_bytes, dp_ranks)
            
            comm_time += dp_time
            breakdown["data_parallel"] = dp_time
        
        # EP communication (if MoE)
        if self.strategy.ep_degree > 1:
            # All-to-all for MoE
            token_bytes = micro_batch * seq_len * hidden_size * dtype_size
            ep_ranks = list(range(self.strategy.ep_degree))
            
            # Dispatch and combine
            dispatch_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
            combine_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
            
            ep_time = dispatch_time + combine_time
            comm_time += ep_time
            breakdown["expert_parallel"] = ep_time
        
        # SP communication (Sequence Parallelism)
        if self.strategy.sp_degree > 1:
            sp_time = self._estimate_sp_communication_time(
                micro_batch, seq_len, hidden_size, dtype_size
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
    ) -> float:
        """
        Estimate sequence parallelism communication time.

        Supports ulysses-SP, ring-SP (p2p/allgather), and unified-2D-SP.
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
        
        sp_ranks = list(range(sp_degree))
        total_time = 0.0
        
        if sp_type == SPType.ULYSSES:
            # 4 all-to-all per attention layer:
            # pre-attention: Q, K, V each needs one all-to-all
            # post-attention: O needs one all-to-all
            # Ulysses communication volume is constant w.r.t. sp_degree
            alltoall_time = self.cluster.estimate_alltoall_time(
                activation_bytes, sp_ranks
            )
            total_time = alltoall_time * 4 * num_layers
            
        elif sp_type == SPType.RING_P2P:
            # (sp_degree - 1) P2P steps per layer
            # Each step moves kv_bytes_per_step
            if sp_degree > 1:
                bw_values = []
                for i in range(sp_degree):
                    for j in range(i + 1, sp_degree):
                        bw = self.cluster.get_bandwidth_between(
                            sp_ranks[i], sp_ranks[j]
                        )
                        if bw > 0:
                            bw_values.append(bw)
                avg_bw = (
                    len(bw_values) / sum(1.0 / bw for bw in bw_values)
                    if bw_values else 100.0
                )
                step_time = kv_bytes_per_step / (avg_bw * 1e9)
                total_time = step_time * (sp_degree - 1) * num_layers
                
        elif sp_type == SPType.RING_ALLGATHER:
            # 1 allgather per layer for KV aggregation
            allgather_time = self.cluster.estimate_allgather_time(
                kv_bytes_per_step * sp_degree, sp_ranks
            )
            total_time = allgather_time * num_layers
            
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

            ulysses_ranks = list(range(ulysses_degree))
            ring_ranks = list(range(ring_degree))

            # Ulysses part: 4 all-to-all (Q/K/V pre + O post)
            ulysses_time = self.cluster.estimate_alltoall_time(
                activation_bytes, ulysses_ranks
            ) * 4 * num_layers

            # Ring part
            ring_kv_bytes = (
                batch_size * (seq_len // sp_degree) *
                num_kv_heads * head_dim * 2 * dtype_size
            )
            if ring_degree > 1:
                bw_values = []
                for i in range(ring_degree):
                    for j in range(i + 1, ring_degree):
                        bw = self.cluster.get_bandwidth_between(
                            ring_ranks[i], ring_ranks[j]
                        )
                        if bw > 0:
                            bw_values.append(bw)
                avg_bw = (
                    len(bw_values) / sum(1.0 / bw for bw in bw_values)
                    if bw_values else 100.0
                )
                ring_step_time = ring_kv_bytes / (avg_bw * 1e9)
                ring_time = ring_step_time * (ring_degree - 1) * num_layers
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
            # 2 communication ops per layer (forward pass)
            total_time = (rs_time + ag_time) * 2 * num_layers

        return total_time
    
    def _estimate_memory(self, batch_size: int, seq_len: int) -> int:
        """Estimate memory per GPU."""
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        # Model parameters
        param_memory = self.model.total_params * dtype_size
        
        # Divide by TP degree (parameters are sharded)
        param_memory //= self.strategy.tp_degree
        
        # Activations - use training mode estimate with proper scaling
        # Note: estimate_memory returns calibrated memory, so we need to extract
        # the base activation component or disable calibration for manual scaling
        is_distributed = (
            self.strategy.tp_degree > 1
            or self.strategy.dp_degree > 1
            or self.strategy.pp_degree > 1
        )
        base_memory = self.model.estimate_memory(
            inference_mode=False,  # Training mode
            batch_size=batch_size,
            is_distributed=is_distributed,
            apply_calibration=False,  # We'll handle scaling manually
        )
        dtype_size = 2 if self.model.config.dtype == "fp16" else 4
        param_memory_total = self.model.total_params * dtype_size
        activation_memory = base_memory - param_memory_total

        # Scale by sequence length ratio
        seq_ratio = seq_len // max(self.model.config.max_seq_len, 1)
        if seq_ratio > 0:
            activation_memory *= seq_ratio

        # Sequence parallelism reduces activation memory
        if self.strategy.sp_degree > 1:
            activation_memory //= self.strategy.sp_degree

        # Megatron-SP: activations are sharded by sp_degree (which equals tp_degree)
        # Memory savings are already accounted for above

        # Gradients (same size as parameters, divided by DP if ZeRO-2/3)
        grad_memory = param_memory
        if self.strategy.zero_stage >= 2:
            grad_memory //= self.strategy.dp_degree

        # Optimizer states (Adam: 2x parameters for momentum and variance)
        optimizer_memory = param_memory * 2 * 4  # fp32 states
        if self.strategy.zero_stage >= 3:
            optimizer_memory //= self.strategy.dp_degree

        # Activation checkpointing: only need to store one layer's activation
        if self.strategy.activation_checkpointing:
            # Find max single layer activation
            max_layer_activation = max(
                layer.activation_bytes for layer in self.model.layers
            ) if self.model.layers else 0
            # Scale by batch and sequence
            max_layer_activation *= batch_size * seq_len // max(self.model.config.max_seq_len, 1)
            if self.strategy.sp_degree > 1:
                max_layer_activation //= self.strategy.sp_degree
            activation_memory = max_layer_activation

        total_memory = param_memory + activation_memory + grad_memory + optimizer_memory

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

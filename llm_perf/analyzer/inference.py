"""Inference performance analyzer."""

from dataclasses import dataclass
from typing import Dict, Any

from ..models.base import BaseModel
from ..hardware.device import Device
from ..hardware.cluster import Cluster
from ..strategy.base import StrategyConfig, SPType
from ..kernels.compute import ComputeKernelRegistry
from ..kernels.communication import CommKernelRegistry
from ..utils.constants import DTYPE_SIZES, PHASE_PREFILL, PHASE_DECODE
from .breakdown import PerformanceBreakdown, LayerBreakdown, KernelBreakdown


@dataclass
class InferenceResult:
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
    
    # Detailed breakdown
    prefill_breakdown: PerformanceBreakdown = None
    decode_breakdown: PerformanceBreakdown = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prefill": {
                "ttft_sec": self.prefill_time_sec,
                "ttft_ms": self.prefill_time_sec * 1000,
                "tokens_per_sec": self.prefill_tokens_per_sec,
            },
            "decode": {
                "tpot_sec": self.decode_time_per_step_sec,
                "tpot_ms": self.decode_time_per_step_sec * 1000,
                "tps": self.decode_tokens_per_sec,
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
            "prefill_breakdown": self.prefill_breakdown.to_dict() if self.prefill_breakdown else None,
            "decode_breakdown": self.decode_breakdown.to_dict() if self.decode_breakdown else None,
        }


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
        
        # End-to-end time
        total_time = prefill_time + decode_time_per_step * generation_len
        
        # Build breakdowns
        prefill_breakdown = self._build_prefill_breakdown(batch_size, prompt_len, prefill_time)
        decode_breakdown = self._build_decode_breakdown(batch_size, decode_time_per_step)
        
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
        )
    
    def _estimate_prefill_time(self, batch_size: int, seq_len: int) -> float:
        """Estimate prefill phase time (prompt processing)."""
        dtype = self.model.config.dtype
        total_time = 0.0
        
        # Prefill processes the full sequence at once
        # Attention is O(batch * seq^2 * heads * dim)
        
        for layer in self.model.layers:
            kernel = self._get_compute_kernel_for_phase(
                layer, batch_size, seq_len, dtype, PHASE_PREFILL
            )
            if kernel:
                time = kernel.estimate_time(
                    layer.input_shape,
                    layer.output_shape,
                    dtype
                )
                total_time += time
        
        # Add communication overhead
        comm_time, _ = self._estimate_communication_time_for_phase(PHASE_PREFILL)
        
        # Apply overlap factor
        overlap_factor = 0.8
        total_time = max(total_time, comm_time * (1 - overlap_factor) + max(total_time, comm_time * overlap_factor))
        
        return total_time
    
    def _estimate_decode_time(self, batch_size: int) -> float:
        """Estimate decode phase time (single token generation)."""
        dtype = self.model.config.dtype
        total_time = 0.0
        
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
        
        # KV cache read overhead (memory bandwidth bound)
        kv_read_time = self._estimate_kv_cache_read_time(batch_size)
        total_time += kv_read_time
        
        # Communication overhead
        comm_time, _ = self._estimate_communication_time_for_phase(PHASE_DECODE)
        
        # Apply overlap factor
        overlap_factor = 0.7
        total_time = max(total_time, comm_time * (1 - overlap_factor) + max(total_time, comm_time * overlap_factor))
        
        return total_time
    
    def _estimate_kv_cache_read_time(self, batch_size: int) -> float:
        """Estimate time to read KV cache during decode."""
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        # KV cache size per token per layer
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_layers
        num_kv_heads = self.model.config.num_key_value_heads or self.model.config.num_attention_heads
        head_dim = hidden_size // self.model.config.num_attention_heads
        
        # K and V per token per layer
        kv_per_token = 2 * num_kv_heads * head_dim * dtype_size
        
        # Read all previous tokens (average case)
        avg_seq_len = self.model.config.max_seq_len // 2
        total_kv_bytes = batch_size * avg_seq_len * num_layers * kv_per_token
        
        # Memory bandwidth bound
        mem_bw = self.device.get_memory_bw_gbps() * 1e9
        return total_kv_bytes / mem_bw
    
    def _estimate_communication_time_for_phase(self, phase: str) -> tuple:
        """Estimate communication time for a phase."""
        comm_time = 0.0
        breakdown = {}
        
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        hidden_size = self.model.config.hidden_size
        
        # TP communication (same as training, less data in decode)
        if self.strategy.tp_degree > 1:
            hidden_size = self.model.config.hidden_size
            batch_size = 1  # Per device for decode
            
            if phase == PHASE_PREFILL:
                seq_len = self.model.config.max_seq_len
                activation_bytes = batch_size * seq_len * hidden_size * dtype_size
            else:
                seq_len = 1
                activation_bytes = batch_size * seq_len * hidden_size * dtype_size
            
            num_layers = self.model.config.num_layers
            tp_bytes = activation_bytes * 2 * num_layers
            
            tp_ranks = list(range(self.strategy.tp_degree))
            tp_time = self.cluster.estimate_allreduce_time(tp_bytes, tp_ranks)
            
            comm_time += tp_time
            breakdown["tensor_parallel"] = tp_time
        
        # PP communication
        if self.strategy.pp_degree > 1:
            hidden_size = self.model.config.hidden_size
            batch_size = 1
            
            pp_bytes = batch_size * hidden_size * dtype_size
            
            # P2P communication
            pp_time = pp_bytes / (self.cluster.network.intra_node_bandwidth_gbps * 1e9)
            
            comm_time += pp_time
            breakdown["pipeline_parallel"] = pp_time
        
        # EP communication for MoE
        if self.strategy.ep_degree > 1:
            hidden_size = self.model.config.hidden_size
            batch_size = 1
            seq_len = 1 if phase == PHASE_DECODE else self.model.config.max_seq_len
            
            token_bytes = batch_size * seq_len * hidden_size * dtype_size
            ep_ranks = list(range(self.strategy.ep_degree))
            
            dispatch_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
            combine_time = self.cluster.estimate_alltoall_time(token_bytes, ep_ranks)
            
            ep_time = dispatch_time + combine_time
            comm_time += ep_time
            breakdown["expert_parallel"] = ep_time
        
        # SP communication (Sequence Parallelism)
        if self.strategy.sp_degree > 1:
            sp_time = self._estimate_sp_communication_time_for_phase(
                phase, hidden_size, dtype_size
            )
            comm_time += sp_time
            breakdown["sequence_parallel"] = sp_time
        
        return comm_time, breakdown
    
    def _estimate_sp_communication_time_for_phase(
        self,
        phase: str,
        hidden_size: int,
        dtype_size: int,
    ) -> float:
        """
        Estimate sequence parallelism communication time for inference.

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
        batch_size = 1
        seq_len = 1 if phase == PHASE_DECODE else self.model.config.max_seq_len
        
        activation_bytes = batch_size * seq_len * hidden_size * dtype_size
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
            alltoall_time = self.cluster.estimate_alltoall_time(
                activation_bytes, sp_ranks
            )
            total_time = alltoall_time * 4 * num_layers
            
        elif sp_type == SPType.RING_P2P:
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
            
            ulysses_ranks = list(range(ulysses_degree))
            ring_ranks = list(range(ring_degree))
            
            # Ulysses part: 4 all-to-all (Q/K/V pre + O post)
            ulysses_time = self.cluster.estimate_alltoall_time(
                activation_bytes, ulysses_ranks
            ) * 4 * num_layers
            
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
        
        return total_time
    
    def _estimate_memory(self, batch_size: int, max_seq_len: int) -> tuple:
        """Estimate memory per GPU. Returns (total_memory, kv_cache_memory)."""
        dtype = self.model.config.dtype
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        
        # Model parameters
        param_memory = self.model.total_params * dtype_size
        param_memory //= self.strategy.tp_degree
        
        # KV cache
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_layers
        num_kv_heads = self.model.config.num_key_value_heads or self.model.config.num_attention_heads
        head_dim = hidden_size // self.model.config.num_attention_heads
        
        # K and V per token per layer
        kv_per_token = 2 * num_kv_heads * head_dim * dtype_size
        kv_cache_memory = batch_size * max_seq_len * num_layers * kv_per_token
        kv_cache_memory //= self.strategy.tp_degree
        
        # Activations - use inference mode estimate
        is_distributed = (
            self.strategy.tp_degree > 1
            or self.strategy.dp_degree > 1
            or self.strategy.pp_degree > 1
        )
        base_memory = self.model.estimate_memory(
            inference_mode=True,  # Inference mode
            batch_size=batch_size,
            is_distributed=is_distributed,
            apply_calibration=False,  # Handle manually for fine-grained control
        )
        dtype_size = 2 if self.model.config.dtype == "fp16" else 4
        param_memory_total = self.model.total_params * dtype_size
        activation_memory = base_memory - param_memory_total

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
        """Get appropriate compute kernel for a layer and phase."""
        name = layer.name
        
        if "proj" in name or "up" in name or "gate" in name or "down" in name:
            m = batch_size * seq_len
            n = layer.output_shape[-1]
            k = layer.input_shape[-1]
            return self.compute_registry.get_or_create_matmul(m, n, k, dtype)
        elif "attention" in name:
            # Different attention pattern for prefill vs decode
            if phase == PHASE_PREFILL:
                return self.compute_registry.get(f"flash_attn_{batch_size}_{seq_len}_32_128_{dtype}")
            else:
                # Decode: seq_len=1 but accesses full KV
                return self.compute_registry.get(f"flash_attn_{batch_size}_1_32_128_{dtype}")
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

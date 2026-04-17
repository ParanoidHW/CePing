"""Detailed performance breakdown for comprehensive analysis.

This module provides fine-grained breakdown of:
- Sub-model level analysis (for pipelines)
- Block-level analysis (within each model)
- Memory breakdown by type (activation, parameter, optimizer, kv_cache)
- Communication breakdown by parallelism type (TP, DP, PP, EP, SP)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum


class ParallelismType(str, Enum):
    """Types of parallelism."""
    TP = "tp"  # Tensor Parallelism
    DP = "dp"  # Data Parallelism
    PP = "pp"  # Pipeline Parallelism
    EP = "ep"  # Expert Parallelism (MoE)
    SP = "sp"  # Sequence Parallelism


class MemoryType(str, Enum):
    """Types of memory usage."""
    PARAMETER = "parameter"
    ACTIVATION = "activation"
    GRADIENT = "gradient"
    OPTIMIZER = "optimizer"
    KV_CACHE = "kv_cache"
    COMM_BUFFER = "comm_buffer"
    WORKSPACE = "workspace"


@dataclass
class CommunicationDetail:
    """Detailed communication operation."""
    parallelism_type: ParallelismType
    operation: str  # allreduce, allgather, reducescatter, alltoall, p2p
    volume_bytes: int  # Total bytes transferred
    source_ranks: List[int] = field(default_factory=list)
    target_ranks: List[int] = field(default_factory=list)
    time_sec: float = 0.0
    bandwidth_gbps: float = 0.0  # Achieved bandwidth
    description: str = ""


@dataclass
class BlockBreakdown:
    """Breakdown for a single block/layer group (e.g., Transformer Block)."""
    block_type: str  # e.g., "transformer", "ffn", "attention", "embedding"
    block_name: str
    layer_indices: List[int] = field(default_factory=list)  # Which layers
    
    # Compute
    flops: int = 0
    compute_time_sec: float = 0.0
    
    # Memory (detailed)
    memory_by_type: Dict[MemoryType, int] = field(default_factory=dict)
    
    # Communication
    comm_operations: List[CommunicationDetail] = field(default_factory=list)
    comm_time_sec: float = 0.0
    
    def total_memory(self) -> int:
        """Total memory across all types."""
        return sum(self.memory_by_type.values())
    
    def comm_volume(self) -> int:
        """Total communication volume."""
        return sum(op.volume_bytes for op in self.comm_operations)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_type": self.block_type,
            "block_name": self.block_name,
            "layer_indices": self.layer_indices,
            "compute": {
                "flops": self.flops,
                "time_sec": self.compute_time_sec,
                "time_ms": self.compute_time_sec * 1000,
            },
            "memory": {
                "by_type": {
                    k.value: v / 1024**3 for k, v in self.memory_by_type.items()
                },
                "total_gb": self.total_memory() / 1024**3,
            },
            "communication": {
                "operations": [
                    {
                        "type": op.parallelism_type.value,
                        "operation": op.operation,
                        "volume_gb": op.volume_bytes / 1024**3,
                        "time_ms": op.time_sec * 1000,
                        "description": op.description,
                    }
                    for op in self.comm_operations
                ],
                "total_volume_gb": self.comm_volume() / 1024**3,
                "total_time_ms": self.comm_time_sec * 1000,
            },
        }


@dataclass
class SubModelBreakdown:
    """Breakdown for a sub-model in a pipeline (e.g., Text Encoder, DiT, VAE)."""
    model_name: str
    model_type: str  # e.g., "text_encoder", "dit", "vae", "llm"
    
    # Overview
    total_time_sec: float = 0.0
    num_iterations: int = 1  # For iterative models (e.g., DiT denoising)
    
    # Compute
    total_flops: int = 0
    compute_time_sec: float = 0.0
    
    # Memory (detailed)
    memory_by_type: Dict[MemoryType, int] = field(default_factory=dict)
    
    # Block-level breakdown
    blocks: List[BlockBreakdown] = field(default_factory=list)
    
    # Communication
    comm_by_parallelism: Dict[ParallelismType, List[CommunicationDetail]] = field(
        default_factory=dict
    )
    
    def total_memory(self) -> int:
        """Total memory across all types."""
        return sum(self.memory_by_type.values())
    
    def total_comm_volume(self) -> int:
        """Total communication volume."""
        total = 0
        for ops in self.comm_by_parallelism.values():
            total += sum(op.volume_bytes for op in ops)
        return total
    
    def total_comm_time(self) -> float:
        """Total communication time."""
        total = 0.0
        for ops in self.comm_by_parallelism.values():
            total += sum(op.time_sec for op in ops)
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "overview": {
                "total_time_sec": self.total_time_sec,
                "num_iterations": self.num_iterations,
            },
            "compute": {
                "total_flops": self.total_flops,
                "compute_time_sec": self.compute_time_sec,
                "compute_time_ms": self.compute_time_sec * 1000,
            },
            "memory": {
                "by_type": {
                    k.value: v / 1024**3 for k, v in self.memory_by_type.items()
                },
                "total_gb": self.total_memory() / 1024**3,
            },
            "blocks": [block.to_dict() for block in self.blocks],
            "communication": {
                "by_parallelism": {
                    k.value: {
                        "operations": [
                            {
                                "operation": op.operation,
                                "volume_gb": op.volume_bytes / 1024**3,
                                "time_ms": op.time_sec * 1000,
                                "description": op.description,
                            }
                            for op in ops
                        ],
                        "total_volume_gb": sum(op.volume_bytes for op in ops) / 1024**3,
                        "total_time_ms": sum(op.time_sec for op in ops) * 1000,
                    }
                    for k, ops in self.comm_by_parallelism.items()
                },
                "total_volume_gb": self.total_comm_volume() / 1024**3,
                "total_time_ms": self.total_comm_time() * 1000,
            },
        }


@dataclass
class MemoryBreakdown:
    """Comprehensive memory breakdown across all components."""
    
    # By memory type (aggregated across all sub-models)
    by_type: Dict[MemoryType, int] = field(default_factory=dict)
    
    # By sub-model
    by_submodel: Dict[str, Dict[MemoryType, int]] = field(default_factory=dict)
    
    # By block type (e.g., all attention blocks across all models)
    by_block_type: Dict[str, Dict[MemoryType, int]] = field(default_factory=dict)
    
    def total(self) -> int:
        """Total memory across all types."""
        return sum(self.by_type.values())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "by_type": {
                k.value: v / 1024**3 for k, v in self.by_type.items()
            },
            "total_gb": self.total() / 1024**3,
            "by_submodel": {
                name: {k.value: v / 1024**3 for k, v in mems.items()}
                for name, mems in self.by_submodel.items()
            },
            "by_block_type": {
                block_type: {k.value: v / 1024**3 for k, v in mems.items()}
                for block_type, mems in self.by_block_type.items()
            },
        }


@dataclass
class CommunicationBreakdown:
    """Comprehensive communication breakdown by parallelism type."""
    
    # By parallelism type
    by_type: Dict[ParallelismType, List[CommunicationDetail]] = field(
        default_factory=dict
    )
    
    # By sub-model
    by_submodel: Dict[str, List[CommunicationDetail]] = field(default_factory=dict)
    
    def total_volume(self) -> int:
        """Total communication volume."""
        total = 0
        for ops in self.by_type.values():
            total += sum(op.volume_bytes for op in ops)
        return total
    
    def total_time(self) -> float:
        """Total communication time."""
        total = 0.0
        for ops in self.by_type.values():
            total += sum(op.time_sec for op in ops)
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "by_parallelism": {
                k.value: {
                    "total_volume_gb": sum(op.volume_bytes for op in ops) / 1024**3,
                    "total_time_ms": sum(op.time_sec for op in ops) * 1000,
                    "operations": [
                        {
                            "operation": op.operation,
                            "volume_gb": op.volume_bytes / 1024**3,
                            "time_ms": op.time_sec * 1000,
                            "bandwidth_gbps": op.bandwidth_gbps,
                            "description": op.description,
                        }
                        for op in ops
                    ],
                }
                for k, ops in self.by_type.items()
            },
            "total_volume_gb": self.total_volume() / 1024**3,
            "total_time_ms": self.total_time() * 1000,
            "by_submodel": {
                name: {
                    "total_volume_gb": sum(op.volume_bytes for op in ops) / 1024**3,
                    "total_time_ms": sum(op.time_sec for op in ops) * 1000,
                }
                for name, ops in self.by_submodel.items()
            },
        }


@dataclass
class DetailedPerformanceResult:
    """Complete detailed performance result for pipeline or single model."""
    
    # Overview
    total_time_sec: float
    throughput: float
    
    # Sub-model breakdown (for pipelines)
    submodels: List[SubModelBreakdown] = field(default_factory=list)
    
    # Aggregated memory breakdown
    memory: MemoryBreakdown = field(default_factory=MemoryBreakdown)
    
    # Aggregated communication breakdown
    communication: CommunicationBreakdown = field(default_factory=CommunicationBreakdown)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overview": {
                "total_time_sec": self.total_time_sec,
                "total_time_ms": self.total_time_sec * 1000,
                "throughput": self.throughput,
            },
            "submodels": [submodel.to_dict() for submodel in self.submodels],
            "memory": self.memory.to_dict(),
            "communication": self.communication.to_dict(),
            "metadata": self.metadata,
        }
    
    def get_summary(self) -> str:
        """Generate a formatted summary."""
        lines = []
        lines.append("=" * 100)
        lines.append("DETAILED PERFORMANCE BREAKDOWN")
        lines.append("=" * 100)
        
        lines.append(f"\nTotal Time: {self.total_time_sec*1000:.2f} ms")
        lines.append(f"Throughput: {self.throughput:.2f}")
        
        # Sub-models
        if self.submodels:
            lines.append("\n--- Sub-Models ---")
            for submodel in self.submodels:
                lines.append(f"\n{submodel.model_name} ({submodel.model_type}):")
                lines.append(f"  Time: {submodel.total_time_sec*1000:.2f} ms")
                if submodel.num_iterations > 1:
                    lines.append(f"  Iterations: {submodel.num_iterations}")
                lines.append(f"  Memory: {submodel.total_memory()/1024**3:.2f} GB")
                lines.append(f"  Comm Volume: {submodel.total_comm_volume()/1024**6:.2f} GB")
        
        # Memory breakdown
        if self.memory.by_type:
            lines.append("\n--- Memory Breakdown ---")
            for mem_type, bytes_val in sorted(self.memory.by_type.items()):
                lines.append(f"  {mem_type.value:15s}: {bytes_val/1024**3:8.2f} GB")
            lines.append(f"  {'TOTAL':15s}: {self.memory.total()/1024**3:8.2f} GB")
        
        # Communication breakdown
        if self.communication.by_type:
            lines.append("\n--- Communication Breakdown ---")
            for para_type, ops in sorted(self.communication.by_type.items()):
                total_vol = sum(op.volume_bytes for op in ops)
                total_time = sum(op.time_sec for op in ops)
                lines.append(f"  {para_type.value.upper():5s}: "
                           f"Volume={total_vol/1024**6:.2f} GB, "
                           f"Time={total_time*1000:.2f} ms")
        
        lines.append("=" * 100)
        return "\n".join(lines)

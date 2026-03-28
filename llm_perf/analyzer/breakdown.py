"""Performance breakdown analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class KernelBreakdown:
    """Performance breakdown for a single kernel."""
    name: str
    kernel_type: str  # compute, communication, memory
    time_sec: float
    memory_bytes: int
    flops: int = 0
    bytes_transferred: int = 0
    is_parallel: bool = True  # Can overlap with other kernels


@dataclass
class LayerBreakdown:
    """Performance breakdown for a model layer."""
    name: str
    kernels: List[KernelBreakdown] = field(default_factory=list)
    
    @property
    def total_time(self) -> float:
        """Total time including overlapped kernels."""
        # Compute and communication may overlap
        compute_time = sum(
            k.time_sec for k in self.kernels 
            if k.kernel_type == "compute"
        )
        comm_time = sum(
            k.time_sec for k in self.kernels 
            if k.kernel_type == "communication"
        )
        # Assume perfect overlap for now
        return max(compute_time, comm_time)
    
    @property
    def total_memory(self) -> int:
        """Total memory usage."""
        return sum(k.memory_bytes for k in self.kernels)
    
    @property
    def compute_time(self) -> float:
        return sum(
            k.time_sec for k in self.kernels 
            if k.kernel_type == "compute"
        )
    
    @property
    def comm_time(self) -> float:
        return sum(
            k.time_sec for k in self.kernels 
            if k.kernel_type == "communication"
        )


@dataclass
class PerformanceBreakdown:
    """
    Complete performance breakdown for a model execution.
    
    This provides detailed timing and memory analysis.
    """
    
    # Overview
    total_time_sec: float
    throughput: float  # tokens/sec or samples/sec
    
    # Breakdown by category
    compute_time_sec: float
    communication_time_sec: float
    memory_time_sec: float  # Time waiting on memory
    
    # Breakdown by layer
    layers: List[LayerBreakdown] = field(default_factory=list)
    
    # Memory usage
    peak_memory_bytes: int = 0
    activation_memory_bytes: int = 0
    parameter_memory_bytes: int = 0
    
    # Communication details
    comm_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overview": {
                "total_time_sec": self.total_time_sec,
                "total_time_ms": self.total_time_sec * 1000,
                "throughput": self.throughput,
            },
            "time_breakdown": {
                "compute_sec": self.compute_time_sec,
                "communication_sec": self.communication_time_sec,
                "memory_sec": self.memory_time_sec,
                "compute_percent": self.compute_time_sec / self.total_time_sec * 100,
                "communication_percent": self.communication_time_sec / self.total_time_sec * 100,
                "memory_percent": self.memory_time_sec / self.total_time_sec * 100,
            },
            "memory_breakdown": {
                "peak_mb": self.peak_memory_bytes / 1024 / 1024,
                "activation_mb": self.activation_memory_bytes / 1024 / 1024,
                "parameter_mb": self.parameter_memory_bytes / 1024 / 1024,
            },
            "communication": self.comm_breakdown,
            "layers": [
                {
                    "name": layer.name,
                    "total_time_ms": layer.total_time * 1000,
                    "compute_ms": layer.compute_time * 1000,
                    "comm_ms": layer.comm_time * 1000,
                    "memory_mb": layer.total_memory / 1024 / 1024,
                }
                for layer in self.layers
            ],
        }
    
    def get_summary_table(self) -> str:
        """Generate a formatted summary table."""
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE BREAKDOWN")
        lines.append("=" * 80)
        
        lines.append(f"\nTotal Time: {self.total_time_sec*1000:.2f} ms")
        lines.append(f"Throughput: {self.throughput:.2f} tokens/sec")
        
        lines.append("\n--- Time Breakdown ---")
        compute_pct = self.compute_time_sec / self.total_time_sec * 100
        comm_pct = self.communication_time_sec / self.total_time_sec * 100
        mem_pct = self.memory_time_sec / self.total_time_sec * 100
        
        lines.append(f"  Compute:       {self.compute_time_sec*1000:8.2f} ms ({compute_pct:5.1f}%)")
        lines.append(f"  Communication: {self.communication_time_sec*1000:8.2f} ms ({comm_pct:5.1f}%)")
        lines.append(f"  Memory Wait:   {self.memory_time_sec*1000:8.2f} ms ({mem_pct:5.1f}%)")
        
        lines.append("\n--- Memory Breakdown ---")
        lines.append(f"  Peak Memory:     {self.peak_memory_bytes/1024/1024:8.2f} MB")
        lines.append(f"  Activations:     {self.activation_memory_bytes/1024/1024:8.2f} MB")
        lines.append(f"  Parameters:      {self.parameter_memory_bytes/1024/1024:8.2f} MB")
        
        if self.comm_breakdown:
            lines.append("\n--- Communication Details ---")
            for name, time_sec in self.comm_breakdown.items():
                lines.append(f"  {name}: {time_sec*1000:.2f} ms")
        
        lines.append("\n--- Top Time-Consuming Layers ---")
        sorted_layers = sorted(self.layers, key=lambda x: x.total_time, reverse=True)[:5]
        for layer in sorted_layers:
            lines.append(f"  {layer.name:40s} {layer.total_time*1000:8.2f} ms")
        
        lines.append("=" * 80)
        return "\n".join(lines)

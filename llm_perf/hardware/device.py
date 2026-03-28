"""Device (GPU/Accelerator) definitions."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DeviceConfig:
    """Hardware device configuration."""
    
    # Device identification
    name: str  # e.g., "A100-SXM-80GB", "H100-SXM-80GB", "MI300X"
    
    # Compute capabilities (in TFLOPS)
    fp32_tflops: float
    fp16_tflops: float
    bf16_tflops: float
    fp8_tflops: Optional[float] = None
    int8_tflops: Optional[float] = None
    
    # Memory specs
    memory_gb: float
    memory_bandwidth_gbps: float  # GB/s
    
    # Interconnect (within node)
    nvlink_bandwidth_gbps: Optional[float] = None  # For NVIDIA
    infinity_fabric_gbps: Optional[float] = None  # For AMD
    
    # L2 cache
    l2_cache_mb: float = 0.0
    
    # Power
    tdp_w: float = 0.0


class Device:
    """Represents a compute device (GPU/Accelerator)."""
    
    # Predefined device configurations
    PRESETS = {
        "A100-SXM-40GB": DeviceConfig(
            name="A100-SXM-40GB",
            fp32_tflops=19.5,
            fp16_tflops=312.0,
            bf16_tflops=312.0,
            fp8_tflops=None,
            int8_tflops=624.0,
            memory_gb=40.0,
            memory_bandwidth_gbps=1555.0,
            nvlink_bandwidth_gbps=600.0,
            l2_cache_mb=40.0,
            tdp_w=400.0,
        ),
        "A100-SXM-80GB": DeviceConfig(
            name="A100-SXM-80GB",
            fp32_tflops=19.5,
            fp16_tflops=312.0,
            bf16_tflops=312.0,
            fp8_tflops=None,
            int8_tflops=624.0,
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
            nvlink_bandwidth_gbps=600.0,
            l2_cache_mb=40.0,
            tdp_w=400.0,
        ),
        "H100-SXM-80GB": DeviceConfig(
            name="H100-SXM-80GB",
            fp32_tflops=67.0,
            fp16_tflops=989.0,
            bf16_tflops=989.0,
            fp8_tflops=1979.0,
            int8_tflops=3958.0,
            memory_gb=80.0,
            memory_bandwidth_gbps=3350.0,
            nvlink_bandwidth_gbps=900.0,
            l2_cache_mb=50.0,
            tdp_w=700.0,
        ),
        "H100-NVL-94GB": DeviceConfig(
            name="H100-NVL-94GB",
            fp32_tflops=67.0,
            fp16_tflops=989.0,
            bf16_tflops=989.0,
            fp8_tflops=1979.0,
            int8_tflops=3958.0,
            memory_gb=94.0,
            memory_bandwidth_gbps=3350.0,
            nvlink_bandwidth_gbps=900.0,
            l2_cache_mb=50.0,
            tdp_w=700.0,
        ),
        "H200-SXM-141GB": DeviceConfig(
            name="H200-SXM-141GB",
            fp32_tflops=67.0,
            fp16_tflops=989.0,
            bf16_tflops=989.0,
            fp8_tflops=1979.0,
            int8_tflops=3958.0,
            memory_gb=141.0,
            memory_bandwidth_gbps=4900.0,
            nvlink_bandwidth_gbps=900.0,
            l2_cache_mb=50.0,
            tdp_w=700.0,
        ),
        "MI300X": DeviceConfig(
            name="MI300X",
            fp32_tflops=163.0,
            fp16_tflops=1307.0,
            bf16_tflops=1307.0,
            fp8_tflops=2613.0,
            int8_tflops=2613.0,
            memory_gb=192.0,
            memory_bandwidth_gbps=5300.0,
            infinity_fabric_gbps=896.0,
            l2_cache_mb=256.0,
            tdp_w=750.0,
        ),
        "L40S": DeviceConfig(
            name="L40S",
            fp32_tflops=91.6,
            fp16_tflops=183.0,
            bf16_tflops=183.0,
            fp8_tflops=366.0,
            int8_tflops=366.0,
            memory_gb=48.0,
            memory_bandwidth_gbps=864.0,
            nvlink_bandwidth_gbps=None,
            l2_cache_mb=48.0,
            tdp_w=350.0,
        ),
    }
    
    def __init__(self, config: DeviceConfig):
        self.config = config
    
    @classmethod
    def from_preset(cls, name: str) -> "Device":
        """Create device from preset configuration."""
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown device preset: {name}. "
                           f"Available: {list(cls.PRESETS.keys())}")
        return cls(cls.PRESETS[name])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Device":
        """Create device from dictionary."""
        config = DeviceConfig(**data)
        return cls(config)
    
    def get_compute_tflops(self, dtype: str) -> float:
        """Get compute throughput for given dtype."""
        dtype_map = {
            "fp32": self.config.fp32_tflops,
            "fp16": self.config.fp16_tflops,
            "bf16": self.config.bf16_tflops,
            "fp8": self.config.fp8_tflops or self.config.fp16_tflops,
            "int8": self.config.int8_tflops or self.config.fp16_tflops,
            "int4": self.config.int8_tflops or self.config.fp16_tflops,
        }
        return dtype_map.get(dtype, self.config.fp16_tflops)
    
    def get_memory_bw_gbps(self) -> float:
        """Get memory bandwidth in GB/s."""
        return self.config.memory_bandwidth_gbps
    
    def estimate_roofline_flops(
        self, 
        arithmetic_intensity: float,
        dtype: str = "fp16"
    ) -> float:
        """
        Estimate achievable FLOPS using roofline model.
        
        Args:
            arithmetic_intensity: FLOPs / byte (operational intensity)
            dtype: Data type for compute
        
        Returns:
            Achievable FLOPS
        """
        peak_flops = self.get_compute_tflops(dtype) * 1e12
        mem_bw = self.get_memory_bw_gbps() * 1e9
        
        # Ridge point: where compute and memory bounds intersect
        ridge_point = peak_flops / mem_bw
        
        if arithmetic_intensity < ridge_point:
            # Memory bound
            return arithmetic_intensity * mem_bw
        else:
            # Compute bound
            return peak_flops
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.config.name,
            "fp32_tflops": self.config.fp32_tflops,
            "fp16_tflops": self.config.fp16_tflops,
            "bf16_tflops": self.config.bf16_tflops,
            "fp8_tflops": self.config.fp8_tflops,
            "int8_tflops": self.config.int8_tflops,
            "memory_gb": self.config.memory_gb,
            "memory_bandwidth_gbps": self.config.memory_bandwidth_gbps,
            "nvlink_bandwidth_gbps": self.config.nvlink_bandwidth_gbps,
            "infinity_fabric_gbps": self.config.infinity_fabric_gbps,
            "l2_cache_mb": self.config.l2_cache_mb,
            "tdp_w": self.config.tdp_w,
        }

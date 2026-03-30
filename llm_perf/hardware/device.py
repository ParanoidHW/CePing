"""Device (GPU/Accelerator) definitions."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class ComputeUnitType(Enum):
    """Type of compute unit used for different operations."""
    CUBE_TENSOR_CORE = "cube_tensor_core"  # Matrix operations (Ascend Cube / GPU Tensor Core)
    VECTOR_CUDA_CORE = "vector_cuda_core"    # Element-wise operations (Ascend Vector / GPU CUDA Core)
    MIXED = "mixed"                          # Mixed usage


@dataclass
class DeviceConfig:
    """Hardware device configuration."""
    
    # Device identification
    name: str  # e.g., "A100-SXM-80GB", "H100-SXM-80GB", "Ascend-910B"
    
    # ========== CUBE / Tensor Core Compute (Matrix Operations) ==========
    # Used for: GEMM, Attention QK^T, PV
    fp32_tflops_cube: float = 0.0
    fp16_tflops_cube: float = 0.0
    bf16_tflops_cube: float = 0.0
    fp8_tflops_cube: Optional[float] = None
    int8_tflops_cube: Optional[float] = None
    
    # ========== VECTOR / CUDA Core Compute (Element-wise Operations) ==========
    # Used for: Activation, LayerNorm, Softmax, element-wise ops
    fp32_tflops_vector: float = 0.0
    fp16_tflops_vector: float = 0.0
    bf16_tflops_vector: float = 0.0
    fp8_tflops_vector: Optional[float] = None
    int8_tflops_vector: Optional[float] = None
    
    # Backward compatibility: if only *_tflops is set, use it for both
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    fp8_tflops: Optional[float] = None
    int8_tflops: Optional[float] = None
    
    # Memory specs
    memory_gb: float
    memory_bandwidth_gbps: float  # GB/s
    
    # Interconnect (within node)
    nvlink_bandwidth_gbps: Optional[float] = None  # For NVIDIA
    hccs_bandwidth_gbps: Optional[float] = None    # For Ascend (Huawei Collective Communication System)
    infinity_fabric_gbps: Optional[float] = None   # For AMD
    
    # L2 cache
    l2_cache_mb: float = 0.0
    
    # Power
    tdp_w: float = 0.0
    
    def __post_init__(self):
        """Ensure backward compatibility for legacy configs."""
        # If cube values are not set but legacy values are, copy them
        if self.fp16_tflops_cube == 0.0 and self.fp16_tflops > 0.0:
            self.fp16_tflops_cube = self.fp16_tflops
            self.fp16_tflops_vector = self.fp16_tflops
        if self.fp32_tflops_cube == 0.0 and self.fp32_tflops > 0.0:
            self.fp32_tflops_cube = self.fp32_tflops
            self.fp32_tflops_vector = self.fp32_tflops
        if self.bf16_tflops_cube == 0.0 and self.bf16_tflops > 0.0:
            self.bf16_tflops_cube = self.bf16_tflops
            self.bf16_tflops_vector = self.bf16_tflops


class Device:
    """Represents a compute device (GPU/Accelerator)."""
    
    # Predefined device configurations
    PRESETS = {
        # ========== NVIDIA GPUs ==========
        # Source: NVIDIA Official Datasheets and Whitepapers
        # Tensor Core (CUBE) for matrix ops: GEMM, Attention
        # CUDA Core (VECTOR) for element-wise ops: Activation, Normalization
        # Note: For Ampere (A100) and Hopper (H100) architectures, 
        #       FP16/BF16 tensor core ops are much faster than CUDA core ops
        #       CUDA Core FP16 is typically 2x FP32 (same as Ampere/Hopper CUDA cores)
        
        "A100-SXM-40GB": DeviceConfig(
            name="A100-SXM-40GB",
            # Tensor Core (CUBE) - for matrix operations
            fp16_tflops_cube=312.0,
            bf16_tflops_cube=312.0,
            fp32_tflops_cube=156.0,  # Tensor Core FP32 is half of FP16 on A100
            int8_tflops_cube=624.0,
            # CUDA Core (VECTOR) - for element-wise operations
            fp32_tflops_vector=19.5,
            fp16_tflops_vector=39.0,  # 2x FP32 for CUDA cores on Ampere
            bf16_tflops_vector=39.0,
            int8_tflops_vector=78.0,
            # Legacy fields for backward compatibility
            fp32_tflops=19.5,
            fp16_tflops=312.0,
            bf16_tflops=312.0,
            int8_tflops=624.0,
            memory_gb=40.0,
            memory_bandwidth_gbps=1555.0,
            nvlink_bandwidth_gbps=600.0,
            l2_cache_mb=40.0,
            tdp_w=400.0,
        ),
        "A100-SXM-80GB": DeviceConfig(
            name="A100-SXM-80GB",
            # Tensor Core (CUBE)
            fp16_tflops_cube=312.0,
            bf16_tflops_cube=312.0,
            fp32_tflops_cube=156.0,
            int8_tflops_cube=624.0,
            # CUDA Core (VECTOR)
            fp32_tflops_vector=19.5,
            fp16_tflops_vector=39.0,
            bf16_tflops_vector=39.0,
            int8_tflops_vector=78.0,
            # Legacy fields
            fp32_tflops=19.5,
            fp16_tflops=312.0,
            bf16_tflops=312.0,
            int8_tflops=624.0,
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
            nvlink_bandwidth_gbps=600.0,
            l2_cache_mb=40.0,
            tdp_w=400.0,
        ),
        "H100-SXM-80GB": DeviceConfig(
            name="H100-SXM-80GB",
            # Tensor Core (CUBE) - significantly faster than A100
            fp16_tflops_cube=989.0,
            bf16_tflops_cube=989.0,
            fp8_tflops_cube=1979.0,
            fp32_tflops_cube=494.5,
            int8_tflops_cube=3958.0,
            # CUDA Core (VECTOR)
            fp32_tflops_vector=67.0,
            fp16_tflops_vector=134.0,  # 2x FP32 on Hopper CUDA cores
            bf16_tflops_vector=134.0,
            int8_tflops_vector=268.0,
            # Legacy fields
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
            # Tensor Core (CUBE)
            fp16_tflops_cube=989.0,
            bf16_tflops_cube=989.0,
            fp8_tflops_cube=1979.0,
            fp32_tflops_cube=494.5,
            int8_tflops_cube=3958.0,
            # CUDA Core (VECTOR)
            fp32_tflops_vector=67.0,
            fp16_tflops_vector=134.0,
            bf16_tflops_vector=134.0,
            int8_tflops_vector=268.0,
            # Legacy fields
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
            # Tensor Core (CUBE) - same compute as H100, more memory
            fp16_tflops_cube=989.0,
            bf16_tflops_cube=989.0,
            fp8_tflops_cube=1979.0,
            fp32_tflops_cube=494.5,
            int8_tflops_cube=3958.0,
            # CUDA Core (VECTOR)
            fp32_tflops_vector=67.0,
            fp16_tflops_vector=134.0,
            bf16_tflops_vector=134.0,
            int8_tflops_vector=268.0,
            # Legacy fields
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
        
        # ========== AMD GPUs ==========
        "MI300X": DeviceConfig(
            name="MI300X",
            # Matrix Core (CUBE equivalent)
            fp16_tflops_cube=1307.0,
            bf16_tflops_cube=1307.0,
            fp8_tflops_cube=2613.0,
            fp32_tflops_cube=653.5,
            int8_tflops_cube=2613.0,
            # Stream Processor (VECTOR equivalent)
            fp32_tflops_vector=163.0,
            fp16_tflops_vector=326.0,
            bf16_tflops_vector=326.0,
            int8_tflops_vector=652.0,
            # Legacy fields
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
        
        # ========== NVIDIA L40S (Ada Lovelace) ==========
        # Note: L40S has less powerful Tensor Cores compared to H100
        # CUDA Core FP32 is higher than A100 but Tensor Core FP16 is lower than H100
        "L40S": DeviceConfig(
            name="L40S",
            # Tensor Core (CUBE)
            fp16_tflops_cube=183.0,
            bf16_tflops_cube=183.0,
            fp8_tflops_cube=366.0,
            fp32_tflops_cube=91.5,
            int8_tflops_cube=366.0,
            # CUDA Core (VECTOR) - Ada Lovelace has strong CUDA cores
            fp32_tflops_vector=91.6,
            fp16_tflops_vector=183.2,
            bf16_tflops_vector=183.2,
            int8_tflops_vector=366.4,
            # Legacy fields
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
        
        # ========== Huawei Ascend NPUs ==========
        # Source: https://blog.ailemon.net/2025/05/24/huawei-ascend-npu-params-for-ai/
        # Note: Ascend uses CUBE core for matrix ops, VECTOR core for element-wise ops
        # CUBE ~= Tensor Core (matrix), VECTOR ~= CUDA Core (vector)
        
        "Ascend-910A": DeviceConfig(
            name="Ascend-910A",
            # CUBE core (matrix operations)
            fp16_tflops_cube=256.0,
            int8_tflops_cube=512.0,
            # VECTOR core (element-wise operations) - roughly 1/10 of CUBE for Ascend
            fp16_tflops_vector=25.6,
            int8_tflops_vector=51.2,
            memory_gb=32.0,
            memory_bandwidth_gbps=1500.0,
            hccs_bandwidth_gbps=100.0,  # HCCS (Huawei Cache Coherence System)
            l2_cache_mb=48.0,
            tdp_w=350.0,
        ),
        "Ascend-910B1": DeviceConfig(
            name="Ascend-910B1",
            # CUBE core
            fp16_tflops_cube=414.0,
            int8_tflops_cube=828.0,
            # VECTOR core
            fp16_tflops_vector=41.4,
            int8_tflops_vector=82.8,
            memory_gb=64.0,
            memory_bandwidth_gbps=392.0,
            hccs_bandwidth_gbps=200.0,
            l2_cache_mb=48.0,
            tdp_w=310.0,
        ),
        "Ascend-910B2": DeviceConfig(
            name="Ascend-910B2",
            # CUBE core
            fp16_tflops_cube=376.0,
            fp32_tflops_cube=94.0,
            int8_tflops_cube=752.0,
            # VECTOR core
            fp16_tflops_vector=37.6,
            fp32_tflops_vector=9.4,
            int8_tflops_vector=75.2,
            memory_gb=64.0,
            memory_bandwidth_gbps=392.0,
            hccs_bandwidth_gbps=200.0,
            l2_cache_mb=48.0,
            tdp_w=310.0,
        ),
        "Ascend-910B3": DeviceConfig(
            name="Ascend-910B3",
            # CUBE core
            fp16_tflops_cube=313.0,
            int8_tflops_cube=626.0,
            # VECTOR core
            fp16_tflops_vector=31.3,
            int8_tflops_vector=62.6,
            memory_gb=64.0,
            memory_bandwidth_gbps=392.0,
            hccs_bandwidth_gbps=200.0,
            l2_cache_mb=48.0,
            tdp_w=310.0,
        ),
        "Ascend-910B4": DeviceConfig(
            name="Ascend-910B4",
            # CUBE core
            fp16_tflops_cube=280.0,
            int8_tflops_cube=560.0,
            # VECTOR core
            fp16_tflops_vector=28.0,
            int8_tflops_vector=56.0,
            memory_gb=32.0,
            memory_bandwidth_gbps=392.0,
            hccs_bandwidth_gbps=200.0,
            l2_cache_mb=48.0,
            tdp_w=310.0,
        ),
        "Ascend-910C": DeviceConfig(
            name="Ascend-910C",
            # CUBE core
            fp16_tflops_cube=800.0,
            int8_tflops_cube=1600.0,
            # VECTOR core
            fp16_tflops_vector=80.0,
            int8_tflops_vector=160.0,
            memory_gb=128.0,
            memory_bandwidth_gbps=784.0,
            hccs_bandwidth_gbps=400.0,
            l2_cache_mb=64.0,
            tdp_w=400.0,
        ),
        # Ascend 950 DT (DataCenter Training) - 144GB HBM, higher memory BW
        "Ascend-950-DT": DeviceConfig(
            name="Ascend-950-DT",
            # CUBE core
            fp16_tflops_cube=500.0,
            fp8_tflops_cube=1000.0,
            int8_tflops_cube=1000.0,
            # VECTOR core
            fp16_tflops_vector=50.0,
            fp8_tflops_vector=100.0,
            int8_tflops_vector=100.0,
            memory_gb=144.0,
            memory_bandwidth_gbps=4000.0,  # 4 TB/s HiZQ 2.0
            hccs_bandwidth_gbps=500.0,
            l2_cache_mb=96.0,
            tdp_w=400.0,
        ),
        # Ascend 950 PR (Production) - 128GB HBM, standard memory BW
        "Ascend-950-PR": DeviceConfig(
            name="Ascend-950-PR",
            # CUBE core
            fp16_tflops_cube=500.0,
            fp8_tflops_cube=1000.0,
            int8_tflops_cube=1000.0,
            # VECTOR core
            fp16_tflops_vector=50.0,
            fp8_tflops_vector=100.0,
            int8_tflops_vector=100.0,
            memory_gb=128.0,
            memory_bandwidth_gbps=1600.0,  # 1.6 TB/s HiBL 1.0
            hccs_bandwidth_gbps=500.0,
            l2_cache_mb=96.0,
            tdp_w=400.0,
        ),
        "Ascend-960": DeviceConfig(
            name="Ascend-960",
            # CUBE core
            fp16_tflops_cube=1000.0,
            fp8_tflops_cube=2000.0,
            int8_tflops_cube=2000.0,
            # VECTOR core
            fp16_tflops_vector=100.0,
            fp8_tflops_vector=200.0,
            int8_tflops_vector=200.0,
            memory_gb=144.0,
            memory_bandwidth_gbps=2400.0,
            hccs_bandwidth_gbps=600.0,
            l2_cache_mb=128.0,
            tdp_w=450.0,
        ),
        "Ascend-970": DeviceConfig(
            name="Ascend-970",
            # CUBE core
            fp16_tflops_cube=2000.0,
            fp8_tflops_cube=4000.0,
            fp4_tflops_cube=8000.0,
            int8_tflops_cube=4000.0,
            # VECTOR core
            fp16_tflops_vector=200.0,
            fp8_tflops_vector=400.0,
            fp4_tflops_vector=800.0,
            int8_tflops_vector=400.0,
            memory_gb=288.0,
            memory_bandwidth_gbps=4800.0,
            hccs_bandwidth_gbps=800.0,
            l2_cache_mb=256.0,
            tdp_w=600.0,
        ),
        "Ascend-310P": DeviceConfig(
            name="Ascend-310P",
            # Inference card, less CUBE power
            fp16_tflops_cube=70.0,
            int8_tflops_cube=140.0,
            # VECTOR core
            fp16_tflops_vector=14.0,
            int8_tflops_vector=28.0,
            memory_gb=24.0,
            memory_bandwidth_gbps=204.8,
            l2_cache_mb=24.0,
            tdp_w=72.0,
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
    
    def get_compute_tflops(self, dtype: str, unit_type: ComputeUnitType = ComputeUnitType.CUBE_TENSOR_CORE) -> float:
        """
        Get compute throughput for given dtype and compute unit type.
        
        Args:
            dtype: Data type (fp32, fp16, bf16, fp8, int8)
            unit_type: Type of compute unit (CUBE_TENSOR_CORE for matrix, VECTOR_CUDA_CORE for element-wise)
        
        Returns:
            Compute throughput in TFLOPS
        """
        if unit_type == ComputeUnitType.CUBE_TENSOR_CORE:
            dtype_map = {
                "fp32": self.config.fp32_tflops_cube or self.config.fp32_tflops,
                "fp16": self.config.fp16_tflops_cube or self.config.fp16_tflops,
                "bf16": self.config.bf16_tflops_cube or self.config.bf16_tflops,
                "fp8": self.config.fp8_tflops_cube or self.config.fp8_tflops or self.config.fp16_tflops_cube,
                "int8": self.config.int8_tflops_cube or self.config.int8_tflops or self.config.fp16_tflops_cube,
                "int4": self.config.int8_tflops_cube or self.config.int8_tflops or self.config.fp16_tflops_cube,
            }
        else:  # VECTOR_CUDA_CORE
            dtype_map = {
                "fp32": self.config.fp32_tflops_vector or self.config.fp32_tflops,
                "fp16": self.config.fp16_tflops_vector or self.config.fp16_tflops,
                "bf16": self.config.bf16_tflops_vector or self.config.bf16_tflops,
                "fp8": self.config.fp8_tflops_vector or self.config.fp8_tflops or self.config.fp16_tflops_vector,
                "int8": self.config.int8_tflops_vector or self.config.int8_tflops or self.config.fp16_tflops_vector,
                "int4": self.config.int8_tflops_vector or self.config.int8_tflops or self.config.fp16_tflops_vector,
            }
        return dtype_map.get(dtype, self.config.fp16_tflops_cube or self.config.fp16_tflops)
    
    def get_memory_bw_gbps(self) -> float:
        """Get memory bandwidth in GB/s."""
        return self.config.memory_bandwidth_gbps
    
    def estimate_roofline_flops(
        self, 
        arithmetic_intensity: float,
        dtype: str = "fp16",
        unit_type: ComputeUnitType = ComputeUnitType.CUBE_TENSOR_CORE
    ) -> float:
        """
        Estimate achievable FLOPS using roofline model.
        
        Args:
            arithmetic_intensity: FLOPs / byte (operational intensity)
            dtype: Data type for compute
            unit_type: Type of compute unit to use
        
        Returns:
            Achievable FLOPS
        """
        peak_flops = self.get_compute_tflops(dtype, unit_type) * 1e12
        mem_bw = self.get_memory_bw_gbps() * 1e9
        
        # Ridge point: where compute and memory bounds intersect
        ridge_point = peak_flops / mem_bw
        
        if arithmetic_intensity < ridge_point:
            # Memory bound
            return arithmetic_intensity * mem_bw
        else:
            # Compute bound
            return peak_flops
    
    def is_ascend_npu(self) -> bool:
        """Check if this is a Huawei Ascend NPU."""
        return self.config.name.startswith("Ascend-")
    
    def is_nvidia_gpu(self) -> bool:
        """Check if this is an NVIDIA GPU."""
        return any(x in self.config.name for x in ["A100", "H100", "H200", "L40S"])
    
    def is_amd_gpu(self) -> bool:
        """Check if this is an AMD GPU."""
        return "MI" in self.config.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.config.name,
            "fp32_tflops_cube": self.config.fp32_tflops_cube,
            "fp16_tflops_cube": self.config.fp16_tflops_cube,
            "bf16_tflops_cube": self.config.bf16_tflops_cube,
            "fp8_tflops_cube": self.config.fp8_tflops_cube,
            "int8_tflops_cube": self.config.int8_tflops_cube,
            "fp32_tflops_vector": self.config.fp32_tflops_vector,
            "fp16_tflops_vector": self.config.fp16_tflops_vector,
            "bf16_tflops_vector": self.config.bf16_tflops_vector,
            "fp8_tflops_vector": self.config.fp8_tflops_vector,
            "int8_tflops_vector": self.config.int8_tflops_vector,
            "memory_gb": self.config.memory_gb,
            "memory_bandwidth_gbps": self.config.memory_bandwidth_gbps,
            "nvlink_bandwidth_gbps": self.config.nvlink_bandwidth_gbps,
            "hccs_bandwidth_gbps": self.config.hccs_bandwidth_gbps,
            "infinity_fabric_gbps": self.config.infinity_fabric_gbps,
            "l2_cache_mb": self.config.l2_cache_mb,
            "tdp_w": self.config.tdp_w,
            "is_ascend": self.is_ascend_npu(),
            "is_nvidia": self.is_nvidia_gpu(),
            "is_amd": self.is_amd_gpu(),
        }

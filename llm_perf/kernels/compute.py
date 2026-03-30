"""Compute kernel evaluation."""

from typing import Dict, Optional, Any
import math

from .base import Kernel, KernelConfig, KernelType
from ..hardware.device import Device, ComputeUnitType
from ..utils.constants import DTYPE_SIZES


class ComputeKernel(Kernel):
    """Compute kernel with roofline model support."""
    
    def __init__(
        self,
        config: KernelConfig,
        device: Device,
        flops: int,
        bytes_accessed: int,
        unit_type: ComputeUnitType = ComputeUnitType.CUBE_TENSOR_CORE,
    ):
        super().__init__(config)
        self.device = device
        self.flops = flops
        self.bytes_accessed = bytes_accessed
        self.unit_type = unit_type
    
    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte."""
        if self.bytes_accessed == 0:
            return float('inf')
        return self.flops / self.bytes_accessed
    
    def estimate_time(
        self,
        input_shape: tuple,
        output_shape: tuple,
        dtype: str,
        **kwargs
    ) -> float:
        """
        Estimate execution time using roofline model.
        
        Returns:
            Time in seconds
        """
        # Use measured FLOPS if available
        if self.config.measured_flops:
            return self.flops / self.config.measured_flops
        
        # Otherwise use theoretical roofline with appropriate compute unit
        achievable_flops = self.device.estimate_roofline_flops(
            self.arithmetic_intensity,
            dtype,
            self.unit_type
        )
        
        compute_time = self.flops / achievable_flops
        
        # Add fixed latency if specified
        if self.config.latency_us:
            compute_time += self.config.latency_us * 1e-6
        
        return compute_time
    
    def estimate_memory(
        self,
        input_shape: tuple,
        output_shape: tuple,
        dtype: str,
        **kwargs
    ) -> int:
        """Estimate memory usage."""
        return self.bytes_accessed
    
    def get_unit_type(self) -> ComputeUnitType:
        """Get the compute unit type used by this kernel."""
        return self.unit_type


class ComputeKernelRegistry:
    """Registry for compute kernels."""
    
    def __init__(self, device: Device):
        self.device = device
        self._kernels: Dict[str, ComputeKernel] = {}
        self._register_default_kernels()
    
    def _register_default_kernels(self):
        """Register default compute kernels."""
        # Matmul kernels - use CUBE/Tensor Core
        self._register_matmul_kernels()
        # Attention kernels - use CUBE/Tensor Core
        self._register_attention_kernels()
        # Activation kernels - use VECTOR/CUDA Core
        self._register_activation_kernels()
        # Normalization kernels - use VECTOR/CUDA Core
        self._register_norm_kernels()
    
    def _register_matmul_kernels(self):
        """Register matrix multiplication kernels."""
        # Standard GEMM: C = A @ B
        # FLOPs = 2 * M * N * K
        # Bytes = (M*K + K*N + M*N) * dtype_size
        
        def create_gemm_kernel(m: int, n: int, k: int, dtype: str) -> ComputeKernel:
            flops = 2 * m * n * k
            dtype_size = DTYPE_SIZES.get(dtype, 2)
            bytes_accessed = (m * k + k * n + m * n) * dtype_size
            
            config = KernelConfig(
                name=f"gemm_{m}x{n}x{k}_{dtype}",
                kernel_type=KernelType.COMPUTE,
            )
            # GEMM uses CUBE/Tensor Core
            return ComputeKernel(config, self.device, flops, bytes_accessed, 
                               ComputeUnitType.CUBE_TENSOR_CORE)
        
        # Register some common GEMM sizes
        common_sizes = [
            (1, 4096, 4096),    # Common projection
            (1, 4096, 11008),   # Llama FFN
            (1, 11008, 4096),   # Llama FFN down
            (4096, 4096, 4096), # Batch matmul
            (8192, 4096, 4096), # Larger batch
        ]
        
        for m, n, k in common_sizes:
            for dtype in ["fp16", "bf16", "fp8"]:
                kernel = create_gemm_kernel(m, n, k, dtype)
                self._kernels[kernel.name] = kernel
    
    def _register_attention_kernels(self):
        """Register attention computation kernels."""
        
        def create_flash_attention_kernel(
            batch: int,
            seq_len: int,
            num_heads: int,
            head_dim: int,
            dtype: str
        ) -> ComputeKernel:
            """
            FlashAttention kernel estimation.
            
            FlashAttention is memory-bound but with better arithmetic intensity
            than naive attention due to tiling.
            """
            # FLOPs for attention: QK^T + softmax + PV
            # Each is roughly batch * heads * seq^2 * head_dim
            flops_per_head = 2 * seq_len * seq_len * head_dim * 2  # QK^T + PV
            flops = flops_per_head * batch * num_heads
            
            # Memory access is reduced due to tiling
            # Roughly O(batch * heads * seq * head_dim) for Q,K,V + O(batch * heads * seq^2) for S
            dtype_size = DTYPE_SIZES.get(dtype, 2)
            bytes_accessed = (
                batch * num_heads * seq_len * head_dim * 3 * dtype_size +  # Q, K, V
                batch * num_heads * seq_len * seq_len * 4 +  # Attention scores (fp32)
                batch * num_heads * seq_len * head_dim * dtype_size  # Output
            )
            
            config = KernelConfig(
                name=f"flash_attn_b{batch}_s{seq_len}_h{num_heads}_d{head_dim}_{dtype}",
                kernel_type=KernelType.COMPUTE,
            )
            # Attention uses CUBE/Tensor Core for QK^T and PV matmuls
            return ComputeKernel(config, self.device, flops, bytes_accessed,
                               ComputeUnitType.CUBE_TENSOR_CORE)
        
        # Common attention configurations
        configs = [
            (1, 4096, 32, 128),   # Llama 7B
            (1, 8192, 32, 128),   # Longer sequence
            (1, 32768, 8, 128),   # GQA + long context
            (8, 4096, 32, 128),   # Batch decode
        ]
        
        for batch, seq, heads, dim in configs:
            for dtype in ["fp16", "bf16"]:
                kernel = create_flash_attention_kernel(batch, seq, heads, dim, dtype)
                self._kernels[kernel.name] = kernel
    
    def _register_activation_kernels(self):
        """Register activation function kernels."""
        
        def create_activation_kernel(
            num_elements: int,
            activation: str,
            dtype: str
        ) -> ComputeKernel:
            """
            Activation function kernel.
            
            These are typically memory-bound (read input, write output).
            Uses VECTOR/CUDA Core for element-wise operations.
            """
            dtype_size = DTYPE_SIZES.get(dtype, 2)
            
            # FLOPs per element for different activation functions
            # These are approximate values based on the mathematical operations required
            # Source: Analysis of common activation function implementations in deep learning frameworks
            # References:
            #   - PyTorch ATen implementation: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native
            #   - CUDA Math Library documentation
            #   - "Performance Analysis of Deep Learning Workloads" (various academic papers)
            flops_per_element = {
                # ReLU: max(0, x) - 1 comparison operation
                # Implementation: single compare + conditional move
                "relu": 1,
                
                # GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                # ~10 FLOPs breakdown:
                #   - x^3: 2 multiplies
                #   - 0.044715 * x^3: 1 multiply
                #   - x + ...: 1 add
                #   - sqrt(2/π) * ...: 1 multiply (constant pre-computed)
                #   - tanh: ~4 operations (polynomial approximation or exp-based)
                #   - 1 + tanh(...): 1 add
                #   - 0.5 * x * (...): 2 multiplies
                # Total: ~10 FLOPs per element
                "gelu": 10,
                
                # SiLU (Swish): x * sigmoid(x)
                # ~8 FLOPs breakdown:
                #   - sigmoid(x) = 1 / (1 + exp(-x)): ~6 operations
                #     * -x: 1 negate
                #     * exp(-x): ~4 operations (polynomial approximation or table lookup)
                #     * 1 + exp(...): 1 add
                #     * 1 / (...): 1 reciprocal (or division)
                #   - x * sigmoid(x): 1 multiply
                # Total: ~8 FLOPs per element
                # Reference: Searching for Activation Functions (Ramachandran et al., 2017)
                "silu": 8,
                
                # SwiGLU: Swish-Gated Linear Unit
                # Formula: (x * W) * sigmoid(x * W) * (y * W)
                # In practice: Swish(x) * Linear(y), where Swish = SiLU
                # ~16 FLOPs breakdown:
                #   - Two linear projections (not counted here, done separately)
                #   - SiLU on gate: ~8 FLOPs (same as above)
                #   - Element-wise multiply of gated values: ~8 FLOPs
                #     (including additional scaling/operations in typical implementations)
                # Total: ~16 FLOPs per element
                # Reference: GLU Variants Improve Transformer (Noam Shazeer, 2020)
                "swiglu": 16,
                
                # Softmax: exp(x_i) / sum(exp(x_j)) for each row
                # ~20 FLOPs breakdown per element:
                #   - exp(x_i): ~4 operations per element
                #   - Sum reduction: O(n) adds, amortized ~2 per element
                #   - Division: 1 operation per element
                #   - Plus additional operations for numerical stability (max subtraction)
                # Total: ~20 FLOPs per element (amortized across the row)
                # Note: This is per-element average; actual implementation may vary
                "softmax": 20,
            }.get(activation, 5)
            
            flops = flops_per_element * num_elements
            bytes_accessed = num_elements * dtype_size * 2  # read + write
            
            config = KernelConfig(
                name=f"{activation}_{num_elements}_{dtype}",
                kernel_type=KernelType.COMPUTE,
            )
            # Activations use VECTOR/CUDA Core
            return ComputeKernel(config, self.device, flops, bytes_accessed,
                               ComputeUnitType.VECTOR_CUDA_CORE)
        
        # Common sizes
        sizes = [4096, 11008, 32000, 131072]
        activations = ["relu", "gelu", "silu", "swiglu", "softmax"]
        
        for size in sizes:
            for act in activations:
                for dtype in ["fp16", "bf16"]:
                    kernel = create_activation_kernel(size, act, dtype)
                    self._kernels[kernel.name] = kernel
    
    def _register_norm_kernels(self):
        """Register normalization kernels."""
        
        def create_norm_kernel(
            num_elements: int,
            norm_type: str,  # "layernorm", "rmsnorm"
            dtype: str
        ) -> ComputeKernel:
            """Layer/RMS normalization kernel."""
            dtype_size = DTYPE_SIZES.get(dtype, 2)
            
            # FLOPs for Layer/RMS Normalization
            # These are approximate values based on the mathematical operations required
            # Source: PyTorch LayerNorm/RMSNorm implementation analysis
            # References:
            #   - PyTorch LayerNorm: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp
            #   - LLaMA/RMSNorm: https://github.com/facebookresearch/llama/blob/main/llama/model.py
            #
            # LayerNorm breakdown (~7 FLOPs per element):
            #   - Mean calculation: sum(x) / N - 1 add per element (amortized)
            #   - Variance calculation: sum((x - mean)^2) / N - ~3 ops per element
            #     * (x - mean): 1 subtract
            #     * (x - mean)^2: 1 multiply
            #     * accumulation: 1 add (amortized)
            #   - Normalization: (x - mean) / sqrt(var + eps) - ~2 ops
            #     * (x - mean): 1 subtract
            #     * division by std: 1 divide (or reciprocal + multiply)
            #   - Scale and shift: x * gamma + beta - 2 ops (only for LayerNorm)
            # RMSNorm (~5 FLOPs per element) - simpler, no mean subtraction:
            #   - RMS calculation: sqrt(sum(x^2) / N) - ~3 ops per element
            #   - Normalization: x / RMS - ~2 ops
            #
            # Using average of 7 as a reasonable estimate for both variants
            flops = num_elements * 7
            bytes_accessed = num_elements * dtype_size * 2
            
            config = KernelConfig(
                name=f"{norm_type}_{num_elements}_{dtype}",
                kernel_type=KernelType.COMPUTE,
            )
            # Normalization uses VECTOR/CUDA Core
            return ComputeKernel(config, self.device, flops, bytes_accessed,
                               ComputeUnitType.VECTOR_CUDA_CORE)
        
        sizes = [4096, 5120, 6144, 8192]
        norms = ["layernorm", "rmsnorm"]
        
        for size in sizes:
            for norm in norms:
                for dtype in ["fp16", "bf16"]:
                    kernel = create_norm_kernel(size, norm, dtype)
                    self._kernels[kernel.name] = kernel
    
    def get(self, name: str) -> Optional[ComputeKernel]:
        """Get a kernel by name."""
        return self._kernels.get(name)
    
    def get_or_create_matmul(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str
    ) -> ComputeKernel:
        """Get or create a matmul kernel for specific dimensions."""
        name = f"gemm_{m}x{n}x{k}_{dtype}"
        if name in self._kernels:
            return self._kernels[name]
        
        # Create new kernel
        flops = 2 * m * n * k
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        bytes_accessed = (m * k + k * n + m * n) * dtype_size
        
        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMPUTE,
        )
        # GEMM uses CUBE/Tensor Core
        kernel = ComputeKernel(config, self.device, flops, bytes_accessed,
                             ComputeUnitType.CUBE_TENSOR_CORE)
        self._kernels[name] = kernel
        return kernel
    
    def list_kernels(self) -> list:
        """List all registered kernel names."""
        return list(self._kernels.keys())

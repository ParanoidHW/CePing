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
        # Conv kernels - use CUBE/Tensor Core
        self._register_conv_kernels()
    
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
    
    def _register_conv_kernels(self):
        """Register convolution kernels for CNN and video workloads."""
        self._register_conv2d_kernels()
        self._register_conv3d_kernels()

    def _register_conv2d_kernels(self):
        """Register 2D convolution kernels for image CNN workloads."""

        def create_conv2d_kernel(
            batch: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            input_h: int,
            input_w: int,
            stride: int,
            padding: int,
            dtype: str,
            groups: int = 1,
        ) -> ComputeKernel:
            """
            Conv2d kernel estimation with detailed memory access breakdown.

            FLOPs = 2 * batch * out_h * out_w * out_channels * kernel_size^2 * in_channels / groups
            Memory = input + weights + output + (workspace for im2col if needed)

            Memory Access Pattern:
            - Input activation: batch * in_c * in_h * in_w * dtype_size
            - Weights: out_c * in_c/groups * k^2 * dtype_size
            - Output activation: batch * out_c * out_h * out_w * dtype_size
            - Workspace (im2col): batch * out_h * out_w * k^2 * in_c * dtype_size (if not implicit gemm)
            """
            # Calculate output dimensions
            out_h = (input_h + 2 * padding - kernel_size) // stride + 1
            out_w = (input_w + 2 * padding - kernel_size) // stride + 1

            # FLOPs calculation
            # Each output pixel requires: kernel_size^2 * (in_channels/groups) mults and adds
            flops = (
                2 * batch * out_h * out_w *
                out_channels * kernel_size * kernel_size * (in_channels // groups)
            )

            dtype_size = DTYPE_SIZES.get(dtype, 2)

            # Detailed memory access breakdown
            input_bytes = batch * in_channels * input_h * input_w * dtype_size
            weight_bytes = (
                out_channels * (in_channels // groups) *
                kernel_size * kernel_size * dtype_size
            )
            output_bytes = batch * out_channels * out_h * out_w * dtype_size

            # Workspace for im2col transformation (when not using implicit GEMM)
            # This is the expanded column matrix for convolution as GEMM
            workspace_bytes = batch * out_h * out_w * kernel_size * kernel_size * in_channels * dtype_size

            # Total memory access includes read+write for each buffer
            # Input: read once (or multiple times if cache miss)
            # Weights: read once
            # Output: write
            # Workspace: read+write (temporary)
            bytes_accessed = (
                input_bytes * 1.0 +      # Input read
                weight_bytes * 1.0 +     # Weight read
                output_bytes * 1.0 +     # Output write
                workspace_bytes * 0.5    # Partial workspace access (depends on algorithm)
            )

            config = KernelConfig(
                name=f"conv2d_b{batch}_ic{in_channels}_oc{out_channels}_"
                     f"k{kernel_size}_h{input_h}_w{input_w}_s{stride}_g{groups}_{dtype}",
                kernel_type=KernelType.COMPUTE,
            )
            # Conv2d uses CUBE/Tensor Core
            return ComputeKernel(config, self.device, int(flops), int(bytes_accessed),
                               ComputeUnitType.CUBE_TENSOR_CORE)

        # Common Conv2d configurations (ResNet-like)
        # Using keyword args similar to torch.nn.Conv2d for better readability
        configs = [
            # ResNet-18/34 early layers
            dict(batch=1, in_channels=3, out_channels=64, kernel_size=7,
                 input_size=(224, 224), stride=2, padding=3),      # Initial conv
            dict(batch=1, in_channels=64, out_channels=64, kernel_size=3,
                 input_size=(56, 56), stride=1, padding=1),        # Layer 1
            dict(batch=1, in_channels=64, out_channels=128, kernel_size=3,
                 input_size=(56, 56), stride=2, padding=1),        # Layer 2 downsample
            dict(batch=1, in_channels=128, out_channels=128, kernel_size=3,
                 input_size=(28, 28), stride=1, padding=1),        # Layer 2
            dict(batch=1, in_channels=128, out_channels=256, kernel_size=3,
                 input_size=(28, 28), stride=2, padding=1),        # Layer 3 downsample
            dict(batch=1, in_channels=256, out_channels=256, kernel_size=3,
                 input_size=(14, 14), stride=1, padding=1),        # Layer 3
            dict(batch=1, in_channels=256, out_channels=512, kernel_size=3,
                 input_size=(14, 14), stride=2, padding=1),        # Layer 4 downsample
            dict(batch=1, in_channels=512, out_channels=512, kernel_size=3,
                 input_size=(7, 7), stride=1, padding=1),          # Layer 4
            # ResNet-50/101/152 bottleneck (1x1, 3x3, 1x1)
            dict(batch=1, in_channels=64, out_channels=64, kernel_size=1,
                 input_size=(56, 56), stride=1, padding=0),        # Bottleneck 1x1
            dict(batch=1, in_channels=64, out_channels=64, kernel_size=3,
                 input_size=(56, 56), stride=1, padding=1),        # Bottleneck 3x3
            dict(batch=1, in_channels=64, out_channels=256, kernel_size=1,
                 input_size=(56, 56), stride=1, padding=0),        # Bottleneck 1x1 expand
            dict(batch=1, in_channels=256, out_channels=512, kernel_size=1,
                 input_size=(56, 56), stride=2, padding=0),        # Bottleneck downsample
            # Batch sizes > 1
            dict(batch=8, in_channels=64, out_channels=64, kernel_size=3,
                 input_size=(56, 56), stride=1, padding=1),
            dict(batch=32, in_channels=64, out_channels=64, kernel_size=3,
                 input_size=(56, 56), stride=1, padding=1),
        ]

        for cfg in configs:
            batch = cfg["batch"]
            in_c = cfg["in_channels"]
            out_c = cfg["out_channels"]
            k = cfg["kernel_size"]
            h, w = cfg["input_size"]
            s = cfg["stride"]
            p = cfg["padding"]
            for dtype in ["fp16", "bf16", "fp32"]:
                kernel = create_conv2d_kernel(batch, in_c, out_c, k, h, w, s, p, dtype)
                self._kernels[kernel.name] = kernel

    def _register_conv3d_kernels(self):
        """Register 3D convolution kernels for video VAE workloads."""

        def create_conv3d_kernel(
            batch: int,
            in_channels: int,
            out_channels: int,
            kernel_size_t: int,
            kernel_size_h: int,
            kernel_size_w: int,
            input_t: int,
            input_h: int,
            input_w: int,
            stride_t: int,
            stride_h: int,
            stride_w: int,
            padding_t: int,
            padding_h: int,
            padding_w: int,
            dtype: str,
        ) -> ComputeKernel:
            """
            Conv3d kernel estimation for video processing.

            FLOPs = 2 * batch * out_t * out_h * out_w * out_c * k_t * k_h * k_w * in_c
            Memory = input + weights + output + workspace

            Memory Access Pattern for 3D Conv:
            - Input activation: batch * in_c * in_t * in_h * in_w * dtype_size
            - Weights: out_c * in_c * k_t * k_h * k_w * dtype_size
            - Output activation: batch * out_c * out_t * out_h * out_w * dtype_size
            - Workspace: batch * out_t * out_h * out_w * k_t * k_h * k_w * in_c * dtype_size

            Note: 3D conv is memory-bound due to large workspace requirements for
            unfolding the 3D kernel into columns.
            """
            # Calculate output dimensions
            out_t = (input_t + 2 * padding_t - kernel_size_t) // stride_t + 1
            out_h = (input_h + 2 * padding_h - kernel_size_h) // stride_h + 1
            out_w = (input_w + 2 * padding_w - kernel_size_w) // stride_w + 1

            # FLOPs calculation
            flops = (
                2 * batch * out_t * out_h * out_w *
                out_channels * kernel_size_t * kernel_size_h * kernel_size_w *
                in_channels
            )

            dtype_size = DTYPE_SIZES.get(dtype, 2)

            # Detailed memory access breakdown
            input_bytes = batch * in_channels * input_t * input_h * input_w * dtype_size
            weight_bytes = (
                out_channels * in_channels *
                kernel_size_t * kernel_size_h * kernel_size_w * dtype_size
            )
            output_bytes = batch * out_channels * out_t * out_h * out_w * dtype_size

            # Workspace for 3D im2col is significantly larger than 2D
            workspace_bytes = (
                batch * out_t * out_h * out_w *
                kernel_size_t * kernel_size_h * kernel_size_w *
                in_channels * dtype_size
            )

            # Total memory access with 3D conv overhead
            # 3D conv has higher memory pressure due to temporal dimension
            bytes_accessed = (
                input_bytes * 1.2 +      # Input read (higher due to temporal overlap)
                weight_bytes * 1.0 +     # Weight read
                output_bytes * 1.0 +     # Output write
                workspace_bytes * 0.8    # Higher workspace usage for 3D unfold
            )

            config = KernelConfig(
                name=f"conv3d_b{batch}_ic{in_channels}_oc{out_channels}_"
                     f"kt{kernel_size_t}_kh{kernel_size_h}_kw{kernel_size_w}_"
                     f"t{input_t}_h{input_h}_w{input_w}_"
                     f"st{stride_t}_sh{stride_h}_sw{stride_w}_{dtype}",
                kernel_type=KernelType.COMPUTE,
            )
            # Conv3d uses CUBE/Tensor Core
            return ComputeKernel(config, self.device, int(flops), int(bytes_accessed),
                               ComputeUnitType.CUBE_TENSOR_CORE)

        # Common Conv3d configurations for Video VAE
        # Using keyword args similar to torch.nn.Conv3d for better readability
        configs = [
            # Video VAE Encoder - downsample blocks
            # Initial conv: 3 channels (RGB) to 128, temporal stride 1
            dict(batch=1, in_channels=3, out_channels=128,
                 kernel_size=(3, 3, 3), input_size=(16, 256, 256),
                 stride=(1, 1, 1), padding=(1, 1, 1)),
            # Downsample 1: 128 -> 256, spatial stride 2
            dict(batch=1, in_channels=128, out_channels=256,
                 kernel_size=(3, 3, 3), input_size=(16, 256, 256),
                 stride=(1, 2, 2), padding=(1, 1, 1)),
            # Downsample 2: 256 -> 512, spatial stride 2
            dict(batch=1, in_channels=256, out_channels=512,
                 kernel_size=(3, 3, 3), input_size=(16, 128, 128),
                 stride=(1, 2, 2), padding=(1, 1, 1)),
            # Temporal downsample: 512 -> 512, temporal stride 2
            dict(batch=1, in_channels=512, out_channels=512,
                 kernel_size=(3, 3, 3), input_size=(16, 64, 64),
                 stride=(2, 1, 1), padding=(1, 1, 1)),

            # Video VAE Decoder - upsample blocks
            # Upsample 1: 512 -> 512, spatial stride 1 with upsampling
            dict(batch=1, in_channels=512, out_channels=512,
                 kernel_size=(3, 3, 3), input_size=(8, 32, 32),
                 stride=(1, 1, 1), padding=(1, 1, 1)),
            # Upsample 2: 512 -> 256, spatial upsampling
            dict(batch=1, in_channels=512, out_channels=256,
                 kernel_size=(3, 3, 3), input_size=(8, 64, 64),
                 stride=(1, 1, 1), padding=(1, 1, 1)),
            # Upsample 3: 256 -> 128, spatial upsampling
            dict(batch=1, in_channels=256, out_channels=128,
                 kernel_size=(3, 3, 3), input_size=(8, 128, 128),
                 stride=(1, 1, 1), padding=(1, 1, 1)),
            # Final conv: 128 -> 3, temporal stride 1
            dict(batch=1, in_channels=128, out_channels=3,
                 kernel_size=(3, 3, 3), input_size=(8, 256, 256),
                 stride=(1, 1, 1), padding=(1, 1, 1)),

            # Common video resolutions
            # 512x512 spatial with different temporal lengths
            dict(batch=1, in_channels=512, out_channels=512,
                 kernel_size=(3, 3, 3), input_size=(8, 64, 64),
                 stride=(1, 1, 1), padding=(1, 1, 1)),
            dict(batch=1, in_channels=512, out_channels=512,
                 kernel_size=(3, 3, 3), input_size=(16, 64, 64),
                 stride=(1, 1, 1), padding=(1, 1, 1)),
            dict(batch=4, in_channels=512, out_channels=512,
                 kernel_size=(3, 3, 3), input_size=(16, 64, 64),
                 stride=(1, 1, 1), padding=(1, 1, 1)),
        ]

        for cfg in configs:
            batch = cfg["batch"]
            ic = cfg["in_channels"]
            oc = cfg["out_channels"]
            kt, kh, kw = cfg["kernel_size"]
            in_t, in_h, in_w = cfg["input_size"]
            st, sh, sw = cfg["stride"]
            pt, ph, pw = cfg["padding"]
            for dtype in ["fp16", "bf16"]:
                kernel = create_conv3d_kernel(
                    batch, ic, oc, kt, kh, kw, in_t, in_h, in_w,
                    st, sh, sw, pt, ph, pw, dtype
                )
                self._kernels[kernel.name] = kernel

    def get_or_create_conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        batch: int = 1,
        input_size: tuple = (224, 224),
        dtype: str = "fp16",
    ) -> ComputeKernel:
        """Get or create a Conv2d kernel for specific dimensions.

        Interface aligned with torch.nn.Conv2d:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolving kernel
            stride: Stride of convolution (default: 1)
            padding: Zero-padding added to both sides (default: 0)
            groups: Number of blocked connections (default: 1)
            batch: Batch size (default: 1)
            input_size: Input spatial dimensions (H, W) (default: (224, 224))
            dtype: Data type (default: "fp16")

        Returns:
            ComputeKernel for 2D convolution
        """
        input_h, input_w = input_size
        name = f"conv2d_b{batch}_ic{in_channels}_oc{out_channels}_" \
               f"k{kernel_size}_h{input_h}_w{input_w}_s{stride}_g{groups}_{dtype}"
        if name in self._kernels:
            return self._kernels[name]

        # Create new kernel
        out_h = (input_h + 2 * padding - kernel_size) // stride + 1
        out_w = (input_w + 2 * padding - kernel_size) // stride + 1
        flops = (
            2 * batch * out_h * out_w *
            out_channels * kernel_size * kernel_size * (in_channels // groups)
        )
        dtype_size = DTYPE_SIZES.get(dtype, 2)
        input_bytes = batch * in_channels * input_h * input_w * dtype_size
        weight_bytes = out_channels * (in_channels // groups) * kernel_size * kernel_size * dtype_size
        output_bytes = batch * out_channels * out_h * out_w * dtype_size
        workspace_bytes = batch * out_h * out_w * kernel_size * kernel_size * in_channels * dtype_size
        bytes_accessed = (
            input_bytes * 1.0 +
            weight_bytes * 1.0 +
            output_bytes * 1.0 +
            workspace_bytes * 0.5
        )

        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMPUTE,
        )
        kernel = ComputeKernel(config, self.device, int(flops), int(bytes_accessed),
                             ComputeUnitType.CUBE_TENSOR_CORE)
        self._kernels[name] = kernel
        return kernel

    def get_or_create_conv3d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        batch: int = 1,
        input_size: tuple = (16, 224, 224),
        dtype: str = "fp16",
    ) -> ComputeKernel:
        """Get or create a Conv3d kernel for video processing.

        Interface aligned with torch.nn.Conv3d:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolving kernel (T, H, W)
            stride: Stride of convolution (T, H, W) (default: (1, 1, 1))
            padding: Zero-padding added to both sides (T, H, W) (default: (0, 0, 0))
            batch: Batch size (default: 1)
            input_size: Input dimensions (T, H, W) (default: (16, 224, 224))
            dtype: Data type (default: "fp16")

        Returns:
            ComputeKernel for 3D convolution
        """
        kernel_size_t, kernel_size_h, kernel_size_w = kernel_size
        input_t, input_h, input_w = input_size
        stride_t, stride_h, stride_w = stride
        padding_t, padding_h, padding_w = padding

        name = f"conv3d_b{batch}_ic{in_channels}_oc{out_channels}_" \
               f"kt{kernel_size_t}_kh{kernel_size_h}_kw{kernel_size_w}_" \
               f"t{input_t}_h{input_h}_w{input_w}_" \
               f"st{stride_t}_sh{stride_h}_sw{stride_w}_{dtype}"
        if name in self._kernels:
            return self._kernels[name]

        # Create new kernel
        out_t = (input_t + 2 * padding_t - kernel_size_t) // stride_t + 1
        out_h = (input_h + 2 * padding_h - kernel_size_h) // stride_h + 1
        out_w = (input_w + 2 * padding_w - kernel_size_w) // stride_w + 1

        flops = (
            2 * batch * out_t * out_h * out_w *
            out_channels * kernel_size_t * kernel_size_h * kernel_size_w *
            in_channels
        )

        dtype_size = DTYPE_SIZES.get(dtype, 2)
        input_bytes = batch * in_channels * input_t * input_h * input_w * dtype_size
        weight_bytes = (
            out_channels * in_channels *
            kernel_size_t * kernel_size_h * kernel_size_w * dtype_size
        )
        output_bytes = batch * out_channels * out_t * out_h * out_w * dtype_size
        workspace_bytes = (
            batch * out_t * out_h * out_w *
            kernel_size_t * kernel_size_h * kernel_size_w *
            in_channels * dtype_size
        )

        bytes_accessed = (
            input_bytes * 1.2 +
            weight_bytes * 1.0 +
            output_bytes * 1.0 +
            workspace_bytes * 0.8
        )

        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMPUTE,
        )
        kernel = ComputeKernel(config, self.device, int(flops), int(bytes_accessed),
                             ComputeUnitType.CUBE_TENSOR_CORE)
        self._kernels[name] = kernel
        return kernel

    def list_kernels(self) -> list:
        """List all registered kernel names."""
        return list(self._kernels.keys())

    def get_or_create_activation(
        self,
        num_elements: int,
        activation: str,
        dtype: str
    ) -> ComputeKernel:
        """
        Get or create an activation kernel for specific parameters.

        Args:
            num_elements: Number of elements to process
            activation: Activation function name ("relu", "gelu", "silu", "swiglu", "softmax")
            dtype: Data type (e.g., "fp16", "bf16")

        Returns:
            ComputeKernel for the activation function
        """
        name = f"{activation}_{num_elements}_{dtype}"
        if name in self._kernels:
            return self._kernels[name]

        # Create new kernel
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        # FLOPs per element for different activation functions
        flops_per_element = {
            "relu": 1,
            "gelu": 10,
            "silu": 8,
            "swiglu": 16,
            "softmax": 20,
        }.get(activation, 5)

        flops = flops_per_element * num_elements
        bytes_accessed = num_elements * dtype_size * 2  # read + write

        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMPUTE,
        )
        # Activations use VECTOR/CUDA Core
        kernel = ComputeKernel(config, self.device, flops, bytes_accessed,
                             ComputeUnitType.VECTOR_CUDA_CORE)
        self._kernels[name] = kernel
        return kernel

    def get_or_create_norm(
        self,
        num_elements: int,
        norm_type: str,
        dtype: str
    ) -> ComputeKernel:
        """
        Get or create a normalization kernel for specific parameters.

        Args:
            num_elements: Number of elements to process
            norm_type: Normalization type ("layernorm", "rmsnorm")
            dtype: Data type (e.g., "fp16", "bf16")

        Returns:
            ComputeKernel for the normalization operation
        """
        name = f"{norm_type}_{num_elements}_{dtype}"
        if name in self._kernels:
            return self._kernels[name]

        # Create new kernel
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        # FLOPs for Layer/RMS Normalization (~7 FLOPs per element)
        flops = num_elements * 7
        bytes_accessed = num_elements * dtype_size * 2

        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMPUTE,
        )
        # Normalization uses VECTOR/CUDA Core
        kernel = ComputeKernel(config, self.device, flops, bytes_accessed,
                             ComputeUnitType.VECTOR_CUDA_CORE)
        self._kernels[name] = kernel
        return kernel

    def get_or_create_softmax(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        dtype: str
    ) -> ComputeKernel:
        """
        Get or create a softmax kernel for specific parameters.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of attention heads (used to calculate total elements)
            dtype: Data type (e.g., "fp16", "bf16")

        Returns:
            ComputeKernel for the softmax operation
        """
        num_elements = batch_size * seq_len * num_heads
        name = f"softmax_{num_elements}_{dtype}"
        if name in self._kernels:
            return self._kernels[name]

        # Create new kernel
        dtype_size = DTYPE_SIZES.get(dtype, 2)

        # Softmax: ~20 FLOPs per element
        flops = num_elements * 20
        bytes_accessed = num_elements * dtype_size * 2  # read + write

        config = KernelConfig(
            name=name,
            kernel_type=KernelType.COMPUTE,
        )
        # Softmax uses VECTOR/CUDA Core
        kernel = ComputeKernel(config, self.device, flops, bytes_accessed,
                             ComputeUnitType.VECTOR_CUDA_CORE)
        self._kernels[name] = kernel
        return kernel

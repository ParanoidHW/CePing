"""Tests for compute kernel module."""

import unittest
from llm_perf.hardware.device import Device, ComputeUnitType
from llm_perf.kernels.compute import ComputeKernelRegistry, ComputeKernel
from llm_perf.kernels.base import KernelConfig, KernelType


class TestComputeKernel(unittest.TestCase):
    """Test cases for ComputeKernel class."""
    
    def test_gemm_uses_cube_core(self):
        """Test that GEMM kernels use CUBE/Tensor Core."""
        device = Device.from_preset("Ascend-910B2")
        registry = ComputeKernelRegistry(device)
        
        kernel = registry.get_or_create_matmul(4096, 4096, 4096, "fp16")
        
        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.get_unit_type(), ComputeUnitType.CUBE_TENSOR_CORE)
    
    def test_activation_uses_vector_core(self):
        """Test that activation kernels use VECTOR/CUDA Core."""
        device = Device.from_preset("Ascend-910B2")
        registry = ComputeKernelRegistry(device)
        
        kernel = registry.get("relu_4096_fp16")
        
        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.get_unit_type(), ComputeUnitType.VECTOR_CUDA_CORE)
    
    def test_norm_uses_vector_core(self):
        """Test that normalization kernels use VECTOR/CUDA Core."""
        device = Device.from_preset("Ascend-910B2")
        registry = ComputeKernelRegistry(device)
        
        kernel = registry.get("rmsnorm_4096_fp16")
        
        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.get_unit_type(), ComputeUnitType.VECTOR_CUDA_CORE)
    
    def test_gemm_time_estimate_ascend(self):
        """Test GEMM time estimation on Ascend NPU."""
        device = Device.from_preset("Ascend-910B2")
        registry = ComputeKernelRegistry(device)
        
        # Small batch GEMM (memory bound)
        kernel = registry.get_or_create_matmul(1, 4096, 4096, "fp16")
        time = kernel.estimate_time((1, 4096), (1, 4096), "fp16")
        
        # Should be positive and reasonable
        self.assertGreater(time, 0)
        
        # Calculate expected: memory bound for small batch
        bytes_accessed = (1 * 4096 + 4096 * 4096 + 1 * 4096) * 2
        mem_bw = device.get_memory_bw_gbps() * 1e9
        expected_time = bytes_accessed / mem_bw
        
        # Should be close to memory-bound estimate
        self.assertAlmostEqual(time, expected_time, delta=expected_time * 0.5)
    
    def test_gemm_time_estimate_nvidia(self):
        """Test GEMM time estimation on NVIDIA GPU."""
        device = Device.from_preset("H100-SXM-80GB")
        registry = ComputeKernelRegistry(device)
        
        kernel = registry.get_or_create_matmul(4096, 4096, 4096, "fp16")
        time = kernel.estimate_time((4096, 4096), (4096, 4096), "fp16")
        
        self.assertGreater(time, 0)
    
    def test_arithmetic_intensity(self):
        """Test arithmetic intensity calculation."""
        device = Device.from_preset("Ascend-910B2")
        
        # Create a kernel with known FLOPs and bytes
        config = KernelConfig(name="test", kernel_type=KernelType.COMPUTE)
        kernel = ComputeKernel(config, device, flops=1000, bytes_accessed=100)
        
        self.assertEqual(kernel.arithmetic_intensity, 10.0)
    
    def test_arithmetic_intensity_zero_bytes(self):
        """Test arithmetic intensity with zero bytes accessed."""
        device = Device.from_preset("Ascend-910B2")
        
        config = KernelConfig(name="test", kernel_type=KernelType.COMPUTE)
        kernel = ComputeKernel(config, device, flops=1000, bytes_accessed=0)
        
        self.assertEqual(kernel.arithmetic_intensity, float('inf'))
    
    def test_attention_kernel_exists(self):
        """Test that attention kernels are registered."""
        device = Device.from_preset("Ascend-910B2")
        registry = ComputeKernelRegistry(device)
        
        kernel = registry.get("flash_attn_b1_s4096_h32_d128_fp16")
        
        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.get_unit_type(), ComputeUnitType.CUBE_TENSOR_CORE)
    
    def test_registry_list_kernels(self):
        """Test listing registered kernels."""
        device = Device.from_preset("Ascend-910B2")
        registry = ComputeKernelRegistry(device)
        
        kernels = registry.list_kernels()
        
        self.assertGreater(len(kernels), 0)
        self.assertIn("gemm_4096x4096x4096_fp16", kernels)
        self.assertIn("relu_4096_fp16", kernels)
        self.assertIn("rmsnorm_4096_fp16", kernels)


class TestComputeKernelPerformanceComparison(unittest.TestCase):
    """Test performance comparison between different devices."""
    
    def test_ascend_vs_nvidia_gemm(self):
        """Compare GEMM performance on Ascend vs NVIDIA."""
        ascend = Device.from_preset("Ascend-910B2")
        nvidia = Device.from_preset("H100-SXM-80GB")
        
        ascend_registry = ComputeKernelRegistry(ascend)
        nvidia_registry = ComputeKernelRegistry(nvidia)
        
        # Large GEMM (compute bound)
        ascend_kernel = ascend_registry.get_or_create_matmul(4096, 4096, 4096, "fp16")
        nvidia_kernel = nvidia_registry.get_or_create_matmul(4096, 4096, 4096, "fp16")
        
        ascend_time = ascend_kernel.estimate_time((4096, 4096), (4096, 4096), "fp16")
        nvidia_time = nvidia_kernel.estimate_time((4096, 4096), (4096, 4096), "fp16")
        
        # Both should have positive time
        self.assertGreater(ascend_time, 0)
        self.assertGreater(nvidia_time, 0)
        
        # NVIDIA H100 should be faster than Ascend 910B2 for compute-bound GEMM
        # (H100: 989 TFLOPS vs 910B2: 376 TFLOPS)
        self.assertLess(nvidia_time, ascend_time)
    
    def test_ascend_cube_vs_vector_performance(self):
        """Test that CUBE is faster than VECTOR for matrix ops on Ascend."""
        device = Device.from_preset("Ascend-910B2")
        
        # Same FLOPs but different compute units
        config = KernelConfig(name="test", kernel_type=KernelType.COMPUTE)
        
        cube_kernel = ComputeKernel(
            config, device, flops=1e12, bytes_accessed=1e9,
            unit_type=ComputeUnitType.CUBE_TENSOR_CORE
        )
        vector_kernel = ComputeKernel(
            config, device, flops=1e12, bytes_accessed=1e9,
            unit_type=ComputeUnitType.VECTOR_CUDA_CORE
        )
        
        cube_time = cube_kernel.estimate_time((1,), (1,), "fp16")
        vector_time = vector_kernel.estimate_time((1,), (1,), "fp16")
        
        # CUBE should be faster (10x more TFLOPS)
        self.assertLess(cube_time, vector_time)
        self.assertLess(cube_time * 5, vector_time)  # At least 5x faster


if __name__ == "__main__":
    unittest.main()

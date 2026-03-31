"""Tests for device module."""

import unittest
from llm_perf.hardware.device import Device, DeviceConfig, ComputeUnitType


class TestDevice(unittest.TestCase):
    """Test cases for Device class."""
    
    def test_nvidia_presets_available(self):
        """Test that NVIDIA GPU presets are available."""
        presets = ["H100-SXM-80GB", "A100-SXM-80GB", "MI300X", "L40S"]
        for preset in presets:
            device = Device.from_preset(preset)
            self.assertIsNotNone(device)
            self.assertEqual(device.config.name, preset)
    
    def test_ascend_presets_available(self):
        """Test that Ascend NPU presets are available."""
        presets = [
            "Ascend-910A", "Ascend-910B1", "Ascend-910B2", "Ascend-910B3",
            "Ascend-910B4", "Ascend-910C", "Ascend-950-DT", "Ascend-950-PR",
            "Ascend-960", "Ascend-970", "Ascend-310P"
        ]
        for preset in presets:
            device = Device.from_preset(preset)
            self.assertIsNotNone(device)
            self.assertEqual(device.config.name, preset)
    
    def test_cube_vector_separation(self):
        """Test that CUBE and VECTOR compute units are properly separated."""
        # Test Ascend 910B2 which has distinct CUBE and VECTOR values
        device = Device.from_preset("Ascend-910B2")
        
        # CUBE should be much higher than VECTOR (typically 10x)
        cube_fp16 = device.get_compute_tflops("fp16", ComputeUnitType.CUBE_TENSOR_CORE)
        vector_fp16 = device.get_compute_tflops("fp16", ComputeUnitType.VECTOR_CUDA_CORE)
        
        self.assertGreater(cube_fp16, vector_fp16)
        self.assertGreater(cube_fp16 / vector_fp16, 5)  # At least 5x difference
    
    def test_ascend_cube_vector_values(self):
        """Test specific CUBE/VECTOR values for Ascend devices."""
        test_cases = [
            ("Ascend-910B1", "fp16", 414.0, 41.4),
            ("Ascend-910B2", "fp16", 376.0, 37.6),
            ("Ascend-910B3", "fp16", 313.0, 31.3),
            ("Ascend-910B4", "fp16", 280.0, 28.0),
            ("Ascend-910C", "fp16", 800.0, 80.0),
        ]
        
        for preset, dtype, expected_cube, expected_vector in test_cases:
            device = Device.from_preset(preset)
            
            cube = device.get_compute_tflops(dtype, ComputeUnitType.CUBE_TENSOR_CORE)
            vector = device.get_compute_tflops(dtype, ComputeUnitType.VECTOR_CUDA_CORE)
            
            self.assertAlmostEqual(cube, expected_cube, places=1,
                                 msg=f"{preset} CUBE {dtype} mismatch")
            self.assertAlmostEqual(vector, expected_vector, places=1,
                                 msg=f"{preset} VECTOR {dtype} mismatch")
    
    def test_nvidia_cube_vector_separation(self):
        """Test that NVIDIA GPUs have proper CUBE/VECTOR separation."""
        # Test H100 - should have significant difference
        h100 = Device.from_preset("H100-SXM-80GB")
        
        cube_fp16 = h100.get_compute_tflops("fp16", ComputeUnitType.CUBE_TENSOR_CORE)
        vector_fp16 = h100.get_compute_tflops("fp16", ComputeUnitType.VECTOR_CUDA_CORE)
        
        # Tensor Core should be much faster than CUDA Core
        # H100: Tensor Core 989T vs CUDA Core 134T (~7.4x difference)
        self.assertGreater(cube_fp16, vector_fp16)
        self.assertGreater(cube_fp16 / vector_fp16, 5)  # At least 5x difference
        self.assertEqual(cube_fp16, 989.0)
        self.assertEqual(vector_fp16, 134.0)
        
        # Test A100
        a100 = Device.from_preset("A100-SXM-80GB")
        a100_cube = a100.get_compute_tflops("fp16", ComputeUnitType.CUBE_TENSOR_CORE)
        a100_vector = a100.get_compute_tflops("fp16", ComputeUnitType.VECTOR_CUDA_CORE)
        
        # A100: Tensor Core 312T vs CUDA Core 39T (~8x difference)
        self.assertEqual(a100_cube, 312.0)
        self.assertEqual(a100_vector, 39.0)
        self.assertGreater(a100_cube / a100_vector, 7)
    
    def test_vector_int8_fp8_fallback_to_fp16(self):
        """Test that VECTOR cores fallback to FP16 for INT8/FP8 (not used for activations)."""
        device = Device.from_preset("H100-SXM-80GB")
        
        # For VECTOR cores, INT8/FP8 should fallback to FP16 value
        # because activation/normalization don't use low-precision
        vector_fp16 = device.get_compute_tflops("fp16", ComputeUnitType.VECTOR_CUDA_CORE)
        vector_fp8 = device.get_compute_tflops("fp8", ComputeUnitType.VECTOR_CUDA_CORE)
        vector_int8 = device.get_compute_tflops("int8", ComputeUnitType.VECTOR_CUDA_CORE)
        vector_int4 = device.get_compute_tflops("int4", ComputeUnitType.VECTOR_CUDA_CORE)
        
        # All should equal FP16 value for VECTOR cores
        self.assertEqual(vector_fp8, vector_fp16)
        self.assertEqual(vector_int8, vector_fp16)
        self.assertEqual(vector_int4, vector_fp16)
        
        # But CUBE cores should have different values
        cube_fp16 = device.get_compute_tflops("fp16", ComputeUnitType.CUBE_TENSOR_CORE)
        cube_fp8 = device.get_compute_tflops("fp8", ComputeUnitType.CUBE_TENSOR_CORE)
        
        # H100 Tensor Core: FP8 should be 2x FP16
        self.assertAlmostEqual(cube_fp8, 2 * cube_fp16, delta=1.0)
    
    def test_backward_compatibility(self):
        """Test that legacy code without CUBE/VECTOR still works."""
        # Create device with only legacy fields
        config = DeviceConfig(
            name="Legacy-GPU",
            fp16_tflops=100.0,
            memory_gb=80.0,
            memory_bandwidth_gbps=1000.0,
        )
        device = Device(config)
        
        # Both CUBE and VECTOR should return the legacy value
        cube = device.get_compute_tflops("fp16", ComputeUnitType.CUBE_TENSOR_CORE)
        vector = device.get_compute_tflops("fp16", ComputeUnitType.VECTOR_CUDA_CORE)
        
        self.assertEqual(cube, 100.0)
        self.assertEqual(vector, 100.0)
    
    def test_device_type_detection(self):
        """Test device type detection methods."""
        ascend = Device.from_preset("Ascend-910B2")
        self.assertTrue(ascend.is_ascend_npu())
        self.assertFalse(ascend.is_nvidia_gpu())
        self.assertFalse(ascend.is_amd_gpu())
        
        nvidia = Device.from_preset("H100-SXM-80GB")
        self.assertFalse(nvidia.is_ascend_npu())
        self.assertTrue(nvidia.is_nvidia_gpu())
        self.assertFalse(nvidia.is_amd_gpu())
        
        amd = Device.from_preset("MI300X")
        self.assertFalse(amd.is_ascend_npu())
        self.assertFalse(amd.is_nvidia_gpu())
        self.assertTrue(amd.is_amd_gpu())
    
    def test_roofline_model(self):
        """Test roofline model calculations."""
        device = Device.from_preset("Ascend-910B2")
        
        # Test memory-bound case (low arithmetic intensity)
        ai_low = 1.0  # FLOPs/byte
        flops_low = device.estimate_roofline_flops(ai_low, "fp16", 
                                                    ComputeUnitType.CUBE_TENSOR_CORE)
        mem_bw = device.get_memory_bw_gbps() * 1e9
        expected_low = ai_low * mem_bw
        self.assertAlmostEqual(flops_low, expected_low, delta=1e6)
        
        # Test compute-bound case (high arithmetic intensity)
        ai_high = 10000.0
        flops_high = device.estimate_roofline_flops(ai_high, "fp16",
                                                     ComputeUnitType.CUBE_TENSOR_CORE)
        peak_flops = device.get_compute_tflops("fp16", ComputeUnitType.CUBE_TENSOR_CORE) * 1e12
        self.assertAlmostEqual(flops_high, peak_flops, delta=1e6)
    
    def test_ascend_memory_bandwidth(self):
        """Test Ascend NPU memory bandwidth values."""
        test_cases = [
            ("Ascend-910A", 1500.0),
            ("Ascend-910B1", 392.0),
            ("Ascend-910B2", 392.0),
            ("Ascend-910C", 784.0),
        ]
        
        for preset, expected_bw in test_cases:
            device = Device.from_preset(preset)
            self.assertEqual(device.get_memory_bw_gbps(), expected_bw,
                           f"{preset} memory bandwidth mismatch")
    
    def test_hccs_bandwidth(self):
        """Test that HCCS bandwidth is set for Ascend devices."""
        ascend = Device.from_preset("Ascend-910B2")
        self.assertIsNotNone(ascend.config.hccs_bandwidth_gbps)
        self.assertGreater(ascend.config.hccs_bandwidth_gbps, 0)
        
        # NVIDIA should not have HCCS
        nvidia = Device.from_preset("H100-SXM-80GB")
        self.assertIsNone(nvidia.config.hccs_bandwidth_gbps)
    
    def test_ascend_950_dt_vs_pr(self):
        """Test differences between Ascend 950-DT and 950-PR."""
        dt = Device.from_preset("Ascend-950-DT")
        pr = Device.from_preset("Ascend-950-PR")
        
        # Same compute power
        self.assertEqual(dt.get_compute_tflops("fp16", ComputeUnitType.CUBE_TENSOR_CORE),
                        pr.get_compute_tflops("fp16", ComputeUnitType.CUBE_TENSOR_CORE))
        self.assertEqual(dt.config.fp16_tflops_cube, 500.0)
        
        # Different memory capacity
        self.assertEqual(dt.config.memory_gb, 144.0)
        self.assertEqual(pr.config.memory_gb, 128.0)
        
        # Different memory bandwidth
        self.assertEqual(dt.config.memory_bandwidth_gbps, 4000.0)  # 4 TB/s HiZQ 2.0
        self.assertEqual(pr.config.memory_bandwidth_gbps, 1600.0)  # 1.6 TB/s HiBL 1.0
        
        # DT should have higher memory bandwidth
        self.assertGreater(dt.config.memory_bandwidth_gbps, pr.config.memory_bandwidth_gbps)
    
    def test_to_dict(self):
        """Test device serialization."""
        device = Device.from_preset("Ascend-910B2")
        data = device.to_dict()
        
        self.assertEqual(data["name"], "Ascend-910B2")
        self.assertIn("fp16_tflops_cube", data)
        self.assertIn("fp16_tflops_vector", data)
        self.assertIn("is_ascend", data)
        self.assertTrue(data["is_ascend"])


class TestDeviceConfig(unittest.TestCase):
    """Test cases for DeviceConfig class."""
    
    def test_post_init_backward_compatibility(self):
        """Test that post_init properly sets CUBE/VECTOR from legacy values."""
        config = DeviceConfig(
            name="Test",
            fp16_tflops=100.0,
            memory_gb=80.0,
            memory_bandwidth_gbps=1000.0,
        )
        
        self.assertEqual(config.fp16_tflops_cube, 100.0)
        self.assertEqual(config.fp16_tflops_vector, 100.0)


if __name__ == "__main__":
    unittest.main()

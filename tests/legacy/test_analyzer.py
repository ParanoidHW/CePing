"""Tests for analyzer breakdown classes."""

import unittest
from llm_perf.analyzer.breakdown import KernelBreakdown, LayerBreakdown, PerformanceBreakdown


class TestKernelBreakdown(unittest.TestCase):
    """Test KernelBreakdown dataclass."""

    def test_basic_creation(self):
        """Test creating a kernel breakdown."""
        kernel = KernelBreakdown(
            name="gemm",
            kernel_type="compute",
            time_sec=0.01,
            memory_bytes=1024,
            flops=1000000,
        )
        self.assertEqual(kernel.name, "gemm")
        self.assertEqual(kernel.kernel_type, "compute")
        self.assertEqual(kernel.time_sec, 0.01)
        self.assertTrue(kernel.is_parallel)

    def test_non_parallel_kernel(self):
        """Test creating a non-parallel kernel."""
        kernel = KernelBreakdown(
            name="sync",
            kernel_type="memory",
            time_sec=0.001,
            memory_bytes=512,
            is_parallel=False,
        )
        self.assertFalse(kernel.is_parallel)


class TestLayerBreakdown(unittest.TestCase):
    """Test LayerBreakdown class."""

    def setUp(self):
        self.layer = LayerBreakdown(
            name="transformer_layer",
            kernels=[
                KernelBreakdown(
                    name="gemm",
                    kernel_type="compute",
                    time_sec=0.01,
                    memory_bytes=1024,
                    flops=1000000,
                ),
                KernelBreakdown(
                    name="allreduce",
                    kernel_type="communication",
                    time_sec=0.005,
                    memory_bytes=512,
                ),
                KernelBreakdown(
                    name="norm",
                    kernel_type="compute",
                    time_sec=0.002,
                    memory_bytes=256,
                ),
            ],
        )

    def test_total_time_with_overlap(self):
        """Test total time accounts for compute/comm overlap."""
        # compute = 0.012, comm = 0.005, max = 0.012
        self.assertAlmostEqual(self.layer.total_time, 0.012)

    def test_total_memory(self):
        """Test total memory aggregation."""
        self.assertEqual(self.layer.total_memory, 1024 + 512 + 256)

    def test_compute_time(self):
        """Test compute time extraction."""
        self.assertAlmostEqual(self.layer.compute_time, 0.012)

    def test_comm_time(self):
        """Test communication time extraction."""
        self.assertAlmostEqual(self.layer.comm_time, 0.005)

    def test_empty_layer(self):
        """Test layer with no kernels."""
        empty = LayerBreakdown(name="empty")
        self.assertEqual(empty.total_time, 0.0)
        self.assertEqual(empty.total_memory, 0)
        self.assertEqual(empty.compute_time, 0.0)
        self.assertEqual(empty.comm_time, 0.0)


class TestPerformanceBreakdown(unittest.TestCase):
    """Test PerformanceBreakdown class."""

    def setUp(self):
        self.breakdown = PerformanceBreakdown(
            total_time_sec=0.1,
            throughput=1000.0,
            compute_time_sec=0.08,
            communication_time_sec=0.01,
            memory_time_sec=0.01,
            layers=[
                LayerBreakdown(
                    name="layer_0",
                    kernels=[
                        KernelBreakdown(
                            name="gemm",
                            kernel_type="compute",
                            time_sec=0.05,
                            memory_bytes=1024,
                        ),
                    ],
                ),
                LayerBreakdown(
                    name="layer_1",
                    kernels=[
                        KernelBreakdown(
                            name="allreduce",
                            kernel_type="communication",
                            time_sec=0.01,
                            memory_bytes=512,
                        ),
                    ],
                ),
            ],
            peak_memory_bytes=10 * 1024 * 1024,
            activation_memory_bytes=2 * 1024 * 1024,
            parameter_memory_bytes=8 * 1024 * 1024,
            comm_breakdown={"tensor_parallel": 0.005, "pipeline_parallel": 0.002},
        )

    def test_to_dict_structure(self):
        """Test dictionary serialization."""
        data = self.breakdown.to_dict()
        self.assertIn("overview", data)
        self.assertIn("time_breakdown", data)
        self.assertIn("memory_breakdown", data)
        self.assertIn("communication", data)
        self.assertIn("layers", data)

        self.assertEqual(data["overview"]["total_time_sec"], 0.1)
        self.assertEqual(data["overview"]["throughput"], 1000.0)
        self.assertEqual(data["time_breakdown"]["compute_sec"], 0.08)
        self.assertEqual(data["time_breakdown"]["communication_sec"], 0.01)

    def test_to_dict_percentages(self):
        """Test percentage calculations in dict."""
        data = self.breakdown.to_dict()
        self.assertEqual(data["time_breakdown"]["compute_percent"], 80.0)
        self.assertEqual(data["time_breakdown"]["communication_percent"], 10.0)
        self.assertEqual(data["time_breakdown"]["memory_percent"], 10.0)

    def test_to_dict_memory_mb(self):
        """Test memory conversion to MB."""
        data = self.breakdown.to_dict()
        self.assertEqual(data["memory_breakdown"]["peak_mb"], 10.0)
        self.assertEqual(data["memory_breakdown"]["activation_mb"], 2.0)
        self.assertEqual(data["memory_breakdown"]["parameter_mb"], 8.0)

    def test_to_dict_layers(self):
        """Test layer serialization."""
        data = self.breakdown.to_dict()
        self.assertEqual(len(data["layers"]), 2)
        self.assertEqual(data["layers"][0]["name"], "layer_0")
        self.assertEqual(data["layers"][0]["compute_ms"], 50.0)

    def test_get_summary_table(self):
        """Test summary table generation."""
        table = self.breakdown.get_summary_table()
        self.assertIn("PERFORMANCE BREAKDOWN", table)
        self.assertIn("Total Time:", table)
        self.assertIn("Compute:", table)
        self.assertIn("Communication:", table)
        self.assertIn("Memory Wait:", table)
        self.assertIn("Peak Memory:", table)
        self.assertIn("tensor_parallel", table)
        self.assertIn("layer_0", table)

    def test_empty_breakdown(self):
        """Test empty breakdown serialization."""
        empty = PerformanceBreakdown(
            total_time_sec=1.0,
            throughput=0.0,
            compute_time_sec=0.0,
            communication_time_sec=0.0,
            memory_time_sec=0.0,
        )
        data = empty.to_dict()
        self.assertEqual(data["time_breakdown"]["compute_percent"], 0.0)
        self.assertEqual(data["memory_breakdown"]["peak_mb"], 0.0)
        self.assertEqual(data["layers"], [])


if __name__ == "__main__":
    unittest.main()

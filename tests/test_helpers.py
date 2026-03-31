"""Tests for utility helper functions."""

import unittest
import json
import tempfile
from pathlib import Path

from llm_perf.utils.helpers import (
    load_json,
    save_json,
    ceil_div,
    format_bytes,
    format_time,
    format_throughput,
)
from llm_perf.utils.constants import (
    DTYPE_SIZES,
    COMPUTE_BOUND,
    MEMORY_BOUND,
    COMMUNICATION_BOUND,
    TP,
    PP,
    DP,
    EP,
    SP,
    PHASE_TRAINING,
    PHASE_PREFILL,
    PHASE_DECODE,
    KERNEL_COMPUTE,
    KERNEL_COMMUNICATION,
    KERNEL_MEMORY,
)


class TestJSONHelpers(unittest.TestCase):
    """Test JSON load/save helpers."""

    def test_save_and_load_json(self):
        """Test round-trip JSON serialization."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_json(data, str(path))
            loaded = load_json(str(path))
            self.assertEqual(loaded, data)

    def test_load_json_unicode(self):
        """Test JSON with unicode content."""
        data = {"text": "中文测试"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_json(data, str(path))
            loaded = load_json(str(path))
            self.assertEqual(loaded["text"], "中文测试")


class TestCeilDiv(unittest.TestCase):
    """Test ceiling division helper."""

    def test_exact_division(self):
        """Test exact division."""
        self.assertEqual(ceil_div(10, 2), 5)
        self.assertEqual(ceil_div(100, 10), 10)

    def test_inexact_division(self):
        """Test inexact division rounds up."""
        self.assertEqual(ceil_div(10, 3), 4)
        self.assertEqual(ceil_div(7, 2), 4)
        self.assertEqual(ceil_div(1, 5), 1)

    def test_zero_dividend(self):
        """Test zero dividend."""
        self.assertEqual(ceil_div(0, 5), 0)


class TestFormatBytes(unittest.TestCase):
    """Test byte formatting helper."""

    def test_bytes(self):
        """Test byte formatting."""
        self.assertEqual(format_bytes(512), "512.00 B")

    def test_kilobytes(self):
        """Test kilobyte formatting."""
        self.assertEqual(format_bytes(1536), "1.50 KB")

    def test_megabytes(self):
        """Test megabyte formatting."""
        self.assertEqual(format_bytes(2 * 1024 * 1024), "2.00 MB")

    def test_gigabytes(self):
        """Test gigabyte formatting."""
        self.assertEqual(format_bytes(3 * 1024 ** 3), "3.00 GB")

    def test_terabytes(self):
        """Test terabyte formatting."""
        self.assertEqual(format_bytes(4 * 1024 ** 4), "4.00 TB")

    def test_zero(self):
        """Test zero bytes."""
        self.assertEqual(format_bytes(0), "0.00 B")

    def test_negative(self):
        """Test negative bytes."""
        self.assertEqual(format_bytes(-1024), "-1.00 KB")


class TestFormatTime(unittest.TestCase):
    """Test time formatting helper."""

    def test_nanoseconds(self):
        """Test nanosecond formatting."""
        self.assertEqual(format_time(1e-9), "1.00 ns")

    def test_microseconds(self):
        """Test microsecond formatting."""
        self.assertEqual(format_time(1e-6), "1.00 us")
        self.assertEqual(format_time(500e-6), "500.00 us")

    def test_milliseconds(self):
        """Test millisecond formatting."""
        self.assertEqual(format_time(1e-3), "1.00 ms")
        self.assertEqual(format_time(500e-3), "500.00 ms")

    def test_seconds(self):
        """Test second formatting."""
        self.assertEqual(format_time(1.5), "1.50 s")
        self.assertEqual(format_time(60.0), "60.00 s")


class TestFormatThroughput(unittest.TestCase):
    """Test throughput formatting helper."""

    def test_tokens_per_sec(self):
        """Test low throughput."""
        self.assertEqual(format_throughput(50.0), "50.00 tokens/s")

    def test_kilo_tokens(self):
        """Test K tokens/s formatting."""
        self.assertEqual(format_throughput(1500.0), "1.50K tokens/s")

    def test_mega_tokens(self):
        """Test M tokens/s formatting."""
        self.assertEqual(format_throughput(2.5e6), "2.50M tokens/s")

    def test_giga_tokens(self):
        """Test B tokens/s formatting."""
        self.assertEqual(format_throughput(3.5e9), "3.50B tokens/s")

    def test_zero(self):
        """Test zero throughput."""
        self.assertEqual(format_throughput(0), "0.00 tokens/s")


class TestConstants(unittest.TestCase):
    """Test that constants are properly defined."""

    def test_dtype_sizes(self):
        """Test data type size constants."""
        self.assertEqual(DTYPE_SIZES["fp32"], 4)
        self.assertEqual(DTYPE_SIZES["fp16"], 2)
        self.assertEqual(DTYPE_SIZES["bf16"], 2)
        self.assertEqual(DTYPE_SIZES["fp8"], 1)
        self.assertEqual(DTYPE_SIZES["int8"], 1)
        self.assertEqual(DTYPE_SIZES["int4"], 0.5)

    def test_bound_constants(self):
        """Test bound category constants."""
        self.assertEqual(COMPUTE_BOUND, "compute_bound")
        self.assertEqual(MEMORY_BOUND, "memory_bound")
        self.assertEqual(COMMUNICATION_BOUND, "communication_bound")

    def test_parallelism_constants(self):
        """Test parallelism type constants."""
        self.assertEqual(TP, "tensor_parallel")
        self.assertEqual(PP, "pipeline_parallel")
        self.assertEqual(DP, "data_parallel")
        self.assertEqual(EP, "expert_parallel")
        self.assertEqual(SP, "sequence_parallel")

    def test_phase_constants(self):
        """Test phase type constants."""
        self.assertEqual(PHASE_TRAINING, "training")
        self.assertEqual(PHASE_PREFILL, "prefill")
        self.assertEqual(PHASE_DECODE, "decode")

    def test_kernel_constants(self):
        """Test kernel category constants."""
        self.assertEqual(KERNEL_COMPUTE, "compute")
        self.assertEqual(KERNEL_COMMUNICATION, "communication")
        self.assertEqual(KERNEL_MEMORY, "memory")


if __name__ == "__main__":
    unittest.main()

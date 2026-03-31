"""Tests for reporter classes."""

import unittest
import json
import tempfile
from pathlib import Path

from llm_perf.analyzer.training import TrainingResult
from llm_perf.analyzer.inference import InferenceResult
from llm_perf.analyzer.breakdown import PerformanceBreakdown, LayerBreakdown, KernelBreakdown
from llm_perf.reporter.table import TableReporter
from llm_perf.reporter.json_reporter import JSONReporter


class TestTableReporter(unittest.TestCase):
    """Test TableReporter formatting."""

    def setUp(self):
        self.reporter = TableReporter(max_width=80)

    def test_report_training(self):
        """Test training report generation."""
        result = TrainingResult(
            samples_per_sec=10.5,
            tokens_per_sec=43008.0,
            time_per_step_sec=0.095,
            time_to_solution_sec=95.0,
            memory_per_gpu_gb=64.5,
        )
        report = self.reporter.report_training(result)
        self.assertIn("Training Performance", report)
        self.assertIn("Samples/sec", report)
        self.assertIn("10.50", report)
        self.assertIn("Memory per GPU", report)

    def test_report_training_with_breakdown(self):
        """Test training report with breakdown."""
        breakdown = PerformanceBreakdown(
            total_time_sec=0.1,
            throughput=43008.0,
            compute_time_sec=0.08,
            communication_time_sec=0.01,
            memory_time_sec=0.01,
        )
        result = TrainingResult(
            samples_per_sec=10.0,
            tokens_per_sec=40000.0,
            time_per_step_sec=0.1,
            time_to_solution_sec=100.0,
            memory_per_gpu_gb=60.0,
            breakdown=breakdown,
        )
        report = self.reporter.report_training(result)
        self.assertIn("PERFORMANCE BREAKDOWN", report)

    def test_report_inference(self):
        """Test inference report generation."""
        result = InferenceResult(
            prefill_time_sec=0.02,
            decode_time_per_step_sec=0.01,
            prefill_tokens_per_sec=51200.0,
            decode_tokens_per_sec=100.0,
            total_time_sec=1.5,
            total_tokens=128,
            memory_per_gpu_gb=48.0,
            kv_cache_memory_gb=2.0,
        )
        report = self.reporter.report_inference(result)
        self.assertIn("Inference Performance", report)
        self.assertIn("TTFT", report)
        self.assertIn("TPOT", report)
        self.assertIn("KV Cache", report)

    def test_report_comparison_training(self):
        """Test comparison report for training results."""
        results = {
            "config_a": TrainingResult(
                samples_per_sec=10.0,
                tokens_per_sec=40000.0,
                time_per_step_sec=0.1,
                time_to_solution_sec=100.0,
                memory_per_gpu_gb=60.0,
            ),
            "config_b": TrainingResult(
                samples_per_sec=20.0,
                tokens_per_sec=80000.0,
                time_per_step_sec=0.05,
                time_to_solution_sec=50.0,
                memory_per_gpu_gb=70.0,
            ),
        }
        report = self.reporter.report_comparison(results, metric="throughput")
        self.assertIn("Comparison by throughput", report)
        self.assertIn("config_b", report)
        self.assertIn("config_a", report)
        # config_b should be first (higher throughput)
        b_index = report.index("config_b")
        a_index = report.index("config_a")
        self.assertLess(b_index, a_index)

    def test_report_comparison_inference(self):
        """Test comparison report for inference results."""
        results = {
            "fast": InferenceResult(
                prefill_time_sec=0.01,
                decode_time_per_step_sec=0.005,
                prefill_tokens_per_sec=100000.0,
                decode_tokens_per_sec=200.0,
                total_time_sec=0.7,
                total_tokens=128,
                memory_per_gpu_gb=50.0,
                kv_cache_memory_gb=2.0,
            ),
            "slow": InferenceResult(
                prefill_time_sec=0.05,
                decode_time_per_step_sec=0.02,
                prefill_tokens_per_sec=20000.0,
                decode_tokens_per_sec=50.0,
                total_time_sec=2.6,
                total_tokens=128,
                memory_per_gpu_gb=50.0,
                kv_cache_memory_gb=2.0,
            ),
        }
        report = self.reporter.report_comparison(results, metric="throughput")
        self.assertIn("Comparison by throughput", report)

    def test_get_metric_value(self):
        """Test metric extraction."""
        training = TrainingResult(
            samples_per_sec=10.0,
            tokens_per_sec=40000.0,
            time_per_step_sec=0.1,
            time_to_solution_sec=100.0,
            memory_per_gpu_gb=60.0,
        )
        inference = InferenceResult(
            prefill_time_sec=0.02,
            decode_time_per_step_sec=0.01,
            prefill_tokens_per_sec=51200.0,
            decode_tokens_per_sec=100.0,
            total_time_sec=1.5,
            total_tokens=128,
            memory_per_gpu_gb=48.0,
            kv_cache_memory_gb=2.0,
        )
        self.assertEqual(self.reporter._get_metric_value(training, "throughput"), 40000.0)
        self.assertEqual(self.reporter._get_metric_value(training, "memory"), 60.0)
        self.assertEqual(self.reporter._get_metric_value(inference, "throughput"), 100.0)
        self.assertEqual(self.reporter._get_metric_value(inference, "ttft"), 0.02)
        self.assertEqual(self.reporter._get_metric_value(None, "unknown"), 0.0)


class TestJSONReporter(unittest.TestCase):
    """Test JSONReporter serialization."""

    def setUp(self):
        self.reporter = JSONReporter(indent=2)

    def test_report_training(self):
        """Test JSON report for training result."""
        result = TrainingResult(
            samples_per_sec=10.0,
            tokens_per_sec=40000.0,
            time_per_step_sec=0.1,
            time_to_solution_sec=100.0,
            memory_per_gpu_gb=60.0,
        )
        json_str = self.reporter.report(result, metadata={"model": "llama-7b"})
        data = json.loads(json_str)
        self.assertEqual(data["metadata"]["model"], "llama-7b")
        self.assertEqual(data["result"]["throughput"]["tokens_per_sec"], 40000.0)

    def test_report_inference(self):
        """Test JSON report for inference result."""
        result = InferenceResult(
            prefill_time_sec=0.02,
            decode_time_per_step_sec=0.01,
            prefill_tokens_per_sec=51200.0,
            decode_tokens_per_sec=100.0,
            total_time_sec=1.5,
            total_tokens=128,
            memory_per_gpu_gb=48.0,
            kv_cache_memory_gb=2.0,
        )
        json_str = self.reporter.report(result)
        data = json.loads(json_str)
        self.assertEqual(data["result"]["prefill"]["ttft_sec"], 0.02)
        self.assertEqual(data["result"]["decode"]["tps"], 100.0)

    def test_report_batch(self):
        """Test batch JSON report."""
        results = {
            "training": TrainingResult(
                samples_per_sec=10.0,
                tokens_per_sec=40000.0,
                time_per_step_sec=0.1,
                time_to_solution_sec=100.0,
                memory_per_gpu_gb=60.0,
            ),
            "inference": InferenceResult(
                prefill_time_sec=0.02,
                decode_time_per_step_sec=0.01,
                prefill_tokens_per_sec=51200.0,
                decode_tokens_per_sec=100.0,
                total_time_sec=1.5,
                total_tokens=128,
                memory_per_gpu_gb=48.0,
                kv_cache_memory_gb=2.0,
            ),
        }
        json_str = self.reporter.report_batch(results, metadata={"batch": True})
        data = json.loads(json_str)
        self.assertTrue(data["metadata"]["batch"])
        self.assertIn("training", data["results"])
        self.assertIn("inference", data["results"])

    def test_save(self):
        """Test saving JSON report to file."""
        result = TrainingResult(
            samples_per_sec=10.0,
            tokens_per_sec=40000.0,
            time_per_step_sec=0.1,
            time_to_solution_sec=100.0,
            memory_per_gpu_gb=60.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            self.reporter.save(result, path, metadata={"test": True})
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["metadata"]["test"], True)

    def test_save_batch(self):
        """Test saving batch JSON report to file."""
        results = {
            "run1": TrainingResult(
                samples_per_sec=10.0,
                tokens_per_sec=40000.0,
                time_per_step_sec=0.1,
                time_to_solution_sec=100.0,
                memory_per_gpu_gb=60.0,
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "batch.json"
            self.reporter.save_batch(results, path)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("run1", data["results"])


if __name__ == "__main__":
    unittest.main()

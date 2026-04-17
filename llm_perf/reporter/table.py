"""Table-based console reporter."""

from typing import Dict, Any, Union
from pathlib import Path

from .base import BaseReporter
from llm_perf.analyzer.training import TrainingResult
from llm_perf.analyzer.inference import InferenceResult
from llm_perf.utils.helpers import format_bytes, format_time, format_throughput


class TableReporter(BaseReporter):
    """Generate formatted table reports for console output."""

    def __init__(self, max_width: int = 100):
        self.max_width = max_width

    def report_training(self, result: TrainingResult, title: str = "Training Performance", **kwargs) -> str:
        """Generate training report."""
        lines = []
        lines.append(self._header(title))

        lines.append(self._section("Throughput"))
        lines.append(self._row("Samples/sec", f"{result.samples_per_sec:.2f}"))
        lines.append(self._row("Tokens/sec", format_throughput(result.tokens_per_sec)))

        lines.append(self._section("Time"))
        lines.append(self._row("Time per step", format_time(result.time_per_step_sec)))

        lines.append(self._section("Memory"))
        lines.append(self._row("Memory per GPU", f"{result.memory_per_gpu_gb:.2f} GB"))

        if result.breakdown:
            lines.append(self._section("Breakdown"))
            lines.append(self._row("Compute time", format_time(result.breakdown.compute_time_sec)))
            lines.append(self._row("Comm time", format_time(result.breakdown.communication_time_sec)))

        lines.append(self._footer())
        return "\n".join(lines)

    def report_inference(
        self, result: InferenceResult, title: str = "Inference Performance", generation_len: int = 128, **kwargs
    ) -> str:
        """Generate inference report."""
        lines = []
        lines.append(self._header(title))

        lines.append(self._section("Prefill Phase"))
        lines.append(self._row("TTFT", format_time(result.prefill_time_sec)))
        lines.append(self._row("Throughput", format_throughput(result.prefill_tokens_per_sec)))

        lines.append(self._section("Decode Phase"))
        lines.append(self._row("TPOT", format_time(result.decode_time_per_step_sec)))
        lines.append(self._row("TPS", format_throughput(result.decode_tokens_per_sec)))

        total_time = result.prefill_time_sec + result.decode_time_per_step_sec * generation_len

        lines.append(self._section("End-to-End"))
        lines.append(self._row("Total time", format_time(total_time)))

        lines.append(self._section("Memory"))
        lines.append(self._row("Memory per GPU", f"{result.memory_per_gpu_gb:.2f} GB"))

        if result.breakdown:
            lines.append(self._section("Breakdown"))
            lines.append(self._row("Compute time", format_time(result.breakdown.compute_time_sec)))

        lines.append(self._footer())
        return "\n".join(lines)

    def report_comparison(self, results: Dict[str, Any], metric: str = "throughput") -> str:
        """Generate comparison report for multiple configurations."""
        lines = []
        lines.append(self._header(f"Comparison by {metric}"))

        sorted_results = sorted(results.items(), key=lambda x: self._get_metric_value(x[1], metric), reverse=True)

        lines.append(f"{'Config':<30} {metric.title():>20}")
        lines.append("-" * 55)

        for name, result in sorted_results:
            value = self._get_metric_value(result, metric)
            lines.append(f"{name:<30} {value:>20.2f}")

        lines.append(self._footer())
        return "\n".join(lines)

    def _header(self, title: str) -> str:
        """Generate header."""
        width = self.max_width
        lines = [
            "=" * width,
            title.center(width),
            "=" * width,
        ]
        return "\n".join(lines)

    def _footer(self) -> str:
        """Generate footer."""
        return "=" * self.max_width

    def _section(self, name: str) -> str:
        """Generate section header."""
        return f"\n[{name}]"

    def _row(self, label: str, value: str) -> str:
        """Generate a table row."""
        return f"  {label:<25} {value:>20}"

    def _get_metric_value(self, result: Any, metric: str) -> float:
        """Extract metric value from result."""
        if isinstance(result, TrainingResult):
            if metric == "throughput":
                return result.tokens_per_sec
            elif metric == "samples":
                return result.samples_per_sec
            elif metric == "memory":
                return result.memory_per_gpu_gb
        elif isinstance(result, InferenceResult):
            if metric == "throughput":
                return result.decode_tokens_per_sec
            elif metric == "ttft":
                return result.prefill_time_sec
            elif metric == "memory":
                return result.memory_per_gpu_gb
        return 0.0

    def save(self, result: Union[TrainingResult, InferenceResult], path: Union[str, Path], **kwargs) -> None:
        """Save table report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report_str = self.report(result, **kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_str)

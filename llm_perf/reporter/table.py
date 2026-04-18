"""Table-based console reporter."""

from typing import Dict, Any, Union
from pathlib import Path

from .base import BaseReporter
from llm_perf.analyzer import UnifiedResult
from llm_perf.utils.helpers import format_bytes, format_time, format_throughput


class TableReporter(BaseReporter):
    """Generate formatted table reports for console output."""

    def __init__(self, max_width: int = 100):
        self.max_width = max_width

    def report(
        self,
        result: UnifiedResult,
        title: str = None,
        generation_len: int = 0,
        **kwargs,
    ) -> str:
        """Generate unified report."""
        if title is None:
            title = f"{result.workload_name} Performance"

        lines = []
        lines.append(self._header(title))

        lines.append(self._section("Workload"))
        lines.append(self._row("Name", result.workload_name))
        lines.append(self._row("Type", result.workload_type.value))

        lines.append(self._section("Performance"))
        lines.append(self._row("Total time", format_time(result.total_time_sec)))
        lines.append(self._row("Peak memory", f"{result.peak_memory_gb:.2f} GB"))

        for metric_name, metric_value in result.throughput.items():
            if metric_name == "tokens_per_sec":
                lines.append(self._row("Tokens/sec", format_throughput(metric_value)))
            elif metric_name == "samples_per_sec":
                lines.append(self._row("Samples/sec", f"{metric_value:.2f}"))
            elif metric_name == "pixels_per_sec":
                lines.append(self._row("Pixels/sec", format_throughput(metric_value)))
            else:
                lines.append(self._row(metric_name, f"{metric_value:.2f}"))

        if result.phases:
            lines.append(self._section("Phase Breakdown"))
            for phase in result.phases:
                lines.append(
                    self._row(f"{phase.name}", f"{phase.total_time_sec * 1000:.2f} ms ({phase.repeat_count} repeats)")
                )

        lines.append(self._footer())
        return "\n".join(lines)

    def report_comparison(
        self,
        results: Dict[str, UnifiedResult],
        metric: str = "throughput",
    ) -> str:
        """Generate comparison report for multiple configurations."""
        lines = []
        lines.append(self._header(f"Comparison by {metric}"))

        def get_metric_value(result: UnifiedResult) -> float:
            throughput = result.throughput.get("tokens_per_sec", result.throughput.get("samples_per_sec", 0))
            if metric == "throughput":
                return throughput
            elif metric == "latency":
                prefill = result.get_phase("prefill")
                return prefill.total_time_sec * 1000 if prefill else result.total_time_sec * 1000
            elif metric == "memory":
                return result.peak_memory_gb
            return throughput

        sorted_results = sorted(results.items(), key=lambda x: get_metric_value(x[1]), reverse=True)

        lines.append(f"{'Config':<30} {metric.title():>20}")
        lines.append("-" * 55)

        for name, result in sorted_results:
            value = get_metric_value(result)
            lines.append(f"{name:<30} {value:>20.2f}")

        lines.append(self._footer())
        return "\n".join(lines)

    def _header(self, title: str) -> str:
        width = self.max_width
        lines = [
            "=" * width,
            title.center(width),
            "=" * width,
        ]
        return "\n".join(lines)

    def _footer(self) -> str:
        return "=" * self.max_width

    def _section(self, name: str) -> str:
        return f"\n[{name}]"

    def _row(self, label: str, value: str) -> str:
        return f"  {label:<25} {value:>20}"

    def save(
        self,
        result: UnifiedResult,
        path: Union[str, Path],
        **kwargs,
    ) -> None:
        """Save table report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report_str = self.report(result, **kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_str)

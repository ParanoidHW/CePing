"""Output formatting for CLI.

Provides:
- YamlReporter: YAML output format
- CLIReporter: Unified reporter for CLI (JSON/YAML/Table)
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import yaml

from llm_perf.reporter import BaseReporter, TableReporter, JSONReporter
from llm_perf.analyzer import UnifiedResult
from llm_perf.workload.breakdown import WorkloadBreakdown, calculate_breakdown


class YamlReporter(BaseReporter):
    """YAML output format reporter."""

    def report(
        self,
        result: UnifiedResult,
        breakdown: Optional[WorkloadBreakdown] = None,
        **kwargs,
    ) -> str:
        """Generate YAML report.

        Args:
            result: UnifiedResult from UnifiedAnalyzer
            breakdown: Optional WorkloadBreakdown (calculated if not provided)
            **kwargs: Additional options

        Returns:
            YAML formatted string
        """
        data = self._build_output(result, breakdown)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def save(
        self,
        result: UnifiedResult,
        path: Union[str, Path],
        breakdown: Optional[WorkloadBreakdown] = None,
        **kwargs,
    ) -> None:
        """Save YAML report to file.

        Args:
            result: UnifiedResult
            path: Output file path
            breakdown: Optional WorkloadBreakdown
            **kwargs: Additional options
        """
        output = self.report(result, breakdown, **kwargs)
        Path(path).write_text(output, encoding="utf-8")

    def _build_output(
        self,
        result: UnifiedResult,
        breakdown: Optional[WorkloadBreakdown] = None,
    ) -> Dict[str, Any]:
        """Build output dictionary."""
        if breakdown is None:
            breakdown = calculate_breakdown(result)

        return {
            "summary": {
                "total_time_sec": result.total_time_sec,
                "peak_memory_gb": result.peak_memory_gb,
                "throughput": result.throughput,
                "mfu": result.mfu,
                "qps": result.qps,
            },
            "stages": [
                {
                    "name": s.name,
                    "total_time_sec": s.total_time_sec,
                    "peak_memory_gb": s.peak_memory_gb,
                }
                for s in breakdown.stages
            ],
            "by_submodule_type": {
                k: {
                    "time_sec": v.time_sec,
                    "flops": v.flops,
                    "params_count": v.params_count,
                    "weight_memory_gb": v.weight_memory_gb,
                }
                for k, v in breakdown.by_submodule_type.items()
            },
            "communication": breakdown.communication.to_dict() if breakdown.communication else None,
            "phases": [p.to_dict() for p in result.phases],
        }


class CLIReporter:
    """Unified reporter for CLI (JSON/YAML/Table)."""

    def __init__(self):
        self.json_reporter = JSONReporter()
        self.yaml_reporter = YamlReporter()
        self.table_reporter = TableReporter()

    def format(
        self,
        result: Union[UnifiedResult, Dict[str, Any]],
        format_type: str = "json",
        breakdown: Optional[WorkloadBreakdown] = None,
        breakdown_level: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Format result in specified format.

        Args:
            result: UnifiedResult or result dict
            format_type: Output format ("json", "yaml", "table")
            breakdown: Optional WorkloadBreakdown
            breakdown_level: Breakdown levels to show ["stage", "phase", "submodule"]
            **kwargs: Additional options

        Returns:
            Formatted string
        """
        if isinstance(result, dict):
            return self._format_dict(result, format_type)

        if format_type == "json":
            return self.yaml_reporter.report(result, breakdown, **kwargs)
        elif format_type == "yaml":
            return self.yaml_reporter.report(result, breakdown, **kwargs)
        elif format_type == "table":
            return self.table_reporter.report(result, **kwargs)
        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def _format_dict(self, result: Dict[str, Any], format_type: str) -> str:
        """Format result dict."""
        if format_type == "json":
            return json.dumps(result, indent=2)
        elif format_type == "yaml":
            return yaml.dump(result, default_flow_style=False, sort_keys=False)
        elif format_type == "table":
            return self._dict_to_table(result)
        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def _dict_to_table(self, result: Dict[str, Any]) -> str:
        """Convert result dict to table format."""
        lines = []

        summary = result.get("result", {}).get("summary", {})
        if summary:
            lines.append("=" * 50)
            lines.append("Results Summary")
            lines.append("=" * 50)
            lines.append(f"Throughput:    {summary.get('throughput', 0):,.2f} tokens/sec")
            lines.append(f"MFU:           {summary.get('mfu', 0) * 100:.1f}%")
            lines.append(f"Peak Memory:   {summary.get('peak_memory_gb', 0):.2f} GB/device")
            lines.append(f"Total Time:    {summary.get('total_time_sec', 0):.2f} sec/batch")

        stages = result.get("result", {}).get("stages", [])
        if stages:
            lines.append("\n" + "-" * 50)
            lines.append("Stage Breakdown")
            lines.append("-" * 50)
            for stage in stages:
                lines.append(f"  {stage.get('name', 'unknown')}:")
                lines.append(f"    Time: {stage.get('total_time_sec', 0):.2f}s")
                lines.append(f"    Memory: {stage.get('peak_memory_gb', 0):.2f}GB")

        validation = result.get("validation", {})
        if validation and not validation.get("is_valid", True):
            lines.append("\n" + "-" * 50)
            lines.append("Validation Warnings")
            lines.append("-" * 50)
            for error in validation.get("errors", []):
                lines.append(f"  - {error}")

        return "\n".join(lines)

    def save(
        self,
        result: Union[UnifiedResult, Dict[str, Any]],
        path: Union[str, Path],
        format_type: str = "json",
        breakdown: Optional[WorkloadBreakdown] = None,
        **kwargs,
    ) -> None:
        """Save result to file.

        Args:
            result: UnifiedResult or result dict
            path: Output file path
            format_type: Output format
            breakdown: Optional WorkloadBreakdown
            **kwargs: Additional options
        """
        output = self.format(result, format_type, breakdown, **kwargs)
        Path(path).write_text(output, encoding="utf-8")
"""JSON report generator."""

import json
from typing import Dict, Any, Union
from pathlib import Path

from .base import BaseReporter
from llm_perf.analyzer import UnifiedResult


class JSONReporter(BaseReporter):
    """Generate JSON reports."""

    def __init__(self, indent: int = 2):
        self.indent = indent

    def report(
        self,
        result: UnifiedResult,
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> str:
        """Generate JSON report."""
        data = {"metadata": metadata or {}, "result": result.to_dict()}
        return json.dumps(data, indent=self.indent, ensure_ascii=False)

    def report_batch(self, results: Dict[str, UnifiedResult], metadata: Dict[str, Any] = None) -> str:
        """Generate JSON report for multiple results."""
        data = {"metadata": metadata or {}, "results": {name: result.to_dict() for name, result in results.items()}}
        return json.dumps(data, indent=self.indent, ensure_ascii=False)

    def save(
        self,
        result: UnifiedResult,
        path: Union[str, Path],
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report_str = self.report(result, metadata=metadata, **kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_str)

    def save_batch(self, results: Dict[str, UnifiedResult], path: Union[str, Path], metadata: Dict[str, Any] = None):
        """Save batch report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.report_batch(results, metadata))

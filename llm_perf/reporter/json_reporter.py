"""JSON report generator."""

import json
from typing import Dict, Any, Union
from pathlib import Path

from ..analyzer.training import TrainingResult
from ..analyzer.inference import InferenceResult


class JSONReporter:
    """Generate JSON reports."""
    
    def __init__(self, indent: int = 2):
        self.indent = indent
    
    def report(
        self,
        result: Union[TrainingResult, InferenceResult],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Generate JSON report."""
        data = {
            "metadata": metadata or {},
            "result": result.to_dict()
        }
        return json.dumps(data, indent=self.indent, ensure_ascii=False)
    
    def report_batch(
        self,
        results: Dict[str, Union[TrainingResult, InferenceResult]],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Generate JSON report for multiple results."""
        data = {
            "metadata": metadata or {},
            "results": {
                name: result.to_dict()
                for name, result in results.items()
            }
        }
        return json.dumps(data, indent=self.indent, ensure_ascii=False)
    
    def save(
        self,
        result: Union[TrainingResult, InferenceResult],
        path: Union[str, Path],
        metadata: Dict[str, Any] = None
    ):
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.report(result, metadata))
    
    def save_batch(
        self,
        results: Dict[str, Union[TrainingResult, InferenceResult]],
        path: Union[str, Path],
        metadata: Dict[str, Any] = None
    ):
        """Save batch report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.report_batch(results, metadata))

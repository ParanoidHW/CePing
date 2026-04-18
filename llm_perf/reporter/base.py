"""Base reporter interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from pathlib import Path

from llm_perf.analyzer import UnifiedResult


class BaseReporter(ABC):
    """Abstract base class for all reporters.

    Defines a unified interface for generating performance reports.
    """

    @abstractmethod
    def report(
        self,
        result: UnifiedResult,
        **kwargs,
    ) -> str:
        """Generate performance report.

        Args:
            result: UnifiedResult from UnifiedAnalyzer
            **kwargs: Additional reporter-specific options

        Returns:
            Formatted report string
        """
        pass

    @abstractmethod
    def save(
        self,
        result: UnifiedResult,
        path: Union[str, Path],
        **kwargs,
    ) -> None:
        """Save report to file.

        Args:
            result: UnifiedResult
            path: Output file path
            **kwargs: Additional options
        """
        pass

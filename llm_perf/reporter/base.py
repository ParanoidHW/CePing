"""Base reporter interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from pathlib import Path

from ..analyzer.training import TrainingResult
from ..analyzer.inference import InferenceResult


class BaseReporter(ABC):
    """Abstract base class for all reporters.

    Defines a unified interface for generating performance reports.
    All reporters (Table, JSON, HTML) should inherit from this class.

    Example:
        >>> class MyReporter(BaseReporter):
        ...     def report_training(self, result, **kwargs):
        ...         return f"Training: {result.tokens_per_sec} tokens/sec"
        ...     def report_inference(self, result, **kwargs):
        ...         return f"Inference: {result.decode_tokens_per_sec} TPS"
        ...     def save(self, result, path, **kwargs):
        ...         Path(path).write_text(self.report(result, **kwargs))
    """

    @abstractmethod
    def report_training(
        self,
        result: TrainingResult,
        **kwargs
    ) -> str:
        """Generate training performance report.

        Args:
            result: TrainingResult from TrainingAnalyzer
            **kwargs: Additional reporter-specific options

        Returns:
            Formatted report string
        """
        pass

    @abstractmethod
    def report_inference(
        self,
        result: InferenceResult,
        **kwargs
    ) -> str:
        """Generate inference performance report.

        Args:
            result: InferenceResult from InferenceAnalyzer
            **kwargs: Additional reporter-specific options

        Returns:
            Formatted report string
        """
        pass

    @abstractmethod
    def save(
        self,
        result: Union[TrainingResult, InferenceResult],
        path: Union[str, Path],
        **kwargs
    ) -> None:
        """Save report to file.

        Args:
            result: Analysis result (Training or Inference)
            path: Output file path
            **kwargs: Additional reporter-specific options
        """
        pass

    def report(
        self,
        result: Union[TrainingResult, InferenceResult],
        **kwargs
    ) -> str:
        """Generate report based on result type.

        This is a convenience method that automatically selects
        the appropriate report method based on result type.

        Args:
            result: Analysis result (Training or Inference)
            **kwargs: Additional reporter-specific options

        Returns:
            Formatted report string
        """
        if isinstance(result, TrainingResult):
            return self.report_training(result, **kwargs)
        elif isinstance(result, InferenceResult):
            return self.report_inference(result, **kwargs)
        else:
            raise ValueError(f"Unknown result type: {type(result)}")
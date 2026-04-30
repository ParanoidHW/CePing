"""CLI module for workload evaluation.

This module provides:
- Command-line interface for evaluation
- Interactive configuration mode
- Output formatting (JSON/YAML/Table)
"""

from .output import CLIReporter, YamlReporter

__all__ = ["CLIReporter", "YamlReporter"]
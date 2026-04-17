"""
LLM Performance Evaluator

A comprehensive tool for evaluating training and inference performance
of Large Language Models under various parallel strategies.
"""

__version__ = "0.1.0"

from .app import Evaluator, StrategyOptimizer, BatchOptimizer, LatencyBudget

__all__ = [
    "Evaluator",
    "StrategyOptimizer",
    "BatchOptimizer",
    "LatencyBudget",
]

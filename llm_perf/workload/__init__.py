"""Workload module for Web2 schema-driven architecture.

This module provides:
- Schema definitions for Web API
- Loader for workload and model configurations
- Registry for workload types and model presets
- Evaluation engine (calls existing UnifiedAnalyzer)
- Breakdown calculation (Stage→Phase→Submodule→Kernel→Communication)
- Validator for configuration validation
"""

from .schema import (
    WorkloadSchema,
    StageSchema,
    HardwareSchema,
    StrategySchema,
    ModelSchema,
    ParamSchemaItem,
    WorkloadCategory,
)
from .loader import WorkloadLoader, get_loader
from .registry import WorkloadRegistry, ModelRegistry, get_workload_registry, get_model_registry
from .engine import EvaluationEngine, EvaluationRequest, EvaluationResult
from .validator import WorkloadValidator, ValidationResult
from .breakdown import BreakdownCalculator, WorkloadBreakdown, calculate_breakdown

__all__ = [
    "WorkloadSchema",
    "StageSchema",
    "HardwareSchema",
    "StrategySchema",
    "ModelSchema",
    "ParamSchemaItem",
    "WorkloadCategory",
    "WorkloadLoader",
    "get_loader",
    "WorkloadRegistry",
    "ModelRegistry",
    "get_workload_registry",
    "get_model_registry",
    "EvaluationEngine",
    "EvaluationRequest",
    "EvaluationResult",
    "WorkloadValidator",
    "ValidationResult",
    "BreakdownCalculator",
    "WorkloadBreakdown",
    "calculate_breakdown",
]
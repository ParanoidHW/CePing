"""Routes module for Web2 API.

All routes follow the principle:
- HTTP adaptation only (parse request, format response)
- Call llm_perf/workload module for core logic
- Never define schema in routes
"""

from .workloads import workloads_bp
from .models import models_bp
from .hardware import hardware_bp
from .evaluate import evaluate_bp
from .resources import resources_bp

__all__ = [
    "workloads_bp",
    "models_bp",
    "hardware_bp",
    "evaluate_bp",
    "resources_bp",
]
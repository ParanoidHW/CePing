"""Web2 API module for schema-driven evaluation service.

This module provides HTTP API endpoints for:
- Workload listing and schema
- Model listing and schema  
- Hardware listing and topology
- Evaluation requests

Core principle: API layer only does HTTP adaptation.
All core logic is in llm_perf/workload module.
"""

from .app import create_app, app

__all__ = ["create_app", "app"]
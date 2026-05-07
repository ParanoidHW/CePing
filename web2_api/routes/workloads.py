"""Workload API routes.

Endpoints:
- GET /api/workloads - List all workload categories
- GET /api/workload/<workload_name> - Get workload schema

Design: Routes only do HTTP adaptation, all logic is in llm_perf/workload.
"""

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify

from llm_perf.workload import get_workload_registry

logger = logging.getLogger(__name__)

workloads_bp = Blueprint("workloads", __name__)


@workloads_bp.route("/workloads", methods=["GET"])
def list_workloads() -> Dict[str, Any]:
    """List all workload categories.

    Returns:
        {
            "categories": {
                "training": ["training", "denoise"],
                "inference": ["inference", "autoregressive"],
                ...
            },
            "total": 15
        }
    """
    registry = get_workload_registry()
    categories = registry.list_workload_categories()
    total = sum(len(v) for v in categories.values())

    return jsonify({
        "categories": categories,
        "total": total,
    })


@workloads_bp.route("/workload/<path:workload_name>", methods=["GET"])
def get_workload(workload_name: str) -> Dict[str, Any]:
    """Get workload schema for frontend rendering.

    Args:
        workload_name: Like "inference/autoregressive" or "training"

    Returns:
        WorkloadSchema dict with:
        - name, description, category, workload_type
        - stages: list of stage schemas
        - parameters: dict of param schemas
        - throughput_metric, supported_models
    """
    registry = get_workload_registry()

    if not registry.is_valid_workload(workload_name):
        return jsonify({
            "error": "Workload not found",
            "workload_name": workload_name,
        }), 404

    try:
        schema = registry.get_workload_schema(workload_name)
        return jsonify(schema.to_dict())
    except Exception as e:
        logger.error(f"Failed to get workload schema: {e}")
        return jsonify({
            "error": "Failed to load workload",
            "message": str(e),
        }), 500
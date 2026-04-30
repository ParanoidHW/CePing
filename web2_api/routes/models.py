"""Model API routes.

Endpoints:
- GET /api/models - List all models
- GET /api/model/<model_name> - Get model schema

Design: Routes only do HTTP adaptation, all logic is in llm_perf/workload.
"""

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from llm_perf.workload import get_model_registry, ModelRegistry

logger = logging.getLogger(__name__)

models_bp = Blueprint("models", __name__)


@models_bp.route("/models", methods=["GET"])
def list_models() -> Dict[str, Any]:
    """List all models.

    Query params:
        workload: (optional) Filter by workload category (e.g., "training")

    Returns:
        {
            "models": [
                {"name": "llama-7b", "architecture": "llama", "sparse_type": "dense", ...},
                ...
            ],
            "total": 20
        }
    """
    registry = get_model_registry()
    workload_category = request.args.get("workload")

    if workload_category:
        models = registry.get_models_for_workload(workload_category)
    else:
        models = registry.list_models()

    return jsonify({
        "models": [m.to_dict() for m in models],
        "total": len(models),
    })


@models_bp.route("/model/<model_name>", methods=["GET"])
def get_model(model_name: str) -> Dict[str, Any]:
    """Get model schema for frontend rendering.

    Args:
        model_name: Like "llama-7b", "deepseek-v3"

    Query params:
        workload: (optional) Get param schema for specific workload type

    Returns:
        ModelSchema dict with:
        - name, description, architecture, sparse_type
        - config: model configuration
        - param_schema: parameter schema by workload type
        - supported_workloads
    """
    registry = get_model_registry()

    if not registry.is_valid_model(model_name):
        return jsonify({
            "error": "Model not found",
            "model_name": model_name,
        }), 404

    try:
        schema = registry.get_model_schema(model_name)
        result = schema.to_dict()

        workload_type = request.args.get("workload")
        if workload_type:
            param_schema = schema.get_param_schema(workload_type)
            result["param_schema_for_workload"] = [p.to_dict() for p in param_schema]

        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to get model schema: {e}")
        return jsonify({
            "error": "Failed to load model",
            "message": str(e),
        }), 500
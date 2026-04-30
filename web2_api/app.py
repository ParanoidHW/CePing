"""Flask application configuration and route registration.

Design principle:
- API layer only does HTTP adaptation (request parsing, response formatting)
- All schema definitions and core logic are in llm_perf/workload module
- Routes call llm_perf/workload module, never define schema locally
"""

import logging
import os
from typing import Any, Dict

from flask import Flask, jsonify
from flask_cors import CORS

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create Flask application instance."""
    app = Flask(__name__)
    CORS(app)

    register_routes(app)
    register_error_handlers(app)

    logger.info("Web2 API application created")
    return app


def register_routes(app: Flask) -> None:
    """Register all API routes."""
    from .routes import (
        workloads_bp,
        models_bp,
        hardware_bp,
        evaluate_bp,
        resources_bp,
    )

    app.register_blueprint(workloads_bp, url_prefix="/api")
    app.register_blueprint(models_bp, url_prefix="/api")
    app.register_blueprint(hardware_bp, url_prefix="/api")
    app.register_blueprint(evaluate_bp, url_prefix="/api")
    app.register_blueprint(resources_bp, url_prefix="/api")

    logger.info("Routes registered: /api/workloads, /api/models, /api/hardware, /api/evaluate, /api/resources")


def register_error_handlers(app: Flask) -> None:
    """Register error handlers."""

    @app.errorhandler(404)
    def not_found(error: Any) -> Dict[str, Any]:
        return jsonify({"error": "Not found", "message": str(error)}), 404

    @app.errorhandler(500)
    def internal_error(error: Any) -> Dict[str, Any]:
        logger.error(f"Internal error: {error}")
        return jsonify({"error": "Internal server error", "message": str(error)}), 500

    @app.errorhandler(Exception)
    def handle_exception(error: Exception) -> Dict[str, Any]:
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        return jsonify({"error": "Exception", "message": str(error)}), 500


app = create_app()
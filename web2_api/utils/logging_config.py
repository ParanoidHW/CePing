"""Logging configuration for web2_api.

Configures JSON-formatted logs for different purposes:
- api.log: General API requests
- evaluate.log: Evaluation process details
- error.log: Errors with full context
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "step"):
            log_data["step"] = record.step

        if hasattr(record, "data"):
            log_data["data"] = record.data

        if record.exc_info:
            log_data["error"] = str(record.exc_info[1])
            import traceback
            log_data["traceback"] = "".join(traceback.format_exception(*record.exc_info))

        return str(log_data).replace("'", '"')


def setup_logging(log_dir: str = "logs") -> None:
    """Setup logging configuration.

    Args:
        log_dir: Directory for log files (default: logs/)
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    (log_path / "api.log").touch()
    (log_path / "evaluate.log").touch()
    (log_path / "error.log").touch()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    api_handler = logging.FileHandler(log_path / "api.log")
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(JsonFormatter())

    evaluate_handler = logging.FileHandler(log_path / "evaluate.log")
    evaluate_handler.setLevel(logging.DEBUG)
    evaluate_handler.setFormatter(JsonFormatter())

    error_handler = logging.FileHandler(log_path / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    api_logger = logging.getLogger("web2_api.routes")
    api_logger.addHandler(api_handler)
    api_logger.addHandler(console_handler)
    api_logger.setLevel(logging.DEBUG)

    eval_logger = logging.getLogger("llm_perf.workload.engine")
    eval_logger.addHandler(evaluate_handler)
    eval_logger.addHandler(console_handler)
    eval_logger.setLevel(logging.DEBUG)

    root_logger.addHandler(error_handler)
import logging
import os
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


def configure_logging() -> None:
    """
    Configures the root logger.
    If LOG_FORMAT=json, uses JSON formatting.
    Otherwise, uses a standard human-readable format.
    """
    log_format_env = os.getenv("LOG_FORMAT", "text").lower()
    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_env)

    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    if log_format_env == "json":
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silence some noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

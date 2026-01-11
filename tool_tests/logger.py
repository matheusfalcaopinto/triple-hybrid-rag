from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(name: str = "tool_tests", level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger that writes both to stdout and to tool_tests/logs.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "tool_tests.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


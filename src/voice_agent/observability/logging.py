from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from logging import LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

_DEFAULT_JSON_PATH = "logs/voice_agent.log"
_CONFIGURED = False


def configure_logging() -> logging.Logger:
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("voice_agent_v4")

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        _CONFIGURED = True
        root.setLevel(level)
        return logging.getLogger("voice_agent_v4")

    root.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(_ConsoleFormatter())
    root.addHandler(console_handler)

    file_path = os.getenv("LOG_JSON_FILE", _DEFAULT_JSON_PATH)
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setLevel(level)
        file_handler.setFormatter(_JsonFormatter())
        root.addHandler(file_handler)

    _CONFIGURED = True
    return logging.getLogger("voice_agent_v4")


class _ConsoleFormatter(logging.Formatter):
    _LEVEL_COLORS = {
        "DEBUG": "\x1b[38;5;245m",
        "INFO": "\x1b[38;5;39m",
        "WARNING": "\x1b[38;5;214m",
        "ERROR": "\x1b[38;5;203m",
        "CRITICAL": "\x1b[38;5;196m",
    }

    def __init__(self) -> None:
        super().__init__()
        self._use_color = self._should_use_color()

    def format(self, record: LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname
        name = record.name
        base = record.getMessage()
        payload = getattr(record, "trace_payload", None)

        if payload:
            formatted = self._format_trace(payload)
        else:
            formatted = base

        if self._use_color:
            level_color = self._LEVEL_COLORS.get(level, "")
            reset = "\x1b[0m" if level_color else ""
            level_text = f"{level_color}{level:<8}{reset}"
        else:
            level_text = f"{level:<8}"

        return f"{timestamp} {level_text} {name}: {formatted}"

    def _format_trace(self, payload: Dict[str, Any]) -> str:
        event = payload.get("event", "trace")
        call_sid = payload.get("call_sid", "-")
        turn = payload.get("turn_id")
        # Shorten UUIDs to first 8 characters for readability
        if call_sid and len(call_sid) > 8:
            call_sid = call_sid[:8]

        # Start with event
        parts = [f"event={event}"]

        # Add turn if present (usually present)
        if turn is not None:
            parts.append(f"turn={turn}")

        # Add other important fields (excluding verbose IDs)
        for key, value in payload.items():
            if key in {"event", "call_sid", "turn_id", "seq", "trace_id"}:
                continue  # Skip already handled or verbose fields
            if value in (None, "", []):
                continue
            parts.append(f"{key}={value}")

        return " ".join(parts)

    @staticmethod
    def _should_use_color() -> bool:
        choice = os.getenv("LOG_COLOR", "auto").lower()
        if choice in {"0", "false", "off"}:
            return False
        if choice in {"1", "true", "on"}:
            return True
        return sys.stderr.isatty()


class _JsonFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_payload = getattr(record, "trace_payload", None)
        if trace_payload:
            payload.update(trace_payload)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

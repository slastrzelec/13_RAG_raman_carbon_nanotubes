"""
Structured logging configuration (Phase 2).

Provides JSON-formatted logs instead of plain text — easier to parse, filter,
and eventually feed into monitoring tools (e.g. Grafana, which is already in
the project's tech stack).

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("something happened", extra={"question": "...", "top_k": 5})
"""
import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    # Standard LogRecord attributes we don't want to duplicate in the "extra" payload
    _RESERVED_ATTRS = set(logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include any custom fields passed via `extra={...}` in the logging call
        for key, value in record.__dict__.items():
            if key not in self._RESERVED_ATTRS:
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger configured to emit single-line JSON to stdout.
    Safe to call multiple times with the same name — won't duplicate handlers.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False  # avoid duplicate logs via the root logger

    return logger

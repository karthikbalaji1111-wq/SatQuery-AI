"""Logging configuration."""

from __future__ import annotations

import json
import logging
import sys

from app.core.observability import RunIdFilter

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for a log platform to index.

    Carries the same fields as the text format and nothing more: a log line
    never gains a payload, a question or an address by changing format.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure root logging once, writing one line per record to stdout."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    # Every line carries the run it belongs to, or "-" outside a workflow. The
    # filter is on the HANDLER so it applies to records from every logger,
    # including third-party ones, without each having to opt in.
    handler.addFilter(RunIdFilter())
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s [%(run_id)s] | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""

    return logging.getLogger(f"satquery.{name}")

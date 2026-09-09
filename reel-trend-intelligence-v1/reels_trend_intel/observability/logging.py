"""Structured logging. Uses structlog when available, falls back to stdlib."""

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover - fallback path
    _HAS_STRUCTLOG = False

_CONFIGURED = False


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    lvl = getattr(logging, level.upper(), logging.INFO)

    if _HAS_STRUCTLOG:
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]
        processors.append(
            structlog.processors.JSONRenderer()
            if json_logs
            else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        )
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(lvl),
            processors=processors,
            cache_logger_on_first_use=True,
        )
    else:  # pragma: no cover
        logging.basicConfig(
            level=lvl,
            stream=sys.stderr,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


class _StdlibLogger:
    """Minimal structlog-compatible shim over stdlib logging."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, **kw: Any) -> None:
        if kw:
            event = event + " " + " ".join(f"{k}={v}" for k, v in kw.items())
        self._log.log(level, event)

    def debug(self, event: str, **kw: Any) -> None:
        self._emit(logging.DEBUG, event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._emit(logging.INFO, event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._emit(logging.WARNING, event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._emit(logging.ERROR, event, **kw)

    def bind(self, **_kw: Any) -> _StdlibLogger:
        return self


def get_logger(name: str = "rti") -> Any:
    if not _CONFIGURED:
        configure_logging()
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _StdlibLogger(name)

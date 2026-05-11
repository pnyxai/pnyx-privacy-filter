"""PCM structured logging using structlog.

Follows the same pattern as a-vert/a_vert/logger.py:
  - Per-module loggers via get_logger(__name__)
  - KeyValueRenderer for compact, human-readable lines:
      2026-04-25 01:00:00 INFO     [app.privacy_manager] event='session created' session_id=abc123
  - configure_logging() called once at app startup; honours PCM_LOG_LEVEL.
  - aiosqlite is always clamped to WARNING — its DEBUG output dumps full SQL
    statements with bound parameters and is pure internal implementation noise.
"""
from __future__ import annotations

import logging
import sys

import structlog

# Loggers that are unconditionally silenced below WARNING regardless of
# the application log level.
_ALWAYS_WARN: tuple[str, ...] = (
    "aiosqlite",
    "tritonclient",
    "httpcore",
)


def configure_logging(level: str) -> None:
    """Configure structlog and the stdlib root logger.

    Call once at application startup (inside create_app).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Always clamp noisy internal loggers.
    for name in _ALWAYS_WARN:
        logging.getLogger(name).setLevel(logging.WARNING)

    # httpx is informational at INFO but chatty at DEBUG; silence unless DEBUG
    # is explicitly requested.
    if numeric_level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.KeyValueRenderer(
                sort_keys=False,
                key_order=["event"],
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger bound to *name*."""
    return structlog.get_logger(name)


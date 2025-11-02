from __future__ import annotations

import contextvars
import logging
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional, Union

_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "dnr_run_id", default=None
)


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        record.run_id = _RUN_ID.get() or "-"
        return True


LogLevel = Union[int, str]


def init_logger(name: str, level: LogLevel = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)

    if isinstance(level, str):
        level_value = logging._nameToLevel.get(level.upper())
        if level_value is None:
            level_value = logging.INFO
        level = level_value

    logger.setLevel(level)

    if not any(isinstance(f, _RunIdFilter) for f in logger.filters):
        logger.addFilter(_RunIdFilter())

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(run_id)s] %(name)s: %(message)s",
            )
        )
        logger.addHandler(handler)

    logger.propagate = False
    return logger


@contextmanager
def run_context(logger: logging.Logger) -> Iterator[str]:
    if not any(isinstance(f, _RunIdFilter) for f in logger.filters):
        logger.addFilter(_RunIdFilter())

    run_id = uuid.uuid4().hex[:12]
    token = _RUN_ID.set(run_id)
    try:
        yield run_id
    finally:
        _RUN_ID.reset(token)


__all__ = ["init_logger", "run_context"]

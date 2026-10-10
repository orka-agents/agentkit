"""Keep OpenAI request diagnostics out of agentsessions execution logs."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock


_logger = logging.getLogger("openai._base_client")
_suppressed: ContextVar[bool] = ContextVar("agentsessions_sdk_diagnostics_suppressed", default=False)
_lock = Lock()
_active_scopes = 0


class _ExecutionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _suppressed.get()


_filter = _ExecutionFilter()


@contextmanager
def suppress_sdk_diagnostics() -> Iterator[None]:
    """Suppress ``openai._base_client`` only in this execution's context.

    Nesting and concurrent scopes share one owned filter without changing other
    filters, handlers, levels, or the environment. Filtering at the emitting
    logger drops records before any handler can format or enqueue their payload,
    even when a queue listener runs later in a different context.

    Keep the scope open until execution tasks and context-propagating workers
    finish. Threads without context propagation must enter their own scope.
    """
    global _active_scopes

    with _lock:
        if _active_scopes == 0:
            # Reject before other filters without mutating an in-flight iteration.
            _logger.filters = [_filter, *_logger.filters]
        _active_scopes += 1
    token = _suppressed.set(True)
    try:
        yield
    finally:
        _suppressed.reset(token)
        with _lock:
            _active_scopes -= 1
            if _active_scopes == 0:
                _logger.filters = [item for item in _logger.filters if item is not _filter]

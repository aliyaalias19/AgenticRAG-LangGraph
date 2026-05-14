"""
Circuit breaker for external service calls.

Prevents cascading failures when the LLM API or vector store is degraded.
After `failure_threshold` consecutive failures the breaker opens and rejects
calls immediately for `timeout` seconds, then enters half-open state.
"""

import threading
import time
from typing import Any, Callable


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0) -> None:
        self._failures = 0
        self._threshold = failure_threshold
        self._timeout = timeout
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._failures >= self._threshold:
                if time.time() - self._opened_at >= self._timeout:
                    # Half-open: allow one probe
                    self._failures = self._threshold - 1
                    return False
                return True
            return False

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if self.is_open:
            raise RuntimeError("Circuit breaker is open — downstream service unavailable")
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._failures = 0
            return result
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self._threshold:
                    self._opened_at = time.time()
            raise

"""Circuit breaker for tool dispatch — prevents hammering failing external services.

Each tool gets its own breaker instance stored in a global registry.
States: CLOSED (normal) → OPEN (blocking, after failure_threshold errors)
        → HALF_OPEN (probe allowed after reset_timeout_s) → CLOSED or OPEN.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import structlog

log = structlog.get_logger()

_FAILURE_THRESHOLD = 3   # consecutive errors to open the circuit
_RESET_TIMEOUT_S = 60.0  # seconds before half-open probe is allowed


class _State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ToolCircuitBreaker:
    failure_threshold: int = _FAILURE_THRESHOLD
    reset_timeout_s: float = _RESET_TIMEOUT_S

    _state: _State = field(default=_State.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def is_open(self, tool_name: str) -> bool:
        """Return True if the circuit is OPEN and the cooldown hasn't elapsed."""
        with self._lock:
            if self._state is _State.OPEN:
                if time.monotonic() - self._opened_at >= self.reset_timeout_s:
                    self._state = _State.HALF_OPEN
                    log.info("circuit_half_open", tool=tool_name)
                    return False
                return True
            return False

    def record_success(self, tool_name: str) -> None:
        with self._lock:
            if self._state is _State.HALF_OPEN:
                log.info("circuit_closed", tool=tool_name)
            self._state = _State.CLOSED
            self._failure_count = 0

    def record_failure(self, tool_name: str) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state is _State.HALF_OPEN or self._failure_count >= self.failure_threshold:
                if self._state is not _State.OPEN:
                    log.warning("circuit_opened", tool=tool_name, failures=self._failure_count)
                self._state = _State.OPEN
                self._opened_at = time.monotonic()
                self._failure_count = 0


_breakers: dict[str, ToolCircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(tool_name: str) -> ToolCircuitBreaker:
    """Return the singleton ToolCircuitBreaker for a given tool (created on first access)."""
    with _breakers_lock:
        if tool_name not in _breakers:
            _breakers[tool_name] = ToolCircuitBreaker()
        return _breakers[tool_name]


def reset_all() -> None:
    """Reset all circuit breakers (useful in tests)."""
    with _breakers_lock:
        _breakers.clear()

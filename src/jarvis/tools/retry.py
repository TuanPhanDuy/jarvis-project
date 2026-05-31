"""Exponential-backoff retry for tool calls.

Transient errors (network blips, rate limits, temporary unavailability) are
retried up to max_retries times with jittered exponential backoff. Permanent
errors (bad input, auth failures, unknown tool) are not retried.

Usage inside _dispatch:
    result = call_with_retry(handler, tool_input, max_retries=2, base_delay=1.0)
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable

_TRANSIENT_PATTERNS = (
    "timeout",
    "timed out",
    "connection",
    "network",
    "temporarily unavailable",
    "service unavailable",
    "too many requests",
    "rate limit",
    "503",
    "502",
    "429",
    "reset by peer",
    "connection refused",
    "broken pipe",
    "eof occurred",
    "remote end closed",
)

_PERMANENT_PATTERNS = (
    "unknown tool",
    "invalid input",
    "unauthorized",
    "forbidden",
    "not found",
    "404",
    "401",
    "403",
    "invalid api key",
    "circuit open",
)


def is_transient(error: str) -> bool:
    """Return True if the error string looks like a transient/retriable failure."""
    low = error.lower()
    if any(p in low for p in _PERMANENT_PATTERNS):
        return False
    return any(p in low for p in _TRANSIENT_PATTERNS)


def call_with_retry(
    handler: Callable[[dict], str],
    tool_input: dict,
    max_retries: int = 2,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> tuple[str, int]:
    """Call handler(tool_input), retrying transient errors with exponential backoff.

    Returns (result, attempts_used).
    attempts_used == 1 means success on first try (no retries needed).
    """
    last_result = ""
    for attempt in range(1, max_retries + 2):  # +2: initial + max_retries retries
        try:
            result = handler(tool_input)
        except Exception as exc:
            result = f"ERROR: {exc}"

        if not result.startswith("ERROR") or not is_transient(result):
            return result, attempt

        last_result = result
        if attempt <= max_retries:
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = delay * 0.25 * random.random()
            time.sleep(delay + jitter)

    return last_result, max_retries + 1

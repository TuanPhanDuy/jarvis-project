"""Tests for the tool circuit breaker."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from jarvis.tools.circuit_breaker import ToolCircuitBreaker, get_breaker, reset_all, _FAILURE_THRESHOLD


class TestToolCircuitBreaker:
    def setup_method(self):
        reset_all()

    def test_starts_closed(self):
        cb = ToolCircuitBreaker()
        assert cb.is_open("tool") is False

    def test_opens_after_failure_threshold(self):
        cb = ToolCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure("tool")
        assert cb.is_open("tool") is True

    def test_does_not_open_before_threshold(self):
        cb = ToolCircuitBreaker(failure_threshold=3)
        for _ in range(2):
            cb.record_failure("tool")
        assert cb.is_open("tool") is False

    def test_success_resets_failure_count(self):
        cb = ToolCircuitBreaker(failure_threshold=3)
        cb.record_failure("tool")
        cb.record_failure("tool")
        cb.record_success("tool")
        cb.record_failure("tool")
        cb.record_failure("tool")
        # Only 2 failures since last success — circuit still closed
        assert cb.is_open("tool") is False

    def test_transitions_to_half_open_after_timeout(self):
        cb = ToolCircuitBreaker(failure_threshold=1, reset_timeout_s=0.01)
        cb.record_failure("tool")
        assert cb.is_open("tool") is True
        time.sleep(0.02)
        # Should now be half-open (returns False = allow probe)
        assert cb.is_open("tool") is False

    def test_half_open_success_closes_circuit(self):
        cb = ToolCircuitBreaker(failure_threshold=1, reset_timeout_s=0.01)
        cb.record_failure("tool")
        time.sleep(0.02)
        cb.is_open("tool")  # transitions to half-open
        cb.record_success("tool")
        assert cb.is_open("tool") is False

    def test_half_open_failure_reopens_circuit(self):
        cb = ToolCircuitBreaker(failure_threshold=1, reset_timeout_s=0.01)
        cb.record_failure("tool")
        time.sleep(0.02)
        cb.is_open("tool")  # half-open
        cb.record_failure("tool")
        assert cb.is_open("tool") is True


class TestGetBreaker:
    def setup_method(self):
        reset_all()

    def test_same_tool_returns_same_instance(self):
        b1 = get_breaker("web_search")
        b2 = get_breaker("web_search")
        assert b1 is b2

    def test_different_tools_return_different_instances(self):
        b1 = get_breaker("web_search")
        b2 = get_breaker("read_url")
        assert b1 is not b2

    def test_reset_all_clears_registry(self):
        b1 = get_breaker("web_search")
        reset_all()
        b2 = get_breaker("web_search")
        assert b1 is not b2

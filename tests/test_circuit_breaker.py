"""Tests for the tool circuit breaker."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from jarvis.tools.circuit_breaker import (
    ToolCircuitBreaker, get_breaker, get_all_states, reset_all, reset_breaker,
    update_breaker_config, _FAILURE_THRESHOLD,
)


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


class TestGetAllStates:
    def setup_method(self):
        reset_all()

    def test_empty_when_no_breakers(self):
        assert get_all_states() == []

    def test_returns_state_for_each_registered_tool(self):
        get_breaker("web_search")
        get_breaker("read_url")
        states = get_all_states()
        names = {s["tool"] for s in states}
        assert names == {"web_search", "read_url"}

    def test_closed_breaker_has_closed_state(self):
        get_breaker("web_search")
        states = get_all_states()
        assert states[0]["state"] == "closed"
        assert states[0]["opened_at"] is None

    def test_open_breaker_has_open_state(self):
        cb = get_breaker("web_search")
        for _ in range(_FAILURE_THRESHOLD):
            cb.record_failure("web_search")
        states = get_all_states()
        assert states[0]["state"] == "open"
        assert states[0]["opened_at"] is not None

    def test_failure_count_reflected(self):
        cb = get_breaker("web_search")
        cb.record_failure("web_search")
        states = get_all_states()
        # after 1 failure, count is 1 (not yet open)
        assert states[0]["failure_count"] == 1


class TestResetBreaker:
    def setup_method(self):
        reset_all()

    def test_returns_false_for_unknown_tool(self):
        assert reset_breaker("no_such_tool") is False

    def test_returns_true_for_known_tool(self):
        get_breaker("web_search")
        assert reset_breaker("web_search") is True

    def test_resets_open_breaker_to_closed(self):
        cb = get_breaker("web_search")
        for _ in range(_FAILURE_THRESHOLD):
            cb.record_failure("web_search")
        assert cb.is_open("web_search") is True
        reset_breaker("web_search")
        # after reset a new breaker is installed — fetch it
        new_cb = get_breaker("web_search")
        assert new_cb.is_open("web_search") is False

    def test_tool_still_tracked_after_reset(self):
        get_breaker("web_search")
        reset_breaker("web_search")
        states = get_all_states()
        names = {s["tool"] for s in states}
        assert "web_search" in names


class TestUpdateBreakerConfig:
    def setup_method(self):
        reset_all()

    def test_creates_breaker_if_absent(self):
        result = update_breaker_config("new_tool", failure_threshold=5)
        assert result["tool"] == "new_tool"
        assert result["failure_threshold"] == 5

    def test_updates_failure_threshold(self):
        get_breaker("web_search")
        result = update_breaker_config("web_search", failure_threshold=10)
        assert result["failure_threshold"] == 10

    def test_updates_reset_timeout_s(self):
        get_breaker("web_search")
        result = update_breaker_config("web_search", reset_timeout_s=120.0)
        assert result["reset_timeout_s"] == 120.0

    def test_updates_both_fields(self):
        result = update_breaker_config("read_url", failure_threshold=7, reset_timeout_s=30.0)
        assert result["failure_threshold"] == 7
        assert result["reset_timeout_s"] == 30.0

    def test_new_threshold_takes_effect(self):
        update_breaker_config("read_url", failure_threshold=2)
        cb = get_breaker("read_url")
        cb.record_failure("read_url")
        assert cb.is_open("read_url") is False
        cb.record_failure("read_url")
        assert cb.is_open("read_url") is True

    def test_returns_current_config_when_no_updates(self):
        update_breaker_config("read_url", failure_threshold=4, reset_timeout_s=45.0)
        # calling again with None values returns unchanged config
        result = update_breaker_config("read_url")
        assert result["failure_threshold"] == 4
        assert result["reset_timeout_s"] == 45.0

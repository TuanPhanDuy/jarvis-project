"""Tests for tool retry with exponential backoff."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from jarvis.tools.retry import call_with_retry, is_transient


class TestIsTransient:
    def test_timeout_is_transient(self):
        assert is_transient("ERROR: tool 'web_search' timed out after 30s")

    def test_connection_error_is_transient(self):
        assert is_transient("ERROR: connection refused")

    def test_rate_limit_is_transient(self):
        assert is_transient("ERROR: 429 too many requests")

    def test_503_is_transient(self):
        assert is_transient("ERROR: 503 service unavailable")

    def test_network_error_is_transient(self):
        assert is_transient("ERROR: network error occurred")

    def test_unknown_tool_is_permanent(self):
        assert not is_transient("ERROR: unknown tool 'foo'")

    def test_404_is_permanent(self):
        assert not is_transient("ERROR: 404 not found")

    def test_auth_failure_is_permanent(self):
        assert not is_transient("ERROR: unauthorized — invalid api key")

    def test_circuit_open_is_permanent(self):
        assert not is_transient("ERROR: tool 'web_search' circuit open")

    def test_success_string_not_transient(self):
        assert not is_transient("Results: 5 items found")

    def test_case_insensitive(self):
        assert is_transient("ERROR: Connection Reset By Peer")


class TestCallWithRetry:
    def test_success_on_first_try(self):
        handler = MagicMock(return_value="ok result")
        result, attempts = call_with_retry(handler, {"q": "test"}, max_retries=2, base_delay=0)
        assert result == "ok result"
        assert attempts == 1
        handler.assert_called_once()

    def test_retries_on_transient_error(self):
        handler = MagicMock(side_effect=[
            "ERROR: connection timeout",
            "success after retry",
        ])
        with patch("jarvis.tools.retry.time.sleep"):
            result, attempts = call_with_retry(handler, {}, max_retries=2, base_delay=0.001)
        assert result == "success after retry"
        assert attempts == 2
        assert handler.call_count == 2

    def test_no_retry_on_permanent_error(self):
        handler = MagicMock(return_value="ERROR: 404 not found")
        result, attempts = call_with_retry(handler, {}, max_retries=3, base_delay=0)
        assert result == "ERROR: 404 not found"
        assert attempts == 1
        handler.assert_called_once()

    def test_exhausts_retries_and_returns_last_error(self):
        handler = MagicMock(return_value="ERROR: connection refused")
        with patch("jarvis.tools.retry.time.sleep"):
            result, attempts = call_with_retry(handler, {}, max_retries=2, base_delay=0.001)
        assert result == "ERROR: connection refused"
        assert attempts == 3  # initial + 2 retries
        assert handler.call_count == 3

    def test_zero_retries_means_no_retry(self):
        handler = MagicMock(return_value="ERROR: connection timeout")
        result, attempts = call_with_retry(handler, {}, max_retries=0, base_delay=0)
        assert attempts == 1
        handler.assert_called_once()

    def test_exception_treated_as_error(self):
        handler = MagicMock(side_effect=RuntimeError("boom"))
        with patch("jarvis.tools.retry.time.sleep"):
            result, attempts = call_with_retry(handler, {}, max_retries=1, base_delay=0.001)
        assert "ERROR:" in result

    def test_sleep_called_between_retries(self):
        handler = MagicMock(return_value="ERROR: connection timeout")
        with patch("jarvis.tools.retry.time.sleep") as mock_sleep:
            call_with_retry(handler, {}, max_retries=2, base_delay=1.0)
        assert mock_sleep.call_count == 2

    def test_success_after_multiple_transient_failures(self):
        handler = MagicMock(side_effect=[
            "ERROR: 503 service unavailable",
            "ERROR: connection reset by peer",
            "final answer",
        ])
        with patch("jarvis.tools.retry.time.sleep"):
            result, attempts = call_with_retry(handler, {}, max_retries=3, base_delay=0.001)
        assert result == "final answer"
        assert attempts == 3

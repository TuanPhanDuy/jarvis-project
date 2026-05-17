"""Tests for failure pattern coaching injected into agent system prompts."""
from __future__ import annotations

import time

import pytest

from jarvis.agents.failure_coach import get_failure_warnings, _FAILURE_THRESHOLD
from jarvis.memory.failures import log_failure


class TestGetFailureWarnings:
    def test_returns_empty_for_none_db_path(self):
        assert get_failure_warnings(None) == ""

    def test_returns_empty_for_missing_db(self, tmp_path):
        result = get_failure_warnings(tmp_path / "nonexistent.db")
        assert result == ""

    def test_returns_empty_when_no_failures(self, tmp_path):
        db = tmp_path / "jarvis.db"
        # Create the db without any failures
        log_failure(db, "web_search", {}, "ERROR: this was one failure")
        result = get_failure_warnings(db, threshold=_FAILURE_THRESHOLD)
        # One failure is below threshold
        assert result == ""

    def test_warning_shown_at_threshold(self, tmp_path):
        db = tmp_path / "jarvis.db"
        for _ in range(_FAILURE_THRESHOLD):
            log_failure(db, "web_search", {"query": "test"}, "ERROR: rate limited")
        result = get_failure_warnings(db)
        assert "web_search" in result
        assert "TOOL WARNINGS" in result

    def test_warning_includes_failure_count(self, tmp_path):
        db = tmp_path / "jarvis.db"
        for _ in range(5):
            log_failure(db, "read_url", {"url": "http://example.com"}, "ERROR: timeout")
        result = get_failure_warnings(db)
        assert "5" in result

    def test_warning_includes_last_error(self, tmp_path):
        db = tmp_path / "jarvis.db"
        for _ in range(_FAILURE_THRESHOLD):
            log_failure(db, "read_url", {}, "ERROR: connection refused")
        result = get_failure_warnings(db)
        assert "connection refused" in result

    def test_multiple_failing_tools_all_listed(self, tmp_path):
        db = tmp_path / "jarvis.db"
        for _ in range(_FAILURE_THRESHOLD):
            log_failure(db, "web_search", {}, "ERROR: api limit")
            log_failure(db, "read_url", {}, "ERROR: timeout")
        result = get_failure_warnings(db)
        assert "web_search" in result
        assert "read_url" in result

    def test_custom_threshold_respected(self, tmp_path):
        db = tmp_path / "jarvis.db"
        for _ in range(2):
            log_failure(db, "web_search", {}, "ERROR: limit")
        # With threshold=2, this should show a warning
        result = get_failure_warnings(db, threshold=2)
        assert "web_search" in result
        # With threshold=5, it should not
        result2 = get_failure_warnings(db, threshold=5)
        assert result2 == ""


class TestBaseAgentIntegration:
    def test_coaching_prefix_called_in_run_turn(self, tmp_path, mock_ollama):
        """BaseAgent._coaching_prefix() is invoked and can return empty without crash."""
        from jarvis.agents.base_agent import BaseAgent

        class ConcreteAgent(BaseAgent):
            def get_system_prompt(self):
                return "You are JARVIS."

        agent = ConcreteAgent(
            model="llama3.2", max_tokens=512,
            tool_schemas=[], tool_registry={},
        )
        # Should not raise — coaching_prefix gracefully handles missing config
        prefix = agent._coaching_prefix()
        assert isinstance(prefix, str)

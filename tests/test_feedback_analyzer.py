"""Unit tests for the feedback analyzer. No API keys needed — Claude call is mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.evals.feedback_analyzer import (
    _fetch_bad_feedback,
    _fetch_failure_patterns,
    _fetch_stats,
    _format_feedback,
    _format_failures,
    run_analysis,
)
from jarvis.memory.feedback import log_feedback
from jarvis.memory.failures import log_failure


class TestFetchHelpers:
    def test_fetch_bad_feedback_empty(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = _fetch_bad_feedback(db)
        assert result == []

    def test_fetch_bad_feedback_filters_low_ratings(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s1", "good response", rating=5)
        log_feedback(db, "s2", "bad response", rating=1, comment="wrong")
        log_feedback(db, "s3", "mediocre", rating=2, comment="meh")
        result = _fetch_bad_feedback(db)
        assert len(result) == 2
        assert all(r["rating"] <= 2 for r in result)

    def test_fetch_failure_patterns_empty(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = _fetch_failure_patterns(db)
        assert result == []

    def test_fetch_failure_patterns_aggregates(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_failure(db, "web_search", {}, "ERROR: timeout")
        log_failure(db, "web_search", {}, "ERROR: timeout")
        log_failure(db, "save_report", {}, "ERROR: disk full")
        result = _fetch_failure_patterns(db)
        assert len(result) >= 1
        web_entry = next((r for r in result if r["tool_name"] == "web_search"), None)
        assert web_entry is not None
        assert web_entry["count"] == 2

    def test_fetch_stats_empty_db(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        stats = _fetch_stats(db)
        assert stats["total_feedback"] == 0
        assert stats["avg_rating"] == 0
        assert stats["total_failures"] == 0

    def test_fetch_stats_with_data(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s", "r1", rating=4)
        log_feedback(db, "s", "r2", rating=2)
        log_failure(db, "tool", {}, "err")
        stats = _fetch_stats(db)
        assert stats["total_feedback"] == 2
        assert stats["avg_rating"] == 3.0
        assert stats["total_failures"] == 1


class TestFormatters:
    def test_format_feedback_empty(self) -> None:
        result = _format_feedback([])
        assert "No low-rated" in result

    def test_format_feedback_with_items(self) -> None:
        import time
        items = [{"rating": 1, "comment": "bad", "ts": time.time()}]
        result = _format_feedback(items)
        assert "rating=1" in result
        assert "bad" in result

    def test_format_failures_empty(self) -> None:
        result = _format_failures([])
        assert "No tool failures" in result

    def test_format_failures_with_items(self) -> None:
        items = [{"tool_name": "web_search", "count": 3, "error_msg": "timeout"}]
        result = _format_failures(items)
        assert "web_search" in result
        assert "×3" in result


class TestRunAnalysis:
    def test_no_data_returns_early(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        mock_client = MagicMock()
        result = run_analysis(db, tmp_path, mock_client, "claude-haiku-4-5-20251001")
        assert "No feedback" in result
        mock_client.messages.create.assert_not_called()

    def test_with_data_calls_claude_and_saves(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s", "bad response", rating=1, comment="wrong answer")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="## Root Cause\nTest improvement report.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        result = run_analysis(db, tmp_path, mock_client, "claude-haiku-4-5-20251001")

        assert "improvement_suggestions.md" in result
        report_path = tmp_path / "improvement_suggestions.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "JARVIS Self-Improvement Analysis" in content
        assert "Test improvement report." in content

    def test_report_overwrites_previous(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s", "bad", rating=1)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Report v1")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        run_analysis(db, tmp_path, mock_client, "model")

        mock_response.content = [MagicMock(text="Report v2")]
        run_analysis(db, tmp_path, mock_client, "model")

        content = (tmp_path / "improvement_suggestions.md").read_text()
        assert "Report v2" in content
        assert "Report v1" not in content

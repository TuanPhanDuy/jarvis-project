"""Tests for GET /api/reminders, POST /api/memory/consolidate/{user_id},
and the get_reminders() helper."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _fake_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.anthropic_api_key = "test-key"
    s.tavily_api_key = "test-key"
    s.model = "llama3.2"
    s.fast_model = "llama3.2"
    s.max_tokens = 512
    s.max_search_calls = 5
    s.routing_strategy = "always_primary"
    s.allowed_commands = []
    s.reports_dir = tmp_path / "reports"
    s.otel_enabled = False
    s.auth_enabled = False
    s.rate_limit_enabled = False
    s.proactive_enabled = False
    s.peer_enabled = False
    s.api_session_ttl_minutes = 60
    s.memory_retention_days = 90
    s.jwt_secret = "test-secret"
    s.chat_rate_limit = "100/minute"
    s.idle_minutes = 30
    s.agent_turn_timeout_seconds = 120
    s.tool_timeout_seconds = 60
    s.peer_port = 8001
    s.vision_model = "llava:13b"
    return s


@pytest.fixture
def client(tmp_path: Path):
    settings = _fake_settings(tmp_path)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    with (
        patch("jarvis.api.server.get_settings", return_value=settings),
        patch("jarvis.config.get_settings", return_value=settings),
        patch("jarvis.scheduler.core.start_scheduler"),
        patch("jarvis.scheduler.core.stop_scheduler"),
        patch("jarvis.tools.registry.build_registry", return_value=([], {})),
    ):
        from fastapi.testclient import TestClient
        from jarvis.api.server import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, settings.reports_dir


# ── Unit: get_reminders ───────────────────────────────────────────────────────

class TestGetReminders:
    def test_returns_list_when_no_scheduler(self) -> None:
        from jarvis.tools.plugins.reminder_manager import get_reminders
        with patch("jarvis.tools.plugins.reminder_manager._get_scheduler", side_effect=Exception("no sched")):
            result = get_reminders()
        assert result == []

    def test_returns_empty_when_no_reminder_jobs(self) -> None:
        from jarvis.tools.plugins.reminder_manager import get_reminders
        mock_sched = MagicMock()
        mock_sched.get_jobs.return_value = []
        with patch("jarvis.tools.plugins.reminder_manager._get_scheduler", return_value=mock_sched):
            result = get_reminders()
        assert result == []

    def test_returns_reminder_entries(self) -> None:
        from datetime import datetime, timezone
        from jarvis.tools.plugins.reminder_manager import get_reminders
        mock_job = MagicMock()
        mock_job.id = "reminder_abc123"
        mock_job.kwargs = {"title": "Team meeting"}
        mock_job.next_run_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
        mock_sched = MagicMock()
        mock_sched.get_jobs.return_value = [mock_job]
        with patch("jarvis.tools.plugins.reminder_manager._get_scheduler", return_value=mock_sched):
            result = get_reminders()
        assert len(result) == 1
        assert result[0]["id"] == "reminder_abc123"
        assert result[0]["title"] == "Team meeting"
        assert result[0]["next_run"] is not None

    def test_non_reminder_jobs_excluded(self) -> None:
        from jarvis.tools.plugins.reminder_manager import get_reminders
        mock_job = MagicMock()
        mock_job.id = "builtin_memory_consolidate"
        mock_sched = MagicMock()
        mock_sched.get_jobs.return_value = [mock_job]
        with patch("jarvis.tools.plugins.reminder_manager._get_scheduler", return_value=mock_sched):
            result = get_reminders()
        assert result == []

    def test_none_next_run_handled(self) -> None:
        from jarvis.tools.plugins.reminder_manager import get_reminders
        mock_job = MagicMock()
        mock_job.id = "reminder_xyz"
        mock_job.kwargs = {"title": "Check-in"}
        mock_job.next_run_time = None
        mock_sched = MagicMock()
        mock_sched.get_jobs.return_value = [mock_job]
        with patch("jarvis.tools.plugins.reminder_manager._get_scheduler", return_value=mock_sched):
            result = get_reminders()
        assert result[0]["next_run"] is None


# ── Endpoint: GET /api/reminders ──────────────────────────────────────────────

class TestRemindersEndpoint:
    def test_returns_200(self, client) -> None:
        c, _ = client
        with patch("jarvis.tools.plugins.reminder_manager.get_reminders", return_value=[]):
            assert c.get("/api/reminders").status_code == 200

    def test_returns_empty_list(self, client) -> None:
        c, _ = client
        with patch("jarvis.tools.plugins.reminder_manager.get_reminders", return_value=[]):
            assert c.get("/api/reminders").json() == []

    def test_returns_reminder_items(self, client) -> None:
        c, _ = client
        fake = [{"id": "reminder_1", "title": "Stand-up", "next_run": "2026-06-01T09:00:00+00:00"}]
        with patch("jarvis.tools.plugins.reminder_manager.get_reminders", return_value=fake):
            data = c.get("/api/reminders").json()
        assert len(data) == 1
        assert data[0]["title"] == "Stand-up"


# ── Endpoint: POST /api/memory/consolidate/{user_id} ─────────────────────────

class TestConsolidateEndpoint:
    def test_returns_202(self, client) -> None:
        c, _ = client
        with patch("jarvis.memory.consolidator.consolidate_user_memory", return_value=5):
            resp = c.post("/api/memory/consolidate/alice")
        assert resp.status_code == 202

    def test_response_has_status_and_user_id(self, client) -> None:
        c, _ = client
        with patch("jarvis.memory.consolidator.consolidate_user_memory", return_value=3):
            data = c.post("/api/memory/consolidate/bob").json()
        assert data["status"] == "started"
        assert data["user_id"] == "bob"

    def test_lookback_hours_param_accepted(self, client) -> None:
        c, _ = client
        with patch("jarvis.memory.consolidator.consolidate_user_memory", return_value=0):
            resp = c.post("/api/memory/consolidate/carol?lookback_hours=48")
        assert resp.status_code == 202
        assert resp.json()["lookback_hours"] == 48

    def test_returns_202_even_if_consolidation_fails(self, client) -> None:
        c, _ = client
        # Consolidation failure is swallowed in background thread — endpoint still 202
        with patch("jarvis.memory.consolidator.consolidate_user_memory", side_effect=RuntimeError("ollama down")):
            resp = c.post("/api/memory/consolidate/dan")
        assert resp.status_code == 202

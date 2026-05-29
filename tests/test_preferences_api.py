"""Tests for preferences API endpoints and delete_preference helper."""
from __future__ import annotations

from pathlib import Path

import pytest


# ── Unit: delete_preference ───────────────────────────────────────────────────

class TestDeletePreference:
    def test_returns_false_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import delete_preference
        result = delete_preference(tmp_path / "missing.db", "alice", "style", "tone")
        assert result is False

    def test_returns_false_when_entry_not_found(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import delete_preference, upsert_preference
        db = tmp_path / "jarvis.db"
        upsert_preference(db, "alice", "style", "format", "markdown")
        result = delete_preference(db, "alice", "style", "nonexistent_key")
        assert result is False

    def test_returns_true_and_removes_entry(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import delete_preference, get_preferences, upsert_preference
        db = tmp_path / "jarvis.db"
        upsert_preference(db, "alice", "style", "tone", "concise")
        assert delete_preference(db, "alice", "style", "tone") is True
        prefs = get_preferences(db, "alice")
        assert "tone" not in prefs.get("style", {})

    def test_does_not_delete_other_users_preferences(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import delete_preference, get_preferences, upsert_preference
        db = tmp_path / "jarvis.db"
        upsert_preference(db, "alice", "style", "tone", "concise")
        upsert_preference(db, "bob", "style", "tone", "verbose")
        delete_preference(db, "alice", "style", "tone")
        bob_prefs = get_preferences(db, "bob")
        assert bob_prefs.get("style", {}).get("tone") == "verbose"

    def test_does_not_delete_other_categories(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import delete_preference, get_preferences, upsert_preference
        db = tmp_path / "jarvis.db"
        upsert_preference(db, "alice", "style", "tone", "concise")
        upsert_preference(db, "alice", "depth", "level", "expert")
        delete_preference(db, "alice", "style", "tone")
        prefs = get_preferences(db, "alice")
        assert prefs.get("depth", {}).get("level") == "expert"


# ── Endpoint tests ────────────────────────────────────────────────────────────

def _fake_settings(tmp_path: Path):
    from unittest.mock import MagicMock
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
    from unittest.mock import patch
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


class TestPreferencesEndpoints:
    def test_get_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/preferences/alice").status_code == 200

    def test_get_empty_for_unknown_user(self, client) -> None:
        c, _ = client
        assert c.get("/api/preferences/nobody").json() == []

    def test_get_returns_preferences_after_upsert(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.preferences import upsert_preference
        db = reports_dir / "jarvis.db"
        upsert_preference(db, "alice", "style", "tone", "concise")
        upsert_preference(db, "alice", "depth", "level", "expert")
        data = c.get("/api/preferences/alice").json()
        assert len(data) == 2
        categories = {e["category"] for e in data}
        assert "style" in categories
        assert "depth" in categories

    def test_get_includes_metadata_fields(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.preferences import upsert_preference
        upsert_preference(reports_dir / "jarvis.db", "bob", "style", "verbosity", "terse")
        data = c.get("/api/preferences/bob").json()
        assert len(data) == 1
        for key in ("category", "key", "value", "confidence", "updated_at", "source"):
            assert key in data[0]

    def test_delete_returns_204(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.preferences import upsert_preference
        upsert_preference(reports_dir / "jarvis.db", "carol", "style", "tone", "formal")
        resp = c.delete("/api/preferences/carol/style/tone")
        assert resp.status_code == 204

    def test_delete_removes_preference(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.preferences import upsert_preference
        db = reports_dir / "jarvis.db"
        upsert_preference(db, "dan", "style", "format", "markdown")
        c.delete("/api/preferences/dan/style/format")
        data = c.get("/api/preferences/dan").json()
        assert data == []

    def test_delete_404_for_nonexistent(self, client) -> None:
        c, _ = client
        resp = c.delete("/api/preferences/nobody/style/tone")
        assert resp.status_code == 404

    def test_get_only_returns_own_user_data(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.preferences import upsert_preference
        db = reports_dir / "jarvis.db"
        upsert_preference(db, "alice", "style", "tone", "concise")
        upsert_preference(db, "bob", "style", "tone", "verbose")
        alice_data = c.get("/api/preferences/alice").json()
        bob_data = c.get("/api/preferences/bob").json()
        assert len(alice_data) == 1
        assert len(bob_data) == 1
        assert alice_data[0]["value"] == "concise"
        assert bob_data[0]["value"] == "verbose"

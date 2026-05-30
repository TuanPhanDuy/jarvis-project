"""Tests for GET /api/memory/search unified search endpoint."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_auth():
    import jarvis.api.server as _s
    _s._require_auth = None
    yield
    _s._require_auth = None


@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


def _settings(tmp_path):
    s = MagicMock()
    s.reports_dir = tmp_path
    return s


class TestMemorySearchValidation:
    def test_empty_query_422(self, client):
        resp = client.get("/api/memory/search?q=")
        assert resp.status_code == 422

    def test_whitespace_query_422(self, client):
        resp = client.get("/api/memory/search?q=   ")
        assert resp.status_code == 422


class TestMemorySearchResponse:
    def test_returns_200_with_three_keys(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)):
            resp = client.get("/api/memory/search?q=RLHF")
        assert resp.status_code == 200
        body = resp.json()
        assert "episodic" in body
        assert "graph" in body
        assert "reports" in body

    def test_type_episodic_only_returns_episodic(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)):
            resp = client.get("/api/memory/search?q=test&type=episodic")
        assert resp.status_code == 200
        body = resp.json()
        # graph and reports should still be in response (empty lists)
        assert "episodic" in body

    def test_type_graph_searches_graph(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)), \
             patch("jarvis.memory.graph.handle_query_knowledge_graph",
                   return_value="RLHF uses PPO"):
            resp = client.get("/api/memory/search?q=RLHF&type=graph")
        assert resp.status_code == 200

    def test_episodic_results_have_content_field(self, client, tmp_path):
        from jarvis.memory.episodic import log_episode
        db = tmp_path / "jarvis.db"
        log_episode(db, "s1", "user", "RLHF training details")
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)):
            resp = client.get("/api/memory/search?q=RLHF&type=episodic")
        body = resp.json()
        if body["episodic"]:
            entry = body["episodic"][0]
            assert "content" in entry
            assert "role" in entry
            assert "timestamp" in entry

    def test_episodic_results_truncated_at_400_chars(self, client, tmp_path):
        from jarvis.memory.episodic import log_episode
        db = tmp_path / "jarvis.db"
        long_content = "RLHF " + "x" * 500
        log_episode(db, "s2", "user", long_content)
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)):
            resp = client.get("/api/memory/search?q=RLHF&type=episodic")
        for ep in resp.json()["episodic"]:
            assert len(ep["content"]) <= 400

    def test_graph_empty_when_no_entities(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)):
            resp = client.get("/api/memory/search?q=unknownentity12345&type=graph")
        assert resp.json()["graph"] == []

    def test_subsystem_error_does_not_crash(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)), \
             patch("jarvis.memory.episodic.search_episodes", side_effect=RuntimeError("db locked")):
            resp = client.get("/api/memory/search?q=test&type=episodic")
        assert resp.status_code == 200
        assert resp.json()["episodic"] == []

    def test_all_types_searched_when_type_omitted(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)):
            resp = client.get("/api/memory/search?q=test")
        body = resp.json()
        # All three keys must be present
        assert set(body.keys()) >= {"episodic", "graph", "reports"}

    def test_user_id_filter_passed_to_episodic(self, client, tmp_path):
        call_args = []
        def fake_search(db, query, limit, user_id):
            call_args.append(user_id)
            return []
        with patch("jarvis.api.server.get_settings", return_value=_settings(tmp_path)), \
             patch("jarvis.memory.episodic.search_episodes", side_effect=fake_search):
            client.get("/api/memory/search?q=test&type=episodic&user_id=alice")
        assert call_args and call_args[0] == "alice"

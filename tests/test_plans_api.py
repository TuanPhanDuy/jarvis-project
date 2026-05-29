"""Tests for GET /api/plans, GET /api/plans/{plan_id}, and executor plan helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


# ── Unit: get_plans / get_plan ────────────────────────────────────────────────

class TestGetPlans:
    def _insert_plan(self, db_path: Path, plan_id: str, goal: str, status: str = "done",
                     session_id: str = "s1", user_id: str = "alice") -> None:
        from jarvis.agents.executor import _get_conn
        conn = _get_conn(db_path)
        steps = [{"id": "1", "description": "step one", "agent_type": "researcher",
                  "depends_on": [], "n_agents": 1, "result": None, "status": "done"}]
        conn.execute(
            "INSERT INTO plans (id, goal, steps_json, status, created_at, session_id, user_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (plan_id, goal, json.dumps(steps), status, time.time(), session_id, user_id),
        )
        conn.commit()
        conn.close()

    def test_returns_empty_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import get_plans
        assert get_plans(tmp_path / "missing.db") == []

    def test_returns_inserted_plan(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import get_plans
        db = tmp_path / "jarvis.db"
        self._insert_plan(db, "plan-001", "Research AI safety")
        result = get_plans(db)
        assert len(result) == 1
        assert result[0]["id"] == "plan-001"
        assert result[0]["goal"] == "Research AI safety"

    def test_steps_parsed_from_json(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import get_plans
        db = tmp_path / "jarvis.db"
        self._insert_plan(db, "plan-002", "Write report")
        result = get_plans(db)
        assert isinstance(result[0]["steps"], list)
        assert result[0]["steps"][0]["description"] == "step one"

    def test_filter_by_session_id(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import get_plans
        db = tmp_path / "jarvis.db"
        self._insert_plan(db, "p1", "goal1", session_id="session-A")
        self._insert_plan(db, "p2", "goal2", session_id="session-B")
        result = get_plans(db, session_id="session-A")
        assert len(result) == 1
        assert result[0]["id"] == "p1"

    def test_filter_by_user_id(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import get_plans
        db = tmp_path / "jarvis.db"
        self._insert_plan(db, "p1", "goal1", user_id="alice")
        self._insert_plan(db, "p2", "goal2", user_id="bob")
        result = get_plans(db, user_id="bob")
        assert len(result) == 1
        assert result[0]["id"] == "p2"

    def test_limit_respected(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import get_plans
        db = tmp_path / "jarvis.db"
        for i in range(5):
            self._insert_plan(db, f"plan-{i}", f"goal {i}")
        result = get_plans(db, limit=3)
        assert len(result) == 3

    def test_ordered_newest_first(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import _get_conn, get_plans
        db = tmp_path / "jarvis.db"
        conn = _get_conn(db)
        for i, ts in enumerate([1000.0, 2000.0, 3000.0]):
            conn.execute(
                "INSERT INTO plans (id, goal, steps_json, status, created_at) VALUES (?,?,?,?,?)",
                (f"p{i}", f"goal{i}", "[]", "done", ts),
            )
        conn.commit()
        conn.close()
        result = get_plans(db)
        assert result[0]["id"] == "p2"  # newest first


class TestGetPlan:
    def test_returns_none_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import get_plan
        assert get_plan(tmp_path / "missing.db", "x") is None

    def test_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import _get_conn, get_plan
        db = tmp_path / "jarvis.db"
        _get_conn(db).close()  # create tables
        assert get_plan(db, "no-such-plan") is None

    def test_returns_plan_by_id(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import _get_conn, get_plan
        db = tmp_path / "jarvis.db"
        conn = _get_conn(db)
        conn.execute(
            "INSERT INTO plans (id, goal, steps_json, status, created_at) VALUES (?,?,?,?,?)",
            ("my-plan", "build something", "[]", "running", time.time()),
        )
        conn.commit()
        conn.close()
        result = get_plan(db, "my-plan")
        assert result is not None
        assert result["goal"] == "build something"
        assert result["status"] == "running"

    def test_steps_deserialized(self, tmp_path: Path) -> None:
        from jarvis.agents.executor import _get_conn, get_plan
        db = tmp_path / "jarvis.db"
        steps = [{"id": "s1", "description": "do thing"}]
        conn = _get_conn(db)
        conn.execute(
            "INSERT INTO plans (id, goal, steps_json, status, created_at) VALUES (?,?,?,?,?)",
            ("plan-x", "goal", json.dumps(steps), "done", time.time()),
        )
        conn.commit()
        conn.close()
        result = get_plan(db, "plan-x")
        assert result["steps"] == steps


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


class TestPlansEndpoints:
    def _seed(self, db_path: Path, plan_id: str = "p1", goal: str = "do work") -> None:
        from jarvis.agents.executor import _get_conn
        conn = _get_conn(db_path)
        conn.execute(
            "INSERT INTO plans (id, goal, steps_json, status, created_at) VALUES (?,?,?,?,?)",
            (plan_id, goal, "[]", "done", time.time()),
        )
        conn.commit()
        conn.close()

    def test_list_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/plans").status_code == 200

    def test_list_empty_when_no_db(self, client) -> None:
        c, _ = client
        assert c.get("/api/plans").json() == []

    def test_list_returns_plan(self, client) -> None:
        c, reports_dir = client
        self._seed(reports_dir / "jarvis.db", "abc", "my goal")
        data = c.get("/api/plans").json()
        assert len(data) == 1
        assert data[0]["goal"] == "my goal"

    def test_get_by_id_returns_plan(self, client) -> None:
        c, reports_dir = client
        self._seed(reports_dir / "jarvis.db", "xyz", "specific goal")
        data = c.get("/api/plans/xyz").json()
        assert data["id"] == "xyz"
        assert data["goal"] == "specific goal"

    def test_get_by_id_404_for_unknown(self, client) -> None:
        c, _ = client
        assert c.get("/api/plans/no-such-plan").status_code == 404

    def test_list_filter_session_id(self, client) -> None:
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.agents.executor import _get_conn
        conn = _get_conn(db)
        for pid, sid in [("p1", "s-alpha"), ("p2", "s-beta")]:
            conn.execute(
                "INSERT INTO plans (id, goal, steps_json, status, created_at, session_id) VALUES (?,?,?,?,?,?)",
                (pid, "goal", "[]", "done", time.time(), sid),
            )
        conn.commit()
        conn.close()
        data = c.get("/api/plans?session_id=s-alpha").json()
        assert len(data) == 1
        assert data[0]["id"] == "p1"

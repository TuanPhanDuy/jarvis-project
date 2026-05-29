"""Tests for:
  - PATCH /api/tools/circuit-breakers/{tool_name}
  - PATCH /api/schedules/{job_id}
  - GET/PUT /api/tools/cache/ttls[/{tool_name}]
  - GET/POST/DELETE /api/evals/cases
"""
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


# ── PATCH /api/tools/circuit-breakers/{tool_name} ─────────────────────────────

class TestPatchCircuitBreaker:
    def test_returns_200(self, client) -> None:
        c, _ = client
        resp = c.patch("/api/tools/circuit-breakers/web_search",
                       json={"failure_threshold": 5})
        assert resp.status_code == 200

    def test_updates_failure_threshold(self, client) -> None:
        c, _ = client
        data = c.patch("/api/tools/circuit-breakers/web_search",
                       json={"failure_threshold": 7}).json()
        assert data["failure_threshold"] == 7
        assert data["tool"] == "web_search"

    def test_updates_reset_timeout_s(self, client) -> None:
        c, _ = client
        data = c.patch("/api/tools/circuit-breakers/web_search",
                       json={"reset_timeout_s": 90.0}).json()
        assert data["reset_timeout_s"] == 90.0

    def test_updates_both_fields(self, client) -> None:
        c, _ = client
        data = c.patch("/api/tools/circuit-breakers/read_url",
                       json={"failure_threshold": 4, "reset_timeout_s": 30.0}).json()
        assert data["failure_threshold"] == 4
        assert data["reset_timeout_s"] == 30.0

    def test_422_when_no_fields_provided(self, client) -> None:
        c, _ = client
        resp = c.patch("/api/tools/circuit-breakers/web_search", json={})
        assert resp.status_code == 422

    def test_422_for_zero_failure_threshold(self, client) -> None:
        c, _ = client
        resp = c.patch("/api/tools/circuit-breakers/web_search",
                       json={"failure_threshold": 0})
        assert resp.status_code == 422

    def test_422_for_negative_reset_timeout(self, client) -> None:
        c, _ = client
        resp = c.patch("/api/tools/circuit-breakers/web_search",
                       json={"reset_timeout_s": -1.0})
        assert resp.status_code == 422

    def test_creates_breaker_for_new_tool(self, client) -> None:
        c, _ = client
        data = c.patch("/api/tools/circuit-breakers/brand_new_tool",
                       json={"failure_threshold": 3}).json()
        assert data["tool"] == "brand_new_tool"


# ── PATCH /api/schedules/{job_id} ─────────────────────────────────────────────

class TestPatchSchedule:
    def test_returns_200(self, client) -> None:
        c, _ = client
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()
        with patch("jarvis.scheduler.core.get_scheduler", return_value=mock_sched):
            resp = c.patch("/api/schedules/some-job-id", json={"cron": "0 9 * * 1"})
        assert resp.status_code == 200

    def test_response_contains_job_id(self, client) -> None:
        c, _ = client
        mock_sched = MagicMock()
        with patch("jarvis.scheduler.core.get_scheduler", return_value=mock_sched):
            data = c.patch("/api/schedules/my-job", json={"cron": "0 9 * * 1"}).json()
        assert data["job_id"] == "my-job"

    def test_422_for_missing_cron(self, client) -> None:
        c, _ = client
        resp = c.patch("/api/schedules/my-job", json={})
        assert resp.status_code == 422

    def test_422_for_invalid_cron(self, client) -> None:
        c, _ = client
        mock_sched = MagicMock()
        with patch("jarvis.scheduler.core.get_scheduler", return_value=mock_sched):
            resp = c.patch("/api/schedules/my-job", json={"cron": "not-a-cron"})
        assert resp.status_code == 422

    def test_404_for_unknown_job(self, client) -> None:
        c, _ = client
        from apscheduler.jobstores.base import JobLookupError
        mock_sched = MagicMock()
        mock_sched.reschedule_job.side_effect = JobLookupError("no-job")
        with patch("jarvis.scheduler.core.get_scheduler", return_value=mock_sched):
            resp = c.patch("/api/schedules/no-such-job", json={"cron": "0 9 * * 1"})
        assert resp.status_code == 404

    def test_503_when_scheduler_not_running(self, client) -> None:
        c, _ = client
        with patch("jarvis.scheduler.core.get_scheduler", return_value=None):
            resp = c.patch("/api/schedules/any-job", json={"cron": "0 9 * * 1"})
        assert resp.status_code == 503

    def test_calls_reschedule_job_with_trigger(self, client) -> None:
        c, _ = client
        mock_sched = MagicMock()
        with patch("jarvis.scheduler.core.get_scheduler", return_value=mock_sched):
            c.patch("/api/schedules/target-job", json={"cron": "30 6 * * *"})
        mock_sched.reschedule_job.assert_called_once()
        call_args = mock_sched.reschedule_job.call_args
        assert call_args[0][0] == "target-job"


# ── GET/PUT /api/tools/cache/ttls ─────────────────────────────────────────────

class TestCacheTtlEndpoints:
    def test_get_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/tools/cache/ttls").status_code == 200

    def test_get_returns_dict(self, client) -> None:
        c, _ = client
        data = c.get("/api/tools/cache/ttls").json()
        assert isinstance(data, dict)

    def test_get_includes_web_search(self, client) -> None:
        c, _ = client
        data = c.get("/api/tools/cache/ttls").json()
        assert "web_search" in data

    def test_put_returns_200(self, client) -> None:
        c, _ = client
        resp = c.put("/api/tools/cache/ttls/web_search", json={"ttl_seconds": 7200})
        assert resp.status_code == 200

    def test_put_updates_ttl(self, client) -> None:
        c, _ = client
        c.put("/api/tools/cache/ttls/web_search", json={"ttl_seconds": 9999})
        data = c.get("/api/tools/cache/ttls").json()
        assert data["web_search"] == 9999

    def test_put_zero_removes_tool(self, client) -> None:
        c, _ = client
        c.put("/api/tools/cache/ttls/web_search", json={"ttl_seconds": 0})
        data = c.get("/api/tools/cache/ttls").json()
        assert "web_search" not in data

    def test_put_422_when_missing_ttl_seconds(self, client) -> None:
        c, _ = client
        resp = c.put("/api/tools/cache/ttls/web_search", json={})
        assert resp.status_code == 422

    def test_put_422_for_negative_ttl(self, client) -> None:
        c, _ = client
        resp = c.put("/api/tools/cache/ttls/web_search", json={"ttl_seconds": -1})
        assert resp.status_code == 422


# ── GET/POST/DELETE /api/evals/cases ─────────────────────────────────────────

class TestEvalCasesEndpoints:
    def test_get_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/evals/cases").status_code == 200

    def test_get_returns_list(self, client) -> None:
        c, _ = client
        data = c.get("/api/evals/cases").json()
        assert isinstance(data, list)

    def test_get_includes_baseline_cases(self, client) -> None:
        c, _ = client
        data = c.get("/api/evals/cases").json()
        ids = {c["id"] for c in data}
        assert "rlhf_basics" in ids

    def test_post_returns_201(self, client) -> None:
        c, _ = client
        resp = c.post("/api/evals/cases", json={"id": "my_test", "prompt": "What is RLHF?"})
        assert resp.status_code == 201

    def test_post_returns_created_case(self, client) -> None:
        c, _ = client
        data = c.post("/api/evals/cases",
                      json={"id": "new_case", "prompt": "Explain attention"}).json()
        assert data["id"] == "new_case"
        assert data["prompt"] == "Explain attention"

    def test_post_case_appears_in_list(self, client) -> None:
        c, _ = client
        c.post("/api/evals/cases", json={"id": "another_case", "prompt": "test prompt"})
        ids = {item["id"] for item in c.get("/api/evals/cases").json()}
        assert "another_case" in ids

    def test_post_422_for_missing_id(self, client) -> None:
        c, _ = client
        resp = c.post("/api/evals/cases", json={"prompt": "no id"})
        assert resp.status_code == 422

    def test_post_422_for_missing_prompt(self, client) -> None:
        c, _ = client
        resp = c.post("/api/evals/cases", json={"id": "no_prompt"})
        assert resp.status_code == 422

    def test_post_409_for_duplicate_id(self, client) -> None:
        c, _ = client
        c.post("/api/evals/cases", json={"id": "dup", "prompt": "first"})
        resp = c.post("/api/evals/cases", json={"id": "dup", "prompt": "second"})
        assert resp.status_code == 409

    def test_delete_returns_204(self, client) -> None:
        c, _ = client
        c.post("/api/evals/cases", json={"id": "del_me", "prompt": "remove this"})
        resp = c.delete("/api/evals/cases/del_me")
        assert resp.status_code == 204

    def test_delete_removes_case_from_list(self, client) -> None:
        c, _ = client
        c.post("/api/evals/cases", json={"id": "temp_case", "prompt": "ephemeral"})
        c.delete("/api/evals/cases/temp_case")
        ids = {item["id"] for item in c.get("/api/evals/cases").json()}
        assert "temp_case" not in ids

    def test_delete_404_for_unknown_case(self, client) -> None:
        c, _ = client
        resp = c.delete("/api/evals/cases/no_such_case")
        assert resp.status_code == 404

    def test_post_stores_optional_fields(self, client) -> None:
        c, _ = client
        payload = {
            "id": "full_case",
            "prompt": "Explain RLHF",
            "expected_contains": ["reward", "human"],
            "forbidden": ["I don't know"],
            "tags": ["ml"],
            "timeout_seconds": 60,
        }
        data = c.post("/api/evals/cases", json=payload).json()
        assert data["expected_contains"] == ["reward", "human"]
        assert data["tags"] == ["ml"]
        assert data["timeout_seconds"] == 60

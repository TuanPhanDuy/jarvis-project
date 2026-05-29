"""Unit tests for eval framework scoring logic. No API keys needed."""
from __future__ import annotations

import json
from pathlib import Path

from jarvis.evals.suite import EvalCase, BASELINE_SUITE
from jarvis.evals.runner import EvalResult, _score_case, _git_hash, persist_results, summarize


class TestEvalCase:
    def test_baseline_suite_not_empty(self) -> None:
        assert len(BASELINE_SUITE) >= 4

    def test_evalcase_defaults(self) -> None:
        case = EvalCase(id="test", prompt="What is X?")
        assert case.expected_contains == []
        assert case.forbidden == []
        assert case.tags == []
        assert case.judge_rubric == ""


class TestScoreCase:
    def _case(self, expected=None, forbidden=None):
        return EvalCase(
            id="t",
            prompt="test",
            expected_contains=expected or [],
            forbidden=forbidden or [],
        )

    def test_all_expected_present(self) -> None:
        case = self._case(expected=["reward", "human feedback"])
        ok, _, failed, _ = _score_case(case, "RLHF uses reward modeling and human feedback signals.")
        assert ok is True
        assert failed == []

    def test_missing_expected_fails(self) -> None:
        case = self._case(expected=["reward", "PPO"])
        ok, _, failed, _ = _score_case(case, "Only reward is mentioned here.")
        assert ok is False
        assert "PPO" in failed

    def test_forbidden_present_fails(self) -> None:
        case = self._case(forbidden=["I don't know"])
        _, ok, _, found = _score_case(case, "I don't know the answer.")
        assert ok is False
        assert "I don't know" in found

    def test_forbidden_absent_passes(self) -> None:
        case = self._case(forbidden=["ERROR"])
        _, ok, _, found = _score_case(case, "Here is a helpful answer.")
        assert ok is True
        assert found == []

    def test_case_insensitive_matching(self) -> None:
        case = self._case(expected=["RLHF"])
        ok, _, failed, _ = _score_case(case, "rlhf is a technique.")
        assert ok is True

    def test_empty_response_fails_expected(self) -> None:
        case = self._case(expected=["something"])
        ok, _, failed, _ = _score_case(case, "")
        assert ok is False

    def test_no_constraints_always_passes(self) -> None:
        case = self._case()
        c_ok, f_ok, failed, found = _score_case(case, "anything at all")
        assert c_ok is True
        assert f_ok is True


class TestGitHash:
    def test_returns_string(self):
        result = _git_hash()
        assert isinstance(result, str)

    def test_persist_results_includes_git_hash(self, tmp_path: Path):
        results: list[EvalResult] = []
        summary = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0}
        persist_results(results, summary, tmp_path)
        line = (tmp_path / "eval_history.jsonl").read_text().strip()
        record = json.loads(line)
        assert "git_hash" in record
        assert isinstance(record["git_hash"], str)

    def test_persist_results_appends_multiple_runs(self, tmp_path: Path):
        summary = {"total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0}
        persist_results([], summary, tmp_path)
        persist_results([], summary, tmp_path)
        lines = (tmp_path / "eval_history.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2


class TestSummarize:
    def _result(self, passed: bool, latency=1.0, cost=0.001, judge=None) -> EvalResult:
        return EvalResult(
            case_id="x", prompt="p", response="r",
            contains_pass=passed, forbidden_pass=passed, overall_pass=passed,
            latency_s=latency, cost_usd=cost, judge_score=judge,
        )

    def test_all_pass(self) -> None:
        results = [self._result(True), self._result(True)]
        s = summarize(results)
        assert s["passed"] == 2
        assert s["failed"] == 0
        assert s["pass_rate"] == 1.0

    def test_partial_pass(self) -> None:
        results = [self._result(True), self._result(False), self._result(False)]
        s = summarize(results)
        assert s["passed"] == 1
        assert s["failed"] == 2
        assert abs(s["pass_rate"] - 0.333) < 0.001

    def test_empty_results(self) -> None:
        s = summarize([])
        assert s["total"] == 0
        assert s["pass_rate"] == 0

    def test_cost_summed(self) -> None:
        results = [self._result(True, cost=0.001), self._result(True, cost=0.002)]
        s = summarize(results)
        assert abs(s["total_cost_usd"] - 0.003) < 0.000001

    def test_avg_latency(self) -> None:
        results = [self._result(True, latency=2.0), self._result(True, latency=4.0)]
        s = summarize(results)
        assert s["avg_latency_s"] == 3.0

    def test_judge_score_averaged(self) -> None:
        results = [self._result(True, judge=4), self._result(True, judge=2)]
        s = summarize(results)
        assert s["avg_judge_score"] == 3.0

    def test_no_judge_scores_is_none(self) -> None:
        results = [self._result(True), self._result(False)]
        s = summarize(results)
        assert s["avg_judge_score"] is None


# ── API endpoint tests ────────────────────────────────────────────────────────

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


def _fake_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.anthropic_api_key = "test-key"
    s.tavily_api_key = "test-key"
    s.model = "claude-sonnet-4-6"
    s.fast_model = "claude-haiku-4-5-20251001"
    s.max_tokens = 1024
    s.max_search_calls = 20
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


def _make_fake_eval_result(case_id: str = "test_case", passed: bool = True) -> "EvalResult":
    return EvalResult(
        case_id=case_id,
        prompt="test prompt",
        response="test response",
        contains_pass=passed,
        forbidden_pass=passed,
        overall_pass=passed,
        latency_s=0.5,
        cost_usd=0.001,
    )


@pytest.fixture
def eval_client(tmp_path: Path):
    settings = _fake_settings(tmp_path)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("jarvis.api.server.get_settings", return_value=settings),
        patch("jarvis.config.get_settings", return_value=settings),
        patch("jarvis.scheduler.core.start_scheduler"),
        patch("jarvis.scheduler.core.stop_scheduler"),
    ):
        from jarvis.api.server import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, settings


class TestEvalsEndpoint:
    def _mock_run_suite(self, results=None):
        if results is None:
            results = [_make_fake_eval_result("rlhf_basics", True)]
        return results

    def test_post_evals_returns_summary(self, eval_client) -> None:
        client, settings = eval_client
        fake_results = [_make_fake_eval_result("rlhf_basics", True)]

        with (
            patch("jarvis.evals.runner.run_suite", return_value=fake_results),
            patch("jarvis.evals.runner.persist_results"),
        ):
            resp = client.post("/api/evals", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["passed"] == 1
        assert body["pass_rate"] == 1.0
        assert "run_id" in body
        assert len(body["results"]) == 1

    def test_post_evals_failed_case_reflected(self, eval_client) -> None:
        client, settings = eval_client
        fake_results = [_make_fake_eval_result("bad_case", False)]

        with (
            patch("jarvis.evals.runner.run_suite", return_value=fake_results),
            patch("jarvis.evals.runner.persist_results"),
        ):
            resp = client.post("/api/evals", json={})
        body = resp.json()
        assert body["failed"] == 1
        assert body["pass_rate"] == 0.0
        assert body["results"][0]["overall_pass"] is False

    def test_post_evals_passes_tags_to_run_suite(self, eval_client) -> None:
        client, settings = eval_client
        fake_results = [_make_fake_eval_result()]

        with (
            patch("jarvis.evals.runner.run_suite", return_value=fake_results) as mock_run,
            patch("jarvis.evals.runner.persist_results"),
        ):
            client.post("/api/evals", json={"tags": ["ml", "rlhf"]})
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["tags_filter"] == ["ml", "rlhf"]

    def test_post_evals_passes_use_judge(self, eval_client) -> None:
        client, settings = eval_client
        fake_results = [_make_fake_eval_result()]

        with (
            patch("jarvis.evals.runner.run_suite", return_value=fake_results) as mock_run,
            patch("jarvis.evals.runner.persist_results"),
        ):
            client.post("/api/evals", json={"use_judge": True})
        assert mock_run.call_args.kwargs["use_judge"] is True

    def test_get_evals_results_empty_when_no_history(self, eval_client) -> None:
        client, settings = eval_client
        resp = client.get("/api/evals/results")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_evals_results_returns_written_records(self, eval_client) -> None:
        client, settings = eval_client
        # Write a record directly
        history_path = settings.reports_dir / "eval_history.jsonl"
        import json as _json
        record = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "git_hash": "abc123",
            "summary": {"total": 2, "passed": 2, "failed": 0, "pass_rate": 1.0,
                        "avg_latency_s": 1.0, "total_cost_usd": 0.002, "avg_judge_score": None},
            "results": [],
        }
        history_path.write_text(_json.dumps(record) + "\n")

        resp = client.get("/api/evals/results")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["total"] == 2
        assert data[0]["git_hash"] == "abc123"
        assert data[0]["timestamp"] == "2026-01-01T00:00:00+00:00"

    def test_get_evals_results_limit(self, eval_client) -> None:
        client, settings = eval_client
        history_path = settings.reports_dir / "eval_history.jsonl"
        import json as _json
        summary = {"total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0,
                   "avg_latency_s": 0.5, "total_cost_usd": 0.001, "avg_judge_score": None}
        for i in range(5):
            record = {"timestamp": f"2026-01-0{i+1}T00:00:00+00:00", "git_hash": "", "summary": summary, "results": []}
            history_path.open("a").write(_json.dumps(record) + "\n")

        resp = client.get("/api/evals/results?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

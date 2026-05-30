"""Tests for eval trend tracking: SQLite persistence and API endpoints."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from jarvis.evals.trend import get_run, get_trend, record_run


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "jarvis.db"


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestRecordRun:
    def test_record_and_retrieve(self, db):
        record_run(db, "run1", total=10, passed=8, failed=2, pass_rate=0.8)
        trend = get_trend(db, last_n=5)
        assert len(trend) == 1
        assert trend[0]["run_id"] == "run1"
        assert trend[0]["pass_rate"] == 0.8

    def test_stores_all_fields(self, db):
        record_run(db, "r2", total=5, passed=5, failed=0, pass_rate=1.0,
                   avg_latency_s=1.23, total_cost_usd=0.01,
                   avg_judge_score=4.5, tags=["ml", "safety"])
        row = get_trend(db, last_n=1)[0]
        assert row["avg_latency_s"] == pytest.approx(1.23)
        assert row["total_cost_usd"] == pytest.approx(0.01)
        assert row["avg_judge_score"] == pytest.approx(4.5)
        assert "ml" in row["tags"]

    def test_upsert_on_duplicate_run_id(self, db):
        record_run(db, "r3", total=5, passed=3, failed=2, pass_rate=0.6)
        record_run(db, "r3", total=5, passed=5, failed=0, pass_rate=1.0)
        trend = get_trend(db, last_n=10)
        assert len(trend) == 1
        assert trend[0]["pass_rate"] == 1.0

    def test_never_raises(self):
        record_run(Path("/nonexistent/path/db.sqlite"),
                   "r0", total=1, passed=1, failed=0, pass_rate=1.0)


class TestGetTrend:
    def test_empty_db_returns_empty(self, db):
        assert get_trend(db) == []

    def test_nonexistent_db_returns_empty(self, tmp_path):
        assert get_trend(tmp_path / "missing.db") == []

    def test_ordered_newest_first(self, db):
        record_run(db, "a", total=1, passed=1, failed=0, pass_rate=1.0)
        time.sleep(0.01)
        record_run(db, "b", total=1, passed=0, failed=1, pass_rate=0.0)
        trend = get_trend(db)
        assert trend[0]["run_id"] == "b"
        assert trend[1]["run_id"] == "a"

    def test_last_n_respected(self, db):
        for i in range(5):
            record_run(db, f"run{i}", total=1, passed=1, failed=0, pass_rate=1.0)
        assert len(get_trend(db, last_n=3)) == 3

    def test_required_fields_present(self, db):
        record_run(db, "check", total=2, passed=1, failed=1, pass_rate=0.5)
        row = get_trend(db)[0]
        for f in ("run_id", "timestamp", "total", "passed", "failed", "pass_rate"):
            assert f in row


class TestGetRun:
    def test_returns_run_with_results(self, db):
        results = [{"case_id": "c1", "overall_pass": True}]
        record_run(db, "r1", total=1, passed=1, failed=0, pass_rate=1.0, results=results)
        run = get_run(db, "r1")
        assert run is not None
        assert run["run_id"] == "r1"
        assert run["results"][0]["case_id"] == "c1"

    def test_returns_none_for_unknown(self, db):
        assert get_run(db, "no-such-run") is None

    def test_nonexistent_db_returns_none(self, tmp_path):
        assert get_run(tmp_path / "missing.db", "r0") is None


# ── API endpoints ─────────────────────────────────────────────────────────────

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


@pytest.fixture()
def patched_db(tmp_path):
    import jarvis.api.server as _s
    settings = type("S", (), {"reports_dir": tmp_path})()
    with patch("jarvis.api.server.get_settings", return_value=settings):
        yield tmp_path


class TestEvalTrendEndpoint:
    def test_trend_returns_200(self, client, patched_db):
        resp = client.get("/api/evals/trend")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_trend_empty_initially(self, client, patched_db):
        resp = client.get("/api/evals/trend")
        assert resp.json() == []

    def test_trend_returns_recorded_runs(self, client, patched_db):
        record_run(patched_db / "jarvis.db", "t1", 5, 4, 1, 0.8)
        resp = client.get("/api/evals/trend")
        assert len(resp.json()) == 1
        assert resp.json()[0]["run_id"] == "t1"

    def test_trend_last_n_param(self, client, patched_db):
        for i in range(5):
            record_run(patched_db / "jarvis.db", f"t{i}", 1, 1, 0, 1.0)
        resp = client.get("/api/evals/trend?last_n=3")
        assert len(resp.json()) == 3


class TestEvalRunDetailEndpoint:
    def test_existing_run_200(self, client, patched_db):
        results = [{"case_id": "c1", "overall_pass": True}]
        record_run(patched_db / "jarvis.db", "det1", 1, 1, 0, 1.0, results=results)
        resp = client.get("/api/evals/runs/det1")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "det1"

    def test_run_includes_results(self, client, patched_db):
        results = [{"case_id": "c2", "overall_pass": False}]
        record_run(patched_db / "jarvis.db", "det2", 1, 0, 1, 0.0, results=results)
        resp = client.get("/api/evals/runs/det2")
        assert resp.json()["results"][0]["case_id"] == "c2"

    def test_missing_run_404(self, client, patched_db):
        resp = client.get("/api/evals/runs/nonexistent-run")
        assert resp.status_code == 404

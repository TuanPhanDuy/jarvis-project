"""Tests for webhook and notification wiring into real system events."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "jarvis.db"


# ── _fire helper ──────────────────────────────────────────────────────────────

class TestFireHelper:
    def test_fire_calls_push_notification(self, db):
        from jarvis.scheduler.core import _fire
        with patch("jarvis.events.notifications.push_notification") as mock_push:
            _fire("system.info", {"x": 1}, db, "Test title")
        mock_push.assert_called_once()
        args = mock_push.call_args
        assert args[1]["event"] == "system.info" or args[0][1] == "system.info"

    def test_fire_calls_fire_event(self, db):
        from jarvis.scheduler.core import _fire
        with patch("jarvis.events.webhooks.fire_event") as mock_fire:
            _fire("training.complete", {"docs": 5}, db, "Crawl done")
        mock_fire.assert_called_once()

    def test_fire_never_raises_on_notification_error(self, db):
        from jarvis.scheduler.core import _fire
        with patch("jarvis.events.notifications.push_notification",
                   side_effect=RuntimeError("db locked")):
            _fire("tool.error", {}, db, "Error")  # must not raise

    def test_fire_never_raises_on_webhook_error(self, db):
        from jarvis.scheduler.core import _fire
        with patch("jarvis.events.webhooks.fire_event",
                   side_effect=RuntimeError("network timeout")):
            _fire("tool.error", {}, db, "Error")  # must not raise

    def test_fire_passes_severity(self, db):
        from jarvis.scheduler.core import _fire
        captured = {}
        def fake_push(db, event, title, severity, body):
            captured["severity"] = severity
        with patch("jarvis.events.notifications.push_notification", side_effect=fake_push):
            _fire("tool.error", {}, db, "Error!", severity="error")
        assert captured.get("severity") == "error"


# ── Auto-crawl job ────────────────────────────────────────────────────────────

class TestAutoCrawlEventFiring:
    def test_fires_training_complete_on_success(self, db, tmp_path):
        from jarvis.scheduler.core import _auto_crawl_job

        with patch("jarvis.scheduler.core.get_settings") as mock_cfg, \
             patch("jarvis.scheduler.core.start_run", return_value=1), \
             patch("jarvis.scheduler.core.complete_run"), \
             patch("jarvis.scheduler.core.ResearchCrawler") as mock_crawler_cls, \
             patch("jarvis.scheduler.core._fire") as mock_fire:

            settings = MagicMock()
            settings.auto_training_topics = "RLHF"
            settings.training_max_papers_per_source = 3
            mock_cfg.return_value = settings

            crawler = MagicMock()
            crawler.crawl_arxiv.return_value = []
            crawler.crawl_hf_blog.return_value = []
            crawler.crawl_anthropic.return_value = []
            crawler.crawl_papers_with_code.return_value = []
            crawler.ingest_all.return_value = []
            mock_crawler_cls.return_value = crawler

            _auto_crawl_job(str(db), str(tmp_path))

        events = [c[0][0] for c in mock_fire.call_args_list]
        assert "training.complete" in events

    def test_fires_tool_error_on_failure(self, db, tmp_path):
        from jarvis.scheduler.core import _auto_crawl_job

        with patch("jarvis.scheduler.core.get_settings", side_effect=RuntimeError("config error")), \
             patch("jarvis.scheduler.core._fire") as mock_fire:
            _auto_crawl_job(str(db), str(tmp_path))

        events = [c[0][0] for c in mock_fire.call_args_list]
        assert "tool.error" in events


# ── Auto-finetune job ─────────────────────────────────────────────────────────

class TestAutoFinetuneEventFiring:
    def test_fires_tool_error_when_settings_fail(self, db, tmp_path):
        from jarvis.scheduler.core import _auto_finetune_job

        with patch("jarvis.scheduler.core.get_settings", side_effect=RuntimeError("cfg error")), \
             patch("jarvis.scheduler.core._fire") as mock_fire:
            _auto_finetune_job(str(db), str(tmp_path))

        events = [c[0][0] for c in mock_fire.call_args_list]
        assert "tool.error" in events


# ── Eval API event firing ──────────────────────────────────────────────────────

class TestEvalApiEventFiring:
    @pytest.fixture(autouse=True)
    def reset_auth(self):
        import jarvis.api.server as _s
        _s._require_auth = None
        yield
        _s._require_auth = None

    def test_eval_run_pushes_notification(self, tmp_path):
        from fastapi.testclient import TestClient
        from jarvis.api.server import app

        mock_result = MagicMock()
        mock_result.case_id = "c1"
        mock_result.overall_pass = True
        mock_result.contains_pass = True
        mock_result.forbidden_pass = True
        mock_result.latency_s = 0.5
        mock_result.cost_usd = 0.0
        mock_result.judge_score = None
        mock_result.error = ""

        settings = MagicMock()
        settings.reports_dir = tmp_path

        with patch("jarvis.api.server.get_settings", return_value=settings), \
             patch("jarvis.evals.runner.run_suite", return_value=[mock_result]), \
             patch("jarvis.evals.runner.summarize", return_value={
                 "total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0,
                 "avg_latency_s": 0.5, "total_cost_usd": 0.0, "avg_judge_score": None,
             }), \
             patch("jarvis.evals.runner.persist_results"), \
             patch("jarvis.evals.trend.record_run"), \
             patch("jarvis.events.notifications.push_notification") as mock_push, \
             patch("jarvis.events.webhooks.fire_event") as mock_fire:

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/evals", json={"tags": [], "use_judge": False})

        assert resp.status_code == 200
        mock_push.assert_called_once()
        mock_fire.assert_called_once()

    def test_eval_notification_has_correct_event(self, tmp_path):
        from fastapi.testclient import TestClient
        from jarvis.api.server import app

        mock_result = MagicMock()
        mock_result.case_id = "c1"
        mock_result.overall_pass = True
        mock_result.contains_pass = True
        mock_result.forbidden_pass = True
        mock_result.latency_s = 0.5
        mock_result.cost_usd = 0.0
        mock_result.judge_score = None
        mock_result.error = ""

        settings = MagicMock()
        settings.reports_dir = tmp_path
        captured = {}

        def fake_push(db, event, title, severity, **kw):
            captured["event"] = event

        with patch("jarvis.api.server.get_settings", return_value=settings), \
             patch("jarvis.evals.runner.run_suite", return_value=[mock_result]), \
             patch("jarvis.evals.runner.summarize", return_value={
                 "total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0,
                 "avg_latency_s": 0.5, "total_cost_usd": 0.0,
             }), \
             patch("jarvis.evals.runner.persist_results"), \
             patch("jarvis.evals.trend.record_run"), \
             patch("jarvis.events.notifications.push_notification", side_effect=fake_push), \
             patch("jarvis.events.webhooks.fire_event"):

            client = TestClient(app, raise_server_exceptions=False)
            client.post("/api/evals", json={"tags": [], "use_judge": False})

        assert captured.get("event") == "eval.complete"

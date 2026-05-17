"""Tests for the auto-training system: tracking, scheduler jobs, and API endpoints."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── TrainingRun tracker ──────────────────────────────────────────────────────

class TestTrainingTracker:
    def test_start_run_inserts_record(self, tmp_path):
        from jarvis.training.tracking import get_history, start_run
        db = tmp_path / "jarvis.db"
        run_id = start_run(db, "crawl")
        assert run_id > 0
        history = get_history(db)
        assert len(history) == 1
        assert history[0].run_type == "crawl"
        assert history[0].status == "running"

    def test_complete_run_updates_record(self, tmp_path):
        from jarvis.training.tracking import complete_run, get_history, start_run
        db = tmp_path / "jarvis.db"
        run_id = start_run(db, "finetune")
        complete_run(db, run_id, docs_crawled=10, pairs_generated=50, model_name="jarvis-ft")
        history = get_history(db)
        r = history[0]
        assert r.status == "completed"
        assert r.docs_crawled == 10
        assert r.pairs_generated == 50
        assert r.model_name == "jarvis-ft"
        assert r.completed_at is not None

    def test_complete_run_with_failed_status(self, tmp_path):
        from jarvis.training.tracking import complete_run, get_history, start_run
        db = tmp_path / "jarvis.db"
        run_id = start_run(db, "crawl")
        complete_run(db, run_id, status="failed", notes="connection error")
        assert get_history(db)[0].status == "failed"

    def test_get_last_run_returns_most_recent_completed(self, tmp_path):
        from jarvis.training.tracking import complete_run, get_last_run, start_run
        db = tmp_path / "jarvis.db"
        id1 = start_run(db, "crawl")
        complete_run(db, id1, docs_crawled=3)
        id2 = start_run(db, "crawl")
        complete_run(db, id2, docs_crawled=7)
        last = get_last_run(db, "crawl")
        assert last.docs_crawled == 7

    def test_get_last_run_returns_none_when_no_completed(self, tmp_path):
        from jarvis.training.tracking import get_last_run, start_run
        db = tmp_path / "jarvis.db"
        start_run(db, "crawl")  # running, not completed
        assert get_last_run(db, "crawl") is None

    def test_get_history_respects_limit(self, tmp_path):
        from jarvis.training.tracking import complete_run, get_history, start_run
        db = tmp_path / "jarvis.db"
        for _ in range(5):
            rid = start_run(db, "crawl")
            complete_run(db, rid)
        assert len(get_history(db, limit=3)) == 3

    def test_count_new_docs_since(self, tmp_path):
        from jarvis.training.tracking import count_new_docs_since
        reports = tmp_path / "reports"
        reports.mkdir()
        old_file = reports / "research_old.md"
        new_file = reports / "research_new.md"
        old_file.write_text("old content")
        time.sleep(0.05)
        cutoff = time.time()
        time.sleep(0.05)
        new_file.write_text("new content")
        count = count_new_docs_since(tmp_path / "jarvis.db", cutoff, reports)
        assert count == 1

    def test_count_new_docs_ignores_non_research_files(self, tmp_path):
        from jarvis.training.tracking import count_new_docs_since
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "other_report.md").write_text("not a research file")
        count = count_new_docs_since(tmp_path / "jarvis.db", 0.0, reports)
        assert count == 0


# ── Scheduler job functions ──────────────────────────────────────────────────

class TestAutoCrawlJob:
    def test_auto_crawl_job_records_run_in_db(self, tmp_path):
        from jarvis.scheduler.core import _auto_crawl_job
        from jarvis.training.tracking import get_history

        db = tmp_path / "jarvis.db"
        reports = tmp_path / "reports"
        reports.mkdir()

        mock_settings = MagicMock()
        mock_settings.auto_training_topics = "RLHF"
        mock_settings.training_max_papers_per_source = 2
        mock_settings.reports_dir = reports

        mock_crawler = MagicMock()
        mock_crawler.crawl_arxiv.return_value = [MagicMock()]
        mock_crawler.crawl_hf_blog.return_value = []
        mock_crawler.crawl_anthropic.return_value = []
        mock_crawler.crawl_papers_with_code.return_value = []
        mock_crawler.ingest_all.return_value = ["doc_paper.md"]

        with (
            patch("jarvis.scheduler.core.get_settings", return_value=mock_settings),
            patch("jarvis.scheduler.core.ResearchCrawler", return_value=mock_crawler),
            patch("jarvis.scheduler.core.start_run", return_value=1) as mock_start,
            patch("jarvis.scheduler.core.complete_run") as mock_complete,
        ):
            _auto_crawl_job(str(db), str(reports))

        mock_start.assert_called_once_with(db, "crawl")
        mock_complete.assert_called_once()
        kwargs = mock_complete.call_args.kwargs
        assert kwargs.get("docs_crawled", 0) >= 1

    def test_auto_crawl_job_handles_exception_gracefully(self, tmp_path):
        from jarvis.scheduler.core import _auto_crawl_job
        db = tmp_path / "jarvis.db"
        reports = tmp_path / "reports"
        reports.mkdir()

        with patch("jarvis.scheduler.core.get_settings", side_effect=RuntimeError("boom")):
            # Should not raise
            _auto_crawl_job(str(db), str(reports))


class TestAutoFinetuneJob:
    def test_auto_finetune_job_skips_when_no_api_key(self, tmp_path):
        from jarvis.scheduler.core import _auto_finetune_job

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = ""

        with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
            _auto_finetune_job(str(tmp_path / "jarvis.db"), str(tmp_path))
        # Should return early without error

    def test_auto_finetune_job_skips_when_insufficient_new_docs(self, tmp_path):
        from jarvis.scheduler.core import _auto_finetune_job

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.auto_training_min_new_docs = 10
        mock_settings.training_data_dir = tmp_path / "training"

        with (
            patch("jarvis.scheduler.core.get_settings", return_value=mock_settings),
            patch("jarvis.scheduler.core.get_last_run", return_value=None),
            patch("jarvis.scheduler.core.count_new_docs_since", return_value=3),
            patch("jarvis.training.data_generator.TrainingDataGenerator") as mock_gen,
        ):
            _auto_finetune_job(str(tmp_path / "jarvis.db"), str(tmp_path))
            mock_gen.assert_not_called()


# ── _parse_cron ──────────────────────────────────────────────────────────────

class TestParseCron:
    def test_parses_valid_five_field_expression(self):
        from jarvis.scheduler.core import _parse_cron
        trigger = _parse_cron("0 3 * * 0")
        assert trigger is not None

    def test_raises_on_invalid_field_count(self):
        import pytest
        from jarvis.scheduler.core import _parse_cron
        with pytest.raises(ValueError, match="5 fields"):
            _parse_cron("* * *")

    def test_daily_midnight_expression(self):
        from jarvis.scheduler.core import _parse_cron
        trigger = _parse_cron("0 0 * * *")
        assert trigger is not None

    def test_weekly_sunday_expression(self):
        from jarvis.scheduler.core import _parse_cron
        trigger = _parse_cron("0 3 * * 0")
        assert trigger is not None


# ── Register auto-training jobs ──────────────────────────────────────────────

class TestRegisterAutoTrainingJobs:
    def test_registers_jobs_when_enabled(self, tmp_path):
        from jarvis.scheduler.core import _register_auto_training_jobs

        mock_settings = MagicMock()
        mock_settings.auto_training_enabled = True
        mock_settings.auto_crawl_cron = "0 1 * * *"
        mock_settings.auto_finetune_cron = "0 3 * * 0"

        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None  # not registered yet

        with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
            _register_auto_training_jobs(mock_scheduler, tmp_path / "jarvis.db", tmp_path)

        assert mock_scheduler.add_job.call_count == 2

    def test_skips_registration_when_disabled(self, tmp_path):
        from jarvis.scheduler.core import _register_auto_training_jobs

        mock_settings = MagicMock()
        mock_settings.auto_training_enabled = False
        mock_scheduler = MagicMock()

        with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
            _register_auto_training_jobs(mock_scheduler, tmp_path / "jarvis.db", tmp_path)

        mock_scheduler.add_job.assert_not_called()

    def test_skips_already_registered_jobs(self, tmp_path):
        from jarvis.scheduler.core import _register_auto_training_jobs

        mock_settings = MagicMock()
        mock_settings.auto_training_enabled = True
        mock_settings.auto_crawl_cron = "0 1 * * *"
        mock_settings.auto_finetune_cron = "0 3 * * 0"

        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = MagicMock()  # already registered

        with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
            _register_auto_training_jobs(mock_scheduler, tmp_path / "jarvis.db", tmp_path)

        mock_scheduler.add_job.assert_not_called()


# ── API endpoints ─────────────────────────────────────────────────────────────

class TestTrainingAPI:
    def _client(self):
        from fastapi.testclient import TestClient
        from jarvis.api.server import app
        return TestClient(app)

    def test_training_status_returns_config(self):
        client = self._client()
        with (
            patch("jarvis.api.server.get_settings") as mock_settings,
            patch("jarvis.training.tracking.get_last_run", return_value=None),
        ):
            s = MagicMock()
            s.auto_training_enabled = False
            s.auto_training_topics = "RLHF"
            s.auto_crawl_cron = "0 1 * * *"
            s.auto_finetune_cron = "0 3 * * 0"
            s.auto_training_model_name = "jarvis-ft"
            s.auto_training_min_new_docs = 5
            s.reports_dir = Path("/tmp")
            mock_settings.return_value = s

            resp = client.get("/api/training/status")

        assert resp.status_code == 200
        data = resp.json()
        assert "auto_training_enabled" in data
        assert "topics" in data
        assert "next_crawl" in data

    def test_training_history_returns_list(self):
        client = self._client()
        with (
            patch("jarvis.api.server.get_settings") as mock_settings,
            patch("jarvis.training.tracking.get_history", return_value=[]),
        ):
            s = MagicMock()
            s.reports_dir = Path("/tmp")
            mock_settings.return_value = s
            resp = client.get("/api/training/history")

        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

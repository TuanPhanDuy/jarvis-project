"""Tests for scheduler/core.py: built-in job registration and job functions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from jarvis.scheduler.core import (
    JOB_FUNCTIONS,
    _add_builtin_jobs,
    _eval_check_job,
    _feedback_analyze_job,
    _memory_consolidation_job,
    _prune_memory_job,
    _system_snapshot_job,
    stop_scheduler,
)


# ── JOB_FUNCTIONS registry ────────────────────────────────────────────────────


def test_job_functions_registry_has_all_six_types() -> None:
    expected = {"research", "monitor", "memory_consolidate", "digest",
                "feedback_analyze", "system_snapshot", "eval_check", "prune_memory"}
    assert expected.issubset(set(JOB_FUNCTIONS.keys()))


# ── _add_builtin_jobs ─────────────────────────────────────────────────────────


def test_add_builtin_jobs_registers_five_jobs(tmp_path: Path) -> None:
    scheduler = MagicMock()
    scheduler.get_job.return_value = None  # no existing jobs
    db_path = tmp_path / "jarvis.db"

    mock_settings = MagicMock()
    mock_settings.auto_training_enabled = False

    with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
        _add_builtin_jobs(scheduler, db_path, tmp_path)

    assert scheduler.add_job.call_count == 5
    job_ids = [kw["id"] for _, kw in scheduler.add_job.call_args_list]
    assert "builtin_memory_consolidate" in job_ids
    assert "builtin_feedback_analyze" in job_ids
    assert "builtin_system_snapshot" in job_ids
    assert "builtin_graph_dedup" in job_ids
    assert "builtin_prune_memory" in job_ids


def test_add_builtin_jobs_skips_already_registered(tmp_path: Path) -> None:
    scheduler = MagicMock()
    scheduler.get_job.return_value = MagicMock()  # all jobs already exist
    db_path = tmp_path / "jarvis.db"

    _add_builtin_jobs(scheduler, db_path, tmp_path)

    scheduler.add_job.assert_not_called()


def test_add_builtin_jobs_partially_skips(tmp_path: Path) -> None:
    def _get_job(job_id):
        return MagicMock() if job_id == "builtin_memory_consolidate" else None

    scheduler = MagicMock()
    scheduler.get_job.side_effect = _get_job
    db_path = tmp_path / "jarvis.db"

    mock_settings = MagicMock()
    mock_settings.auto_training_enabled = False

    with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
        _add_builtin_jobs(scheduler, db_path, tmp_path)

    assert scheduler.add_job.call_count == 4  # only the four missing ones added


# ── Individual job functions ──────────────────────────────────────────────────


def test_memory_consolidation_job_calls_consolidator(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    with (
        patch("jarvis.config.get_settings") as mock_settings,
        patch("jarvis.memory.consolidator.get_all_user_ids", return_value=["alice", "bob"]) as mock_users,
        patch("jarvis.memory.consolidator.consolidate_user_memory", return_value=3) as mock_consolidate,
    ):
        mock_settings.return_value.model = "llama3.2"
        _memory_consolidation_job(str(db))

    mock_users.assert_called_once()
    assert mock_consolidate.call_count == 2


def test_system_snapshot_job_calls_take_snapshot(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    with patch("jarvis.twin.main.take_snapshot") as mock_snap:
        _system_snapshot_job(str(db))
    mock_snap.assert_called_once_with(db_path=db)


def test_feedback_analyze_job_calls_run_analysis(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    with (
        patch("jarvis.config.get_settings") as mock_settings,
        patch("jarvis.evals.feedback_analyzer.run_analysis", return_value="ok") as mock_run,
    ):
        mock_settings.return_value.model = "llama3.2"
        _feedback_analyze_job(str(db), str(tmp_path))

    mock_run.assert_called_once()


# ── stop_scheduler ────────────────────────────────────────────────────────────


def test_stop_scheduler_shuts_down_running_scheduler() -> None:
    import jarvis.scheduler.core as sched_mod

    mock_sched = MagicMock()
    mock_sched.running = True
    sched_mod._scheduler = mock_sched

    stop_scheduler()

    mock_sched.shutdown.assert_called_once_with(wait=False)
    sched_mod._scheduler = None  # restore


# ── _eval_check_job ───────────────────────────────────────────────────────────


def _make_eval_result(passed: bool):
    from jarvis.evals.runner import EvalResult
    return EvalResult(
        case_id="test", prompt="p", response="r",
        contains_pass=passed, forbidden_pass=passed, overall_pass=passed,
        latency_s=0.1, cost_usd=0.001,
    )


def test_eval_check_job_no_finetune_when_pass_rate_above_threshold(tmp_path: Path) -> None:
    results = [_make_eval_result(True), _make_eval_result(True)]  # pass_rate = 1.0

    with (
        patch("jarvis.scheduler.core.get_settings") as mock_settings,
        patch("jarvis.evals.runner.run_suite", return_value=results),
        patch("jarvis.evals.runner.persist_results"),
        patch("jarvis.scheduler.core._auto_finetune_job") as mock_ft,
    ):
        mock_settings.return_value.eval_pass_rate_threshold = 0.8
        _eval_check_job(str(tmp_path / "jarvis.db"), str(tmp_path))

    mock_ft.assert_not_called()


def test_eval_check_job_triggers_finetune_when_below_threshold(tmp_path: Path) -> None:
    results = [_make_eval_result(True), _make_eval_result(False)]  # pass_rate = 0.5

    with (
        patch("jarvis.scheduler.core.get_settings") as mock_settings,
        patch("jarvis.evals.runner.run_suite", return_value=results),
        patch("jarvis.evals.runner.persist_results"),
        patch("jarvis.scheduler.core._auto_finetune_job") as mock_ft,
    ):
        mock_settings.return_value.eval_pass_rate_threshold = 0.8
        _eval_check_job(str(tmp_path / "jarvis.db"), str(tmp_path))

    mock_ft.assert_called_once()


def test_eval_check_job_exactly_at_threshold_no_finetune(tmp_path: Path) -> None:
    results = [_make_eval_result(True), _make_eval_result(True),
               _make_eval_result(False), _make_eval_result(False),
               _make_eval_result(True)]  # pass_rate = 0.6

    with (
        patch("jarvis.scheduler.core.get_settings") as mock_settings,
        patch("jarvis.evals.runner.run_suite", return_value=results),
        patch("jarvis.evals.runner.persist_results"),
        patch("jarvis.scheduler.core._auto_finetune_job") as mock_ft,
    ):
        mock_settings.return_value.eval_pass_rate_threshold = 0.5  # 0.6 >= 0.5 → no trigger
        _eval_check_job(str(tmp_path / "jarvis.db"), str(tmp_path))

    mock_ft.assert_not_called()


def test_eval_check_job_swallows_exception(tmp_path: Path) -> None:
    with (
        patch("jarvis.scheduler.core.get_settings") as mock_settings,
        patch("jarvis.evals.runner.run_suite", side_effect=RuntimeError("boom")),
    ):
        mock_settings.return_value.eval_pass_rate_threshold = 0.8
        _eval_check_job(str(tmp_path / "jarvis.db"), str(tmp_path))  # should not raise


def test_auto_eval_job_registered_when_enabled(tmp_path: Path) -> None:
    scheduler = MagicMock()
    scheduler.get_job.return_value = None
    db_path = tmp_path / "jarvis.db"

    mock_settings = MagicMock()
    mock_settings.auto_training_enabled = True
    mock_settings.auto_eval_enabled = True
    mock_settings.auto_crawl_cron = "0 1 * * *"
    mock_settings.auto_finetune_cron = "0 3 * * 0"
    mock_settings.eval_check_cron = "0 22 * * 6"

    with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
        _add_builtin_jobs(scheduler, db_path, tmp_path)

    job_ids = [kw["id"] for _, kw in scheduler.add_job.call_args_list]
    assert "builtin_eval_check" in job_ids


def test_auto_eval_job_not_registered_when_disabled(tmp_path: Path) -> None:
    scheduler = MagicMock()
    scheduler.get_job.return_value = None
    db_path = tmp_path / "jarvis.db"

    mock_settings = MagicMock()
    mock_settings.auto_training_enabled = True
    mock_settings.auto_eval_enabled = False
    mock_settings.auto_crawl_cron = "0 1 * * *"
    mock_settings.auto_finetune_cron = "0 3 * * 0"

    with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
        _add_builtin_jobs(scheduler, db_path, tmp_path)

    job_ids = [kw["id"] for _, kw in scheduler.add_job.call_args_list]
    assert "builtin_eval_check" not in job_ids


# ── _prune_memory_job ─────────────────────────────────────────────────────────


def test_prune_memory_job_calls_all_four_prune_functions(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    with (
        patch("jarvis.scheduler.core.get_settings") as mock_settings,
        patch("jarvis.memory.episodic.prune_old_episodes", return_value=5) as mock_ep,
        patch("jarvis.memory.feedback.prune_old_feedback", return_value=2) as mock_fb,
        patch("jarvis.memory.failures.prune_old_failures", return_value=1) as mock_fail,
        patch("jarvis.memory.preferences.prune_old_preferences", return_value=3) as mock_pref,
    ):
        mock_settings.return_value.memory_retention_days = 30
        _prune_memory_job(str(db))

    mock_ep.assert_called_once_with(db, 30)
    mock_fb.assert_called_once_with(db, 30)
    mock_fail.assert_called_once_with(db, 30)
    mock_pref.assert_called_once_with(db, 30)


def test_prune_memory_job_swallows_exception(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    with (
        patch("jarvis.scheduler.core.get_settings") as mock_settings,
        patch("jarvis.memory.episodic.prune_old_episodes", side_effect=RuntimeError("boom")),
    ):
        mock_settings.return_value.memory_retention_days = 30
        _prune_memory_job(str(db))  # should not raise


def test_prune_memory_registered_as_builtin(tmp_path: Path) -> None:
    scheduler = MagicMock()
    scheduler.get_job.return_value = None
    db_path = tmp_path / "jarvis.db"

    mock_settings = MagicMock()
    mock_settings.auto_training_enabled = False

    with patch("jarvis.scheduler.core.get_settings", return_value=mock_settings):
        _add_builtin_jobs(scheduler, db_path, tmp_path)

    job_ids = [kw["id"] for _, kw in scheduler.add_job.call_args_list]
    assert "builtin_prune_memory" in job_ids

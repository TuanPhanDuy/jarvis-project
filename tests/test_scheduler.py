"""Tests for scheduler/core.py: built-in job registration and job functions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from jarvis.scheduler.core import (
    JOB_FUNCTIONS,
    _add_builtin_jobs,
    _feedback_analyze_job,
    _memory_consolidation_job,
    _system_snapshot_job,
    stop_scheduler,
)


# ── JOB_FUNCTIONS registry ────────────────────────────────────────────────────


def test_job_functions_registry_has_all_six_types() -> None:
    expected = {"research", "monitor", "memory_consolidate", "digest", "feedback_analyze", "system_snapshot"}
    assert expected.issubset(set(JOB_FUNCTIONS.keys()))


# ── _add_builtin_jobs ─────────────────────────────────────────────────────────


def test_add_builtin_jobs_registers_four_jobs(tmp_path: Path) -> None:
    scheduler = MagicMock()
    scheduler.get_job.return_value = None  # no existing jobs
    db_path = tmp_path / "jarvis.db"

    _add_builtin_jobs(scheduler, db_path, tmp_path)

    assert scheduler.add_job.call_count == 4
    job_ids = [kw["id"] for _, kw in scheduler.add_job.call_args_list]
    assert "builtin_memory_consolidate" in job_ids
    assert "builtin_feedback_analyze" in job_ids
    assert "builtin_system_snapshot" in job_ids
    assert "builtin_graph_dedup" in job_ids


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

    _add_builtin_jobs(scheduler, db_path, tmp_path)

    assert scheduler.add_job.call_count == 3  # only the three missing ones added


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

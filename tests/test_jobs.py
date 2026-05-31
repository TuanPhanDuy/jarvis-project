"""Tests for the async job queue (api/jobs.py)."""
from __future__ import annotations

import time

import pytest

from jarvis.api.jobs import (
    create_job,
    get_job,
    list_jobs,
    mark_done,
    mark_failed,
    mark_running,
)


class TestCreateJob:
    def test_returns_nonempty_id(self, tmp_path):
        job_id = create_job(tmp_path / "db", "research RLHF")
        assert isinstance(job_id, str) and len(job_id) > 0

    def test_default_status_is_pending(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "hello")
        job = get_job(db, job_id)
        assert job["status"] == "pending"

    def test_stores_message(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "explain attention mechanism")
        assert get_job(db, job_id)["message"] == "explain attention mechanism"

    def test_stores_agent_type(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "msg", agent_type="researcher")
        assert get_job(db, job_id)["agent_type"] == "researcher"

    def test_stores_user_id(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "msg", user_id="alice")
        assert get_job(db, job_id)["user_id"] == "alice"


class TestMarkRunning:
    def test_status_becomes_running(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task")
        mark_running(db, job_id)
        assert get_job(db, job_id)["status"] == "running"

    def test_started_at_is_set(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task")
        before = time.time()
        mark_running(db, job_id)
        after = time.time()
        started = get_job(db, job_id)["started_at"]
        assert before <= started <= after


class TestMarkDone:
    def test_status_becomes_done(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task")
        mark_done(db, job_id, "great answer", {})
        assert get_job(db, job_id)["status"] == "done"

    def test_result_stored(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task")
        mark_done(db, job_id, "my result", {})
        assert get_job(db, job_id)["result"] == "my result"

    def test_usage_stored(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task")
        mark_done(db, job_id, "ok", {"input_tokens": 42})
        job = get_job(db, job_id)
        assert job["usage"]["input_tokens"] == 42

    def test_finished_at_is_set(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task")
        before = time.time()
        mark_done(db, job_id, "ok", {})
        assert get_job(db, job_id)["finished_at"] >= before


class TestMarkFailed:
    def test_status_becomes_failed(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task")
        mark_failed(db, job_id, "model offline")
        assert get_job(db, job_id)["status"] == "failed"

    def test_error_stored(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task")
        mark_failed(db, job_id, "model offline")
        assert get_job(db, job_id)["error"] == "model offline"


class TestGetJob:
    def test_nonexistent_returns_none(self, tmp_path):
        assert get_job(tmp_path / "db", "fake-id") is None

    def test_returns_full_record(self, tmp_path):
        db = tmp_path / "db"
        job_id = create_job(db, "task", agent_type="coder", user_id="bob")
        job = get_job(db, job_id)
        assert job["id"] == job_id
        assert job["agent_type"] == "coder"
        assert job["user_id"] == "bob"
        assert "usage" in job


class TestListJobs:
    def test_empty_db_returns_empty(self, tmp_path):
        assert list_jobs(tmp_path / "db") == []

    def test_returns_all_jobs(self, tmp_path):
        db = tmp_path / "db"
        for i in range(3):
            create_job(db, f"task {i}")
        assert len(list_jobs(db)) == 3

    def test_filter_by_status(self, tmp_path):
        db = tmp_path / "db"
        j1 = create_job(db, "t1")
        j2 = create_job(db, "t2")
        mark_done(db, j1, "ok", {})
        pending = list_jobs(db, status="pending")
        done = list_jobs(db, status="done")
        assert len(pending) == 1
        assert len(done) == 1

    def test_filter_by_user_id(self, tmp_path):
        db = tmp_path / "db"
        create_job(db, "task", user_id="alice")
        create_job(db, "task", user_id="bob")
        assert len(list_jobs(db, user_id="alice")) == 1
        assert len(list_jobs(db, user_id="bob")) == 1

    def test_limit_respected(self, tmp_path):
        db = tmp_path / "db"
        for i in range(10):
            create_job(db, f"task {i}")
        assert len(list_jobs(db, limit=3)) == 3

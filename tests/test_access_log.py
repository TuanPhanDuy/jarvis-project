"""Tests for API request access log."""
from __future__ import annotations

import time

import pytest

from jarvis.api.access_log import (
    get_access_log,
    get_access_log_stats,
    record_request,
)


class TestRecordRequest:
    def test_records_entry(self, tmp_path):
        db = tmp_path / "db"
        record_request(db, "GET", "/api/chat", 200, 42.5, "alice")
        entries = get_access_log(db)
        assert len(entries) == 1
        e = entries[0]
        assert e["method"] == "GET"
        assert e["path"] == "/api/chat"
        assert e["status_code"] == 200
        assert abs(e["latency_ms"] - 42.5) < 0.01
        assert e["user_id"] == "alice"

    def test_never_raises_on_bad_path(self, tmp_path):
        record_request(tmp_path / "no" / "sub" / "db", "GET", "/x", 200, 0.0)

    def test_default_user_is_anonymous(self, tmp_path):
        db = tmp_path / "db"
        record_request(db, "POST", "/api/chat", 200, 10.0)
        assert get_access_log(db)[0]["user_id"] == "anonymous"


class TestGetAccessLog:
    def _seed(self, db, entries):
        for method, path, status, latency, user in entries:
            record_request(db, method, path, status, latency, user)

    def test_empty_db_returns_empty(self, tmp_path):
        assert get_access_log(tmp_path / "db") == []

    def test_returns_newest_first(self, tmp_path):
        db = tmp_path / "db"
        self._seed(db, [
            ("GET", "/api/chat", 200, 10.0, "u"),
            ("GET", "/api/stats", 200, 20.0, "u"),
        ])
        entries = get_access_log(db)
        assert entries[0]["path"] == "/api/stats"

    def test_filter_by_path(self, tmp_path):
        db = tmp_path / "db"
        self._seed(db, [
            ("GET", "/api/chat", 200, 10.0, "u"),
            ("GET", "/api/stats", 200, 20.0, "u"),
        ])
        entries = get_access_log(db, path_filter="chat")
        assert all("chat" in e["path"] for e in entries)
        assert len(entries) == 1

    def test_filter_by_status(self, tmp_path):
        db = tmp_path / "db"
        self._seed(db, [
            ("GET", "/api/x", 200, 5.0, "u"),
            ("GET", "/api/y", 404, 5.0, "u"),
            ("GET", "/api/z", 500, 5.0, "u"),
        ])
        errors = get_access_log(db, status_filter=404)
        assert len(errors) == 1
        assert errors[0]["status_code"] == 404

    def test_filter_by_since_ts(self, tmp_path):
        db = tmp_path / "db"
        record_request(db, "GET", "/old", 200, 5.0)
        future_ts = time.time() + 9999
        entries = get_access_log(db, since_ts=future_ts)
        assert entries == []

    def test_limit_respected(self, tmp_path):
        db = tmp_path / "db"
        for i in range(20):
            record_request(db, "GET", f"/api/{i}", 200, float(i))
        assert len(get_access_log(db, limit=5)) == 5


class TestGetAccessLogStats:
    def test_empty_db_returns_zeros(self, tmp_path):
        stats = get_access_log_stats(tmp_path / "db")
        assert stats["total_requests"] == 0
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0.0

    def test_counts_requests(self, tmp_path):
        db = tmp_path / "db"
        for _ in range(5):
            record_request(db, "GET", "/api/chat", 200, 10.0)
        stats = get_access_log_stats(db)
        assert stats["total_requests"] == 5

    def test_counts_errors(self, tmp_path):
        db = tmp_path / "db"
        record_request(db, "GET", "/api/chat", 200, 10.0)
        record_request(db, "GET", "/api/bad", 500, 5.0)
        stats = get_access_log_stats(db)
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.5

    def test_avg_latency_computed(self, tmp_path):
        db = tmp_path / "db"
        record_request(db, "GET", "/api/x", 200, 10.0)
        record_request(db, "GET", "/api/y", 200, 30.0)
        stats = get_access_log_stats(db)
        assert abs(stats["avg_latency_ms"] - 20.0) < 0.1

    def test_top_paths_present(self, tmp_path):
        db = tmp_path / "db"
        for _ in range(3):
            record_request(db, "GET", "/api/chat", 200, 5.0)
        record_request(db, "GET", "/api/stats", 200, 5.0)
        stats = get_access_log_stats(db)
        assert stats["top_paths"][0]["path"] == "/api/chat"
        assert stats["top_paths"][0]["count"] == 3

    def test_since_ts_filters(self, tmp_path):
        db = tmp_path / "db"
        record_request(db, "GET", "/api/old", 200, 5.0)
        future_ts = time.time() + 9999
        stats = get_access_log_stats(db, since_ts=future_ts)
        assert stats["total_requests"] == 0

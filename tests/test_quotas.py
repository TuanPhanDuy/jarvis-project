"""Tests for tool quota management."""
from __future__ import annotations

import time

import pytest

from jarvis.tools.quotas import (
    QuotaExceededError,
    check_quota,
    delete_quota,
    get_quotas,
    record_call,
    set_quota,
)


class TestSetQuota:
    def test_creates_quota(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", max_calls=10, window_seconds=3600)
        quotas = get_quotas(db, "alice")
        assert len(quotas) == 1
        assert quotas[0]["max_calls"] == 10
        assert quotas[0]["window_seconds"] == 3600

    def test_upsert_updates_existing(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 10, 3600)
        set_quota(db, "alice", "web_search", 20, 7200)
        quotas = get_quotas(db, "alice")
        assert len(quotas) == 1
        assert quotas[0]["max_calls"] == 20

    def test_multiple_tools_stored(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 10, 3600)
        set_quota(db, "alice", "read_url", 5, 1800)
        assert len(get_quotas(db, "alice")) == 2


class TestDeleteQuota:
    def test_deletes_existing(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 10, 3600)
        assert delete_quota(db, "alice", "web_search") is True
        assert get_quotas(db, "alice") == []

    def test_returns_false_when_not_found(self, tmp_path):
        assert delete_quota(tmp_path / "db", "alice", "nonexistent") is False


class TestGetQuotas:
    def test_empty_user_returns_empty(self, tmp_path):
        assert get_quotas(tmp_path / "db", "nobody") == []

    def test_includes_usage_fields(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 10, 3600)
        quota = get_quotas(db, "alice")[0]
        assert "calls_used" in quota
        assert "remaining" in quota

    def test_calls_used_reflects_log(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 10, 3600)
        record_call(db, "alice", "web_search")
        record_call(db, "alice", "web_search")
        quota = get_quotas(db, "alice")[0]
        assert quota["calls_used"] == 2
        assert quota["remaining"] == 8

    def test_user_isolation(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 10, 3600)
        set_quota(db, "bob", "web_search", 5, 1800)
        assert get_quotas(db, "alice")[0]["max_calls"] == 10
        assert get_quotas(db, "bob")[0]["max_calls"] == 5


class TestCheckQuota:
    def test_no_quota_set_does_not_raise(self, tmp_path):
        check_quota(tmp_path / "db", "alice", "web_search")  # must not raise

    def test_within_limit_does_not_raise(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 3, 3600)
        record_call(db, "alice", "web_search")
        record_call(db, "alice", "web_search")
        check_quota(db, "alice", "web_search")  # used 2/3 — should not raise

    def test_at_limit_raises(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 2, 3600)
        record_call(db, "alice", "web_search")
        record_call(db, "alice", "web_search")
        with pytest.raises(QuotaExceededError):
            check_quota(db, "alice", "web_search")

    def test_error_message_contains_details(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 1, 60)
        record_call(db, "alice", "web_search")
        try:
            check_quota(db, "alice", "web_search")
            assert False, "Should have raised"
        except QuotaExceededError as e:
            assert "alice" in str(e)
            assert "web_search" in str(e)
            assert "1" in str(e)

    def test_different_tool_not_affected(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 1, 3600)
        record_call(db, "alice", "web_search")
        check_quota(db, "alice", "read_url")  # different tool — no quota set

    def test_different_user_not_affected(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 1, 3600)
        record_call(db, "alice", "web_search")
        check_quota(db, "bob", "web_search")  # different user — no quota


class TestRecordCall:
    def test_increments_count(self, tmp_path):
        db = tmp_path / "db"
        set_quota(db, "alice", "web_search", 100, 3600)
        for _ in range(3):
            record_call(db, "alice", "web_search")
        assert get_quotas(db, "alice")[0]["calls_used"] == 3

    def test_never_raises_on_bad_path(self, tmp_path):
        record_call(tmp_path / "no" / "sub" / "db", "u", "t")

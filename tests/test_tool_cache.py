"""Tests for SQLite-backed tool result cache."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from jarvis.tools.cache import _cache_key, _TOOL_TTLS, get_cached, set_cached


class TestCacheKey:
    def test_same_input_produces_same_key(self):
        key1 = _cache_key("web_search", {"query": "RLHF"})
        key2 = _cache_key("web_search", {"query": "RLHF"})
        assert key1 == key2

    def test_different_inputs_differ(self):
        key1 = _cache_key("web_search", {"query": "RLHF"})
        key2 = _cache_key("web_search", {"query": "PPO"})
        assert key1 != key2

    def test_different_tools_differ(self):
        key1 = _cache_key("web_search", {"query": "RLHF"})
        key2 = _cache_key("read_url", {"query": "RLHF"})
        assert key1 != key2

    def test_dict_ordering_is_stable(self):
        key1 = _cache_key("web_search", {"b": 2, "a": 1})
        key2 = _cache_key("web_search", {"a": 1, "b": 2})
        assert key1 == key2


class TestGetSetCached:
    def test_miss_returns_none(self, tmp_path):
        result = get_cached(tmp_path / "db", "web_search", {"query": "RLHF"})
        assert result is None

    def test_set_then_get_returns_result(self, tmp_path):
        db = tmp_path / "db"
        set_cached(db, "web_search", {"query": "RLHF"}, "RLHF is a technique")
        assert get_cached(db, "web_search", {"query": "RLHF"}) == "RLHF is a technique"

    def test_uncacheable_tool_never_stored(self, tmp_path):
        db = tmp_path / "db"
        set_cached(db, "run_command", {"cmd": "ls"}, "file.txt")
        assert get_cached(db, "run_command", {"cmd": "ls"}) is None

    def test_expired_entry_returns_none(self, tmp_path):
        db = tmp_path / "db"
        # Override TTL map to use a very short TTL for testing
        short_ttls = {"web_search": 1}
        with patch("jarvis.tools.cache._TOOL_TTLS", short_ttls):
            set_cached(db, "web_search", {"query": "X"}, "result")
            # Expired? No — TTL hasn't elapsed yet in this test run.
            # Directly write an expired row to verify eviction:
            import sqlite3, hashlib, json as _json, time as _time
            from jarvis.tools.cache import _get_conn, _cache_key
            conn = _get_conn(db)
            key = _cache_key("web_search", {"query": "stale"})
            now = _time.time()
            conn.execute(
                "INSERT OR REPLACE INTO tool_cache (key, tool_name, result, created_at, expires_at) VALUES (?,?,?,?,?)",
                (key, "web_search", "old result", now - 200, now - 100),
            )
            conn.commit()
            conn.close()
        # Expired row should not be returned
        assert get_cached(db, "web_search", {"query": "stale"}) is None

    def test_all_listed_tools_are_cacheable(self):
        for tool_name in _TOOL_TTLS:
            assert _TOOL_TTLS[tool_name] > 0

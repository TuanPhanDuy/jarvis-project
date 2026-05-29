"""Tests for tool cache: get_cached, set_cached, get_cache_stats, clear_cache."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from jarvis.tools.cache import (
    _TOOL_TTLS,
    clear_cache,
    get_cache_stats,
    get_cached,
    get_cache_ttls,
    set_cache_ttl,
    set_cached,
)


# ── get_cached / set_cached ───────────────────────────────────────────────────

class TestGetSetCached:
    def test_miss_for_uncacheable_tool(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = get_cached(db, "create_plan", {"steps": []})
        assert result is None

    def test_miss_before_set(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = get_cached(db, "web_search", {"query": "RLHF"})
        assert result is None

    def test_hit_after_set(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cached(db, "web_search", {"query": "RLHF"}, "some result")
        result = get_cached(db, "web_search", {"query": "RLHF"})
        assert result == "some result"

    def test_miss_for_different_input(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cached(db, "web_search", {"query": "RLHF"}, "result A")
        result = get_cached(db, "web_search", {"query": "transformers"})
        assert result is None

    def test_uncacheable_tool_set_is_noop(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cached(db, "create_plan", {"steps": []}, "should not be stored")
        assert get_cached(db, "create_plan", {"steps": []}) is None


# ── get_cache_stats ───────────────────────────────────────────────────────────

class TestGetCacheStats:
    def test_returns_zeros_when_db_missing(self, tmp_path: Path) -> None:
        stats = get_cache_stats(tmp_path / "missing.db")
        assert stats["total"] == 0
        assert stats["live"] == 0
        assert stats["expired"] == 0
        assert stats["by_tool"] == {}

    def test_counts_live_entries(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cached(db, "web_search", {"query": "RLHF"}, "r1")
        set_cached(db, "web_search", {"query": "transformers"}, "r2")
        stats = get_cache_stats(db)
        assert stats["total"] >= 2
        assert stats["live"] >= 2

    def test_by_tool_breakdown(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cached(db, "web_search", {"query": "RLHF"}, "r1")
        set_cached(db, "read_url", {"url": "https://example.com"}, "r2")
        stats = get_cache_stats(db)
        assert "web_search" in stats["by_tool"]
        assert "read_url" in stats["by_tool"]
        assert stats["by_tool"]["web_search"]["live"] >= 1
        assert stats["by_tool"]["read_url"]["live"] >= 1

    def test_expired_count_for_stale_entries(self, tmp_path: Path) -> None:
        import sqlite3
        db = tmp_path / "jarvis.db"
        # Insert a row that is already expired
        set_cached(db, "web_search", {"query": "old"}, "stale result")
        # Manually backdate the expires_at to the past
        conn = sqlite3.connect(str(db))
        conn.execute("UPDATE tool_cache SET expires_at = 0")
        conn.commit()
        conn.close()
        stats = get_cache_stats(db)
        assert stats["expired"] >= 1
        assert stats["live"] == 0


# ── clear_cache ───────────────────────────────────────────────────────────────

class TestClearCache:
    def test_returns_zero_when_db_missing(self, tmp_path: Path) -> None:
        deleted = clear_cache(tmp_path / "missing.db")
        assert deleted == 0

    def test_returns_count_of_deleted_rows(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cached(db, "web_search", {"query": "RLHF"}, "r1")
        set_cached(db, "web_search", {"query": "transformers"}, "r2")
        deleted = clear_cache(db)
        assert deleted == 2

    def test_cache_empty_after_clear(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cached(db, "web_search", {"query": "RLHF"}, "r1")
        clear_cache(db)
        assert get_cached(db, "web_search", {"query": "RLHF"}) is None

    def test_stats_zero_after_clear(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cached(db, "web_search", {"query": "RLHF"}, "r1")
        clear_cache(db)
        stats = get_cache_stats(db)
        assert stats["total"] == 0


# ── get_cache_ttls / set_cache_ttl ────────────────────────────────────────────

class TestCacheTtls:
    def setup_method(self):
        # restore defaults between tests
        self._original = dict(_TOOL_TTLS)

    def teardown_method(self):
        _TOOL_TTLS.clear()
        _TOOL_TTLS.update(self._original)

    def test_get_returns_dict(self) -> None:
        ttls = get_cache_ttls()
        assert isinstance(ttls, dict)

    def test_get_includes_known_tools(self) -> None:
        ttls = get_cache_ttls()
        assert "web_search" in ttls
        assert "read_url" in ttls

    def test_get_returns_copy_not_reference(self) -> None:
        ttls = get_cache_ttls()
        ttls["injected"] = 999
        assert "injected" not in get_cache_ttls()

    def test_set_adds_new_tool(self) -> None:
        set_cache_ttl("my_tool", 500)
        assert get_cache_ttls()["my_tool"] == 500

    def test_set_updates_existing_tool(self) -> None:
        original = get_cache_ttls()["web_search"]
        set_cache_ttl("web_search", original * 2)
        assert get_cache_ttls()["web_search"] == original * 2

    def test_set_zero_removes_tool(self) -> None:
        set_cache_ttl("web_search", 0)
        assert "web_search" not in get_cache_ttls()

    def test_set_returns_tool_and_ttl(self) -> None:
        result = set_cache_ttl("my_tool", 300)
        assert result == {"my_tool": 300}

    def test_disabled_tool_no_longer_cached(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_cache_ttl("web_search", 0)
        set_cached(db, "web_search", {"query": "test"}, "result")
        assert get_cached(db, "web_search", {"query": "test"}) is None

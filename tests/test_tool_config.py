"""Tests for per-tool runtime configuration."""
from __future__ import annotations

import pytest

from jarvis.tools.tool_config import (
    delete_tool_config,
    get_tool_config,
    get_tool_max_retries,
    get_tool_timeout,
    set_tool_config,
)


class TestSetAndGetToolConfig:
    def test_returns_defaults_when_no_config(self, tmp_path):
        cfg = get_tool_config(tmp_path / "db", "web_search")
        assert cfg["tool_name"] == "web_search"
        assert cfg["timeout_seconds"] is None
        assert cfg["max_retries"] is None
        assert cfg["cache_ttl_seconds"] is None

    def test_set_timeout(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", timeout_seconds=45)
        assert get_tool_config(db, "web_search")["timeout_seconds"] == 45

    def test_set_max_retries(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", max_retries=5)
        assert get_tool_config(db, "web_search")["max_retries"] == 5

    def test_set_cache_ttl(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", cache_ttl_seconds=3600)
        assert get_tool_config(db, "web_search")["cache_ttl_seconds"] == 3600

    def test_partial_update_preserves_other_fields(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", timeout_seconds=30, max_retries=3)
        set_tool_config(db, "web_search", cache_ttl_seconds=600)
        cfg = get_tool_config(db, "web_search")
        assert cfg["timeout_seconds"] == 30
        assert cfg["max_retries"] == 3
        assert cfg["cache_ttl_seconds"] == 600

    def test_upsert_updates_existing(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", timeout_seconds=30)
        set_tool_config(db, "web_search", timeout_seconds=90)
        assert get_tool_config(db, "web_search")["timeout_seconds"] == 90

    def test_tool_isolation(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", timeout_seconds=10)
        set_tool_config(db, "read_url", timeout_seconds=20)
        assert get_tool_config(db, "web_search")["timeout_seconds"] == 10
        assert get_tool_config(db, "read_url")["timeout_seconds"] == 20


class TestDeleteToolConfig:
    def test_delete_existing(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", timeout_seconds=30)
        assert delete_tool_config(db, "web_search") is True
        assert get_tool_config(db, "web_search")["timeout_seconds"] is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        assert delete_tool_config(tmp_path / "db", "ghost_tool") is False


class TestGetToolTimeout:
    def test_returns_global_default_when_no_config(self, tmp_path):
        assert get_tool_timeout(tmp_path / "db", "web_search", global_default=60) == 60

    def test_returns_per_tool_override(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", timeout_seconds=45)
        assert get_tool_timeout(db, "web_search", global_default=60) == 45

    def test_override_takes_precedence_over_global(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", timeout_seconds=10)
        assert get_tool_timeout(db, "web_search", global_default=999) == 10


class TestGetToolMaxRetries:
    def test_returns_global_default_when_no_config(self, tmp_path):
        assert get_tool_max_retries(tmp_path / "db", "web_search", global_default=2) == 2

    def test_returns_per_tool_override(self, tmp_path):
        db = tmp_path / "db"
        set_tool_config(db, "web_search", max_retries=5)
        assert get_tool_max_retries(db, "web_search", global_default=2) == 5

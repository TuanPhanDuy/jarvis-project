"""Tests for database maintenance (stats, vacuum, prune)."""
from __future__ import annotations

import sqlite3
import time

import pytest

from jarvis.api.maintenance import get_db_stats, prune_data, vacuum_db


def _create_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE agent_turns (id TEXT, timestamp REAL)")
    conn.execute("CREATE TABLE episodic_memory (id TEXT, timestamp REAL)")
    conn.execute("CREATE TABLE audit_log (id TEXT, timestamp REAL)")
    conn.execute("CREATE TABLE agent_checkpoints (id TEXT, created_at REAL)")
    conn.execute("CREATE TABLE agent_jobs (id TEXT, status TEXT, finished_at REAL)")
    conn.commit()
    conn.close()


class TestGetDbStats:
    def test_nonexistent_db_returns_zeros(self, tmp_path):
        stats = get_db_stats(tmp_path / "db")
        assert stats["db_size_bytes"] == 0
        assert stats["tables"] == []

    def test_lists_all_tables(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        stats = get_db_stats(db)
        table_names = {t["name"] for t in stats["tables"]}
        assert "agent_turns" in table_names
        assert "episodic_memory" in table_names

    def test_row_counts_accurate(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        conn = sqlite3.connect(str(db))
        conn.execute("INSERT INTO agent_turns VALUES ('t1', 1.0)")
        conn.execute("INSERT INTO agent_turns VALUES ('t2', 2.0)")
        conn.commit()
        conn.close()
        stats = get_db_stats(db)
        turns = next(t for t in stats["tables"] if t["name"] == "agent_turns")
        assert turns["row_count"] == 2

    def test_db_size_nonzero(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        stats = get_db_stats(db)
        assert stats["db_size_bytes"] > 0

    def test_db_path_in_result(self, tmp_path):
        db = tmp_path / "db"
        stats = get_db_stats(db)
        assert str(db) in stats["db_path"]


class TestVacuumDb:
    def test_nonexistent_db_returns_zeros(self, tmp_path):
        result = vacuum_db(tmp_path / "db")
        assert result["size_before"] == 0
        assert result["reclaimed_bytes"] == 0

    def test_returns_required_keys(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        result = vacuum_db(db)
        assert "size_before" in result
        assert "size_after" in result
        assert "reclaimed_bytes" in result

    def test_reclaimed_bytes_nonnegative(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        result = vacuum_db(db)
        assert result["reclaimed_bytes"] >= 0


class TestPruneData:
    def _seed(self, db_path, table: str, ts_col: str, timestamps: list[float]):
        conn = sqlite3.connect(str(db_path))
        for i, ts in enumerate(timestamps):
            conn.execute(f"INSERT INTO {table} VALUES (?,?)", (f"id-{i}", ts))
        conn.commit()
        conn.close()

    def test_nonexistent_db_returns_zeros(self, tmp_path):
        result = prune_data(tmp_path / "db", "turns", older_than_days=30)
        assert result["deleted_counts"]["turns"] == 0

    def test_prunes_old_turns(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        old_ts = time.time() - 40 * 86400
        recent_ts = time.time() - 1 * 86400
        self._seed(db, "agent_turns", "timestamp", [old_ts, old_ts, recent_ts])
        result = prune_data(db, "turns", older_than_days=30)
        assert result["deleted_counts"]["turns"] == 2

    def test_prunes_old_episodes(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        old_ts = time.time() - 40 * 86400
        self._seed(db, "episodic_memory", "timestamp", [old_ts])
        result = prune_data(db, "episodes", older_than_days=30)
        assert result["deleted_counts"]["episodes"] == 1

    def test_target_all_covers_all_tables(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        old_ts = time.time() - 100 * 86400
        self._seed(db, "agent_turns", "timestamp", [old_ts])
        self._seed(db, "audit_log", "timestamp", [old_ts])
        result = prune_data(db, "all", older_than_days=30)
        assert result["deleted_counts"]["turns"] == 1
        assert result["deleted_counts"]["audit"] == 1

    def test_recent_rows_not_deleted(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        recent_ts = time.time() - 1 * 86400
        self._seed(db, "agent_turns", "timestamp", [recent_ts, recent_ts])
        result = prune_data(db, "turns", older_than_days=30)
        assert result["deleted_counts"]["turns"] == 0

    def test_jobs_prune_only_finished(self, tmp_path):
        db = tmp_path / "db"
        _create_db(db)
        old_ts = time.time() - 100 * 86400
        conn = sqlite3.connect(str(db))
        conn.execute("INSERT INTO agent_jobs VALUES (?,?,?)", ("j1", "done", old_ts))
        conn.execute("INSERT INTO agent_jobs VALUES (?,?,?)", ("j2", "running", old_ts))
        conn.commit()
        conn.close()
        result = prune_data(db, "jobs", older_than_days=30)
        assert result["deleted_counts"]["jobs"] == 1  # only done/failed/cancelled

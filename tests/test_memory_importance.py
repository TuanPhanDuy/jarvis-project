"""Tests for episodic memory importance scoring and decay."""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from jarvis.memory.episodic import (
    _HALF_LIFE_SECONDS,
    _IMPORTANCE_BOOST,
    _effective_importance,
    apply_importance_decay,
    boost_importance,
    log_episode,
    search_episodes,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "jarvis.db"


# ── _effective_importance ─────────────────────────────────────────────────────

class TestEffectiveImportance:
    def test_fresh_episode_full_importance(self):
        now = time.time()
        eff = _effective_importance(1.0, now, now)
        assert abs(eff - 1.0) < 1e-9

    def test_importance_halved_after_half_life(self):
        now = time.time()
        old_ts = now - _HALF_LIFE_SECONDS
        eff = _effective_importance(1.0, old_ts, now)
        assert abs(eff - 0.5) < 1e-6

    def test_importance_quarter_after_two_half_lives(self):
        now = time.time()
        old_ts = now - 2 * _HALF_LIFE_SECONDS
        eff = _effective_importance(1.0, old_ts, now)
        assert abs(eff - 0.25) < 1e-6

    def test_higher_base_importance_proportional(self):
        now = time.time()
        eff2 = _effective_importance(2.0, now, now)
        eff1 = _effective_importance(1.0, now, now)
        assert abs(eff2 / eff1 - 2.0) < 1e-9

    def test_future_timestamp_clamped_at_full(self):
        now = time.time()
        future = now + 3600
        eff = _effective_importance(1.0, future, now)
        assert eff == pytest.approx(1.0)


# ── boost_importance ──────────────────────────────────────────────────────────

class TestBoostImportance:
    def test_boost_increases_importance(self, db):
        log_episode(db, "s1", "user", "hello world")
        from jarvis.memory.episodic import _get_conn
        conn = _get_conn(db)
        ep_id = conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()["id"]
        conn.close()

        boost_importance(db, ep_id, delta=0.3)

        conn = _get_conn(db)
        row = conn.execute("SELECT importance FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        conn.close()
        assert row["importance"] == pytest.approx(1.3)

    def test_boost_capped_at_5(self, db):
        log_episode(db, "s2", "user", "test")
        from jarvis.memory.episodic import _get_conn
        conn = _get_conn(db)
        ep_id = conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()["id"]
        conn.execute("UPDATE episodes SET importance = 4.9 WHERE id = ?", (ep_id,))
        conn.commit()
        conn.close()

        boost_importance(db, ep_id, delta=1.0)

        conn = _get_conn(db)
        row = conn.execute("SELECT importance FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        conn.close()
        assert row["importance"] == pytest.approx(5.0)

    def test_boost_nonexistent_id_no_raise(self, db):
        boost_importance(db, 99999)  # should not raise

    def test_default_boost_delta(self, db):
        log_episode(db, "s3", "user", "default boost test")
        from jarvis.memory.episodic import _get_conn
        conn = _get_conn(db)
        ep_id = conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()["id"]
        conn.close()

        boost_importance(db, ep_id)

        conn = _get_conn(db)
        row = conn.execute("SELECT importance FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        conn.close()
        assert row["importance"] == pytest.approx(1.0 + _IMPORTANCE_BOOST)


# ── apply_importance_decay ────────────────────────────────────────────────────

class TestApplyImportanceDecay:
    def test_decay_reduces_importance(self, db):
        from jarvis.memory.episodic import _get_conn
        # Insert an episode that is 14 days old
        old_ts = time.time() - _HALF_LIFE_SECONDS
        conn = _get_conn(db)
        conn.execute(
            "INSERT INTO episodes (timestamp, session_id, user_id, role, content, importance)"
            " VALUES (?, 's1', 'u1', 'user', 'decay test', 1.0)",
            (old_ts,),
        )
        conn.commit()
        ep_id = conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()["id"]
        conn.close()

        apply_importance_decay(db)

        conn = _get_conn(db)
        row = conn.execute("SELECT importance FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        conn.close()
        assert row["importance"] == pytest.approx(0.5, abs=0.01)

    def test_decay_returns_updated_count(self, db):
        log_episode(db, "s1", "user", "msg1")
        log_episode(db, "s1", "user", "msg2")
        updated = apply_importance_decay(db)
        assert updated == 2

    def test_decay_does_not_raise_on_empty_db(self, db):
        result = apply_importance_decay(db)
        assert result == 0


# ── search ranking by importance ──────────────────────────────────────────────

class TestSearchRankingByImportance:
    def test_new_episodes_have_importance_column(self, db):
        log_episode(db, "s1", "user", "ranking test content")
        results = search_episodes(db, "ranking test")
        assert len(results) == 1
        assert "importance" in results[0]

    def test_search_boosts_returned_episodes(self, db):
        from jarvis.memory.episodic import _get_conn
        log_episode(db, "s1", "user", "boost on search test")
        conn = _get_conn(db)
        ep_id = conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()["id"]
        conn.close()

        search_episodes(db, "boost on search")

        conn = _get_conn(db)
        row = conn.execute("SELECT importance FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        conn.close()
        assert row["importance"] > 1.0

    def test_high_importance_episode_ranked_first(self, db):
        from jarvis.memory.episodic import _get_conn
        # Insert two episodes with same content but different importance
        old_ts = time.time() - 86400  # 1 day old
        conn = _get_conn(db)
        conn.execute(
            "INSERT INTO episodes (timestamp, session_id, user_id, role, content, importance)"
            " VALUES (?, 's1', 'u1', 'user', 'important keyword', 3.0)",
            (old_ts,),
        )
        conn.execute(
            "INSERT INTO episodes (timestamp, session_id, user_id, role, content, importance)"
            " VALUES (?, 's2', 'u1', 'user', 'important keyword', 0.1)",
            (old_ts,),
        )
        conn.commit()
        conn.close()

        results = search_episodes(db, "important keyword", limit=2)
        assert len(results) == 2
        # Higher importance episode should come first
        assert results[0]["importance"] > results[1]["importance"]


# ── schema migration ──────────────────────────────────────────────────────────

class TestSchemaMigration:
    def test_existing_db_without_importance_gets_column(self, tmp_path):
        import sqlite3
        db = tmp_path / "old.db"
        # Create a DB without the importance column (simulates legacy DB)
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'anonymous',
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        # Opening via _get_conn should migrate
        from jarvis.memory.episodic import _get_conn
        conn = _get_conn(db)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
        conn.close()
        assert "importance" in columns

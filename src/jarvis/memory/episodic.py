"""Episodic memory: timestamped conversation log in SQLite with FTS5 search.

All conversations are automatically logged by the API server. Claude can search
this log to recall what was discussed in previous sessions.

Each episode carries an `importance` score (float, default 1.0):
  - Boosted (+0.2) when surfaced by memory retrieval or positive feedback.
  - Decayed by exponential half-life (default 14 days) via apply_importance_decay().
  - Search results are ranked by importance × recency_weight so frequently
    recalled, recent memories surface ahead of stale, unreferenced ones.

DB location: reports_dir/jarvis.db (shared with graph and other memory tables).
"""
from __future__ import annotations

import datetime
import math
import sqlite3
import time
from pathlib import Path

_IMPORTANCE_BOOST = 0.2
_HALF_LIFE_DAYS = 14.0
_HALF_LIFE_SECONDS = _HALF_LIFE_DAYS * 86400.0


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            session_id  TEXT    NOT NULL,
            user_id     TEXT    NOT NULL DEFAULT 'anonymous',
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ep_ts      ON episodes(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_ep_session ON episodes(session_id);
        CREATE INDEX IF NOT EXISTS idx_ep_user    ON episodes(user_id);
    """)
    # Safe migration: add importance column if it doesn't exist yet
    existing = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    if "importance" not in existing:
        conn.execute("ALTER TABLE episodes ADD COLUMN importance REAL NOT NULL DEFAULT 1.0")
        conn.commit()

    # FTS5 is optional — fall back to LIKE search if unavailable
    try:
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                content,
                content='episodes',
                content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn


def _effective_importance(importance: float, timestamp: float, now: float | None = None) -> float:
    """Compute decayed importance: importance × 0.5^(age_days / half_life_days)."""
    age_seconds = (now or time.time()) - timestamp
    decay = math.pow(0.5, max(0.0, age_seconds) / _HALF_LIFE_SECONDS)
    return importance * decay


def log_episode(
    db_path: Path, session_id: str, role: str, content: str, user_id: str = "anonymous"
) -> None:
    """Append a conversation turn to episodic memory. Best-effort — never raises."""
    try:
        conn = _get_conn(db_path)
        try:
            conn.execute(
                "INSERT INTO episodes (timestamp, session_id, user_id, role, content, importance)"
                " VALUES (?,?,?,?,?,1.0)",
                (time.time(), session_id, user_id, role, content),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def boost_importance(db_path: Path, episode_id: int, delta: float = _IMPORTANCE_BOOST) -> None:
    """Increase the importance of an episode. Best-effort — never raises."""
    try:
        conn = _get_conn(db_path)
        try:
            conn.execute(
                "UPDATE episodes SET importance = MIN(importance + ?, 5.0) WHERE id = ?",
                (delta, episode_id),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def apply_importance_decay(db_path: Path, half_life_days: float = _HALF_LIFE_DAYS) -> int:
    """Apply exponential decay to all episode importance scores.

    New importance = old_importance × 0.5^(age_days / half_life_days).
    Returns the number of rows updated.
    """
    now = time.time()
    half_life_s = half_life_days * 86400.0
    try:
        conn = _get_conn(db_path)
        try:
            cur = conn.execute(
                """UPDATE episodes
                   SET importance = importance * POWER(0.5, MAX(0.0, ? - timestamp) / ?)
                   WHERE importance > 0.01""",
                (now, half_life_s),
            )
            updated = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return updated
    except Exception:
        return 0


def _search(db_path: Path, query: str, limit: int, user_id: str | None = None) -> list[sqlite3.Row]:
    """Search episodes by FTS5 (preferred) or LIKE fallback.

    Results are ranked by effective_importance (importance × decay) descending,
    so high-importance recent episodes surface above stale low-importance ones.
    """
    conn = _get_conn(db_path)
    now = time.time()
    user_filter = "AND e.user_id = ?" if user_id else ""
    user_arg = (user_id,) if user_id else ()
    try:
        try:
            # Fetch a larger candidate set from FTS5, then re-rank by importance × decay
            candidates = conn.execute(
                f"""
                SELECT e.id, e.timestamp, e.session_id, e.user_id, e.role, e.content,
                       e.importance
                FROM episodes_fts f
                JOIN episodes e ON e.id = f.rowid
                WHERE episodes_fts MATCH ? {user_filter}
                ORDER BY rank
                LIMIT ?
                """,
                (query, *user_arg, limit * 3),
            ).fetchall()
            if candidates:
                ranked = sorted(
                    candidates,
                    key=lambda r: _effective_importance(r["importance"], r["timestamp"], now),
                    reverse=True,
                )
                # Boost the importance of returned episodes (they were referenced)
                for row in ranked[:limit]:
                    boost_importance(db_path, row["id"])
                return ranked[:limit]
        except sqlite3.OperationalError:
            pass

        # LIKE fallback: sort by effective_importance using POWER() in SQL
        rows = conn.execute(
            f"""SELECT *, importance * POWER(0.5, MAX(0.0, ? - timestamp) / ?) AS eff_imp
               FROM episodes
               WHERE content LIKE ? {user_filter}
               ORDER BY eff_imp DESC
               LIMIT ?""",
            (now, _HALF_LIFE_SECONDS, f"%{query}%", *user_arg, limit),
        ).fetchall()
        for row in rows:
            boost_importance(db_path, row["id"])
        return rows
    finally:
        conn.close()


def handle_search_episodic_memory(tool_input: dict, db_path: Path, user_id: str | None = None) -> str:
    try:
        query = tool_input["query"]
        limit = int(tool_input.get("limit", 10))
        rows = _search(db_path, query, limit, user_id=user_id)

        if not rows:
            return f"No episodes found matching '{query}'."

        lines = [f"Found {len(rows)} episode(s) matching '{query}':\n"]
        for row in rows:
            ts = datetime.datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M")
            imp = row["importance"] if "importance" in row.keys() else 1.0
            lines.append(
                f"[{ts}] [{row['role']}] (session: {row['session_id'][:8]}… | importance: {imp:.2f})"
            )
            lines.append(row["content"][:400].strip())
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: search_episodic_memory failed — {e}"


def list_episodes(
    db_path: Path,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Return episodes newest-first with optional user/session filter."""
    try:
        conn = _get_conn(db_path)
        try:
            clauses, params = [], []
            if user_id:
                clauses.append("user_id = ?")
                params.append(user_id)
            if session_id:
                clauses.append("session_id = ?")
                params.append(session_id)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params += [limit, offset]
            rows = conn.execute(
                f"SELECT id, session_id, user_id, role, content, importance, timestamp "
                f"FROM episodes {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_episode(db_path: Path, episode_id: int) -> dict | None:
    """Return a single episode by ID."""
    try:
        conn = _get_conn(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def delete_episode(db_path: Path, episode_id: int) -> bool:
    """Delete one episode by ID. Returns True if it existed."""
    try:
        conn = _get_conn(db_path)
        try:
            cur = conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False


def search_episodes(
    db_path: Path,
    query: str,
    limit: int = 20,
    user_id: str | None = None,
) -> list[dict]:
    """Full-text search episodes. Returns plain dicts ranked by importance × decay."""
    try:
        rows = _search(db_path, query, limit, user_id=user_id)
        return [dict(r) for r in rows]
    except Exception:
        return []


def delete_episodes(
    db_path: Path,
    session_id: str | None = None,
    user_id: str | None = None,
) -> int:
    """Delete episodes filtered by session_id and/or user_id.

    At least one filter must be provided. Returns number of rows deleted.
    """
    if not session_id and not user_id:
        raise ValueError("Provide at least one of session_id or user_id")
    try:
        conn = _get_conn(db_path)
        try:
            where_parts, params = [], []
            if session_id:
                where_parts.append("session_id = ?")
                params.append(session_id)
            if user_id:
                where_parts.append("user_id = ?")
                params.append(user_id)
            cur = conn.execute(
                f"DELETE FROM episodes WHERE {' AND '.join(where_parts)}", params
            )
            deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted
    except Exception:
        return 0


def prune_old_episodes(db_path: Path, retention_days: int) -> int:
    """Delete episodes older than retention_days. Returns number of rows deleted."""
    cutoff = time.time() - retention_days * 86400
    try:
        conn = _get_conn(db_path)
        cur = conn.execute("DELETE FROM episodes WHERE timestamp < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        return deleted
    except Exception:
        return 0


SCHEMA: dict = {
    "name": "search_episodic_memory",
    "description": (
        "Search JARVIS's episodic memory — a timestamped log of all past conversations. "
        "Use this to recall what was discussed in previous sessions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords or phrase to search in past conversations.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}

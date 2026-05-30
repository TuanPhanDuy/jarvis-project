"""Session persistence — saves in-memory sessions to SQLite and reloads on startup.

Each session row stores the full message history as JSON so the server can
resume conversations after a restart.  Rows are soft-expiry: loading only
restores sessions younger than the configured TTL.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import structlog

log = structlog.get_logger()


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persisted_sessions (
            session_id  TEXT PRIMARY KEY,
            agent_type  TEXT NOT NULL DEFAULT 'PlannerAgent',
            user_id     TEXT,
            messages    TEXT NOT NULL DEFAULT '[]',
            fork_of     TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ps_updated ON persisted_sessions(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_ps_user    ON persisted_sessions(user_id);
    """)
    conn.commit()
    return conn


def save_session(
    db_path: Path,
    session_id: str,
    messages: list[dict],
    agent_type: str = "PlannerAgent",
    user_id: str | None = None,
    fork_of: str | None = None,
    created_at: float | None = None,
) -> None:
    """Upsert a session's message history. Best-effort — never raises."""
    try:
        now = time.time()
        conn = _conn(db_path)
        try:
            conn.execute(
                """INSERT INTO persisted_sessions
                       (session_id, agent_type, user_id, messages, fork_of, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       messages   = excluded.messages,
                       agent_type = excluded.agent_type,
                       updated_at = excluded.updated_at""",
                (
                    session_id,
                    agent_type,
                    user_id,
                    json.dumps(messages),
                    fork_of,
                    created_at or now,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("session_save_failed", session_id=session_id, error=str(exc))


def load_sessions(db_path: Path, ttl_minutes: int = 60) -> list[dict]:
    """Return all sessions updated within ttl_minutes, ordered newest-first."""
    if not db_path.exists():
        return []
    cutoff = time.time() - ttl_minutes * 60
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM persisted_sessions WHERE updated_at >= ? ORDER BY updated_at DESC",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        result = []
        for row in rows:
            try:
                messages = json.loads(row["messages"])
            except Exception:
                messages = []
            result.append({
                "session_id": row["session_id"],
                "agent_type": row["agent_type"],
                "user_id": row["user_id"],
                "messages": messages,
                "fork_of": row["fork_of"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return result
    except Exception as exc:
        log.warning("session_load_failed", error=str(exc))
        return []


def get_session_history(
    db_path: Path,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Return paginated messages for a persisted session, newest-first."""
    if not db_path.exists():
        return []
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT messages FROM persisted_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return []
        messages: list[dict] = json.loads(row["messages"])
        messages.reverse()
        return messages[offset: offset + limit]
    except Exception:
        return []


def delete_persisted_session(db_path: Path, session_id: str) -> bool:
    """Remove a persisted session. Returns True if it existed."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM persisted_sessions WHERE session_id = ?", (session_id,)
            )
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False

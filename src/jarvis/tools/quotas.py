"""Per-user, per-tool call quotas with a rolling time window.

Quotas are stored in SQLite so they survive worker restarts. The call log
is a lightweight append-only table pruned on each check.

Usage in _dispatch:
    check_quota(db_path, user_id, tool_name)  # raises QuotaExceededError if over limit
    record_call(db_path, user_id, tool_name)  # call after a successful tool execution
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class QuotaExceededError(Exception):
    def __init__(self, user_id: str, tool_name: str, limit: int, window_seconds: int) -> None:
        self.user_id = user_id
        self.tool_name = tool_name
        self.limit = limit
        self.window_seconds = window_seconds
        super().__init__(
            f"Quota exceeded: user '{user_id}' has reached the limit of "
            f"{limit} calls to '{tool_name}' per {window_seconds}s window"
        )


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tool_quotas (
            user_id        TEXT NOT NULL,
            tool_name      TEXT NOT NULL,
            max_calls      INTEGER NOT NULL,
            window_seconds INTEGER NOT NULL,
            PRIMARY KEY (user_id, tool_name)
        );
        CREATE TABLE IF NOT EXISTS tool_call_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            tool_name  TEXT NOT NULL,
            called_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tcl_lookup
            ON tool_call_log(user_id, tool_name, called_at DESC);
    """)
    conn.commit()
    return conn


def set_quota(
    db_path: Path,
    user_id: str,
    tool_name: str,
    max_calls: int,
    window_seconds: int,
) -> None:
    """Create or update a quota. Best-effort — never raises."""
    try:
        conn = _conn(db_path)
        try:
            conn.execute(
                """INSERT INTO tool_quotas (user_id, tool_name, max_calls, window_seconds)
                   VALUES (?,?,?,?)
                   ON CONFLICT(user_id, tool_name) DO UPDATE SET
                       max_calls=excluded.max_calls,
                       window_seconds=excluded.window_seconds""",
                (user_id, tool_name, max_calls, window_seconds),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def delete_quota(db_path: Path, user_id: str, tool_name: str) -> bool:
    """Remove a quota. Returns True if it existed."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM tool_quotas WHERE user_id=? AND tool_name=?",
                (user_id, tool_name),
            )
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False


def get_quotas(db_path: Path, user_id: str) -> list[dict]:
    """Return all quotas for a user, including current usage."""
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM tool_quotas WHERE user_id=? ORDER BY tool_name",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        result = []
        for r in rows:
            now = time.time()
            window_start = now - r["window_seconds"]
            calls_used = _count_calls(db_path, user_id, r["tool_name"], window_start)
            result.append({
                "user_id": r["user_id"],
                "tool_name": r["tool_name"],
                "max_calls": r["max_calls"],
                "window_seconds": r["window_seconds"],
                "calls_used": calls_used,
                "remaining": max(0, r["max_calls"] - calls_used),
            })
        return result
    except Exception:
        return []


def _count_calls(db_path: Path, user_id: str, tool_name: str, since: float) -> int:
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM tool_call_log "
                "WHERE user_id=? AND tool_name=? AND called_at>=?",
                (user_id, tool_name, since),
            ).fetchone()
        finally:
            conn.close()
        return row["cnt"] if row else 0
    except Exception:
        return 0


def check_quota(db_path: Path, user_id: str, tool_name: str) -> None:
    """Raise QuotaExceededError if the user has exceeded their quota for this tool.

    No-ops if no quota is set for this user/tool pair. Never raises for DB errors.
    """
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT max_calls, window_seconds FROM tool_quotas "
                "WHERE user_id=? AND tool_name=?",
                (user_id, tool_name),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return
        max_calls = row["max_calls"]
        window_seconds = row["window_seconds"]
        window_start = time.time() - window_seconds
        used = _count_calls(db_path, user_id, tool_name, window_start)
        if used >= max_calls:
            raise QuotaExceededError(user_id, tool_name, max_calls, window_seconds)
    except QuotaExceededError:
        raise
    except Exception:
        pass


def record_call(db_path: Path, user_id: str, tool_name: str) -> None:
    """Log one tool call for quota tracking. Best-effort — never raises."""
    try:
        conn = _conn(db_path)
        try:
            conn.execute(
                "INSERT INTO tool_call_log (user_id, tool_name, called_at) VALUES (?,?,?)",
                (user_id, tool_name, time.time()),
            )
            conn.execute(
                "DELETE FROM tool_call_log WHERE called_at < ? AND user_id=? AND tool_name=?",
                (time.time() - 86400, user_id, tool_name),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

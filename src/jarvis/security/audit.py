"""Audit log — immutable record of every tool call (approved, denied, or auto).

Every tool dispatch writes one row regardless of outcome.  The audit log is
stored in the shared jarvis.db SQLite database alongside episodic memory.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    REAL    NOT NULL,
            session_id   TEXT    NOT NULL,
            user_id      TEXT,
            tool_name    TEXT    NOT NULL,
            tool_input   TEXT    NOT NULL,
            risk_level   TEXT    NOT NULL,
            approved     INTEGER NOT NULL,
            approver     TEXT    NOT NULL DEFAULT 'auto',
            result_ok    INTEGER,
            duration_ms  REAL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_ts      ON audit_log(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
        CREATE INDEX IF NOT EXISTS idx_audit_tool    ON audit_log(tool_name);
    """)
    conn.commit()
    return conn


def log_tool_call(
    db_path: Path,
    session_id: str,
    tool_name: str,
    tool_input: dict,
    risk_level: str,
    approved: int,
    approver: str = "auto",
    result_ok: int | None = None,
    duration_ms: float | None = None,
    user_id: str | None = None,
) -> None:
    """Write one audit entry. Best-effort — never raises."""
    try:
        conn = _get_conn(db_path)
        conn.execute(
            """
            INSERT INTO audit_log
                (timestamp, session_id, user_id, tool_name, tool_input,
                 risk_level, approved, approver, result_ok, duration_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                time.time(),
                session_id,
                user_id,
                tool_name,
                json.dumps(tool_input),
                risk_level,
                approved,
                approver,
                result_ok,
                duration_ms,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_recent_audit(db_path: Path, limit: int = 50, session_id: str | None = None) -> list[dict]:
    """Return recent audit entries as plain dicts."""
    try:
        conn = _get_conn(db_path)
        where = "WHERE session_id = ?" if session_id else ""
        args = (session_id, limit) if session_id else (limit,)
        rows = conn.execute(
            f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ?", args
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

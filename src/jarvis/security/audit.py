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
        try:
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
        finally:
            conn.close()
    except Exception:
        pass


def prune_old_audit(db_path: Path, retention_days: int) -> int:
    """Delete audit entries older than retention_days. Returns number of rows deleted."""
    cutoff = time.time() - retention_days * 86400
    try:
        conn = _get_conn(db_path)
        try:
            cur = conn.execute("DELETE FROM audit_log WHERE timestamp < ?", (cutoff,))
            deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted
    except Exception:
        return 0


def get_audit_stats(db_path: Path, since_ts: float | None = None) -> dict:
    """Return aggregated audit stats: total calls, approval/denial rates, top tools, risk breakdown."""
    if not db_path.exists():
        return {"total_calls": 0, "approved": 0, "denied": 0, "approval_rate": 0.0,
                "top_tools": [], "risk_breakdown": {}}
    try:
        conn = _get_conn(db_path)
        try:
            where = "WHERE timestamp >= ?" if since_ts else ""
            params: list = [since_ts] if since_ts else []

            rows = conn.execute(
                f"SELECT tool_name, risk_level, approved FROM audit_log {where}", params
            ).fetchall()
        finally:
            conn.close()

        total = len(rows)
        if total == 0:
            return {"total_calls": 0, "approved": 0, "denied": 0, "approval_rate": 0.0,
                    "top_tools": [], "risk_breakdown": {}}

        approved = sum(1 for r in rows if r["approved"])
        denied = total - approved

        tool_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        for r in rows:
            tool_counts[r["tool_name"]] = tool_counts.get(r["tool_name"], 0) + 1
            risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1

        top_tools = sorted(
            [{"tool_name": t, "count": c} for t, c in tool_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10]

        return {
            "total_calls": total,
            "approved": approved,
            "denied": denied,
            "approval_rate": round(approved / total, 4) if total else 0.0,
            "top_tools": top_tools,
            "risk_breakdown": risk_counts,
        }
    except Exception:
        return {"total_calls": 0, "approved": 0, "denied": 0, "approval_rate": 0.0,
                "top_tools": [], "risk_breakdown": {}}


def get_session_timeline(db_path: Path, session_id: str) -> list[dict]:
    """Return ordered tool-call timeline for a session.

    Each entry: {tool, timestamp, duration_ms, risk_level, result_ok}.
    Ordered oldest-first so callers can reconstruct the execution sequence.
    """
    try:
        conn = _get_conn(db_path)
        try:
            rows = conn.execute(
                """SELECT tool_name, timestamp, duration_ms, risk_level, result_ok
                   FROM audit_log
                   WHERE session_id = ?
                   ORDER BY timestamp ASC""",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "tool": r["tool_name"],
                "timestamp": r["timestamp"],
                "duration_ms": r["duration_ms"],
                "risk_level": r["risk_level"],
                "result_ok": bool(r["result_ok"]),
            }
            for r in rows
        ]
    except Exception:
        return []


def get_slow_tools(db_path: Path, threshold_ms: float = 5000.0) -> list[dict]:
    """Return tools whose average duration exceeds threshold_ms.

    Returns list of {tool_name, avg_duration_ms, call_count, max_duration_ms},
    sorted by avg_duration_ms descending.
    """
    try:
        conn = _get_conn(db_path)
        try:
            rows = conn.execute(
                """SELECT tool_name,
                          AVG(duration_ms) AS avg_ms,
                          COUNT(*) AS call_count,
                          MAX(duration_ms) AS max_ms
                   FROM audit_log
                   WHERE duration_ms IS NOT NULL
                   GROUP BY tool_name
                   HAVING AVG(duration_ms) > ?
                   ORDER BY avg_ms DESC""",
                (threshold_ms,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "tool_name": r["tool_name"],
                "avg_duration_ms": round(r["avg_ms"], 1),
                "call_count": r["call_count"],
                "max_duration_ms": round(r["max_ms"], 1),
            }
            for r in rows
        ]
    except Exception:
        return []


def get_recent_audit(
    db_path: Path,
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = None,
) -> list[dict]:
    """Return paginated audit entries as plain dicts, newest first."""
    try:
        conn = _get_conn(db_path)
        try:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE session_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (session_id, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

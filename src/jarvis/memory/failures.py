"""Failure memory: log tool errors to SQLite for self-improvement pattern analysis.

Every tool call that returns an "ERROR: ..." string is automatically logged here
by BaseAgent._dispatch(). Claude can query patterns to learn what's failing and why.

DB location: reports_dir/jarvis.db (shared SQLite file, tool_failures table).
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_failures (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            tool_name   TEXT    NOT NULL,
            tool_input  TEXT    NOT NULL,
            error_msg   TEXT    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fail_tool ON tool_failures(tool_name)")
    conn.commit()
    return conn


def log_failure(db_path: Path, tool_name: str, tool_input: dict, error_msg: str) -> None:
    """Log a tool failure. Best-effort — never raises."""
    try:
        conn = _get_conn(db_path)
        try:
            conn.execute(
                "INSERT INTO tool_failures (timestamp, tool_name, tool_input, error_msg) VALUES (?, ?, ?, ?)",
                (time.time(), tool_name, json.dumps(tool_input, default=str), error_msg),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def handle_analyze_failures(tool_input: dict, db_path: Path) -> str:
    """Tool handler: return failure patterns grouped by tool + error."""
    try:
        tool_name = tool_input.get("tool_name")
        conn = _get_conn(db_path)
        try:
            if tool_name:
                rows = conn.execute(
                    """
                    SELECT tool_name, error_msg, COUNT(*) AS count
                    FROM tool_failures WHERE tool_name = ?
                    GROUP BY tool_name, error_msg
                    ORDER BY count DESC LIMIT 20
                    """,
                    (tool_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT tool_name, error_msg, COUNT(*) AS count
                    FROM tool_failures
                    GROUP BY tool_name, error_msg
                    ORDER BY count DESC LIMIT 20
                    """
                ).fetchall()
        finally:
            conn.close()

        if not rows:
            subject = f"'{tool_name}'" if tool_name else "any tool"
            return f"No failures recorded for {subject}."

        lines = [f"Top failure patterns ({len(rows)} entries):\n"]
        for row in rows:
            lines.append(f"**{row['tool_name']}** (×{row['count']}): {row['error_msg'][:200]}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: analyze_failures failed — {e}"


def get_failure_patterns(
    db_path: Path,
    tool_name: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Return failure patterns as a list of {tool_name, error_msg, count} dicts, sorted by count desc."""
    try:
        if not db_path.exists():
            return []
        conn = _get_conn(db_path)
        try:
            if tool_name:
                rows = conn.execute(
                    """
                    SELECT tool_name, error_msg, COUNT(*) AS count
                    FROM tool_failures WHERE tool_name = ?
                    GROUP BY tool_name, error_msg
                    ORDER BY count DESC LIMIT ?
                    """,
                    (tool_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT tool_name, error_msg, COUNT(*) AS count
                    FROM tool_failures
                    GROUP BY tool_name, error_msg
                    ORDER BY count DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        finally:
            conn.close()
        return [{"tool_name": r["tool_name"], "error_msg": r["error_msg"], "count": r["count"]} for r in rows]
    except Exception:
        return []


def get_failure_heatmap(
    db_path: Path,
    tool_name: str | None = None,
    days: int = 30,
) -> dict:
    """Return failure counts bucketed by hour-of-day (0-23) and day-of-week (0-6).

    day-of-week: 0=Monday … 6=Sunday (strftime %w returns 0=Sunday, we remap).
    Returns:
      {tool_name?: str, days: int,
       by_hour: {0: count, ..., 23: count},
       by_dow:  {0: count, ..., 6: count},   # 0=Monday
       total: int}
    """
    try:
        cutoff = time.time() - days * 86400
        conn = _get_conn(db_path)
        try:
            if tool_name:
                rows = conn.execute(
                    "SELECT timestamp FROM tool_failures WHERE tool_name=? AND timestamp>=?",
                    (tool_name, cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT timestamp FROM tool_failures WHERE timestamp>=?", (cutoff,)
                ).fetchall()
        finally:
            conn.close()

        from datetime import datetime, timezone
        by_hour = {h: 0 for h in range(24)}
        by_dow = {d: 0 for d in range(7)}

        for r in rows:
            dt = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
            by_hour[dt.hour] += 1
            # remap: strftime Sunday=0 → Monday=0
            dow = (dt.weekday())  # already Mon=0 in Python
            by_dow[dow] += 1

        return {
            "tool_name": tool_name,
            "days": days,
            "by_hour": by_hour,
            "by_dow": by_dow,
            "total": len(rows),
        }
    except Exception:
        return {"tool_name": tool_name, "days": days, "by_hour": {}, "by_dow": {}, "total": 0}


def prune_old_failures(db_path: Path, retention_days: int) -> int:
    """Delete failure records older than retention_days. Returns number of rows deleted."""
    cutoff = time.time() - retention_days * 86400
    try:
        conn = _get_conn(db_path)
        try:
            cur = conn.execute("DELETE FROM tool_failures WHERE timestamp < ?", (cutoff,))
            deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted
    except Exception:
        return 0


SCHEMA: dict = {
    "name": "analyze_failures",
    "description": (
        "Analyze JARVIS's tool failure history to find recurring error patterns. "
        "Use this to identify broken tools or common misuse patterns for self-improvement."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Filter to a specific tool name, or omit to see all failures.",
            }
        },
        "required": [],
    },
}

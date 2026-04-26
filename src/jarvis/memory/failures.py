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
        conn.execute(
            "INSERT INTO tool_failures (timestamp, tool_name, tool_input, error_msg) VALUES (?, ?, ?, ?)",
            (time.time(), tool_name, json.dumps(tool_input, default=str), error_msg),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def handle_analyze_failures(tool_input: dict, db_path: Path) -> str:
    """Tool handler: return failure patterns grouped by tool + error."""
    try:
        tool_name = tool_input.get("tool_name")
        conn = _get_conn(db_path)

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

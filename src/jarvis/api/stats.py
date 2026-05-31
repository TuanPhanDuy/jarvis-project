"""System-wide statistics aggregated from SQLite tables.

Pulled together into a single snapshot for GET /api/stats.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def _query(db_path: Path, sql: str, params: tuple = ()) -> list:
    """Run a SELECT and return rows as dicts. Returns [] if DB or table absent."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _scalar(db_path: Path, sql: str, params: tuple = (), default=0):
    rows = _query(db_path, sql, params)
    if rows:
        return list(rows[0].values())[0] or default
    return default


def get_system_stats(db_path: Path, sessions_active: int = 0) -> dict:
    """Return a comprehensive system snapshot.

    Args:
        db_path:         Path to jarvis.db.
        sessions_active: Number of currently active in-memory sessions
                         (passed in from the server, not derivable from DB).
    """
    # ── Sessions ──────────────────────────────────────────────────────────────
    sessions_persisted = _scalar(db_path, "SELECT COUNT(*) FROM persisted_sessions")

    # ── Tokens & cost ─────────────────────────────────────────────────────────
    tok_row = _query(
        db_path,
        "SELECT COALESCE(SUM(input_tokens),0) AS inp, COALESCE(SUM(output_tokens),0) AS out "
        "FROM agent_turns",
    )
    total_input = tok_row[0]["inp"] if tok_row else 0
    total_output = tok_row[0]["out"] if tok_row else 0

    # Estimate cost using Sonnet pricing as a safe default
    cost_usd = (total_input * 3.0 + total_output * 15.0) / 1_000_000

    # ── Tools ─────────────────────────────────────────────────────────────────
    tool_rows = _query(
        db_path,
        "SELECT tool_name, COUNT(*) AS calls, SUM(CASE WHEN result_ok=0 THEN 1 ELSE 0 END) AS errors "
        "FROM audit_log GROUP BY tool_name ORDER BY calls DESC LIMIT 10",
    )
    total_tool_calls = _scalar(db_path, "SELECT COUNT(*) FROM audit_log")
    total_tool_errors = _scalar(
        db_path, "SELECT COUNT(*) FROM audit_log WHERE result_ok=0"
    )

    # ── Memory ────────────────────────────────────────────────────────────────
    episodes = _scalar(db_path, "SELECT COUNT(*) FROM episodic_memory")
    graph_entities = _scalar(db_path, "SELECT COUNT(DISTINCT name) FROM kg_entities")
    preferences = _scalar(db_path, "SELECT COUNT(*) FROM user_preferences")

    # ── Jobs ──────────────────────────────────────────────────────────────────
    job_rows = _query(
        db_path,
        "SELECT status, COUNT(*) AS cnt FROM agent_jobs GROUP BY status",
    )
    jobs_by_status = {r["status"]: r["cnt"] for r in job_rows}

    # ── DB size ───────────────────────────────────────────────────────────────
    db_size_bytes = db_path.stat().st_size if db_path.exists() else 0

    return {
        "generated_at": time.time(),
        "sessions": {
            "active": sessions_active,
            "persisted_total": sessions_persisted,
        },
        "tokens": {
            "total_input": int(total_input),
            "total_output": int(total_output),
            "total": int(total_input + total_output),
            "estimated_cost_usd": round(cost_usd, 6),
        },
        "tools": {
            "total_calls": int(total_tool_calls),
            "total_errors": int(total_tool_errors),
            "error_rate": round(total_tool_errors / total_tool_calls, 4) if total_tool_calls else 0.0,
            "top_10": [
                {
                    "tool_name": r["tool_name"],
                    "calls": r["calls"],
                    "errors": r["errors"],
                }
                for r in tool_rows
            ],
        },
        "memory": {
            "episodes": int(episodes),
            "graph_entities": int(graph_entities),
            "preferences": int(preferences),
        },
        "jobs": jobs_by_status,
        "db_size_bytes": db_size_bytes,
    }

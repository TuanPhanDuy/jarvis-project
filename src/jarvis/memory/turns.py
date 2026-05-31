"""Per-agent-turn audit table: tokens, latency, model, tool calls.

Complements OTel spans with a lightweight SQLite record that is queryable
without any external infrastructure. Exposed via GET /api/turns.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_turns (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL DEFAULT '',
            agent_type      TEXT NOT NULL DEFAULT '',
            model           TEXT NOT NULL DEFAULT '',
            input_tokens    INTEGER NOT NULL DEFAULT 0,
            output_tokens   INTEGER NOT NULL DEFAULT 0,
            tool_calls_json TEXT NOT NULL DEFAULT '[]',
            latency_ms      REAL NOT NULL DEFAULT 0,
            timestamp       REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON agent_turns(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_ts ON agent_turns(timestamp DESC)")
    conn.commit()
    return conn


def log_turn(
    db_path: Path,
    session_id: str,
    agent_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    tool_calls: list[str],
    latency_ms: float,
) -> None:
    """Record one agent turn. Best-effort — never raises."""
    try:
        conn = _get_conn(db_path)
        try:
            conn.execute(
                """INSERT INTO agent_turns
                   (id, session_id, agent_type, model, input_tokens, output_tokens,
                    tool_calls_json, latency_ms, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), session_id, agent_type, model,
                 input_tokens, output_tokens, json.dumps(tool_calls), latency_ms, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def prune_old_turns(db_path: Path, retention_days: int) -> int:
    """Delete agent_turns older than retention_days. Returns number of rows deleted."""
    cutoff = time.time() - retention_days * 86400
    try:
        conn = _get_conn(db_path)
        try:
            cur = conn.execute("DELETE FROM agent_turns WHERE timestamp < ?", (cutoff,))
            deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted
    except Exception:
        return 0


# USD per 1M tokens: {model_substring: (input_price, output_price)}
_MODEL_PRICING: list[tuple[str, float, float]] = [
    ("haiku",  0.80,  4.00),
    ("sonnet",  3.00, 15.00),
    ("opus",   15.00, 75.00),
]


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    model_lower = model.lower()
    for fragment, in_price, out_price in _MODEL_PRICING:
        if fragment in model_lower:
            return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return (input_tokens * 3.00 + output_tokens * 15.00) / 1_000_000


def get_session_cost(db_path: Path, session_id: str) -> dict:
    """Return per-turn cost breakdown and totals for a session."""
    try:
        conn = _get_conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM agent_turns WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        turns = []
        total_input = total_output = total_cost = 0.0
        for i, r in enumerate(rows):
            d = dict(r)
            cost = _cost_usd(d["model"], d["input_tokens"], d["output_tokens"])
            turns.append({
                "turn_index": i,
                "model": d["model"],
                "agent_type": d["agent_type"],
                "input_tokens": d["input_tokens"],
                "output_tokens": d["output_tokens"],
                "cost_usd": round(cost, 8),
                "latency_ms": d["latency_ms"],
                "timestamp": d["timestamp"],
            })
            total_input += d["input_tokens"]
            total_output += d["output_tokens"]
            total_cost += cost
        return {
            "session_id": session_id,
            "turns": turns,
            "totals": {
                "turn_count": len(turns),
                "input_tokens": int(total_input),
                "output_tokens": int(total_output),
                "total_tokens": int(total_input + total_output),
                "cost_usd": round(total_cost, 8),
            },
        }
    except Exception:
        return {"session_id": session_id, "turns": [], "totals": {}}


def get_turn_stats(
    db_path: Path,
    session_id: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Return recent turns, newest first. Optionally filter by session."""
    try:
        conn = _get_conn(db_path)
        try:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM agent_turns WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM agent_turns ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        finally:
            conn.close()
        result = []
        for r in rows:
            d = dict(r)
            d["tool_calls"] = json.loads(d.pop("tool_calls_json", "[]"))
            result.append(d)
        return result
    except Exception:
        return []

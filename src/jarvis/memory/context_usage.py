"""Per-turn context window usage tracking.

Records input/output tokens after each agent turn so callers can see how
quickly a session is approaching the model's context limit.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

# Context window sizes in tokens for common models
_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3-5-sonnet", 200_000),
    ("claude-3-haiku", 200_000),
    ("claude-3-opus", 200_000),
    ("llama", 128_000),
    ("mistral", 32_000),
]

_DEFAULT_CONTEXT = 200_000


def _context_window(model: str) -> int:
    model_lower = (model or "").lower()
    for fragment, size in _CONTEXT_WINDOWS:
        if fragment in model_lower:
            return size
    return _DEFAULT_CONTEXT


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_usage (
            id             TEXT PRIMARY KEY,
            session_id     TEXT NOT NULL,
            turn_index     INTEGER NOT NULL,
            model          TEXT NOT NULL DEFAULT '',
            input_tokens   INTEGER NOT NULL DEFAULT 0,
            output_tokens  INTEGER NOT NULL DEFAULT 0,
            context_window INTEGER NOT NULL DEFAULT 200000,
            timestamp      REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cu_session ON context_usage(session_id, turn_index)")
    conn.commit()
    return conn


def record_usage(
    db_path: Path,
    session_id: str,
    turn_index: int,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Record token usage for one agent turn. Best-effort — never raises."""
    try:
        window = _context_window(model)
        conn = _conn(db_path)
        try:
            conn.execute(
                """INSERT INTO context_usage
                   (id, session_id, turn_index, model, input_tokens, output_tokens, context_window, timestamp)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), session_id, turn_index, model,
                 input_tokens, output_tokens, window, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def get_context_pressure(db_path: Path, session_id: str) -> dict:
    """Return per-turn context usage history and current pressure for a session.

    Returns:
      {session_id, context_window, turns: [{turn_index, input_tokens, output_tokens,
        cumulative_tokens, pressure_pct, timestamp}], current_pressure_pct}
    """
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM context_usage WHERE session_id=? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {"session_id": session_id, "context_window": _DEFAULT_CONTEXT, "turns": [], "current_pressure_pct": 0.0}

        context_window = rows[-1]["context_window"]
        cumulative = 0
        turns = []
        for r in rows:
            cumulative += r["input_tokens"] + r["output_tokens"]
            pressure = round(r["input_tokens"] / context_window * 100, 1)
            turns.append({
                "turn_index": r["turn_index"],
                "model": r["model"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cumulative_tokens": cumulative,
                "pressure_pct": pressure,
                "timestamp": r["timestamp"],
            })

        current_pressure = turns[-1]["pressure_pct"] if turns else 0.0
        return {
            "session_id": session_id,
            "context_window": context_window,
            "turns": turns,
            "current_pressure_pct": current_pressure,
        }
    except Exception:
        return {"session_id": session_id, "context_window": _DEFAULT_CONTEXT, "turns": [], "current_pressure_pct": 0.0}

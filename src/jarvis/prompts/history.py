"""Persistent prompt version history.

Each time a prompt override is set via the API, the previous value is saved
here so it can be browsed and rolled back.

This complements the in-memory override store (overrides.py) with a durable
audit trail of what prompts looked like over time.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_history (
            version_id  TEXT PRIMARY KEY,
            agent_type  TEXT NOT NULL,
            prompt      TEXT NOT NULL,
            set_at      REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ph_agent ON prompt_history(agent_type, set_at DESC)"
    )
    conn.commit()
    return conn


def record_version(db_path: Path, agent_type: str, prompt: str) -> str:
    """Save a prompt snapshot. Returns the version_id."""
    version_id = str(uuid.uuid4())
    try:
        conn = _conn(db_path)
        try:
            conn.execute(
                "INSERT INTO prompt_history (version_id, agent_type, prompt, set_at) VALUES (?,?,?,?)",
                (version_id, agent_type.lower(), prompt, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
    return version_id


def get_history(db_path: Path, agent_type: str) -> list[dict]:
    """Return all saved versions for an agent type, newest first."""
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT version_id, agent_type, set_at, LENGTH(prompt) AS length_chars "
                "FROM prompt_history WHERE agent_type = ? ORDER BY set_at DESC",
                (agent_type.lower(),),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_version(db_path: Path, version_id: str) -> dict | None:
    """Return a single version including the full prompt text."""
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM prompt_history WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None
    except Exception:
        return None

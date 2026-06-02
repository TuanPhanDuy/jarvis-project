"""Agent conversation checkpointing — save and restore mid-run message state.

Each checkpoint captures the full messages list at a specific turn count.
Checkpoints are stored in SQLite and can be used to fork a new session from
any prior snapshot (useful for resuming interrupted long research runs).
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
        CREATE TABLE IF NOT EXISTS agent_checkpoints (
            id           TEXT PRIMARY KEY,
            session_id   TEXT NOT NULL,
            turn_count   INTEGER NOT NULL,
            agent_type   TEXT NOT NULL DEFAULT '',
            messages_json TEXT NOT NULL DEFAULT '[]',
            created_at   REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cp_session ON agent_checkpoints(session_id, turn_count)"
    )
    conn.commit()
    return conn


def save_checkpoint(
    db_path: Path,
    session_id: str,
    turn_count: int,
    agent_type: str,
    messages: list[dict],
) -> str:
    """Persist a snapshot; returns the checkpoint ID. Best-effort — never raises."""
    cp_id = str(uuid.uuid4())
    try:
        conn = _get_conn(db_path)
        try:
            conn.execute(
                """INSERT INTO agent_checkpoints
                   (id, session_id, turn_count, agent_type, messages_json, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (cp_id, session_id, turn_count, agent_type, json.dumps(messages), time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
    return cp_id


def list_checkpoints(db_path: Path, session_id: str) -> list[dict]:
    """Return all checkpoints for a session, oldest first."""
    try:
        conn = _get_conn(db_path)
        try:
            rows = conn.execute(
                """SELECT id, session_id, turn_count, agent_type, created_at
                   FROM agent_checkpoints
                   WHERE session_id = ?
                   ORDER BY turn_count ASC""",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def load_checkpoint(db_path: Path, checkpoint_id: str) -> dict | None:
    """Load a checkpoint by ID; returns None if not found."""
    try:
        conn = _get_conn(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM agent_checkpoints WHERE id = ?",
                (checkpoint_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        d = dict(row)
        d["messages"] = json.loads(d.pop("messages_json", "[]"))
        return d
    except Exception:
        return None


def diff_checkpoints(db_path: Path, checkpoint_a: str, checkpoint_b: str) -> dict:
    """Compare two checkpoints and return a structured delta.

    Returns:
      {checkpoint_a, checkpoint_b, added: [...], removed: [...], common_count: int}

    Messages are compared by (role, content) identity.  Order is preserved.
    """
    cp_a = load_checkpoint(db_path, checkpoint_a)
    cp_b = load_checkpoint(db_path, checkpoint_b)

    if cp_a is None:
        raise ValueError(f"Checkpoint '{checkpoint_a}' not found")
    if cp_b is None:
        raise ValueError(f"Checkpoint '{checkpoint_b}' not found")

    msgs_a = cp_a["messages"]
    msgs_b = cp_b["messages"]

    def _key(m: dict) -> str:
        content = m.get("content", "")
        if isinstance(content, list):
            content = str(content)
        return f"{m.get('role', '')}::{content[:200]}"

    set_a = {_key(m) for m in msgs_a}
    set_b = {_key(m) for m in msgs_b}

    added = [m for m in msgs_b if _key(m) not in set_a]
    removed = [m for m in msgs_a if _key(m) not in set_b]

    return {
        "checkpoint_a": {"id": checkpoint_a, "turn_count": cp_a["turn_count"], "created_at": cp_a["created_at"]},
        "checkpoint_b": {"id": checkpoint_b, "turn_count": cp_b["turn_count"], "created_at": cp_b["created_at"]},
        "message_count_a": len(msgs_a),
        "message_count_b": len(msgs_b),
        "common_count": len(set_a & set_b),
        "added": added,
        "removed": removed,
    }


def delete_checkpoints(db_path: Path, session_id: str) -> int:
    """Delete all checkpoints for a session. Returns number of rows deleted."""
    try:
        conn = _get_conn(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM agent_checkpoints WHERE session_id = ?",
                (session_id,),
            )
            deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted
    except Exception:
        return 0

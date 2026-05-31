"""Free-text session notes / annotations.

Notes are independent of the message history — they are observer comments
added by users or automated processes (e.g., "reviewed and correct",
"needs follow-up", "key insight"). They survive export/import.
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
        CREATE TABLE IF NOT EXISTS session_notes (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            content     TEXT NOT NULL,
            author      TEXT NOT NULL DEFAULT 'user',
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sn_session ON session_notes(session_id, created_at DESC)")
    conn.commit()
    return conn


def add_note(
    db_path: Path,
    session_id: str,
    content: str,
    author: str = "user",
) -> dict:
    """Create a note and return it. Raises ValueError if content is empty."""
    content = content.strip()
    if not content:
        raise ValueError("Note content cannot be empty")
    note_id = str(uuid.uuid4())
    created_at = time.time()
    conn = _conn(db_path)
    try:
        conn.execute(
            "INSERT INTO session_notes (id, session_id, content, author, created_at) VALUES (?,?,?,?,?)",
            (note_id, session_id, content, author.strip() or "user", created_at),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": note_id, "session_id": session_id, "content": content,
            "author": author, "created_at": created_at}


def list_notes(db_path: Path, session_id: str) -> list[dict]:
    """Return all notes for a session, oldest first."""
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM session_notes WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def delete_note(db_path: Path, note_id: str) -> bool:
    """Delete a note by ID. Returns True if it existed."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute("DELETE FROM session_notes WHERE id = ?", (note_id,))
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False


def get_notes_for_export(db_path: Path, session_id: str) -> list[dict]:
    """Return notes in export-friendly format (same as list_notes)."""
    return list_notes(db_path, session_id)

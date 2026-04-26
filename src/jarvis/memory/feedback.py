"""Feedback memory: stores user ratings on JARVIS responses.

Ratings feed into quality tracking and can inform prompt improvements.
Claude can call record_feedback to log explicit user signals.

DB: reports_dir/jarvis.db (shared SQLite, feedback table).
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            user_id     TEXT    NOT NULL DEFAULT 'anonymous',
            msg_hash    TEXT    NOT NULL,
            rating      INTEGER NOT NULL,   -- 1 (bad) to 5 (excellent), or -1/+1 thumbs
            comment     TEXT    NOT NULL DEFAULT '',
            ts          REAL    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_session ON feedback(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_ts      ON feedback(ts DESC)")
    conn.commit()
    return conn


def log_feedback(
    db_path: Path,
    session_id: str,
    response_text: str,
    rating: int,
    comment: str = "",
    user_id: str = "anonymous",
) -> None:
    """Store a rating for a JARVIS response. Rating: 1-5 or -1/+1."""
    msg_hash = hashlib.sha256(response_text.encode()).hexdigest()[:16]
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO feedback (session_id, user_id, msg_hash, rating, comment, ts) VALUES (?,?,?,?,?,?)",
        (session_id, user_id, msg_hash, rating, comment, time.time()),
    )
    conn.commit()
    conn.close()


def get_feedback_stats(db_path: Path, session_id: str | None = None) -> dict:
    """Return aggregate feedback statistics, optionally scoped to a session."""
    conn = _get_conn(db_path)
    if session_id:
        row = conn.execute(
            "SELECT COUNT(*) as cnt, AVG(rating) as avg_rating FROM feedback WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        recent = conn.execute(
            "SELECT rating, comment, ts FROM feedback WHERE session_id = ? ORDER BY ts DESC LIMIT 5",
            (session_id,),
        ).fetchall()
    else:
        row = conn.execute("SELECT COUNT(*) as cnt, AVG(rating) as avg_rating FROM feedback").fetchone()
        recent = conn.execute(
            "SELECT rating, comment, ts FROM feedback ORDER BY ts DESC LIMIT 5"
        ).fetchall()
    conn.close()
    return {
        "total": row["cnt"] or 0,
        "avg_rating": round(row["avg_rating"] or 0, 2),
        "recent": [{"rating": r["rating"], "comment": r["comment"]} for r in recent],
    }


def handle_record_feedback(tool_input: dict, db_path: Path) -> str:
    try:
        session_id = tool_input.get("session_id", "unknown")
        response_snippet = tool_input.get("response_snippet", "")
        rating = int(tool_input["rating"])
        comment = tool_input.get("comment", "")
        if not (-1 <= rating <= 5):
            return "ERROR: rating must be -1 (thumbs down), +1 (thumbs up), or 1-5."
        log_feedback(db_path, session_id, response_snippet, rating, comment)
        return f"Feedback recorded (rating: {rating}). Thank you!"
    except Exception as e:
        return f"ERROR: record_feedback failed — {e}"


SCHEMA: dict = {
    "name": "record_feedback",
    "description": (
        "Record user feedback on a JARVIS response. Call this when the user explicitly "
        "rates a response (thumbs up/down, 1-5 stars, or says 'that was wrong/great'). "
        "This helps JARVIS improve over time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Current session ID.",
            },
            "response_snippet": {
                "type": "string",
                "description": "First 200 chars of the response being rated.",
            },
            "rating": {
                "type": "integer",
                "description": "Rating: -1 (thumbs down), 1 (thumbs up), or 1-5 (stars).",
            },
            "comment": {
                "type": "string",
                "description": "Optional user comment explaining the rating.",
            },
        },
        "required": ["session_id", "rating"],
    },
}

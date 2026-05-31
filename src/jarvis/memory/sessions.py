"""Session persistence — saves in-memory sessions to SQLite and reloads on startup.

Each session row stores the full message history as JSON so the server can
resume conversations after a restart.  Rows are soft-expiry: loading only
restores sessions younger than the configured TTL.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import structlog

log = structlog.get_logger()


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persisted_sessions (
            session_id  TEXT PRIMARY KEY,
            agent_type  TEXT NOT NULL DEFAULT 'PlannerAgent',
            user_id     TEXT,
            messages    TEXT NOT NULL DEFAULT '[]',
            fork_of     TEXT,
            title       TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ps_updated ON persisted_sessions(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_ps_user    ON persisted_sessions(user_id);

        CREATE TABLE IF NOT EXISTS session_tags (
            session_id  TEXT NOT NULL,
            tag         TEXT NOT NULL,
            created_at  REAL NOT NULL,
            PRIMARY KEY (session_id, tag)
        );
        CREATE INDEX IF NOT EXISTS idx_st_tag ON session_tags(tag);

        CREATE VIRTUAL TABLE IF NOT EXISTS session_fts
            USING fts5(session_id UNINDEXED, content, tokenize='porter ascii');
    """)
    conn.commit()
    return conn


def save_session(
    db_path: Path,
    session_id: str,
    messages: list[dict],
    agent_type: str = "PlannerAgent",
    user_id: str | None = None,
    fork_of: str | None = None,
    created_at: float | None = None,
) -> None:
    """Upsert a session's message history. Best-effort — never raises."""
    try:
        now = time.time()
        conn = _conn(db_path)
        try:
            conn.execute(
                """INSERT INTO persisted_sessions
                       (session_id, agent_type, user_id, messages, fork_of, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       messages   = excluded.messages,
                       agent_type = excluded.agent_type,
                       updated_at = excluded.updated_at""",
                (
                    session_id,
                    agent_type,
                    user_id,
                    json.dumps(messages),
                    fork_of,
                    created_at or now,
                    now,
                ),
            )
            # Keep FTS index in sync
            content = " ".join(
                str(m.get("content", ""))
                for m in messages
                if isinstance(m.get("content"), str)
            )
            conn.execute("DELETE FROM session_fts WHERE session_id = ?", (session_id,))
            conn.execute(
                "INSERT INTO session_fts (session_id, content) VALUES (?,?)",
                (session_id, content),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("session_save_failed", session_id=session_id, error=str(exc))


def load_sessions(db_path: Path, ttl_minutes: int = 60) -> list[dict]:
    """Return all sessions updated within ttl_minutes, ordered newest-first."""
    if not db_path.exists():
        return []
    cutoff = time.time() - ttl_minutes * 60
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM persisted_sessions WHERE updated_at >= ? ORDER BY updated_at DESC",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        result = []
        for row in rows:
            try:
                messages = json.loads(row["messages"])
            except Exception:
                messages = []
            result.append({
                "session_id": row["session_id"],
                "agent_type": row["agent_type"],
                "user_id": row["user_id"],
                "messages": messages,
                "fork_of": row["fork_of"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return result
    except Exception as exc:
        log.warning("session_load_failed", error=str(exc))
        return []


def get_session_history(
    db_path: Path,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Return paginated messages for a persisted session, newest-first."""
    if not db_path.exists():
        return []
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT messages FROM persisted_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return []
        messages: list[dict] = json.loads(row["messages"])
        messages.reverse()
        return messages[offset: offset + limit]
    except Exception:
        return []


def delete_persisted_session(db_path: Path, session_id: str) -> bool:
    """Remove a persisted session. Returns True if it existed."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM persisted_sessions WHERE session_id = ?", (session_id,)
            )
            conn.execute("DELETE FROM session_tags WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_fts WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False


# ── Title / metadata ──────────────────────────────────────────────────────────

def generate_title(messages: list[dict]) -> str:
    """Derive a short title (≤10 words) from the first user message.

    Rules:
    - Strip leading filler phrases ("can you", "please", "jarvis", etc.)
    - Truncate to 10 words
    - Title-case the result
    """
    import re
    first = next(
        (str(m.get("content", "")) for m in messages if m.get("role") == "user"),
        "",
    ).strip()
    if not first:
        return "Untitled session"
    # Remove common filler prefixes
    fillers = r"^(hey|hi|hello|jarvis|please|can you|could you|would you|i want you to|i need you to)[,.]?\s+"
    text = re.sub(fillers, "", first, flags=re.IGNORECASE).strip()
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text)
    # Truncate to 10 words
    words = text.split()
    title = " ".join(words[:10])
    if len(words) > 10:
        title += "…"
    return title.capitalize() if title else "Untitled session"


def set_title(db_path: Path, session_id: str, title: str) -> bool:
    """Set or update the title for a persisted session. Returns False if not found."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute(
                "UPDATE persisted_sessions SET title = ? WHERE session_id = ?",
                (title.strip(), session_id),
            )
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False


def get_metadata(db_path: Path, session_id: str) -> dict | None:
    """Return metadata (excluding messages) for a persisted session."""
    if not db_path.exists():
        return None
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                """SELECT session_id, agent_type, user_id, fork_of, title, created_at, updated_at
                   FROM persisted_sessions WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        tags = get_tags(db_path, session_id)
        return {
            "session_id": row["session_id"],
            "agent_type": row["agent_type"],
            "user_id": row["user_id"],
            "fork_of": row["fork_of"],
            "title": row["title"],
            "tags": tags,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    except Exception:
        return None


# ── Tags ─────────────────────────────────────────────────────────────────────

def add_tag(db_path: Path, session_id: str, tag: str) -> bool:
    """Add a tag to a session. Returns False if session does not exist."""
    try:
        conn = _conn(db_path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM persisted_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not exists:
                return False
            conn.execute(
                "INSERT OR IGNORE INTO session_tags (session_id, tag, created_at) VALUES (?,?,?)",
                (session_id, tag.strip().lower(), time.time()),
            )
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception:
        return False


def remove_tag(db_path: Path, session_id: str, tag: str) -> bool:
    """Remove a tag from a session. Returns True if the tag existed."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM session_tags WHERE session_id = ? AND tag = ?",
                (session_id, tag.strip().lower()),
            )
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False


def get_tags(db_path: Path, session_id: str) -> list[str]:
    """Return all tags for a session, sorted alphabetically."""
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT tag FROM session_tags WHERE session_id = ? ORDER BY tag",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        return [r["tag"] for r in rows]
    except Exception:
        return []


def get_sessions_by_tag(db_path: Path, tag: str) -> list[str]:
    """Return session_ids that have the given tag."""
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT session_id FROM session_tags WHERE tag = ? ORDER BY created_at DESC",
                (tag.strip().lower(),),
            ).fetchall()
        finally:
            conn.close()
        return [r["session_id"] for r in rows]
    except Exception:
        return []


# ── Full-text search ──────────────────────────────────────────────────────────

def index_session(db_path: Path, session_id: str, messages: list[dict]) -> None:
    """Update the FTS index for a session. Best-effort — never raises."""
    try:
        content = " ".join(
            str(m.get("content", ""))
            for m in messages
            if isinstance(m.get("content"), str)
        )
        conn = _conn(db_path)
        try:
            conn.execute("DELETE FROM session_fts WHERE session_id = ?", (session_id,))
            conn.execute(
                "INSERT INTO session_fts (session_id, content) VALUES (?,?)",
                (session_id, content),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def search_sessions(db_path: Path, query: str, tag: str | None = None) -> list[dict]:
    """Full-text search across session message content.

    Returns [{session_id, snippet, rank}], ranked by relevance.
    Optionally filter to sessions with a specific tag.
    """
    if not query.strip():
        return []
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                """SELECT session_id,
                          snippet(session_fts, 1, '<b>', '</b>', '…', 20) AS snippet,
                          rank
                   FROM session_fts
                   WHERE content MATCH ?
                   ORDER BY rank
                   LIMIT 50""",
                (query,),
            ).fetchall()
        finally:
            conn.close()
        results = [dict(r) for r in rows]
        if tag:
            tagged = set(get_sessions_by_tag(db_path, tag))
            results = [r for r in results if r["session_id"] in tagged]
        return results
    except Exception:
        return []

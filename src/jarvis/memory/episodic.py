"""Episodic memory: timestamped conversation log in SQLite with FTS5 search.

All conversations are automatically logged by the API server. Claude can search
this log to recall what was discussed in previous sessions.

DB location: reports_dir/jarvis.db (shared with graph and memory_access tables).
"""
from __future__ import annotations

import datetime
import sqlite3
import time
from pathlib import Path


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            session_id  TEXT    NOT NULL,
            user_id     TEXT    NOT NULL DEFAULT 'anonymous',
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ep_ts      ON episodes(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_ep_session ON episodes(session_id);
        CREATE INDEX IF NOT EXISTS idx_ep_user    ON episodes(user_id);
    """)
    # FTS5 is optional — fall back to LIKE search if unavailable
    try:
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                content,
                content='episodes',
                content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn


def log_episode(
    db_path: Path, session_id: str, role: str, content: str, user_id: str = "anonymous"
) -> None:
    """Append a conversation turn to episodic memory. Best-effort — never raises."""
    try:
        conn = _get_conn(db_path)
        conn.execute(
            "INSERT INTO episodes (timestamp, session_id, user_id, role, content) VALUES (?,?,?,?,?)",
            (time.time(), session_id, user_id, role, content),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _search(db_path: Path, query: str, limit: int, user_id: str | None = None) -> list[sqlite3.Row]:
    conn = _get_conn(db_path)
    user_filter = "AND e.user_id = ?" if user_id else ""
    user_arg = (user_id,) if user_id else ()
    try:
        rows = conn.execute(
            f"""
            SELECT e.id, e.timestamp, e.session_id, e.user_id, e.role, e.content
            FROM episodes_fts f
            JOIN episodes e ON e.id = f.rowid
            WHERE episodes_fts MATCH ? {user_filter}
            ORDER BY rank
            LIMIT ?
            """,
            (query, *user_arg, limit),
        ).fetchall()
        if rows:
            conn.close()
            return rows
    except sqlite3.OperationalError:
        pass
    rows = conn.execute(
        f"SELECT * FROM episodes WHERE content LIKE ? {user_filter} ORDER BY timestamp DESC LIMIT ?",
        (f"%{query}%", *user_arg, limit),
    ).fetchall()
    conn.close()
    return rows


def handle_search_episodic_memory(tool_input: dict, db_path: Path, user_id: str | None = None) -> str:
    try:
        query = tool_input["query"]
        limit = int(tool_input.get("limit", 10))
        rows = _search(db_path, query, limit, user_id=user_id)

        if not rows:
            return f"No episodes found matching '{query}'."

        lines = [f"Found {len(rows)} episode(s) matching '{query}':\n"]
        for row in rows:
            ts = datetime.datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{ts}] [{row['role']}] (session: {row['session_id'][:8]}…)")
            lines.append(row["content"][:400].strip())
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: search_episodic_memory failed — {e}"


def prune_old_episodes(db_path: Path, retention_days: int) -> int:
    """Delete episodes older than retention_days. Returns number of rows deleted."""
    cutoff = time.time() - retention_days * 86400
    try:
        conn = _get_conn(db_path)
        cur = conn.execute("DELETE FROM episodes WHERE timestamp < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        return deleted
    except Exception:
        return 0


SCHEMA: dict = {
    "name": "search_episodic_memory",
    "description": (
        "Search JARVIS's episodic memory — a timestamped log of all past conversations. "
        "Use this to recall what was discussed in previous sessions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords or phrase to search in past conversations.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}

"""Topic watchlist — monitored search queries that fire notifications on new hits.

Each watchlist entry holds a topic string, optional keywords, and a check
interval. The scheduler calls check_watchlist() periodically; any new results
since the last check are persisted and a notification is created.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id           TEXT PRIMARY KEY,
            topic        TEXT NOT NULL,
            keywords     TEXT NOT NULL DEFAULT '[]',
            user_id      TEXT NOT NULL DEFAULT 'anonymous',
            enabled      INTEGER NOT NULL DEFAULT 1,
            last_checked REAL,
            hit_count    INTEGER NOT NULL DEFAULT 0,
            created_at   REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_hits (
            id           TEXT PRIMARY KEY,
            watch_id     TEXT NOT NULL,
            title        TEXT NOT NULL DEFAULT '',
            url          TEXT NOT NULL DEFAULT '',
            snippet      TEXT NOT NULL DEFAULT '',
            found_at     REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wl_user ON watchlist(user_id, enabled)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wlh_watch ON watchlist_hits(watch_id, found_at DESC)")
    conn.commit()
    return conn


def create_watch(
    db_path: Path,
    topic: str,
    keywords: list[str] | None = None,
    user_id: str = "anonymous",
) -> dict:
    wid = str(uuid.uuid4())
    now = time.time()
    conn = _conn(db_path)
    try:
        conn.execute(
            "INSERT INTO watchlist (id, topic, keywords, user_id, created_at) VALUES (?,?,?,?,?)",
            (wid, topic, json.dumps(keywords or []), user_id, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": wid, "topic": topic, "keywords": keywords or [], "user_id": user_id,
            "enabled": True, "hit_count": 0, "last_checked": None, "created_at": now}


def list_watches(db_path: Path, user_id: str | None = None) -> list[dict]:
    try:
        conn = _conn(db_path)
        try:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM watchlist WHERE user_id=? ORDER BY created_at DESC", (user_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM watchlist ORDER BY created_at DESC"
                ).fetchall()
        finally:
            conn.close()
        result = []
        for r in rows:
            d = dict(r)
            d["keywords"] = json.loads(d["keywords"])
            d["enabled"] = bool(d["enabled"])
            result.append(d)
        return result
    except Exception:
        return []


def get_watch(db_path: Path, watch_id: str) -> dict | None:
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute("SELECT * FROM watchlist WHERE id=?", (watch_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        d = dict(row)
        d["keywords"] = json.loads(d["keywords"])
        d["enabled"] = bool(d["enabled"])
        return d
    except Exception:
        return None


def delete_watch(db_path: Path, watch_id: str) -> bool:
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute("DELETE FROM watchlist WHERE id=?", (watch_id,))
            conn.execute("DELETE FROM watchlist_hits WHERE watch_id=?", (watch_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    except Exception:
        return False


def toggle_watch(db_path: Path, watch_id: str, enabled: bool) -> bool:
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute("UPDATE watchlist SET enabled=? WHERE id=?", (int(enabled), watch_id))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    except Exception:
        return False


def get_hits(db_path: Path, watch_id: str, limit: int = 20) -> list[dict]:
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM watchlist_hits WHERE watch_id=? ORDER BY found_at DESC LIMIT ?",
                (watch_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def check_watch(db_path: Path, watch_id: str) -> list[dict]:
    """Run a search for one watchlist entry and persist new hits.

    Returns the list of new hits found (empty if nothing new).
    """
    watch = get_watch(db_path, watch_id)
    if not watch or not watch["enabled"]:
        return []

    try:
        from jarvis.tools.web_search import handle_web_search
        query = watch["topic"]
        if watch.get("keywords"):
            query += " " + " ".join(watch["keywords"])

        raw = handle_web_search({"query": query, "max_results": 5})
        results: list[dict] = []
        if isinstance(raw, str) and raw.startswith("["):
            results = json.loads(raw)
        elif isinstance(raw, list):
            results = raw
    except Exception:
        results = []

    # Deduplicate by URL against existing hits
    conn = _conn(db_path)
    try:
        existing_urls = {
            r["url"] for r in conn.execute(
                "SELECT url FROM watchlist_hits WHERE watch_id=?", (watch_id,)
            ).fetchall()
        }
        new_hits = []
        now = time.time()
        for r in results:
            url = r.get("url", "") or r.get("link", "")
            if url and url not in existing_urls:
                hit_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO watchlist_hits (id, watch_id, title, url, snippet, found_at) VALUES (?,?,?,?,?,?)",
                    (hit_id, watch_id, r.get("title", "")[:200], url[:500],
                     r.get("content", r.get("snippet", ""))[:400], now),
                )
                new_hits.append({"id": hit_id, "watch_id": watch_id, "title": r.get("title", ""),
                                  "url": url, "snippet": r.get("content", r.get("snippet", ""))[:400],
                                  "found_at": now})
        conn.execute(
            "UPDATE watchlist SET last_checked=?, hit_count=hit_count+? WHERE id=?",
            (now, len(new_hits), watch_id),
        )
        conn.commit()
    finally:
        conn.close()

    return new_hits

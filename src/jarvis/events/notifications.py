"""System notification center — stores system events in SQLite.

Events are fired from scheduler completions, eval runs, training runs,
webhook delivery failures, and tool errors.  Clients poll GET /api/notifications
or subscribe to the SSE stream.

Severity levels: info | warning | error
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import structlog

log = structlog.get_logger()

_VALID_SEVERITIES = frozenset(["info", "warning", "error"])
_VALID_EVENTS = frozenset([
    "scheduler.complete",
    "eval.complete",
    "training.complete",
    "tool.error",
    "webhook.failed",
    "session.created",
    "system.info",
])


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notifications (
            id          TEXT PRIMARY KEY,
            event       TEXT NOT NULL,
            title       TEXT NOT NULL,
            body        TEXT NOT NULL DEFAULT '',
            severity    TEXT NOT NULL DEFAULT 'info',
            read        INTEGER NOT NULL DEFAULT 0,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_notif_created ON notifications(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_notif_read    ON notifications(read);
    """)
    conn.commit()
    return conn


def push_notification(
    db_path: Path,
    event: str,
    title: str,
    body: str = "",
    severity: str = "info",
) -> dict:
    """Create a new notification. Returns the created record."""
    if severity not in _VALID_SEVERITIES:
        severity = "info"
    nid = str(uuid.uuid4())
    now = time.time()
    try:
        conn = _conn(db_path)
        try:
            conn.execute(
                "INSERT INTO notifications (id, event, title, body, severity, read, created_at)"
                " VALUES (?, ?, ?, ?, ?, 0, ?)",
                (nid, event, title, body, severity, now),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("notification_push_failed", error=str(exc))
    log.info("notification_pushed", notif_id=nid, notif_event=event, title=title, severity=severity)
    return {"id": nid, "event": event, "title": title, "body": body,
            "severity": severity, "read": False, "created_at": now}


def list_notifications(
    db_path: Path,
    unread_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Return notifications newest-first."""
    if not db_path.exists():
        return []
    try:
        conn = _conn(db_path)
        try:
            where = "WHERE read = 0" if unread_only else ""
            rows = conn.execute(
                f"SELECT * FROM notifications {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": r["id"], "event": r["event"], "title": r["title"],
                "body": r["body"], "severity": r["severity"],
                "read": bool(r["read"]), "created_at": r["created_at"],
            }
            for r in rows
        ]
    except Exception:
        return []


def mark_read(db_path: Path, notification_id: str) -> bool:
    """Mark a notification as read. Returns True if it existed."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute(
                "UPDATE notifications SET read = 1 WHERE id = ?", (notification_id,)
            )
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False


def clear_read(db_path: Path) -> int:
    """Delete all read notifications. Returns count deleted."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute("DELETE FROM notifications WHERE read = 1")
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount
    except Exception:
        return 0


def unread_count(db_path: Path) -> int:
    """Return the number of unread notifications."""
    if not db_path.exists():
        return 0
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute("SELECT COUNT(*) FROM notifications WHERE read = 0").fetchone()
        finally:
            conn.close()
        return row[0] if row else 0
    except Exception:
        return 0

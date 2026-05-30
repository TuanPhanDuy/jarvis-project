"""Webhook notification system.

Stores HTTP callback registrations in SQLite and fires them asynchronously
when system events occur (schedule.complete, eval.complete, training.complete, tool.error).

Delivery retries: up to 3 attempts with exponential backoff (2s, 4s, 8s).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from pathlib import Path

import structlog

log = structlog.get_logger()

_VALID_EVENTS = frozenset([
    "schedule.complete",
    "eval.complete",
    "training.complete",
    "tool.error",
    "chat.complete",
])
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 2.0


def _conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(db_path: Path) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhooks (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                events TEXT NOT NULL,
                secret TEXT,
                created_at REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhook_deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                webhook_id TEXT NOT NULL,
                event TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                delivered_at REAL,
                error TEXT
            )
        """)


def register_webhook(
    db_path: Path,
    url: str,
    events: list[str],
    secret: str | None = None,
) -> dict:
    """Register a new webhook. Returns the created record."""
    _ensure_table(db_path)
    unknown = set(events) - _valid_events_set()
    if unknown:
        raise ValueError(f"Unknown event types: {sorted(unknown)}")
    webhook_id = str(uuid.uuid4())
    now = time.time()
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO webhooks (id, url, events, secret, created_at, active) VALUES (?,?,?,?,?,1)",
            (webhook_id, url, json.dumps(sorted(events)), secret, now),
        )
    log.info("webhook_registered", id=webhook_id, url=url, events=events)
    return {"id": webhook_id, "url": url, "events": events, "active": True, "created_at": now}


def list_webhooks(db_path: Path, event: str | None = None) -> list[dict]:
    """Return all active webhooks, optionally filtered by event type."""
    _ensure_table(db_path)
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM webhooks WHERE active = 1 ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for row in rows:
        events = json.loads(row["events"])
        if event and event not in events:
            continue
        result.append({
            "id": row["id"],
            "url": row["url"],
            "events": events,
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        })
    return result


def delete_webhook(db_path: Path, webhook_id: str) -> bool:
    """Soft-delete a webhook. Returns True if it existed."""
    _ensure_table(db_path)
    with _conn(db_path) as conn:
        result = conn.execute(
            "UPDATE webhooks SET active = 0 WHERE id = ? AND active = 1",
            (webhook_id,),
        )
    return result.rowcount > 0


def fire_event(db_path: Path, event: str, payload: dict) -> None:
    """Dispatch an event to all matching active webhooks (synchronous, with retry).

    Call this from background threads/tasks — it blocks until all deliveries complete.
    """
    _ensure_table(db_path)
    hooks = list_webhooks(db_path, event=event)
    if not hooks:
        return
    full_payload = {"event": event, "timestamp": time.time(), "data": payload}
    body = json.dumps(full_payload)
    for hook in hooks:
        _deliver(db_path, hook, event, body)


def _valid_events_set() -> frozenset[str]:
    return _VALID_EVENTS


def _sign(secret: str, body: str) -> str:
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


def _deliver(db_path: Path, hook: dict, event: str, body: str) -> None:
    import urllib.request
    import urllib.error

    headers = {"Content-Type": "application/json", "X-Jarvis-Event": event}
    if hook.get("secret"):
        headers["X-Jarvis-Signature"] = _sign(hook["secret"], body)

    last_error = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                hook["url"],
                data=body.encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    _record_delivery(db_path, hook["id"], event, body, "success", attempt, None)
                    log.info("webhook_delivered", id=hook["id"], url=hook["url"], event=event, attempt=attempt)
                    return
                last_error = f"HTTP {resp.status}"
        except Exception as exc:
            last_error = str(exc)
            log.warning("webhook_attempt_failed", id=hook["id"], attempt=attempt, error=last_error)

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))

    _record_delivery(db_path, hook["id"], event, body, "failed", _MAX_RETRIES, last_error)
    log.error("webhook_delivery_failed", id=hook["id"], url=hook["url"], event=event, error=last_error)


def _record_delivery(
    db_path: Path,
    webhook_id: str,
    event: str,
    payload: str,
    status: str,
    attempts: int,
    error: str | None,
) -> None:
    try:
        with _conn(db_path) as conn:
            conn.execute(
                """INSERT INTO webhook_deliveries
                   (webhook_id, event, payload, status, attempts, delivered_at, error)
                   VALUES (?,?,?,?,?,?,?)""",
                (webhook_id, event, payload, status, attempts, time.time(), error),
            )
    except Exception:
        pass


def get_deliveries(db_path: Path, webhook_id: str, limit: int = 20) -> list[dict]:
    """Return recent delivery records for a webhook."""
    _ensure_table(db_path)
    with _conn(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM webhook_deliveries WHERE webhook_id = ?
               ORDER BY delivered_at DESC LIMIT ?""",
            (webhook_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]

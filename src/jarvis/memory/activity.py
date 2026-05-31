"""Session activity timeline — unified chronological event feed for a session.

Merges five event types into a single sorted list:
  • messages   — user/assistant turns from the session message history
  • tool_calls — tool dispatches from the audit_log
  • checkpoints— agent state snapshots from agent_checkpoints
  • notes      — free-text annotations from session_notes
  • tags        — tag additions from session_tags (all shown at the same timestamp)

This gives operators a complete "what happened" view without loading the
full message history.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _safe_query(db_path: Path, sql: str, params: tuple = ()) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_activity(
    db_path: Path,
    session_id: str,
    messages: list[dict] | None = None,
) -> list[dict]:
    """Build a merged chronological activity timeline for a session.

    Args:
        db_path:   Path to jarvis.db.
        session_id: The session to query.
        messages:  Optional pre-loaded message list (avoids a DB round-trip).
                   If None, we skip message events (caller should pass them in
                   when available from in-memory sessions).

    Returns:
        List of activity items sorted by timestamp ascending:
        [{timestamp, type, summary, detail?}]
    """
    events: list[dict] = []

    # ── Messages ──────────────────────────────────────────────────────────────
    if messages:
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            events.append({
                "timestamp": None,  # no timestamp on in-memory messages
                "type": "message",
                "summary": f"{role.capitalize()}: {str(content)[:120]}",
                "detail": {"role": role, "index": i},
            })

    # ── Tool calls (audit_log) ─────────────────────────────────────────────
    tool_rows = _safe_query(
        db_path,
        "SELECT tool_name, result_ok, duration_ms, timestamp FROM audit_log "
        "WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,),
    )
    for r in tool_rows:
        status = "ok" if r["result_ok"] else "error"
        events.append({
            "timestamp": r["timestamp"],
            "type": "tool_call",
            "summary": f"Tool: {r['tool_name']} ({status}, {r['duration_ms']:.0f}ms)",
            "detail": {"tool_name": r["tool_name"], "result_ok": bool(r["result_ok"]),
                       "duration_ms": r["duration_ms"]},
        })

    # ── Checkpoints ───────────────────────────────────────────────────────
    cp_rows = _safe_query(
        db_path,
        "SELECT id, turn_count, agent_type, created_at FROM agent_checkpoints "
        "WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    for r in cp_rows:
        events.append({
            "timestamp": r["created_at"],
            "type": "checkpoint",
            "summary": f"Checkpoint saved at turn {r['turn_count']} ({r['agent_type']})",
            "detail": {"checkpoint_id": r["id"], "turn_count": r["turn_count"]},
        })

    # ── Notes ────────────────────────────────────────────────────────────
    note_rows = _safe_query(
        db_path,
        "SELECT id, content, author, created_at FROM session_notes "
        "WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    for r in note_rows:
        events.append({
            "timestamp": r["created_at"],
            "type": "note",
            "summary": f"Note ({r['author']}): {r['content'][:80]}",
            "detail": {"note_id": r["id"], "author": r["author"]},
        })

    # ── Tags ─────────────────────────────────────────────────────────────
    tag_rows = _safe_query(
        db_path,
        "SELECT tag, created_at FROM session_tags WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    for r in tag_rows:
        events.append({
            "timestamp": r["created_at"],
            "type": "tag",
            "summary": f"Tag added: {r['tag']}",
            "detail": {"tag": r["tag"]},
        })

    # Sort by timestamp (None timestamps — in-memory messages — go first)
    events.sort(key=lambda e: (e["timestamp"] is None, e["timestamp"] or 0))
    return events

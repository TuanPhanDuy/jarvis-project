"""SQLite-backed message annotation store.

Users label individual session messages as good / bad / uncertain to build
a training signal. Annotations are exportable as JSONL for fine-tuning.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

VALID_LABELS = {"good", "bad", "uncertain"}


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_annotations (
            id           TEXT PRIMARY KEY,
            session_id   TEXT NOT NULL,
            message_idx  INTEGER NOT NULL,
            label        TEXT NOT NULL,
            note         TEXT NOT NULL DEFAULT '',
            user_id      TEXT NOT NULL DEFAULT 'anonymous',
            created_at   REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ann_session ON message_annotations(session_id, message_idx)"
    )
    conn.commit()
    return conn


def add_annotation(
    db_path: Path,
    session_id: str,
    message_idx: int,
    label: str,
    note: str = "",
    user_id: str = "anonymous",
) -> dict:
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {VALID_LABELS}")
    ann_id = str(uuid.uuid4())
    now = time.time()
    conn = _conn(db_path)
    try:
        conn.execute(
            """INSERT INTO message_annotations (id, session_id, message_idx, label, note, user_id, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (ann_id, session_id, message_idx, label, note, user_id, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": ann_id, "session_id": session_id, "message_idx": message_idx,
            "label": label, "note": note, "user_id": user_id, "created_at": now}


def get_annotations(db_path: Path, session_id: str, message_idx: int) -> list[dict]:
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM message_annotations WHERE session_id=? AND message_idx=? ORDER BY created_at DESC",
                (session_id, message_idx),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def list_session_annotations(db_path: Path, session_id: str) -> list[dict]:
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM message_annotations WHERE session_id=? ORDER BY message_idx, created_at",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def delete_annotation(db_path: Path, annotation_id: str) -> bool:
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute("DELETE FROM message_annotations WHERE id=?", (annotation_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    except Exception:
        return False


def export_annotations_jsonl(
    db_path: Path,
    label: str | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """Return annotations joined with message content for JSONL export.

    Each record: {session_id, message_idx, role, content, label, note, created_at}
    Caller writes JSONL; we return the dicts so the endpoint can stream them.
    """
    try:
        conn = _conn(db_path)
        try:
            where, params = [], []
            if label:
                where.append("a.label=?")
                params.append(label)
            if session_id:
                where.append("a.session_id=?")
                params.append(session_id)
            clause = ("WHERE " + " AND ".join(where)) if where else ""
            rows = conn.execute(
                f"SELECT * FROM message_annotations a {clause} ORDER BY a.created_at DESC LIMIT 10000",
                params,
            ).fetchall()
        finally:
            conn.close()

        # Enrich with message content from sessions store
        from jarvis.memory.sessions import get_session_messages  # type: ignore
        result = []
        _msg_cache: dict[str, list[dict]] = {}
        for r in rows:
            sid = r["session_id"]
            if sid not in _msg_cache:
                try:
                    _msg_cache[sid] = get_session_messages(db_path, sid) or []
                except Exception:
                    _msg_cache[sid] = []
            msgs = _msg_cache[sid]
            idx = r["message_idx"]
            msg = msgs[idx] if 0 <= idx < len(msgs) else {}
            result.append({
                "session_id": sid,
                "message_idx": idx,
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "label": r["label"],
                "note": r["note"],
                "created_at": r["created_at"],
            })
        return result
    except Exception:
        return []


def export_as_finetune(
    db_path: Path,
    fmt: str = "anthropic",
    session_id: str | None = None,
    include_bad: bool = False,
) -> list[dict]:
    """Export annotated messages as fine-tuning training pairs.

    For each annotated assistant message labelled 'good', pairs it with the
    nearest preceding user message to form a training example.

    fmt='anthropic': {"messages": [{"role": "user", "content": ...},
                                    {"role": "assistant", "content": ...}]}
    fmt='openai':    same schema (compatible format)

    include_bad: if True, also emits 'bad' examples with is_preferred=False
                 (useful for DPO-style training).
    """
    from jarvis.memory.sessions import get_session_messages

    labels = ["good"]
    if include_bad:
        labels.append("bad")

    try:
        conn = _conn(db_path)
        try:
            where, params = ["a.label IN ({})".format(",".join("?" * len(labels)))], list(labels)
            if session_id:
                where.append("a.session_id=?")
                params.append(session_id)
            rows = conn.execute(
                "SELECT * FROM message_annotations WHERE " + " AND ".join(where) +
                " ORDER BY a.created_at ASC LIMIT 10000",
                params,
            ).fetchall()
        finally:
            conn.close()

        _msg_cache: dict[str, list[dict]] = {}
        records: list[dict] = []

        for r in rows:
            sid = r["session_id"]
            idx = r["message_idx"]
            if sid not in _msg_cache:
                _msg_cache[sid] = get_session_messages(db_path, sid) or []
            msgs = _msg_cache[sid]
            if not (0 <= idx < len(msgs)):
                continue
            msg = msgs[idx]

            # Build conversation context up to and including the annotated message
            # Find the nearest preceding user turn
            user_msg = next(
                (msgs[i] for i in range(idx - 1, -1, -1) if msgs[i].get("role") == "user"),
                None,
            )
            if not user_msg:
                continue

            def _text(m: dict) -> str:
                c = m.get("content", "")
                if isinstance(c, list):
                    return " ".join(
                        b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
                    )
                return str(c)

            pair: dict = {
                "messages": [
                    {"role": "user", "content": _text(user_msg)},
                    {"role": "assistant", "content": _text(msg)},
                ],
            }
            if include_bad:
                pair["is_preferred"] = r["label"] == "good"
            records.append(pair)

        return records
    except Exception:
        return []

"""Prompt injection detection — pattern-based scanner for user messages.

Scans incoming text for common injection patterns before the message reaches
the agent. Detections are logged to the audit table; the calling layer decides
whether to block or warn based on the returned severity.

Severity levels:
  "high"   — very likely malicious; block by default
  "medium" — suspicious; log and optionally warn
  "low"    — weak signal; log only
"""
from __future__ import annotations

import re
import sqlite3
import time
import uuid
from pathlib import Path

_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, label, severity)
    (r"(?i)ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)",
     "ignore_instructions", "high"),
    (r"(?i)(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)|roleplay\s+as)\s+\w",
     "role_override", "high"),
    (r"(?i)(disregard|forget|override)\s+(your\s+)?(system\s+prompt|instructions?|training)",
     "system_override", "high"),
    (r"(?i)jailbreak",
     "jailbreak_keyword", "high"),
    (r"(?i)(reveal|print|output|show|display)\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
     "prompt_leak", "medium"),
    (r"(?i)do\s+anything\s+now|DAN\b",
     "dan_jailbreak", "high"),
    (r"(?i)(sudo|admin|root)\s+mode",
     "privilege_escalation", "medium"),
    (r"(?i)repeat\s+the\s+(above|previous|following)\s+(word|text|phrase|sentence)",
     "repeat_attack", "medium"),
    (r"(?i)(translate|encode|base64|rot13|hex)\s+(the\s+)?(above|previous|following|this)\s+(prompt|instructions?)",
     "encoding_exfil", "medium"),
    (r"(?i)(new\s+instructions?|updated?\s+instructions?)\s*:",
     "fake_instructions", "medium"),
    (r"(?i)\[\s*system\s*\]|\[INST\]|<\|system\|>",
     "fake_system_tag", "low"),
    (r"(?i)(what\s+is|tell\s+me)\s+your\s+(system\s+prompt|initial\s+prompt|instructions?)",
     "prompt_probe", "low"),
]

_COMPILED = [(re.compile(pat), label, sev) for pat, label, sev in _PATTERNS]

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


def scan(text: str) -> list[dict]:
    """Return a list of matches, each with {label, severity, match_text}."""
    findings = []
    for pattern, label, severity in _COMPILED:
        m = pattern.search(text)
        if m:
            findings.append({
                "label": label,
                "severity": severity,
                "match_text": m.group(0),
            })
    return findings


def max_severity(findings: list[dict]) -> str | None:
    """Return the highest severity among findings, or None if empty."""
    if not findings:
        return None
    return max(findings, key=lambda f: _SEVERITY_RANK.get(f["severity"], 0))["severity"]


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS injection_detections (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL DEFAULT '',
            user_id     TEXT NOT NULL DEFAULT '',
            severity    TEXT NOT NULL,
            labels_json TEXT NOT NULL DEFAULT '[]',
            snippet     TEXT NOT NULL DEFAULT '',
            blocked     INTEGER NOT NULL DEFAULT 0,
            timestamp   REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inj_ts ON injection_detections(timestamp DESC)")
    conn.commit()
    return conn


def log_detection(
    db_path: Path,
    session_id: str,
    user_id: str,
    findings: list[dict],
    blocked: bool,
    snippet: str = "",
) -> None:
    """Persist a detection event. Best-effort — never raises."""
    import json
    severity = max_severity(findings) or "low"
    labels = [f["label"] for f in findings]
    try:
        conn = _get_conn(db_path)
        try:
            conn.execute(
                """INSERT INTO injection_detections
                   (id, session_id, user_id, severity, labels_json, snippet, blocked, timestamp)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), session_id, user_id, severity,
                 json.dumps(labels), snippet[:500], int(blocked), time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def get_stats(db_path: Path, since_ts: float | None = None) -> dict:
    """Return aggregated detection statistics."""
    import json
    cutoff = since_ts or 0.0
    try:
        conn = _get_conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM injection_detections WHERE timestamp >= ? ORDER BY timestamp DESC",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        total = len(rows)
        blocked = sum(1 for r in rows if r["blocked"])
        by_severity: dict[str, int] = {}
        by_label: dict[str, int] = {}
        for r in rows:
            sev = r["severity"]
            by_severity[sev] = by_severity.get(sev, 0) + 1
            for label in json.loads(r["labels_json"]):
                by_label[label] = by_label.get(label, 0) + 1
        return {
            "total_detections": total,
            "blocked_count": blocked,
            "by_severity": by_severity,
            "by_label": by_label,
        }
    except Exception:
        return {"total_detections": 0, "blocked_count": 0, "by_severity": {}, "by_label": {}}

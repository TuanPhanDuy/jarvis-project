"""User preference store — persists cross-session behavioral patterns.

Preferences are stored in SQLite (shared jarvis.db) and loaded at session
start to give JARVIS context about who it's talking to.

Categories:
  communication_style — verbosity, tone, format preferences
  technical_depth     — beginner / intermediate / expert
  domain_interest     — topics the user cares about
  schedule            — timezone, working hours, routines
  tool_prefs          — preferred languages, frameworks, tools
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id    TEXT    NOT NULL,
            category   TEXT    NOT NULL,
            key        TEXT    NOT NULL,
            value      TEXT    NOT NULL,
            confidence REAL    NOT NULL DEFAULT 0.5,
            source     TEXT    NOT NULL DEFAULT 'inferred',
            updated_at REAL    NOT NULL,
            PRIMARY KEY (user_id, category, key)
        );
        CREATE INDEX IF NOT EXISTS idx_pref_user ON user_preferences(user_id);
        CREATE TABLE IF NOT EXISTS session_summaries (
            session_id  TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            summary     TEXT NOT NULL,
            key_topics  TEXT NOT NULL DEFAULT '[]',
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_summ_user ON session_summaries(user_id);
    """)
    conn.commit()
    return conn


def upsert_preference(
    db_path: Path,
    user_id: str,
    category: str,
    key: str,
    value: str,
    confidence: float = 0.5,
    source: str = "inferred",
) -> None:
    """Insert or update a user preference. Best-effort — never raises."""
    try:
        conn = _get_conn(db_path)
        conn.execute(
            """
            INSERT INTO user_preferences (user_id, category, key, value, confidence, source, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(user_id, category, key) DO UPDATE SET
                value      = excluded.value,
                confidence = MAX(user_preferences.confidence, excluded.confidence),
                source     = excluded.source,
                updated_at = excluded.updated_at
            """,
            (user_id, category, key, value, confidence, source, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_preferences(db_path: Path, user_id: str) -> dict[str, dict[str, str]]:
    """Return all preferences for a user as {category: {key: value}}."""
    try:
        conn = _get_conn(db_path)
        rows = conn.execute(
            "SELECT category, key, value, confidence FROM user_preferences WHERE user_id = ? ORDER BY confidence DESC",
            (user_id,),
        ).fetchall()
        conn.close()
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            result.setdefault(row["category"], {})[row["key"]] = row["value"]
        return result
    except Exception:
        return {}


def get_preference_context(db_path: Path, user_id: str) -> str:
    """Return a formatted string suitable for injection into a system prompt.

    Returns empty string if no preferences are stored yet.
    """
    prefs = get_preferences(db_path, user_id)
    if not prefs:
        return ""

    lines = ["## What I know about you"]
    category_labels = {
        "communication_style": "Communication style",
        "technical_depth": "Technical depth",
        "domain_interest": "Areas of interest",
        "schedule": "Schedule / timezone",
        "tool_prefs": "Preferred tools / languages",
    }
    for category, kv in prefs.items():
        label = category_labels.get(category, category.replace("_", " ").title())
        for key, value in kv.items():
            lines.append(f"- {label} → {key}: {value}")
    return "\n".join(lines)


def save_session_summary(
    db_path: Path,
    session_id: str,
    user_id: str,
    summary: str,
    key_topics: list[str],
) -> None:
    """Persist a session summary. Best-effort — never raises."""
    import json
    try:
        conn = _get_conn(db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO session_summaries (session_id, user_id, summary, key_topics, created_at)
            VALUES (?,?,?,?,?)
            """,
            (session_id, user_id, summary, json.dumps(key_topics), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def prune_old_preferences(db_path: Path, retention_days: int) -> int:
    """Delete preferences not updated in the last retention_days. Returns rows deleted."""
    cutoff = time.time() - retention_days * 86400
    try:
        conn = _get_conn(db_path)
        cur = conn.execute("DELETE FROM user_preferences WHERE updated_at < ?", (cutoff,))
        deleted = cur.rowcount
        cur2 = conn.execute("DELETE FROM session_summaries WHERE created_at < ?", (cutoff,))
        deleted += cur2.rowcount
        conn.commit()
        conn.close()
        return deleted
    except Exception:
        return 0


def handle_update_user_preference(tool_input: dict, db_path: Path, user_id: str) -> str:
    try:
        category = tool_input["category"]
        key = tool_input["key"]
        value = tool_input["value"]
        confidence = float(tool_input.get("confidence", 0.5))
        source = tool_input.get("source", "inferred")
        valid_categories = {"communication_style", "technical_depth", "domain_interest", "schedule", "tool_prefs"}
        if category not in valid_categories:
            return f"ERROR: unknown category '{category}'. Valid: {', '.join(sorted(valid_categories))}"
        upsert_preference(db_path, user_id, category, key, value, confidence, source)
        return f"Preference recorded: [{category}] {key} = {value}"
    except Exception as e:
        return f"ERROR: update_user_preference failed — {e}"


def handle_recall_user_preferences(tool_input: dict, db_path: Path, user_id: str) -> str:
    try:
        context = get_preference_context(db_path, user_id)
        return context if context else "No preferences stored yet for this user."
    except Exception as e:
        return f"ERROR: recall_user_preferences failed — {e}"


UPDATE_SCHEMA: dict = {
    "name": "update_user_preference",
    "description": (
        "Store or update a preference or behavioral pattern observed about the user. "
        "Call this when the user explicitly states a preference or when you infer one from their behavior. "
        "Examples: preferred programming language, verbosity preference, timezone, areas of interest."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["communication_style", "technical_depth", "domain_interest", "schedule", "tool_prefs"],
                "description": "Preference category.",
            },
            "key": {
                "type": "string",
                "description": "Short key, e.g. 'preferred_language', 'verbosity', 'timezone'.",
            },
            "value": {
                "type": "string",
                "description": "The preference value, e.g. 'Python', 'concise', 'EST'.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence 0-1. Use 1.0 for explicit statements, 0.5-0.7 for inferred.",
            },
            "source": {
                "type": "string",
                "enum": ["explicit", "inferred"],
                "description": "Whether the user stated this explicitly or you inferred it.",
            },
        },
        "required": ["category", "key", "value"],
    },
}

RECALL_SCHEMA: dict = {
    "name": "recall_user_preferences",
    "description": (
        "Retrieve all stored preferences and behavioral patterns for the current user. "
        "Call this at the start of a session to personalize your responses."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

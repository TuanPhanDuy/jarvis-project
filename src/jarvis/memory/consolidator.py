"""Memory consolidator — runs periodically to extract user preferences from episodes.

Called by the APScheduler `memory_consolidate` job. Sends recent conversation
episodes to the local model to extract preference signals, then writes them
to user_preferences via preferences.upsert_preference().
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import structlog

log = structlog.get_logger()

_EXTRACTION_PROMPT = """\
You are analyzing a conversation log to extract user preferences and behavioral patterns.

Below are recent conversation episodes. Identify any user preferences, interests, or habits mentioned explicitly or implied by their behavior.

For each preference found, output one line in this exact format:
PREFERENCE|<category>|<key>|<value>|<confidence>|<source>

Categories: communication_style, technical_depth, domain_interest, schedule, tool_prefs
Confidence: 0.9 for explicit statements, 0.6 for clear inferences, 0.4 for weak signals
Source: "explicit" if user stated it directly, "inferred" otherwise

Only output PREFERENCE lines — no other text.  If nothing useful is found, output nothing.

Conversation:
{episodes}
"""


def consolidate_user_memory(
    db_path: Path,
    user_id: str,
    model: str,
    lookback_hours: int = 24,
) -> int:
    """Extract preferences from recent episodes and save them.

    Returns the number of preferences upserted.
    """
    import ollama
    from jarvis.memory.episodic import _get_conn as ep_conn
    from jarvis.memory.preferences import upsert_preference, save_session_summary

    cutoff = time.time() - lookback_hours * 3600
    try:
        conn = ep_conn(db_path)
        rows = conn.execute(
            "SELECT session_id, role, content, timestamp FROM episodes "
            "WHERE user_id = ? AND timestamp >= ? ORDER BY timestamp ASC LIMIT 200",
            (user_id, cutoff),
        ).fetchall()
        conn.close()
    except Exception as exc:
        log.error("consolidate_fetch_failed", user_id=user_id, error=str(exc))
        return 0

    if not rows:
        return 0

    episode_text = "\n".join(
        f"[{row['role'].upper()}]: {row['content'][:300]}" for row in rows
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Extract user preferences from conversations."},
                {"role": "user", "content": _EXTRACTION_PROMPT.format(episodes=episode_text)},
            ],
            options={"temperature": 0.1},
        )
        text = response.message.content.strip()
    except Exception as exc:
        log.error("consolidate_llm_failed", user_id=user_id, error=str(exc))
        return 0

    count = 0
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 6 and parts[0] == "PREFERENCE":
            _, category, key, value, confidence_str, source = parts
            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.5
            upsert_preference(db_path, user_id, category.strip(), key.strip(), value.strip(), confidence, source.strip())
            count += 1

    if rows:
        session_ids = list({row["session_id"] for row in rows})
        summary_prompt = f"Summarize this conversation in 2-3 sentences:\n\n{episode_text[:2000]}"
        try:
            summ_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": summary_prompt}],
                options={"temperature": 0.2},
            )
            summary = summ_resp.message.content.strip()
            for sid in session_ids[:5]:
                save_session_summary(db_path, sid, user_id, summary, [])
        except Exception:
            pass

    log.info("consolidation_complete", user_id=user_id, preferences_found=count)
    return count


def get_all_user_ids(db_path: Path) -> list[str]:
    """Return distinct user_ids from the episodes table."""
    try:
        from jarvis.memory.episodic import _get_conn
        conn = _get_conn(db_path)
        rows = conn.execute(
            "SELECT DISTINCT user_id FROM episodes WHERE user_id != 'anonymous'"
        ).fetchall()
        conn.close()
        return [r["user_id"] for r in rows]
    except Exception:
        return []

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


def _cluster_by_session(rows) -> dict[str, list]:
    """Group episode rows by session_id, preserving timestamp order."""
    clusters: dict[str, list] = {}
    for row in rows:
        clusters.setdefault(row["session_id"], []).append(row)
    return clusters


def _parse_preference_lines(text: str) -> list[dict]:
    """Parse PREFERENCE|category|key|value|confidence|source lines from LLM output."""
    results = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 6 and parts[0] == "PREFERENCE":
            _, category, key, value, conf_str, source = parts
            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.5
            results.append({
                "category": category.strip(),
                "key": key.strip(),
                "value": value.strip(),
                "confidence": confidence,
                "source": source.strip(),
            })
    return results


def consolidate_user_memory(
    db_path: Path,
    user_id: str,
    model: str,
    lookback_hours: int = 24,
) -> int:
    """Extract preferences from recent episodes and save them.

    Processes episodes session-by-session to preserve context, then merges
    extracted preferences by highest confidence. Logs conflicts when two sessions
    disagree on the same key at high confidence.

    Returns the number of unique preferences upserted.
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

    clusters = _cluster_by_session(rows)

    # Merge preferences across sessions: (category, key) → best candidate
    best: dict[tuple, dict] = {}

    for session_id, session_rows in clusters.items():
        episode_text = "\n".join(
            f"[{row['role'].upper()}]: {row['content'][:300]}" for row in session_rows
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
            extracted = _parse_preference_lines(response.message.content.strip())
        except Exception as exc:
            log.error("consolidate_llm_failed", user_id=user_id, session=session_id, error=str(exc))
            continue

        for pref in extracted:
            k = (pref["category"], pref["key"])
            if k not in best:
                best[k] = pref
            else:
                existing = best[k]
                if (
                    existing["confidence"] >= 0.6
                    and pref["confidence"] >= 0.6
                    and existing["value"] != pref["value"]
                ):
                    log.warning(
                        "preference_conflict",
                        user_id=user_id,
                        key=pref["key"],
                        old_value=existing["value"],
                        new_value=pref["value"],
                    )
                if pref["confidence"] > existing["confidence"]:
                    best[k] = pref

        # Session summary
        summary_text = "\n".join(
            f"[{row['role'].upper()}]: {row['content'][:300]}" for row in session_rows
        )
        try:
            summ_resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": f"Summarize this conversation in 2-3 sentences:\n\n{summary_text[:2000]}"}],
                options={"temperature": 0.2},
            )
            save_session_summary(db_path, session_id, user_id, summ_resp.message.content.strip(), [])
        except Exception:
            pass

    for pref in best.values():
        upsert_preference(
            db_path, user_id,
            pref["category"], pref["key"], pref["value"],
            pref["confidence"], pref["source"],
        )

    count = len(best)
    log.info("consolidation_complete", user_id=user_id, preferences_found=count)
    try:
        record_consolidation_run(db_path, user_id, count, lookback_hours, started_at=cutoff + lookback_hours * 3600)
    except Exception:
        pass
    return count


def _runs_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_runs (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            preferences_found INTEGER NOT NULL DEFAULT 0,
            lookback_hours  INTEGER NOT NULL DEFAULT 24,
            status          TEXT NOT NULL DEFAULT 'done',
            error           TEXT,
            started_at      REAL NOT NULL,
            finished_at     REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cr_user ON consolidation_runs(user_id, started_at DESC)")
    conn.commit()
    return conn


def record_consolidation_run(
    db_path: Path,
    user_id: str,
    preferences_found: int,
    lookback_hours: int,
    started_at: float,
    error: str | None = None,
) -> None:
    import uuid as _uuid
    conn = _runs_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO consolidation_runs (id, user_id, preferences_found, lookback_hours, status, error, started_at, finished_at) VALUES (?,?,?,?,?,?,?,?)",
            (str(_uuid.uuid4()), user_id, preferences_found, lookback_hours,
             "failed" if error else "done", error, started_at, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def get_consolidation_history(db_path: Path, user_id: str, limit: int = 20) -> list[dict]:
    """Return past consolidation runs for a user, newest first."""
    try:
        conn = _runs_conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM consolidation_runs WHERE user_id=? ORDER BY started_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


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

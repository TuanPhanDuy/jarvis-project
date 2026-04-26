"""Dynamic personality context generator.

Returns a short string injected into the system prompt that adapts JARVIS's
tone and style based on:
  - Time of day (working hours vs. off-hours)
  - User relationship depth (first session vs. long-time user)
  - Known user preferences
"""
from __future__ import annotations

import time
from pathlib import Path


def get_personality_context(
    user_id: str | None,
    user_prefs: dict[str, dict[str, str]],
    session_count: int = 0,
) -> str:
    """Return a personality adaptation block for the system prompt."""
    lines = []

    # Time-of-day tone adjustment
    hour = _local_hour()
    if 9 <= hour < 18:
        lines.append("- Tone: professional and precise. You're in working hours mode.")
    elif 18 <= hour < 23:
        lines.append("- Tone: slightly more relaxed, but still sharp.")
    else:
        lines.append("- Tone: brief and to the point. Minimal pleasantries.")

    # Relationship depth
    if session_count == 0:
        lines.append("- This appears to be a new user. Introduce capabilities briefly if relevant.")
    elif session_count > 10:
        lines.append("- You know this user well. Skip pleasantries and get straight to it.")

    # Communication style from preferences
    style_prefs = user_prefs.get("communication_style", {})
    verbosity = style_prefs.get("verbosity", "")
    if verbosity == "concise":
        lines.append("- This user prefers concise responses. Be brief.")
    elif verbosity == "detailed":
        lines.append("- This user likes detailed explanations.")

    # Technical depth
    depth_prefs = user_prefs.get("technical_depth", {})
    level = depth_prefs.get("level", "")
    if level == "expert":
        lines.append("- The user is an expert. Skip basics, use technical terminology freely.")
    elif level == "beginner":
        lines.append("- The user is a beginner. Explain concepts clearly without jargon.")

    if not lines:
        return ""

    return "## Personality adaptation for this session\n" + "\n".join(lines)


def _local_hour() -> int:
    """Return the current local hour (0-23)."""
    import datetime
    return datetime.datetime.now().hour

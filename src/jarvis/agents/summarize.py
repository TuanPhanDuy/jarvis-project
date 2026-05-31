"""Session summarizer — condense a conversation into structured insights.

Uses a direct Ollama chat call (no agentic loop) so it is fast and cheap.
Falls back to a heuristic extraction when the model is unavailable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SessionSummary:
    summary: str
    key_topics: list[str]
    action_items: list[str]
    message_count: int


_SYSTEM_PROMPT = """\
You are a concise summarization assistant.
Given a conversation, return ONLY valid JSON with exactly these keys:
  "summary"      – 2-4 sentence overview of what was discussed
  "key_topics"   – list of 3-7 short topic strings (noun phrases)
  "action_items" – list of 0-5 actionable next steps (imperative sentences), empty list if none
Do not add any text outside the JSON object.\
"""


def _heuristic_summary(messages: list[dict]) -> SessionSummary:
    """Fallback: extract topics from user messages without a model call."""
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    combined = " ".join(str(m) for m in user_msgs)[:1000]
    words = re.findall(r"\b[A-Za-z]{4,}\b", combined)
    freq: dict[str, int] = {}
    for w in words:
        freq[w.lower()] = freq.get(w.lower(), 0) + 1
    topics = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:6]]
    summary = f"Session with {len(messages)} messages covering: {', '.join(topics[:3])}." if topics else \
              f"Session with {len(messages)} messages."
    return SessionSummary(
        summary=summary,
        key_topics=topics,
        action_items=[],
        message_count=len(messages),
    )


def summarize_session(
    messages: list[dict],
    model: str = "",
    max_context_chars: int = 6000,
) -> SessionSummary:
    """Summarize a session's message history.

    Args:
        messages:          Full message list (role/content dicts).
        model:             Ollama model name; falls back to heuristic if empty or unavailable.
        max_context_chars: Truncate conversation text to this many chars before sending.

    Returns:
        SessionSummary with summary, key_topics, action_items, message_count.
    """
    import json

    message_count = len(messages)
    if not messages:
        return SessionSummary("No messages.", [], [], 0)

    # Build a compact conversation transcript
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", ""))
        if role in ("user", "assistant"):
            lines.append(f"{role.upper()}: {content}")
    transcript = "\n".join(lines)[:max_context_chars]

    if not model:
        return _heuristic_summary(messages)

    try:
        import ollama
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this conversation:\n\n{transcript}"},
            ],
        )
        text = resp.message.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return SessionSummary(
                summary=str(data.get("summary", "")),
                key_topics=[str(t) for t in data.get("key_topics", [])],
                action_items=[str(a) for a in data.get("action_items", [])],
                message_count=message_count,
            )
    except Exception:
        pass

    return _heuristic_summary(messages)

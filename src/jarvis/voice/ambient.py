"""Ambient mode: continuous always-listening JARVIS with end-of-day summary.

Loop:
  1. Wait for wake word ("jarvis")
  2. Record + transcribe speech
  3. Run agent turn
  4. Speak response
  5. Repeat

End-of-day summary:
  Queries today's episodic memory and speaks a brief digest of what was discussed.
  Triggered by --ambient flag or by saying "summarize my day".

Usage:
    from jarvis.voice.ambient import run_ambient_mode
    run_ambient_mode(agent, settings)
"""
from __future__ import annotations

import datetime


def summarize_today(db_path, tts_fn=None) -> str:
    """Generate and optionally speak a summary of today's JARVIS activity."""
    from jarvis.memory.episodic import _get_conn

    start_ts = datetime.datetime.combine(datetime.date.today(), datetime.time.min).timestamp()
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT role, content FROM episodes WHERE timestamp >= ? ORDER BY timestamp",
        (start_ts,),
    ).fetchall()
    conn.close()

    if not rows:
        summary = "No JARVIS activity recorded today."
    else:
        user_turns = [r["content"] for r in rows if r["role"] == "user"]
        count = len(user_turns)
        preview = ". ".join(t[:80] for t in user_turns[:3])
        summary = f"Today you had {count} conversation turn{'s' if count != 1 else ''} with JARVIS. Topics included: {preview}."

    if tts_fn:
        tts_fn(summary)
    return summary


def run_ambient_mode(agent, settings) -> None:
    """Continuous wake-word loop. Press Ctrl-C to exit."""
    from jarvis.voice.wake_word import wait_for_wake_word
    from jarvis.voice.stt import record_and_transcribe
    from jarvis.voice.tts import speak

    db_path = settings.reports_dir / "jarvis.db"

    def tts(text: str) -> None:
        speak(
            text,
            engine=settings.tts_engine,
            elevenlabs_api_key=settings.elevenlabs_api_key,
            elevenlabs_voice_id=settings.elevenlabs_voice_id,
        )

    print("JARVIS ambient mode active. Say 'jarvis' to speak.")
    messages: list[dict] = []

    try:
        while True:
            wait_for_wake_word(access_key=settings.wake_word_key)
            print("Listening…")
            text = record_and_transcribe()
            if not text:
                continue
            print(f"You: {text}")

            if "summarize my day" in text.lower() or "summary" in text.lower():
                response = summarize_today(db_path, tts_fn=None)
            else:
                messages.append({"role": "user", "content": text})
                response, messages = agent.run_turn(messages)

            print(f"JARVIS: {response}\n")
            tts(response)
    except KeyboardInterrupt:
        print("\nAmbient mode stopped.")

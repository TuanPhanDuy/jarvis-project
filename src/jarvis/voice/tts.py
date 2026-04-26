"""Text-to-Speech with two engine options.

Engine A — local (pyttsx3):
    Free, offline, works immediately. Voice is robotic but instant.
    Install: pip install pyttsx3

Engine B — ElevenLabs (cloud):
    Natural, human-quality voice. Requires ELEVENLABS_API_KEY and internet.
    Install: pip install elevenlabs

The engine is chosen via JARVIS_TTS_ENGINE env var (default: "local").
If the chosen engine is unavailable, falls back to printing the text.

Usage:
    from jarvis.voice.tts import speak
    speak("Hello, I am JARVIS.")
"""
from __future__ import annotations

import re

_pyttsx3_engine = None


def _clean_text(text: str) -> str:
    """Strip markdown formatting so TTS reads clean prose."""
    text = re.sub(r"#+\s*", "", text)          # headings
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)   # bold/italic
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)         # code blocks
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text) # links
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _speak_local(text: str) -> None:
    global _pyttsx3_engine
    import pyttsx3

    if _pyttsx3_engine is None:
        _pyttsx3_engine = pyttsx3.init()
        _pyttsx3_engine.setProperty("rate", 175)   # words per minute
        _pyttsx3_engine.setProperty("volume", 0.9)
        # Try to pick a decent voice (index 1 is often better on Windows)
        voices = _pyttsx3_engine.getProperty("voices")
        if len(voices) > 1:
            _pyttsx3_engine.setProperty("voice", voices[1].id)

    _pyttsx3_engine.say(text)
    _pyttsx3_engine.runAndWait()


def _speak_elevenlabs(text: str, api_key: str, voice_id: str) -> None:
    from elevenlabs import ElevenLabs, play

    client = ElevenLabs(api_key=api_key)
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_turbo_v2_5",
        output_format="mp3_44100_128",
    )
    play(audio)


def speak(
    text: str,
    engine: str = "local",
    elevenlabs_api_key: str | None = None,
    elevenlabs_voice_id: str = "Rachel",
) -> None:
    """Convert text to speech and play it.

    Args:
        text: Text to speak. Markdown formatting is stripped automatically.
        engine: "local" (pyttsx3) or "elevenlabs".
        elevenlabs_api_key: Required when engine="elevenlabs".
        elevenlabs_voice_id: ElevenLabs voice ID or name (default "Rachel").
    """
    clean = _clean_text(text)
    if not clean:
        return

    # Truncate very long responses — speak a summary instead of 3 pages of text
    if len(clean) > 800:
        clean = clean[:800] + "... and more. Check the terminal for the full response."

    try:
        if engine == "elevenlabs" and elevenlabs_api_key:
            _speak_elevenlabs(clean, elevenlabs_api_key, elevenlabs_voice_id)
        else:
            _speak_local(clean)
    except ImportError as e:
        # Engine not installed — silent fallback (text already shown in terminal)
        pass
    except Exception:
        pass  # TTS failures are non-fatal; terminal output is always the source of truth


def is_available(engine: str = "local") -> bool:
    """Return True if the requested TTS engine is installed."""
    try:
        if engine == "elevenlabs":
            import elevenlabs  # noqa: F401
        else:
            import pyttsx3  # noqa: F401
        return True
    except ImportError:
        return False

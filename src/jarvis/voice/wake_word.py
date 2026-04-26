"""Wake-word detection for always-listening ambient mode.

Uses pvporcupine (Picovoice) when available, falls back to a keyboard trigger.
Get a free access key at: https://console.picovoice.ai/

Install (optional):
    pip install pvporcupine pyaudio

Configure:
    PICOVOICE_ACCESS_KEY=your-key-here

Usage:
    from jarvis.voice.wake_word import wait_for_wake_word
    wait_for_wake_word()   # blocks until "jarvis" is spoken
    # … then start recording …
"""
from __future__ import annotations


def wait_for_wake_word(access_key: str | None = None, keyword: str = "jarvis") -> None:
    """Block until the wake word is detected. Falls back to Enter key if unavailable."""
    if _try_porcupine(access_key, keyword):
        return
    _fallback_enter()


def _try_porcupine(access_key: str | None, keyword: str) -> bool:
    try:
        import struct
        import pvporcupine
        import pyaudio

        porcupine = pvporcupine.create(access_key=access_key or "", keywords=[keyword])
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
        )
        print(f'Listening for wake word "{keyword}"…')
        try:
            while True:
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                samples = struct.unpack_from("h" * porcupine.frame_length, pcm)
                if porcupine.process(samples) >= 0:
                    return True
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            porcupine.delete()
    except ImportError:
        return False
    except Exception as e:
        print(f"Wake word detection unavailable: {e}")
        return False


def _fallback_enter() -> None:
    try:
        input("Press Enter to speak to JARVIS… ")
    except (EOFError, KeyboardInterrupt):
        pass


def is_available() -> bool:
    try:
        import pvporcupine  # noqa: F401
        return True
    except ImportError:
        return False

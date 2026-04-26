"""Speech-to-Text via OpenAI Whisper.

Records from the default microphone until silence is detected, then transcribes
using a local Whisper model. No internet required for transcription.

Dependencies (install separately):
    pip install openai-whisper pyaudio numpy
    # On Windows PyAudio may need: pip install pipwin && pipwin install pyaudio

Usage:
    from jarvis.voice.stt import record_and_transcribe
    text = record_and_transcribe()   # blocks until speech + silence
"""
from __future__ import annotations

import io
import wave

_whisper_model = None
_MODEL_NAME = "base"  # tiny | base | small | medium | large


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(_MODEL_NAME)
    return _whisper_model


def record_and_transcribe(
    silence_threshold: int = 500,
    silence_duration_s: float = 1.5,
    sample_rate: int = 16000,
    chunk_size: int = 1024,
) -> str:
    """Record from the microphone until silence, then return the transcribed text.

    Args:
        silence_threshold: RMS amplitude below which audio is considered silence.
        silence_duration_s: Seconds of consecutive silence that triggers end of recording.
        sample_rate: Audio sample rate (Whisper expects 16 kHz).
        chunk_size: PyAudio frames per buffer read.

    Returns:
        Transcribed text string, or empty string if nothing was heard.
    """
    import numpy as np
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    frames: list[bytes] = []
    silent_chunks = 0
    silence_limit = int(silence_duration_s * sample_rate / chunk_size)
    recording_started = False

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            amplitude = np.frombuffer(data, dtype=np.int16).max()

            if amplitude > silence_threshold:
                recording_started = True
                silent_chunks = 0
                frames.append(data)
            elif recording_started:
                frames.append(data)
                silent_chunks += 1
                if silent_chunks >= silence_limit:
                    break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    if not frames:
        return ""

    # Write recorded frames to an in-memory WAV buffer for Whisper
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    buf.seek(0)

    # Whisper expects a numpy float32 array
    audio_bytes = buf.read()
    audio_np = np.frombuffer(audio_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0

    model = _get_model()
    result = model.transcribe(audio_np, language="en", fp16=False)
    return result.get("text", "").strip()


def is_available() -> bool:
    """Return True if whisper and pyaudio are installed."""
    try:
        import whisper  # noqa: F401
        import pyaudio  # noqa: F401
        return True
    except ImportError:
        return False

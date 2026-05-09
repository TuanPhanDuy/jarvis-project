"""Plugin: summarize_youtube — extract and summarise YouTube transcripts locally."""
from __future__ import annotations

import re


_STYLE_PROMPTS = {
    "brief": "Summarize this YouTube video transcript in 3-5 sentences. Be concise.",
    "detailed": (
        "Write a detailed summary of this YouTube video transcript. "
        "Cover the main topics, key points, and any conclusions or recommendations. "
        "Use bullet points for key takeaways."
    ),
    "bullets": (
        "Extract the key points from this YouTube video transcript as a bullet list. "
        "Each bullet should be one clear, standalone insight or fact."
    ),
}


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def handle(tool_input: dict) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
        import ollama
        from jarvis.config import get_settings

        url = str(tool_input.get("url", "")).strip()
        style = str(tool_input.get("style", "brief")).lower().strip()

        if not url:
            return "ERROR: 'url' is required"
        if style not in _STYLE_PROMPTS:
            return f"ERROR: unknown style '{style}'. Use: brief, detailed, bullets"

        video_id = _extract_video_id(url)
        if not video_id:
            return f"ERROR: could not extract video ID from URL — {url}"

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except TranscriptsDisabled:
            return f"ERROR: transcripts are disabled for video {video_id}"
        except NoTranscriptFound:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id).find_generated_transcript(["en"]).fetch()
            except Exception:
                return f"ERROR: no transcript available for video {video_id}"

        full_text = " ".join(entry["text"] for entry in transcript_list)
        truncated = full_text[:6000]

        prompt = f"{_STYLE_PROMPTS[style]}\n\nTranscript:\n{truncated}"
        model = get_settings().model

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        summary = response.message.content.strip()

        return f"**YouTube Summary** (video: {video_id}, style: {style})\n\n{summary}"

    except ImportError:
        return "ERROR: youtube-transcript-api not installed. Run: uv add youtube-transcript-api"
    except Exception as e:
        return f"ERROR: summarize_youtube failed — {e}"


SCHEMA: dict = {
    "name": "summarize_youtube",
    "description": (
        "Extract and summarise a YouTube video's transcript locally — no API key needed. "
        "Styles: 'brief' (3-5 sentences), 'detailed' (full breakdown with bullets), "
        "'bullets' (key points as a bullet list)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "YouTube video URL (any format: watch?v=, youtu.be/, shorts/).",
            },
            "style": {
                "type": "string",
                "enum": ["brief", "detailed", "bullets"],
                "description": "Summary style. Default: 'brief'.",
            },
        },
        "required": ["url"],
    },
}

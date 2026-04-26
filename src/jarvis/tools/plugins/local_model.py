"""Plugin: ask_local_model — query a local Ollama model for privacy-sensitive tasks.

Use this when the task involves confidential data that must not leave the machine.
Requires Ollama running locally: https://ollama.com

Configure via:
  OLLAMA_BASE_URL  (default: http://localhost:11434)
  OLLAMA_MODEL     (default: llama3.2)
"""
from __future__ import annotations

import json
import urllib.request


def handle(tool_input: dict) -> str:
    try:
        prompt = tool_input.get("prompt", "").strip()
        if not prompt:
            return "ERROR: prompt is required."

        from jarvis.config import get_settings
        settings = get_settings()

        base_url = settings.ollama_base_url.rstrip("/")
        model = settings.ollama_model

        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            response = data.get("response", "").strip()
            if not response:
                return f"ERROR: Ollama returned empty response for model '{model}'."
            return f"[{model}] {response}"

    except urllib.error.URLError as e:
        return (
            f"ERROR: Could not reach Ollama at {settings.ollama_base_url}. "
            f"Is Ollama running? Start with: ollama serve — {e}"
        )
    except Exception as e:
        return f"ERROR: ask_local_model failed — {e}"


SCHEMA: dict = {
    "name": "ask_local_model",
    "description": (
        "Query a local Ollama language model. Use this for privacy-sensitive tasks "
        "where data must not be sent to external APIs. Requires Ollama running locally."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The prompt to send to the local model.",
            }
        },
        "required": ["prompt"],
    },
}

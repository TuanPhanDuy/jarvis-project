"""Example plugin: get_weather — demonstrates the JARVIS plugin interface.

To create your own plugin:
  1. Copy this file into src/jarvis/tools/plugins/
  2. Replace SCHEMA and handle() with your tool's logic
  3. Restart JARVIS — it will be auto-discovered

Rules (same as all JARVIS tools):
  - handle() must never raise exceptions; return "ERROR: ..." strings instead
  - Use get_settings() for any config (API keys, paths, etc.)
  - SCHEMA["name"] must be unique across all tools
"""
from __future__ import annotations


def handle(tool_input: dict) -> str:
    location = tool_input.get("location", "").strip()
    if not location:
        return "ERROR: location is required."
    # Replace this stub with a real weather API call, e.g. Open-Meteo or OpenWeatherMap
    return (
        f"[Example plugin] Weather for '{location}' is not implemented yet. "
        "Replace this plugin with a real weather API call."
    )


SCHEMA: dict = {
    "name": "get_weather",
    "description": (
        "Example plugin — look up current weather for a location. "
        "Replace the implementation with a real weather service."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location, e.g. 'London' or 'New York, NY'.",
            }
        },
        "required": ["location"],
    },
}

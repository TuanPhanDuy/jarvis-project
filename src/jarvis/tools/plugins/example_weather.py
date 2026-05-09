"""Weather plugin using the free Open-Meteo API (no key required)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

_WMO: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Heavy showers",
    85: "Light snow showers",
    86: "Snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def handle(tool_input: dict) -> str:
    location = tool_input.get("location", "").strip()
    if not location:
        return "ERROR: location is required."
    try:
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode(
                {"name": location, "count": 1, "language": "en", "format": "json"}
            )
        )
        geo = _fetch(geo_url)
        results = geo.get("results") or []
        if not results:
            return f"ERROR: City '{location}' not found."
        r = results[0]
        lat, lon = r["latitude"], r["longitude"]

        wx_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                    "wind_speed_unit": "kmh",
                    "temperature_unit": "celsius",
                }
            )
        )
        wx = _fetch(wx_url)
        cw = wx["current_weather"]
        code = int(cw["weathercode"])
        description = _WMO.get(code, f"Unknown (code {code})")
        return f"{r['name']}: {cw['temperature']}°C, {description}, wind {cw['windspeed']} km/h"
    except Exception as e:
        return f"ERROR: {e}"


SCHEMA: dict = {
    "name": "get_weather",
    "description": "Get current weather for a city using the free Open-Meteo API (no API key needed).",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'London' or 'New York'.",
            }
        },
        "required": ["location"],
    },
}

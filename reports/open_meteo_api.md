# Open-Meteo API Research

## Overview

Open-Meteo is a free, open-source weather API that requires no API key. It provides current and forecast weather data using WMO weather codes.

## Endpoints

### 1. Geocoding — city name → lat/lon

```
GET https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json
```

**Response structure:**
```json
{
  "results": [
    {
      "name": "London",
      "latitude": 51.5085,
      "longitude": -0.1257,
      "country": "United Kingdom",
      "timezone": "Europe/London"
    }
  ]
}
```

If `results` is absent or empty, the city was not found.

### 2. Current weather — lat/lon → weather data

```
GET https://api.open-meteo.com/v1/forecast
    ?latitude={lat}
    &longitude={lon}
    &current_weather=true
    &wind_speed_unit=kmh
    &temperature_unit=celsius
```

**Response structure:**
```json
{
  "current_weather": {
    "temperature": 12.3,
    "windspeed": 18.5,
    "weathercode": 3,
    "time": "2024-01-15T14:00"
  }
}
```

## WMO Weather Code → Description Mapping

| Code(s) | Description |
|---------|-------------|
| 0 | Clear sky |
| 1 | Mainly clear |
| 2 | Partly cloudy |
| 3 | Overcast |
| 45, 48 | Fog |
| 51, 53, 55 | Drizzle |
| 61, 63, 65 | Rain |
| 66, 67 | Freezing rain |
| 71, 73, 75 | Snow |
| 77 | Snow grains |
| 80, 81, 82 | Rain showers |
| 85, 86 | Snow showers |
| 95 | Thunderstorm |
| 96, 99 | Thunderstorm with hail |

## Python Implementation (stdlib only — no `requests`)

```python
import json
import urllib.parse
import urllib.request

def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())

def get_weather(city: str) -> str:
    # Step 1: geocode
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urllib.parse.urlencode({"name": city, "count": 1, "language": "en", "format": "json"})
    )
    geo = _fetch(geo_url)
    results = geo.get("results") or []
    if not results:
        return f"ERROR: City '{city}' not found."
    r = results[0]
    lat, lon = r["latitude"], r["longitude"]

    # Step 2: fetch weather
    wx_url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current_weather": "true",
            "wind_speed_unit": "kmh",
            "temperature_unit": "celsius",
        })
    )
    wx = _fetch(wx_url)
    cw = wx["current_weather"]
    code = int(cw["weathercode"])
    description = WMO_DESCRIPTIONS.get(code, f"Unknown (code {code})")
    return f"{r['name']}: {cw['temperature']}°C, {description}, wind {cw['windspeed']} km/h"
```

## Notes

- No authentication required — completely free
- Rate limiting: 10 000 requests/day per IP (generous for personal use)
- `timeout=10` prevents hanging on slow responses
- Both endpoints return JSON; parse errors are caught by caller

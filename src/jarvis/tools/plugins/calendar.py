"""Plugin: read_calendar — read upcoming events from .ics files.

Looks for .ics files in:
  1. JARVIS_CALENDAR_PATH env var (if set)
  2. ~/Calendar/
  3. Current directory

Requires: pip install icalendar (optional — gracefully degrades if missing)
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path


def _find_ics_files() -> list[Path]:
    """Locate .ics calendar files from known locations."""
    search_dirs = []

    calendar_path = os.environ.get("JARVIS_CALENDAR_PATH")
    if calendar_path:
        search_dirs.append(Path(calendar_path))

    home_cal = Path.home() / "Calendar"
    if home_cal.exists():
        search_dirs.append(home_cal)

    search_dirs.append(Path.cwd())

    files = []
    seen: set[str] = set()
    for d in search_dirs:
        if d.is_file() and d.suffix == ".ics":
            if str(d) not in seen:
                seen.add(str(d))
                files.append(d)
        elif d.is_dir():
            for f in d.glob("*.ics"):
                if str(f) not in seen:
                    seen.add(str(f))
                    files.append(f)
    return files[:10]


def _parse_ics(path: Path, days_ahead: int) -> list[dict]:
    """Parse an .ics file and return events within days_ahead days."""
    from icalendar import Calendar  # type: ignore

    now = datetime.now(tz=timezone.utc)
    cutoff = datetime(
        now.year, now.month, now.day, tzinfo=timezone.utc
    )
    future = datetime(
        (now.date() + __import__("datetime").timedelta(days=days_ahead)).timetuple()[:3]
        + (23, 59, 59),
        tzinfo=timezone.utc,
    )
    # Simpler: just compute future as a naive date comparison
    cutoff_date = now.date()

    events = []
    cal = Calendar.from_ical(path.read_bytes())
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        summary = str(component.get("SUMMARY", "Untitled"))
        dtstart = component.get("DTSTART")
        if dtstart is None:
            continue
        dt = dtstart.dt
        if isinstance(dt, datetime):
            event_date = dt.date() if hasattr(dt, "date") else dt
        elif isinstance(dt, date):
            event_date = dt
        else:
            continue

        if cutoff_date <= event_date <= cutoff_date + __import__("datetime").timedelta(days=days_ahead):
            events.append({
                "date": str(event_date),
                "summary": summary,
                "location": str(component.get("LOCATION", "")),
                "description": str(component.get("DESCRIPTION", ""))[:200],
            })

    events.sort(key=lambda e: e["date"])
    return events


def handle(tool_input: dict) -> str:
    days_ahead = int(tool_input.get("days_ahead", 7))
    ics_files = _find_ics_files()

    if not ics_files:
        return (
            "No calendar files found. Set JARVIS_CALENDAR_PATH to your .ics file "
            "or place .ics files in ~/Calendar/."
        )

    try:
        import icalendar  # noqa: F401
    except ImportError:
        return (
            "ERROR: icalendar package not installed. "
            "Run: pip install icalendar"
        )

    all_events: list[dict] = []
    for f in ics_files:
        try:
            all_events.extend(_parse_ics(f, days_ahead))
        except Exception as e:
            all_events.append({"date": "?", "summary": f"ERROR reading {f.name}: {e}", "location": "", "description": ""})

    if not all_events:
        return f"No events found in the next {days_ahead} days."

    lines = [f"Upcoming events (next {days_ahead} days):\n"]
    for ev in all_events[:25]:
        loc = f" @ {ev['location']}" if ev["location"] else ""
        lines.append(f"• {ev['date']}: {ev['summary']}{loc}")
        if ev["description"]:
            lines.append(f"  {ev['description']}")
    return "\n".join(lines)


SCHEMA: dict = {
    "name": "read_calendar",
    "description": (
        "Read upcoming calendar events from .ics files. "
        "Returns events within the specified number of days."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "days_ahead": {
                "type": "integer",
                "description": "Number of days ahead to look for events (default 7).",
                "default": 7,
            },
        },
        "required": [],
    },
}

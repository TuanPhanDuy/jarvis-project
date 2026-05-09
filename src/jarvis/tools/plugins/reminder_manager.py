"""Plugin: manage_reminder — set, cancel, and list reminders via APScheduler."""
from __future__ import annotations

import json
import uuid
from datetime import datetime


def handle(tool_input: dict) -> str:
    try:
        action = str(tool_input.get("action", "list")).lower().strip()

        if action == "set":
            return _set_reminder(tool_input)
        elif action == "cancel":
            return _cancel_reminder(tool_input)
        elif action == "list":
            return _list_reminders()
        else:
            return f"ERROR: unknown action '{action}'. Use: set, cancel, list"

    except Exception as e:
        return f"ERROR: manage_reminder failed — {e}"


def _get_scheduler():
    from jarvis.scheduler.core import get_scheduler
    sched = get_scheduler()
    if sched is None:
        raise RuntimeError("Scheduler is not running. Start the JARVIS API server first.")
    return sched


def _set_reminder(tool_input: dict) -> str:
    title = str(tool_input.get("title", "")).strip()
    remind_at_str = str(tool_input.get("remind_at", "")).strip()

    if not title:
        return "ERROR: 'title' is required for action=set"
    if not remind_at_str:
        return "ERROR: 'remind_at' (ISO 8601 datetime) is required for action=set"

    try:
        remind_at = datetime.fromisoformat(remind_at_str)
    except ValueError:
        return f"ERROR: invalid datetime format '{remind_at_str}'. Use ISO 8601 (e.g. 2025-06-01T09:00:00)"

    sched = _get_scheduler()
    reminder_id = f"reminder_{uuid.uuid4().hex[:8]}"

    sched.add_job(
        _fire_reminder,
        trigger="date",
        run_date=remind_at,
        id=reminder_id,
        kwargs={"reminder_id": reminder_id, "title": title},
        replace_existing=False,
    )

    return (
        f"Reminder set.\n"
        f"ID: {reminder_id}\n"
        f"Title: {title}\n"
        f"Fires at: {remind_at.isoformat()}"
    )


def _cancel_reminder(tool_input: dict) -> str:
    reminder_id = str(tool_input.get("reminder_id", "")).strip()
    if not reminder_id:
        return "ERROR: 'reminder_id' is required for action=cancel"

    sched = _get_scheduler()
    try:
        sched.remove_job(reminder_id)
        return f"Reminder '{reminder_id}' cancelled."
    except Exception:
        return f"ERROR: reminder '{reminder_id}' not found."


def _list_reminders() -> str:
    sched = _get_scheduler()
    jobs = [j for j in sched.get_jobs() if j.id.startswith("reminder_")]
    if not jobs:
        return "No pending reminders."

    lines = ["**Pending Reminders:**"]
    for job in jobs:
        next_run = job.next_run_time.isoformat() if job.next_run_time else "n/a"
        title = (job.kwargs or {}).get("title", job.id)
        lines.append(f"- [{job.id}] \"{title}\" → {next_run}")
    return "\n".join(lines)


def _fire_reminder(reminder_id: str, title: str) -> None:
    """Called by APScheduler when the reminder fires. Pushes a WsProactive notification."""
    try:
        from jarvis.api.server import _active_websockets
        import asyncio

        msg = {
            "type": "proactive",
            "severity": "info",
            "message": f"Reminder: {title}",
            "source": "reminder",
        }
        for ws in list(_active_websockets.values()):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        ws.send_json(msg),
                    )
            except Exception:
                pass
    except Exception:
        pass


SCHEMA: dict = {
    "name": "manage_reminder",
    "description": (
        "Set, cancel, or list reminders. Reminders fire at a specific datetime and "
        "push a notification to any connected dashboard. "
        "Actions: 'set' (requires title + remind_at), 'cancel' (requires reminder_id), 'list'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "cancel", "list"],
                "description": "What to do. Default: 'list'.",
            },
            "title": {
                "type": "string",
                "description": "Reminder text shown when it fires (required for 'set').",
            },
            "remind_at": {
                "type": "string",
                "description": "ISO 8601 datetime when the reminder should fire, e.g. '2025-06-01T09:00:00' (required for 'set').",
            },
            "reminder_id": {
                "type": "string",
                "description": "ID returned by a previous 'set' call (required for 'cancel').",
            },
        },
        "required": ["action"],
    },
}

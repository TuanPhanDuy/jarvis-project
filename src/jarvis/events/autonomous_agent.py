"""AutonomousDecisionAgent — handles events from the bus and pushes proactive messages.

When a JarvisEvent arrives, this module composes a context message, runs a
single PlannerAgent turn, and pushes the result to all connected WebSocket
clients as a WsProactive message.

The agent turn is run in a ThreadPoolExecutor (same pattern as the WS handler)
so it does not block the asyncio event loop.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import structlog

from jarvis.events.types import JarvisEvent, SystemEvent, UserEvent, ExternalEvent

log = structlog.get_logger()

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-autonomous")


def _compose_prompt(event: JarvisEvent) -> str | None:
    """Build the prompt for the agent turn. Returns None to skip the event."""
    if isinstance(event, SystemEvent):
        if event.metric == "cpu":
            return (
                f"SYSTEM ALERT: CPU usage is at {event.value:.0f}% (threshold {event.threshold:.0f}%). "
                "Briefly assess what this might mean and whether any action is warranted. "
                "Keep the response under 3 sentences."
            )
        if event.metric == "memory":
            return (
                f"SYSTEM ALERT: Memory usage is at {event.value:.0f}% (threshold {event.threshold:.0f}%). "
                "Briefly note this and suggest whether the user should be concerned."
            )
        if event.metric == "disk":
            return (
                f"SYSTEM ALERT: Disk free space is at {event.value:.0f}% (threshold {event.threshold:.0f}% free). "
                "Advise the user on this condition briefly."
            )

    if isinstance(event, UserEvent) and event.sub_type == "long_idle":
        idle_min = event.data.get("idle_seconds", 0) // 60
        return (
            f"The user has been idle for {idle_min} minutes. "
            "Search episodic memory for any unfinished tasks or items they mentioned earlier. "
            "If anything is worth following up on, summarize it briefly — otherwise skip this."
        )

    if isinstance(event, ExternalEvent) and event.sub_type == "new_report":
        fname = event.payload.get("filename", "unknown")
        return (
            f"A new research report was saved: '{fname}'. "
            "Look at it briefly and tell the user what it covers in one sentence."
        )

    return None


async def handle_event(
    event: JarvisEvent,
    active_websockets: dict,
    build_agent_fn,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Handle a bus event: run agent turn and push WsProactive to connected clients."""
    from jarvis.api.models import WsProactive

    if not active_websockets:
        return

    prompt = _compose_prompt(event)
    if not prompt:
        return

    severity = "info"
    if isinstance(event, SystemEvent) and event.severity == "alert":
        severity = "alert"
    elif isinstance(event, SystemEvent):
        severity = "warning"

    result_holder: list = []

    def run() -> None:
        try:
            agent = build_agent_fn()
            messages = [{"role": "user", "content": prompt}]
            text, _ = agent.run_turn(messages)
            result_holder.append(text)
        except Exception as exc:
            result_holder.append(f"[JARVIS autonomous response failed: {exc}]")

    await asyncio.get_event_loop().run_in_executor(_executor, run)

    if not result_holder:
        return

    text = result_holder[0]
    msg = WsProactive(trigger=event.event_type, text=text, severity=severity).model_dump()

    for session_id, ws in list(active_websockets.items()):
        try:
            await ws.send_json(msg)
            log.info("proactive_push_sent", session_id=session_id, trigger=event.event_type)
        except Exception:
            pass

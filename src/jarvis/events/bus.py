"""Async pub/sub event bus for JARVIS proactive behavior.

Handlers are registered per event_type and called when a matching event
is published.  The dispatch loop runs as an asyncio background task.

Usage:
    bus = get_event_bus()
    bus.subscribe("system_alert", my_async_handler)
    await bus.publish(SystemEvent(...))
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from jarvis.events.types import JarvisEvent

log = structlog.get_logger()

Handler = Callable[[JarvisEvent], Awaitable[None]]

_bus: "EventBus | None" = None


class EventBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[JarvisEvent | None] = asyncio.Queue()
        self._handlers: dict[str, list[Handler]] = {}

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: JarvisEvent) -> None:
        await self._queue.put(event)

    def publish_sync(self, event: JarvisEvent, loop: asyncio.AbstractEventLoop) -> None:
        """Thread-safe publish from synchronous code."""
        loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def _dispatch_loop(self) -> None:
        """Drain the queue and call registered handlers. Runs until shutdown."""
        log.info("event_bus_started")
        while True:
            event = await self._queue.get()
            if event is None:
                log.info("event_bus_stopped")
                break
            handlers = self._handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as exc:
                    log.error("event_handler_error", event_type=event.event_type, error=str(exc))

    async def shutdown(self) -> None:
        await self._queue.put(None)


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus

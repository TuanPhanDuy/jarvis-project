"""Tests for event bus SSE stream and bus fan-out."""
from __future__ import annotations

import asyncio

import pytest

from jarvis.events.bus import EventBus
from jarvis.events.types import JarvisEvent, SystemEvent, UserEvent


class TestEventBusFanOut:
    def _make_bus(self) -> EventBus:
        return EventBus()

    @pytest.mark.asyncio
    async def test_sse_listener_receives_published_event(self):
        bus = self._make_bus()
        q = bus.add_sse_listener()
        event = JarvisEvent(event_type="test_event")
        await bus.publish(event)
        received = q.get_nowait()
        assert received.event_type == "test_event"

    @pytest.mark.asyncio
    async def test_multiple_listeners_all_receive(self):
        bus = self._make_bus()
        q1 = bus.add_sse_listener()
        q2 = bus.add_sse_listener()
        await bus.publish(JarvisEvent(event_type="broadcast"))
        assert q1.get_nowait().event_type == "broadcast"
        assert q2.get_nowait().event_type == "broadcast"

    @pytest.mark.asyncio
    async def test_removed_listener_does_not_receive(self):
        bus = self._make_bus()
        q = bus.add_sse_listener()
        bus.remove_sse_listener(q)
        await bus.publish(JarvisEvent(event_type="missed"))
        assert q.empty()

    @pytest.mark.asyncio
    async def test_full_queue_does_not_block_publish(self):
        bus = self._make_bus()
        q = bus.add_sse_listener()
        # Fill the queue to max (100) + 1
        for _ in range(101):
            await bus.publish(JarvisEvent(event_type="flood"))
        # Should not raise — drops events for slow consumers

    @pytest.mark.asyncio
    async def test_system_event_forwarded(self):
        bus = self._make_bus()
        q = bus.add_sse_listener()
        event = SystemEvent(metric="cpu", value=95.0, threshold=80.0)
        await bus.publish(event)
        received = q.get_nowait()
        assert received.event_type == "system_alert"

    @pytest.mark.asyncio
    async def test_to_dict_has_required_fields(self):
        event = SystemEvent(metric="disk", value=90.0, threshold=85.0, severity="warning")
        d = event.to_dict()
        assert "event_id" in d
        assert "timestamp" in d
        assert "event_type" in d
        assert d["metric"] == "disk"

    @pytest.mark.asyncio
    async def test_user_event_to_dict(self):
        event = UserEvent(user_id="alice", session_id="s1", sub_type="long_idle")
        d = event.to_dict()
        assert d["user_id"] == "alice"
        assert d["sub_type"] == "long_idle"

    def test_add_listener_returns_queue(self):
        bus = self._make_bus()
        q = bus.add_sse_listener()
        assert isinstance(q, asyncio.Queue)

    def test_remove_nonexistent_listener_no_error(self):
        bus = self._make_bus()
        bus.remove_sse_listener(asyncio.Queue())  # should not raise

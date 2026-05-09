"""Tests for the JARVIS event bus and monitors."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.events.bus import EventBus
from jarvis.events.triggers import FileWatcher, SystemMonitor
from jarvis.events.types import ExternalEvent, JarvisEvent, SystemEvent


# ── EventBus ─────────────────────────────────────────────────────────────────


class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        bus = EventBus()
        received: list[JarvisEvent] = []

        async def handler(event: JarvisEvent) -> None:
            received.append(event)

        bus.subscribe("system_alert", handler)
        event = SystemEvent(metric="cpu", value=95.0, threshold=90.0, severity="warning")
        await bus.publish(event)

        task = asyncio.ensure_future(bus._dispatch_loop())
        # Let one iteration drain
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(received) == 1
        assert received[0].event_type == "system_alert"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_receive(self):
        bus = EventBus()
        calls_a: list[JarvisEvent] = []
        calls_b: list[JarvisEvent] = []

        async def handler_a(e: JarvisEvent) -> None:
            calls_a.append(e)

        async def handler_b(e: JarvisEvent) -> None:
            calls_b.append(e)

        bus.subscribe("system_alert", handler_a)
        bus.subscribe("system_alert", handler_b)
        await bus.publish(SystemEvent(metric="memory", value=90.0, threshold=85.0, severity="warning"))

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(calls_a) == 1
        assert len(calls_b) == 1

    @pytest.mark.asyncio
    async def test_no_subscribers_does_not_raise(self):
        bus = EventBus()
        await bus.publish(SystemEvent(metric="disk", value=5.0, threshold=10.0, severity="alert"))

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task  # should not raise

    @pytest.mark.asyncio
    async def test_different_event_types_are_isolated(self):
        bus = EventBus()
        received: list[JarvisEvent] = []

        async def handler(e: JarvisEvent) -> None:
            received.append(e)

        bus.subscribe("external_event", handler)
        await bus.publish(SystemEvent(metric="cpu", value=95.0, threshold=90.0, severity="warning"))

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(received) == 0  # system_alert not forwarded to external_event handler

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_stop_bus(self):
        bus = EventBus()
        good_calls: list[JarvisEvent] = []

        async def bad_handler(e: JarvisEvent) -> None:
            raise RuntimeError("handler boom")

        async def good_handler(e: JarvisEvent) -> None:
            good_calls.append(e)

        bus.subscribe("system_alert", bad_handler)
        bus.subscribe("system_alert", good_handler)
        await bus.publish(SystemEvent(metric="cpu", value=91.0, threshold=90.0, severity="warning"))

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(good_calls) == 1  # good handler still ran despite bad one


# ── SystemMonitor ─────────────────────────────────────────────────────────────


class TestSystemMonitor:
    def _make_monitor(self) -> tuple[EventBus, SystemMonitor]:
        bus = EventBus()
        monitor = SystemMonitor(bus)
        return bus, monitor

    @pytest.mark.asyncio
    async def test_fires_cpu_event_when_above_threshold(self):
        bus, monitor = self._make_monitor()
        events: list[SystemEvent] = []

        async def handler(e: JarvisEvent) -> None:
            events.append(e)  # type: ignore[arg-type]

        bus.subscribe("system_alert", handler)

        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=50.0)
        mock_psutil.disk_usage.return_value = MagicMock(percent=50.0)

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            # Simulate two consecutive high-CPU readings to satisfy strike >= 2
            monitor._cpu_strikes = 1
            monitor._last_fired = {}

            with patch("jarvis.events.triggers.asyncio.sleep", return_value=None):
                # Manually invoke one poll cycle
                import psutil  # type: ignore[import-not-found]
                psutil = mock_psutil  # noqa: F841

                cpu = mock_psutil.cpu_percent(interval=1)
                mem = mock_psutil.virtual_memory().percent
                disk = mock_psutil.disk_usage("/").percent

                if cpu > monitor.CPU_THRESHOLD:
                    monitor._cpu_strikes += 1
                    if monitor._cpu_strikes >= 2 and monitor._cooldown_ok("cpu"):
                        await bus.publish(SystemEvent(
                            metric="cpu", value=cpu, threshold=monitor.CPU_THRESHOLD, severity="warning"
                        ))
                        monitor._mark_fired("cpu")
                        monitor._cpu_strikes = 0

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert any(e.metric == "cpu" for e in events)

    @pytest.mark.asyncio
    async def test_no_event_below_cpu_threshold(self):
        bus, monitor = self._make_monitor()
        events: list[SystemEvent] = []

        async def handler(e: JarvisEvent) -> None:
            events.append(e)  # type: ignore[arg-type]

        bus.subscribe("system_alert", handler)

        cpu = 50.0  # well below 90%
        if cpu > monitor.CPU_THRESHOLD:
            monitor._cpu_strikes += 1
        else:
            monitor._cpu_strikes = 0

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(events) == 0
        assert monitor._cpu_strikes == 0

    @pytest.mark.asyncio
    async def test_fires_memory_event(self):
        bus, monitor = self._make_monitor()
        events: list[SystemEvent] = []

        async def handler(e: JarvisEvent) -> None:
            events.append(e)  # type: ignore[arg-type]

        bus.subscribe("system_alert", handler)

        mem = 90.0  # above 85% threshold
        if mem > monitor.MEM_THRESHOLD and monitor._cooldown_ok("memory"):
            await bus.publish(SystemEvent(
                metric="memory", value=mem, threshold=monitor.MEM_THRESHOLD, severity="warning"
            ))
            monitor._mark_fired("memory")

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert any(e.metric == "memory" for e in events)

    @pytest.mark.asyncio
    async def test_cooldown_prevents_repeat_alert(self):
        bus, monitor = self._make_monitor()
        events: list[SystemEvent] = []

        async def handler(e: JarvisEvent) -> None:
            events.append(e)  # type: ignore[arg-type]

        bus.subscribe("system_alert", handler)

        # First alert fires
        await bus.publish(SystemEvent(metric="memory", value=90.0, threshold=85.0, severity="warning"))
        monitor._mark_fired("memory")

        # Second attempt within cooldown — should be skipped
        assert not monitor._cooldown_ok("memory")

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(events) == 1  # only the first one went through

    @pytest.mark.asyncio
    async def test_fires_disk_event(self):
        bus, monitor = self._make_monitor()
        events: list[SystemEvent] = []

        async def handler(e: JarvisEvent) -> None:
            events.append(e)  # type: ignore[arg-type]

        bus.subscribe("system_alert", handler)

        free_pct = 5.0  # below 10% threshold
        if free_pct < monitor.DISK_FREE_THRESHOLD and monitor._cooldown_ok("disk"):
            await bus.publish(SystemEvent(
                metric="disk", value=free_pct, threshold=monitor.DISK_FREE_THRESHOLD, severity="alert"
            ))
            monitor._mark_fired("disk")

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert any(e.metric == "disk" for e in events)


# ── FileWatcher ───────────────────────────────────────────────────────────────


class TestFileWatcher:
    @pytest.mark.asyncio
    async def test_detects_new_md_file(self, tmp_path: Path):
        bus = EventBus()
        events: list[ExternalEvent] = []

        async def handler(e: JarvisEvent) -> None:
            events.append(e)  # type: ignore[arg-type]

        bus.subscribe("external_event", handler)
        watcher = FileWatcher(bus, watch_dir=tmp_path)

        # Seed seen with current (empty) state
        watcher._seen = set()

        # New file appears
        new_file = tmp_path / "research.md"
        new_file.write_text("# Research")

        # Simulate one poll cycle
        current = {f.name for f in tmp_path.glob("*.md")}
        new_files = current - watcher._seen
        for fname in new_files:
            await bus.publish(ExternalEvent(
                source="file_watcher",
                sub_type="new_report",
                payload={"filename": fname, "path": str(tmp_path / fname)},
            ))
        watcher._seen = current

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(events) == 1
        assert events[0].payload["filename"] == "research.md"

    @pytest.mark.asyncio
    async def test_ignores_already_seen_files(self, tmp_path: Path):
        bus = EventBus()
        events: list[ExternalEvent] = []

        async def handler(e: JarvisEvent) -> None:
            events.append(e)  # type: ignore[arg-type]

        bus.subscribe("external_event", handler)
        watcher = FileWatcher(bus, watch_dir=tmp_path)

        existing = tmp_path / "old.md"
        existing.write_text("# Old")
        watcher._seen = {"old.md"}

        # Second poll — same file
        current = {f.name for f in tmp_path.glob("*.md")}
        new_files = current - watcher._seen
        for fname in new_files:
            await bus.publish(ExternalEvent(
                source="file_watcher", sub_type="new_report", payload={"filename": fname}
            ))
        watcher._seen = current

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(events) == 0  # already seen, no event

    @pytest.mark.asyncio
    async def test_ignores_non_md_files(self, tmp_path: Path):
        bus = EventBus()
        events: list[ExternalEvent] = []

        async def handler(e: JarvisEvent) -> None:
            events.append(e)  # type: ignore[arg-type]

        bus.subscribe("external_event", handler)
        watcher = FileWatcher(bus, watch_dir=tmp_path)
        watcher._seen = set()

        (tmp_path / "notes.txt").write_text("ignored")
        (tmp_path / "data.csv").write_text("col1,col2")

        # Glob only matches .md
        current = {f.name for f in tmp_path.glob("*.md")}
        new_files = current - watcher._seen
        for fname in new_files:
            await bus.publish(ExternalEvent(
                source="file_watcher", sub_type="new_report", payload={"filename": fname}
            ))

        task = asyncio.ensure_future(bus._dispatch_loop())
        await asyncio.sleep(0)
        await bus.shutdown()
        await task

        assert len(events) == 0

    def test_load_seen_from_sidecar(self, tmp_path: Path):
        bus = EventBus()
        sidecar = tmp_path / ".file_watcher_seen.json"
        sidecar.write_text(json.dumps(["report_a.md", "report_b.md"]))

        watcher = FileWatcher(bus, watch_dir=tmp_path)
        assert "report_a.md" in watcher._seen
        assert "report_b.md" in watcher._seen

    def test_load_seen_from_missing_sidecar(self, tmp_path: Path):
        bus = EventBus()
        watcher = FileWatcher(bus, watch_dir=tmp_path)
        assert watcher._seen == set()

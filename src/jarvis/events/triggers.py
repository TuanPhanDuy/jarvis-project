"""Event-generating monitors that run as asyncio background tasks.

Each monitor polls its data source and publishes typed events to the EventBus
when thresholds are exceeded or notable state changes are detected.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from jarvis.events.bus import EventBus
from jarvis.events.types import ExternalEvent, SystemEvent, UserEvent

log = structlog.get_logger()


class SystemMonitor:
    """Polls psutil metrics every 60 s and fires SystemEvents on threshold breach.

    Thresholds:
        CPU    > 90%  for 2 consecutive readings
        Memory > 85%
        Disk   < 10% free on the primary partition
    """

    CPU_THRESHOLD = 90.0
    MEM_THRESHOLD = 85.0
    DISK_FREE_THRESHOLD = 10.0
    INTERVAL = 60

    ALERT_COOLDOWN = 300  # seconds between repeat alerts for the same metric

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._cpu_strikes = 0
        self._last_fired: dict[str, float] = {}

    def _cooldown_ok(self, metric: str) -> bool:
        import time as _time
        return _time.time() - self._last_fired.get(metric, 0) > self.ALERT_COOLDOWN

    def _mark_fired(self, metric: str) -> None:
        import time as _time
        self._last_fired[metric] = _time.time()

    async def run(self) -> None:
        try:
            import psutil
        except ImportError:
            log.warning("psutil_not_installed", msg="SystemMonitor disabled — install psutil")
            return

        log.info("system_monitor_started")
        while True:
            await asyncio.sleep(self.INTERVAL)
            try:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory().percent
                disk = psutil.disk_usage("/").percent  # percent used

                if cpu > self.CPU_THRESHOLD:
                    self._cpu_strikes += 1
                    if self._cpu_strikes >= 2 and self._cooldown_ok("cpu"):
                        await self._bus.publish(SystemEvent(
                            metric="cpu", value=cpu, threshold=self.CPU_THRESHOLD, severity="warning"
                        ))
                        self._mark_fired("cpu")
                        self._cpu_strikes = 0
                else:
                    self._cpu_strikes = 0

                if mem > self.MEM_THRESHOLD and self._cooldown_ok("memory"):
                    await self._bus.publish(SystemEvent(
                        metric="memory", value=mem, threshold=self.MEM_THRESHOLD, severity="warning"
                    ))
                    self._mark_fired("memory")

                free_pct = 100 - disk
                if free_pct < self.DISK_FREE_THRESHOLD and self._cooldown_ok("disk"):
                    await self._bus.publish(SystemEvent(
                        metric="disk", value=free_pct, threshold=self.DISK_FREE_THRESHOLD, severity="alert"
                    ))
                    self._mark_fired("disk")
            except Exception as exc:
                log.error("system_monitor_error", error=str(exc))


class IdleDetector:
    """Watches active WebSocket sessions and fires UserEvent(long_idle) after N minutes.

    Accepts a callable that returns {session_id: last_activity_timestamp}.
    """

    def __init__(
        self,
        bus: EventBus,
        get_sessions: object,
        idle_minutes: int = 30,
        check_interval: int = 60,
    ) -> None:
        self._bus = bus
        self._get_sessions = get_sessions
        self._idle_seconds = idle_minutes * 60
        self._interval = check_interval
        self._notified: set[str] = set()

    async def run(self) -> None:
        import time
        log.info("idle_detector_started", idle_minutes=self._idle_seconds // 60)
        while True:
            await asyncio.sleep(self._interval)
            try:
                now = time.time()
                sessions = self._get_sessions()
                for sid, last_seen in sessions.items():
                    if (now - last_seen) > self._idle_seconds and sid not in self._notified:
                        await self._bus.publish(UserEvent(
                            session_id=sid,
                            sub_type="long_idle",
                            data={"idle_seconds": int(now - last_seen)},
                        ))
                        self._notified.add(sid)
                # Clear notified set for sessions that became active again
                active = set(sessions.keys())
                self._notified &= active
            except Exception as exc:
                log.error("idle_detector_error", error=str(exc))


class FileWatcher:
    """Watches reports_dir for new .md files and fires ExternalEvent.

    Uses polling (watchdog is optional) — checks every 30 s.
    """

    INTERVAL = 30

    def __init__(self, bus: EventBus, watch_dir: Path) -> None:
        self._bus = bus
        self._watch_dir = watch_dir
        self._sidecar = watch_dir / ".file_watcher_seen.json"
        self._seen: set[str] = self._load_seen()

    def _load_seen(self) -> set[str]:
        try:
            import json
            if self._sidecar.exists():
                return set(json.loads(self._sidecar.read_text()))
        except Exception:
            pass
        return set()

    def _save_seen(self) -> None:
        try:
            import json
            self._sidecar.write_text(json.dumps(list(self._seen)))
        except Exception:
            pass

    async def run(self) -> None:
        if not self._watch_dir.exists():
            return
        if not self._seen:
            self._seen = {f.name for f in self._watch_dir.glob("*.md")}
            self._save_seen()
        log.info("file_watcher_started", dir=str(self._watch_dir))
        while True:
            await asyncio.sleep(self.INTERVAL)
            try:
                current = {f.name for f in self._watch_dir.glob("*.md")}
                new_files = current - self._seen
                for fname in new_files:
                    await self._bus.publish(ExternalEvent(
                        source="file_watcher",
                        sub_type="new_report",
                        payload={"filename": fname, "path": str(self._watch_dir / fname)},
                    ))
                    log.info("new_report_detected", filename=fname)
                self._seen = current
                self._save_seen()
            except Exception as exc:
                log.error("file_watcher_error", error=str(exc))

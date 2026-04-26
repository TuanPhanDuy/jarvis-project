"""Typed event hierarchy for the JARVIS event bus.

Events are published by monitors (system health, idle detection, file watcher)
and consumed by AutonomousDecisionAgent to trigger proactive behavior.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class JarvisEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
        }


@dataclass
class SystemEvent(JarvisEvent):
    """Fired by SystemMonitor when resource thresholds are exceeded."""
    metric: str = ""     # "cpu" | "disk" | "memory"
    value: float = 0.0   # current reading (percentage)
    threshold: float = 0.0
    severity: str = "warning"

    def __post_init__(self):
        self.event_type = "system_alert"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(metric=self.metric, value=self.value, threshold=self.threshold, severity=self.severity)
        return d


@dataclass
class UserEvent(JarvisEvent):
    """Fired when notable user-related activity is detected."""
    user_id: str = "anonymous"
    session_id: str = ""
    sub_type: str = ""   # "long_idle" | "session_started" | "pattern_detected"
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = "user_event"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(user_id=self.user_id, session_id=self.session_id, sub_type=self.sub_type, data=self.data)
        return d


@dataclass
class TaskEvent(JarvisEvent):
    """Fired when an async task completes or fails."""
    task_id: str = ""
    sub_type: str = ""   # "task_completed" | "task_failed"
    result: str = ""

    def __post_init__(self):
        self.event_type = "task_event"


@dataclass
class ExternalEvent(JarvisEvent):
    """Fired by file watchers or webhook triggers."""
    source: str = ""     # "file_watcher" | "webhook"
    payload: dict = field(default_factory=dict)
    sub_type: str = ""

    def __post_init__(self):
        self.event_type = "external_event"

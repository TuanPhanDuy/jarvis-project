"""In-memory prompt override store.

Agents call load_prompt() which checks this registry first; if an override
exists it is returned instead of the .md file. Overrides live only for the
lifetime of the process — they are reset on restart.

Thread-safe: a plain dict protected by a threading.Lock is sufficient for
the read-heavy, write-rare access pattern.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_overrides: dict[str, str] = {}


def set_override(agent_type: str, prompt: str) -> None:
    """Register an in-memory prompt override for the given agent type."""
    with _lock:
        _overrides[agent_type.lower()] = prompt


def get_override(agent_type: str) -> str | None:
    """Return the override prompt or None if no override is set."""
    with _lock:
        return _overrides.get(agent_type.lower())


def clear_override(agent_type: str) -> bool:
    """Remove the override. Returns True if one existed."""
    with _lock:
        return _overrides.pop(agent_type.lower(), None) is not None


def clear_all_overrides() -> int:
    """Remove all overrides. Returns the count removed."""
    with _lock:
        count = len(_overrides)
        _overrides.clear()
        return count


def list_overrides() -> dict[str, str]:
    """Return a copy of the current overrides dict {agent_type: prompt}."""
    with _lock:
        return dict(_overrides)

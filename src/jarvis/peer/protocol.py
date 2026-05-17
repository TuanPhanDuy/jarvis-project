"""Peer sync protocol: HTTP-based graph delta exchange.

Each JARVIS instance exposes POST /api/peer/sync to receive a graph delta
from another node. When we receive a new peer via discovery, we push our
local delta to them and pull theirs back.

Delta format (from edge/sync.py):
    {"entities": [...], "relationships": [...], "since_ts": float, "exported_at": float}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import structlog

log = structlog.get_logger()

_RETRY_DELAYS = (1.0, 2.0, 4.0)  # seconds between attempts (3 total attempts)


def _with_retry(fn, host: str, port: int, label: str):
    """Call fn(), retrying with exponential backoff on exception. Re-raises after all attempts."""
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            log.warning(f"peer_{label}_attempt_failed", host=host, port=port,
                        attempt=attempt, error=str(exc))
            if delay is not None:
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def push_delta(peer_host: str, peer_port: int, delta: dict, timeout: float = 10.0) -> bool:
    """POST our graph delta to a peer. Returns True on success, False after all retries fail."""
    import urllib.request

    url = f"http://{peer_host}:{peer_port}/api/peer/sync"
    body = json.dumps(delta).encode()

    def _do_push() -> bool:
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200

    try:
        return _with_retry(_do_push, peer_host, peer_port, "push")
    except Exception as exc:
        log.warning("peer_push_failed", host=peer_host, port=peer_port, error=str(exc))
        return False


def merge_incoming_delta(delta: dict, db_path: Path) -> int:
    """Merge a received graph delta into local DB. Returns count of merged items."""
    from jarvis.edge.sync import import_delta
    return import_delta(db_path, delta)


def pull_delta(peer_host: str, peer_port: int, since_ts: float = 0.0, timeout: float = 10.0) -> dict | None:
    """GET graph delta from a peer since `since_ts`. Returns delta dict or None after all retries fail."""
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode({"since_ts": since_ts})
    url = f"http://{peer_host}:{peer_port}/api/peer/delta?{params}"

    def _do_pull() -> dict:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())

    try:
        return _with_retry(_do_pull, peer_host, peer_port, "pull")
    except Exception as exc:
        log.warning("peer_pull_failed", host=peer_host, port=peer_port, error=str(exc))
        return None


def sync_with_peer(peer_host: str, peer_port: int, db_path: Path, last_sync_ts: float = 0.0) -> bool:
    """Bidirectional sync: push our delta to a peer and pull their delta back."""
    from jarvis.edge.sync import export_delta

    delta = export_delta(db_path, since_ts=last_sync_ts)
    if delta["entities"] or delta["relationships"]:
        pushed = push_delta(peer_host, peer_port, delta)
        if pushed:
            log.info("peer_sync_pushed", host=peer_host, port=peer_port,
                     entities=len(delta["entities"]), rels=len(delta["relationships"]))
    else:
        log.debug("peer_sync_nothing_to_push", host=peer_host)

    incoming = pull_delta(peer_host, peer_port, since_ts=last_sync_ts)
    if incoming:
        count = merge_incoming_delta(incoming, db_path)
        log.info("peer_sync_pulled", host=peer_host, port=peer_port, count=count)

    return True

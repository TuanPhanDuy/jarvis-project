"""Peer sync protocol: HTTP-based graph delta exchange.

Each JARVIS instance exposes POST /api/peer/sync to receive a graph delta
from another node. When we receive a new peer via discovery, we push our
local delta to them and pull theirs back.

Delta format (from edge/sync.py):
    {"entities": [...], "relationships": [...], "since_ts": float, "exported_at": float}
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

log = structlog.get_logger()


def push_delta(peer_host: str, peer_port: int, delta: dict, timeout: float = 10.0) -> bool:
    """POST our graph delta to a peer. Returns True on success."""
    try:
        import urllib.request
        import urllib.error

        url = f"http://{peer_host}:{peer_port}/api/peer/sync"
        body = json.dumps(delta).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception as exc:
        log.warning("peer_push_failed", host=peer_host, port=peer_port, error=str(exc))
        return False


def merge_incoming_delta(delta: dict, db_path: Path) -> int:
    """Merge a received graph delta into local DB. Returns count of merged items."""
    from jarvis.edge.sync import import_delta
    return import_delta(db_path, delta)


def pull_delta(peer_host: str, peer_port: int, since_ts: float = 0.0, timeout: float = 10.0) -> dict | None:
    """GET graph delta from a peer since `since_ts`. Returns delta dict or None on failure."""
    try:
        import urllib.parse
        import urllib.request

        params = urllib.parse.urlencode({"since_ts": since_ts})
        url = f"http://{peer_host}:{peer_port}/api/peer/delta?{params}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            import json as _json
            return _json.loads(resp.read())
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

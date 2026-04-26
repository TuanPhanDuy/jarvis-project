"""Peer coordinator: manages the discovery loop and triggers graph sync.

Runs as a background asyncio task when JARVIS_PEER_ENABLED=true.
On discovering a new peer it pushes the local knowledge graph delta to them.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import structlog

log = structlog.get_logger()


class PeerCoordinator:
    """Orchestrates peer discovery and knowledge graph sync."""

    def __init__(self, db_path: Path, device_id: str, http_port: int = 7474):
        from jarvis.peer.discovery import PeerDiscovery

        self._db_path = db_path
        self._discovery = PeerDiscovery(device_id=device_id, http_port=http_port)
        self._synced_peers: dict[str, float] = {}  # device_id → last sync ts
        self._running = False

    def get_peer_list(self) -> list[dict]:
        return self._discovery.get_peers_dict()

    async def start(self) -> None:
        self._running = True
        log.info("peer_coordinator_started")
        await asyncio.gather(
            self._discovery.start(),
            self._sync_loop(),
        )

    def stop(self) -> None:
        self._running = False
        self._discovery.stop()

    async def _sync_loop(self) -> None:
        """Periodically sync with newly discovered or long-unseen peers."""
        while self._running:
            await asyncio.sleep(60)
            for peer in self._discovery.peers:
                last_sync = self._synced_peers.get(peer.device_id, 0.0)
                # Sync if we've never synced or haven't synced in the last 5 min
                if (time.time() - last_sync) > 300:
                    await self._do_sync(peer)

    async def _do_sync(self, peer) -> None:
        from jarvis.peer.protocol import sync_with_peer

        try:
            last_sync = self._synced_peers.get(peer.device_id, 0.0)
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                lambda: sync_with_peer(peer.host, peer.port, self._db_path, last_sync),
            )
            if success:
                self._synced_peers[peer.device_id] = time.time()
                log.info("peer_sync_complete", device_id=peer.device_id, host=peer.host)
        except Exception as exc:
            log.error("peer_sync_error", device_id=peer.device_id, error=str(exc))

"""Peer discovery via UDP broadcast.

Each JARVIS instance broadcasts its presence every 30 s on the LAN.
Other instances listen on the same port and add new peers to the coordinator.

Broadcast payload (JSON):
    {"device_id": "abc12345", "host": "192.168.1.5", "port": 7474, "version": "1"}
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid

BROADCAST_PORT = 17474  # separate from the HTTP peer port to avoid conflicts
BROADCAST_INTERVAL = 30
BROADCAST_TTL = 90  # remove peer if not heard from in this many seconds


class PeerInfo:
    def __init__(self, device_id: str, host: str, port: int):
        self.device_id = device_id
        self.host = host
        self.port = port
        self.last_seen: float = time.time()

    def is_stale(self) -> bool:
        return (time.time() - self.last_seen) > BROADCAST_TTL

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "host": self.host,
            "port": self.port,
            "last_seen": self.last_seen,
        }


class PeerDiscovery:
    """Broadcasts our presence and collects peer announcements via UDP."""

    def __init__(self, device_id: str | None = None, http_port: int = 7474):
        self._device_id = device_id or str(uuid.uuid4())[:8]
        self._http_port = http_port
        self._peers: dict[str, PeerInfo] = {}
        self._running = False

    @property
    def peers(self) -> list[PeerInfo]:
        """Return non-stale peers, excluding ourselves."""
        return [p for p in self._peers.values() if not p.is_stale()]

    def get_peers_dict(self) -> list[dict]:
        return [p.to_dict() for p in self.peers]

    async def start(self) -> None:
        self._running = True
        await asyncio.gather(
            self._broadcast_loop(),
            self._listen_loop(),
            self._prune_loop(),
        )

    def stop(self) -> None:
        self._running = False

    async def _broadcast_loop(self) -> None:
        payload = json.dumps({
            "device_id": self._device_id,
            "host": self._get_local_ip(),
            "port": self._http_port,
            "version": "1",
        }).encode()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        try:
            while self._running:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: sock.sendto(payload, ("<broadcast>", BROADCAST_PORT)),
                    )
                except Exception:
                    pass
                await asyncio.sleep(BROADCAST_INTERVAL)
        finally:
            sock.close()

    async def _listen_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", BROADCAST_PORT))
            sock.setblocking(False)
        except Exception:
            return  # port in use or permission denied — discovery disabled

        loop = asyncio.get_event_loop()
        try:
            while self._running:
                try:
                    data, addr = await loop.run_in_executor(None, lambda: sock.recvfrom(1024))
                    peer = json.loads(data.decode())
                    did = peer.get("device_id", "")
                    if did and did != self._device_id:
                        if did in self._peers:
                            self._peers[did].last_seen = time.time()
                        else:
                            self._peers[did] = PeerInfo(
                                device_id=did,
                                host=peer.get("host", addr[0]),
                                port=int(peer.get("port", 7474)),
                            )
                except (BlockingIOError, json.JSONDecodeError, Exception):
                    await asyncio.sleep(1)
        finally:
            sock.close()

    async def _prune_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            stale = [did for did, p in self._peers.items() if p.is_stale()]
            for did in stale:
                del self._peers[did]

    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

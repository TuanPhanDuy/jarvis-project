"""MQTT transport: publish queries from edge → cloud JARVIS, receive responses.

Cloud JARVIS subscribes to: jarvis/query/{device_id}
Cloud JARVIS publishes to:  jarvis/response/{device_id}

Message format (JSON):
  Query:    {"request_id": "uuid", "message": "...", "device_id": "..."}
  Response: {"request_id": "uuid", "response": "..."}

Requires: pip install paho-mqtt
"""
from __future__ import annotations

import json
import threading
import uuid


class MQTTTransport:
    """Publish queries to cloud JARVIS and receive responses over MQTT."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        device_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._host = host
        self._port = port
        self._device_id = device_id or str(uuid.uuid4())[:8]
        self._timeout = timeout
        self._client = None
        self._connected = False
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, str] = {}

    def connect(self) -> bool:
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(client_id=f"jarvis-edge-{self._device_id}")
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.connect(self._host, self._port, keepalive=60)
            self._client.loop_start()
            # Wait briefly for connection
            import time; time.sleep(1.0)
            return self._connected
        except Exception as e:
            print(f"MQTT connect failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()

    def is_connected(self) -> bool:
        return self._connected

    def query(self, message: str, timeout: float | None = None) -> str:
        """Send a query and block until cloud JARVIS responds or timeout."""
        if not self._connected or not self._client:
            return "ERROR: Not connected to MQTT broker."

        request_id = str(uuid.uuid4())
        event = threading.Event()
        self._pending[request_id] = event

        payload = json.dumps({
            "request_id": request_id,
            "message": message,
            "device_id": self._device_id,
        })
        self._client.publish(f"jarvis/query/{self._device_id}", payload)

        received = event.wait(timeout=timeout or self._timeout)
        response = self._responses.pop(request_id, None)
        self._pending.pop(request_id, None)

        if not received or response is None:
            return "ERROR: Timeout — cloud JARVIS did not respond in time."
        return response

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            client.subscribe(f"jarvis/response/{self._device_id}")

    def _on_message(self, client, userdata, msg) -> None:
        try:
            data = json.loads(msg.payload)
            req_id = data.get("request_id", "")
            response = data.get("response", "")
            self._responses[req_id] = response
            if req_id in self._pending:
                self._pending[req_id].set()
        except Exception:
            pass

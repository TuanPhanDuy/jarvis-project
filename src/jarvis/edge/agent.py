"""Edge JARVIS agent — lightweight relay for resource-constrained devices.

Designed for Raspberry Pi, phone, or any edge node. Relays queries to cloud
JARVIS via MQTT when available; degrades gracefully to local responses.
"""
from __future__ import annotations

import uuid


class EdgeAgent:
    """Minimal agent: capture → relay to cloud → speak response."""

    def __init__(
        self,
        cloud_relay=None,    # MQTTTransport instance, or None for offline mode
        use_voice: bool = False,
    ) -> None:
        self._relay = cloud_relay
        self._use_voice = use_voice
        self._device_id = str(uuid.uuid4())[:8]

    def handle(self, message: str) -> str:
        if self._relay and self._relay.is_connected():
            return self._relay.query(message)
        return (
            "Cloud JARVIS is unreachable. "
            "Check MQTT connection or run 'jarvis-edge --no-cloud' for local mode."
        )

    def run(self) -> None:
        print(f"JARVIS Edge (device: {self._device_id})")
        relay_status = "connected" if (self._relay and self._relay.is_connected()) else "offline"
        print(f"Cloud relay: {relay_status}. Type 'quit' to exit.\n")

        while True:
            if self._use_voice:
                from jarvis.voice.wake_word import wait_for_wake_word
                from jarvis.voice.stt import record_and_transcribe
                print("Listening for wake word…")
                wait_for_wake_word()
                print("Recording…")
                user_input = record_and_transcribe()
                if not user_input:
                    continue
                print(f"You: {user_input}")
            else:
                user_input = input("You: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue

            response = self.handle(user_input)
            print(f"JARVIS: {response}\n")

            if self._use_voice:
                from jarvis.voice.tts import speak
                speak(response)

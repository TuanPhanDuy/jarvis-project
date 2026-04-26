"""JARVIS Edge Agent — entrypoint.

Usage:
    jarvis-edge                        # text mode, cloud relay via MQTT
    jarvis-edge --voice                # voice mode (wake word + STT + TTS)
    jarvis-edge --no-cloud             # local text mode, no MQTT
    jarvis-edge --mqtt-host 10.0.0.5  # custom cloud MQTT host
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS Edge Agent — lightweight relay")
    parser.add_argument("--mqtt-host", default="localhost", help="Cloud JARVIS MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--voice", action="store_true", help="Enable wake-word + voice I/O")
    parser.add_argument("--no-cloud", action="store_true", help="Run without cloud relay")
    args = parser.parse_args()

    relay = None
    if not args.no_cloud:
        try:
            from jarvis.edge.mqtt_transport import MQTTTransport
            relay = MQTTTransport(host=args.mqtt_host, port=args.mqtt_port)
            if not relay.connect():
                print("Warning: MQTT connect failed. Running in local-only mode.")
                relay = None
        except ImportError:
            print("paho-mqtt not installed. Running in local-only mode.")
            print("Install: pip install paho-mqtt")

    from jarvis.edge.agent import EdgeAgent
    agent = EdgeAgent(cloud_relay=relay, use_voice=args.voice)

    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nEdge agent stopped.")
    finally:
        if relay:
            relay.disconnect()


if __name__ == "__main__":
    main()

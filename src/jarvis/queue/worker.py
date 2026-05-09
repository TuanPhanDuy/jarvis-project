"""JARVIS queue worker — standalone process that consumes RabbitMQ tasks.

Run with:
    python -m jarvis.queue.worker

The worker:
  1. Connects to RabbitMQ
  2. Consumes messages from the jarvis.tasks queue
  3. Calls process_task() for each message
  4. Publishes result to a results queue (jarvis.results)
  5. Acks the message on success, nacks on unrecoverable error

Environment variables read from .env (via Settings):
  RABBITMQ_URL          — amqp://guest:guest@localhost:5672/
  RABBITMQ_TASK_QUEUE   — jarvis.tasks
"""
from __future__ import annotations

import json
import signal
import sys

import pika
import structlog

from jarvis.api.models import QueueTask
from jarvis.config import get_settings
from jarvis.queue.consumer import process_task

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

RESULTS_QUEUE = "jarvis.results"
_running = True


def _on_message(channel, method, properties, body: bytes) -> None:
    try:
        task = QueueTask(**json.loads(body))
    except Exception as exc:
        log.error("invalid_message", error=str(exc), body=body[:200])
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    result = process_task(task)

    # Publish result: prefer per-task reply_to queue (AMQP RPC pattern); fall back to shared queue
    reply_to = properties.reply_to if properties and properties.reply_to else None
    if reply_to:
        routing_key = reply_to
    else:
        channel.queue_declare(queue=RESULTS_QUEUE, durable=True)
        routing_key = RESULTS_QUEUE
        log.info("task_result_no_reply_to", task_id=task.task_id, fallback_queue=RESULTS_QUEUE)

    channel.basic_publish(
        exchange="",
        routing_key=routing_key,
        body=result.model_dump_json(),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent,
            content_type="application/json",
            correlation_id=task.task_id,
        ),
    )

    channel.basic_ack(delivery_tag=method.delivery_tag)
    log.info("message_acked", task_id=task.task_id, error=result.error)


def _shutdown(signum, frame) -> None:
    global _running
    log.info("shutdown_signal_received")
    _running = False
    sys.exit(0)


def _connect_with_retry(rabbitmq_url: str, max_retries: int = 10) -> pika.BlockingConnection:
    """Connect to RabbitMQ with exponential backoff. Raises on exhausted retries."""
    import time as _time
    for attempt in range(max_retries):
        try:
            return pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = min(2 ** attempt, 60)
            log.warning("rabbitmq_connect_retry", attempt=attempt + 1, delay_s=delay, error=str(exc))
            _time.sleep(delay)


def main() -> None:
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    settings = get_settings()
    log.info(
        "worker_starting",
        rabbitmq_url=settings.rabbitmq_url,
        queue=settings.rabbitmq_task_queue,
    )

    while _running:
        try:
            connection = _connect_with_retry(settings.rabbitmq_url)
            channel = connection.channel()
            channel.queue_declare(queue=settings.rabbitmq_task_queue, durable=True)
            channel.basic_qos(prefetch_count=1)  # one task at a time per worker
            channel.basic_consume(
                queue=settings.rabbitmq_task_queue,
                on_message_callback=_on_message,
            )
            log.info("worker_ready", queue=settings.rabbitmq_task_queue)
            channel.start_consuming()
        except Exception as exc:
            if not _running:
                break
            log.error("worker_disconnected", error=str(exc), action="reconnecting")
            import time as _time
            _time.sleep(5)


if __name__ == "__main__":
    main()

"""RabbitMQ task producer.

Publishes JARVIS tasks to the message queue for async processing by the worker.

Usage:
    from jarvis.queue.producer import publish_task
    task_id = publish_task(message="Research RLHF", session_id="abc123")
"""
from __future__ import annotations

import json
import uuid

import pika

from jarvis.api.metrics import QUEUE_TASKS_PUBLISHED
from jarvis.api.models import QueueTask


def publish_task(
    message: str,
    session_id: str | None = None,
    researcher_mode: bool = False,
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
    queue_name: str = "jarvis.tasks",
    reply_to: str | None = None,
) -> str:
    """Publish a task to RabbitMQ and return the task_id.

    Args:
        message: The user message / task description.
        session_id: Optional session ID for conversation continuity.
        researcher_mode: Use ResearcherAgent instead of PlannerAgent.
        rabbitmq_url: AMQP connection URL.
        queue_name: Name of the task queue.
        reply_to: Optional reply queue name for the AMQP RPC pattern. If set,
            the worker will publish the result to this queue instead of the
            shared results queue, allowing the caller to await a specific result.

    Returns:
        task_id — a UUID string you can use to poll for results.
    """
    task_id = str(uuid.uuid4())
    task = QueueTask(
        task_id=task_id,
        message=message,
        session_id=session_id or str(uuid.uuid4()),
        researcher_mode=researcher_mode,
    )

    connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
    try:
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=task.model_dump_json(),
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent,
                content_type="application/json",
                reply_to=reply_to,
            ),
        )
        QUEUE_TASKS_PUBLISHED.inc()
    finally:
        connection.close()

    return task_id

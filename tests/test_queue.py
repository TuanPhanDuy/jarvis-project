"""Tests for JARVIS RabbitMQ producer and worker dispatch logic."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from jarvis.api.models import QueueTask
from jarvis.queue.worker import _on_message


# ── Producer ──────────────────────────────────────────────────────────────────


class TestQueueProducer:
    def _make_mock_connection(self) -> tuple[MagicMock, MagicMock]:
        mock_channel = MagicMock()
        mock_conn = MagicMock()
        mock_conn.channel.return_value = mock_channel
        return mock_conn, mock_channel

    def test_publish_serializes_task_to_json(self):
        mock_conn, mock_channel = self._make_mock_connection()

        with patch("pika.BlockingConnection", return_value=mock_conn), \
             patch("pika.URLParameters"), \
             patch("jarvis.queue.producer.QUEUE_TASKS_PUBLISHED"):
            from jarvis.queue.producer import publish_task
            task_id = publish_task("Research RLHF", session_id="sess-001")

        assert task_id  # non-empty UUID string
        mock_channel.basic_publish.assert_called_once()
        call_kwargs = mock_channel.basic_publish.call_args[1]
        body = json.loads(call_kwargs["body"])
        assert body["message"] == "Research RLHF"
        assert body["session_id"] == "sess-001"

    def test_publish_sets_persistent_delivery_mode(self):
        mock_conn, mock_channel = self._make_mock_connection()

        with patch("pika.BlockingConnection", return_value=mock_conn), \
             patch("pika.URLParameters"), \
             patch("jarvis.queue.producer.QUEUE_TASKS_PUBLISHED"):
            from jarvis.queue.producer import publish_task
            publish_task("hello")

        props = mock_channel.basic_publish.call_args[1]["properties"]
        # pika stores delivery_mode as int 2 (Persistent); compare by value
        assert props.delivery_mode in (2, "2")

    def test_publish_with_reply_to(self):
        mock_conn, mock_channel = self._make_mock_connection()

        with patch("pika.BlockingConnection", return_value=mock_conn), \
             patch("pika.URLParameters"), \
             patch("jarvis.queue.producer.QUEUE_TASKS_PUBLISHED"):
            from jarvis.queue.producer import publish_task
            publish_task("hello", reply_to="my-reply-queue")

        props = mock_channel.basic_publish.call_args[1]["properties"]
        assert props.reply_to == "my-reply-queue"

    def test_publish_connection_always_closed(self):
        mock_conn, _ = self._make_mock_connection()

        with patch("pika.BlockingConnection", return_value=mock_conn), \
             patch("pika.URLParameters"), \
             patch("jarvis.queue.producer.QUEUE_TASKS_PUBLISHED"):
            from jarvis.queue.producer import publish_task
            publish_task("test")

        mock_conn.close.assert_called_once()

    def test_publish_returns_task_id_string(self):
        mock_conn, _ = self._make_mock_connection()

        with patch("pika.BlockingConnection", return_value=mock_conn), \
             patch("pika.URLParameters"), \
             patch("jarvis.queue.producer.QUEUE_TASKS_PUBLISHED"):
            from jarvis.queue.producer import publish_task
            result = publish_task("test message")

        assert isinstance(result, str)
        assert len(result) == 36  # UUID4 format


# ── Worker dispatch (_on_message) ─────────────────────────────────────────────


class TestWorkerDispatch:
    def _make_pika_parts(
        self,
        body: bytes,
        reply_to: str | None = None,
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        channel = MagicMock()
        method = MagicMock()
        method.delivery_tag = "tag-001"
        properties = MagicMock()
        properties.reply_to = reply_to
        return channel, method, properties

    def _make_valid_task_body(self) -> bytes:
        task = QueueTask(
            task_id="task-xyz",
            message="What is RLHF?",
            session_id="sess-abc",
            researcher_mode=False,
        )
        return task.model_dump_json().encode()

    def test_valid_message_acked_after_processing(self):
        channel, method, properties = self._make_pika_parts(self._make_valid_task_body())
        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"task_id":"task-xyz","session_id":"sess-abc","response":"ok","error":null}'

        with patch("jarvis.queue.worker.process_task", return_value=mock_result):
            _on_message(channel, method, properties, self._make_valid_task_body())

        channel.basic_ack.assert_called_once_with(delivery_tag="tag-001")
        channel.basic_nack.assert_not_called()

    def test_malformed_body_is_nacked(self):
        channel, method, properties = self._make_pika_parts(b"not-json")

        _on_message(channel, method, properties, b"not-json")

        channel.basic_nack.assert_called_once_with(
            delivery_tag="tag-001", requeue=False
        )
        channel.basic_ack.assert_not_called()

    def test_reply_to_routes_result_to_reply_queue(self):
        body = self._make_valid_task_body()
        channel, method, properties = self._make_pika_parts(body, reply_to="rpc-reply-q")
        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"task_id":"task-xyz"}'
        mock_result.error = None

        with patch("jarvis.queue.worker.process_task", return_value=mock_result):
            _on_message(channel, method, properties, body)

        publish_calls = channel.basic_publish.call_args_list
        assert any(
            c[1]["routing_key"] == "rpc-reply-q"
            for c in publish_calls
        )

    def test_no_reply_to_routes_to_shared_results_queue(self):
        body = self._make_valid_task_body()
        channel, method, properties = self._make_pika_parts(body, reply_to=None)
        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"task_id":"task-xyz"}'
        mock_result.error = None

        with patch("jarvis.queue.worker.process_task", return_value=mock_result):
            _on_message(channel, method, properties, body)

        publish_calls = channel.basic_publish.call_args_list
        assert any(
            "jarvis.results" in str(c)
            for c in publish_calls
        )

    def test_processing_error_still_acks(self):
        """Even if process_task raises, _on_message should not crash the worker."""
        body = self._make_valid_task_body()
        channel, method, properties = self._make_pika_parts(body)
        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"error":"something failed"}'
        mock_result.error = "something failed"

        with patch("jarvis.queue.worker.process_task", return_value=mock_result):
            _on_message(channel, method, properties, body)

        # Message is acked even on error (error is in result payload, not exception)
        channel.basic_ack.assert_called_once()

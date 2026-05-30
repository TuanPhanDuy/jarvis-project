"""Tests for webhook notification system."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Core module tests ─────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    return tmp_path / "jarvis.db"


class TestRegisterWebhook:
    def test_register_returns_id_and_url(self, db):
        from jarvis.events.webhooks import register_webhook
        rec = register_webhook(db, "http://example.com/hook", ["schedule.complete"])
        assert rec["id"]
        assert rec["url"] == "http://example.com/hook"
        assert "schedule.complete" in rec["events"]
        assert rec["active"] is True

    def test_register_multiple_events(self, db):
        from jarvis.events.webhooks import register_webhook
        rec = register_webhook(db, "http://a.com", ["eval.complete", "tool.error"])
        assert len(rec["events"]) == 2

    def test_register_unknown_event_raises(self, db):
        from jarvis.events.webhooks import register_webhook
        with pytest.raises(ValueError, match="Unknown event"):
            register_webhook(db, "http://x.com", ["nonexistent.event"])

    def test_register_persisted_to_db(self, db):
        from jarvis.events.webhooks import register_webhook, list_webhooks
        register_webhook(db, "http://b.com", ["chat.complete"])
        hooks = list_webhooks(db)
        assert any(h["url"] == "http://b.com" for h in hooks)


class TestListWebhooks:
    def test_empty_returns_empty_list(self, db):
        from jarvis.events.webhooks import list_webhooks
        assert list_webhooks(db) == []

    def test_filter_by_event(self, db):
        from jarvis.events.webhooks import register_webhook, list_webhooks
        register_webhook(db, "http://a.com", ["schedule.complete"])
        register_webhook(db, "http://b.com", ["eval.complete"])
        result = list_webhooks(db, event="schedule.complete")
        assert len(result) == 1
        assert result[0]["url"] == "http://a.com"

    def test_deleted_not_returned(self, db):
        from jarvis.events.webhooks import register_webhook, list_webhooks, delete_webhook
        rec = register_webhook(db, "http://c.com", ["chat.complete"])
        delete_webhook(db, rec["id"])
        assert list_webhooks(db) == []


class TestDeleteWebhook:
    def test_delete_existing_returns_true(self, db):
        from jarvis.events.webhooks import register_webhook, delete_webhook
        rec = register_webhook(db, "http://d.com", ["tool.error"])
        assert delete_webhook(db, rec["id"]) is True

    def test_delete_nonexistent_returns_false(self, db):
        from jarvis.events.webhooks import delete_webhook
        assert delete_webhook(db, "nonexistent-uuid") is False


class TestFireEvent:
    def test_fire_event_calls_registered_hook(self, db):
        from jarvis.events.webhooks import register_webhook, fire_event
        register_webhook(db, "http://receiver.com/hook", ["training.complete"])

        with patch("jarvis.events.webhooks._deliver") as mock_deliver:
            fire_event(db, "training.complete", {"model": "jarvis-ft"})

        mock_deliver.assert_called_once()
        call_args = mock_deliver.call_args
        assert call_args[0][1]["url"] == "http://receiver.com/hook"

    def test_fire_event_skips_unmatched_hooks(self, db):
        from jarvis.events.webhooks import register_webhook, fire_event
        register_webhook(db, "http://x.com", ["eval.complete"])

        with patch("jarvis.events.webhooks._deliver") as mock_deliver:
            fire_event(db, "tool.error", {"tool": "web_search"})

        mock_deliver.assert_not_called()

    def test_fire_event_no_hooks_is_noop(self, db):
        from jarvis.events.webhooks import fire_event
        fire_event(db, "chat.complete", {"session_id": "abc"})  # should not raise


class TestWebhookSignature:
    def test_sign_produces_hex_string(self):
        from jarvis.events.webhooks import _sign
        sig = _sign("mysecret", '{"event": "test"}')
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex digest

    def test_sign_different_secret_different_signature(self):
        from jarvis.events.webhooks import _sign
        s1 = _sign("secret1", "body")
        s2 = _sign("secret2", "body")
        assert s1 != s2


class TestDeliveryRecord:
    def test_delivery_recorded_on_success(self, db):
        from jarvis.events.webhooks import register_webhook, get_deliveries, _record_delivery
        rec = register_webhook(db, "http://e.com", ["schedule.complete"])
        _record_delivery(db, rec["id"], "schedule.complete", "{}", "success", 1, None)
        deliveries = get_deliveries(db, rec["id"])
        assert len(deliveries) == 1
        assert deliveries[0]["status"] == "success"


# ── API endpoint tests ────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


class TestWebhookApiEndpoints:
    def test_create_webhook_201(self, client, tmp_path):
        db = tmp_path / "jarvis.db"
        with patch("jarvis.api.server.get_settings") as mock_cfg:
            mock_cfg.return_value.reports_dir = tmp_path
            resp = client.post("/api/webhooks", json={
                "url": "http://example.com/cb",
                "events": ["schedule.complete"],
            })
        assert resp.status_code == 201
        body = resp.json()
        assert body["url"] == "http://example.com/cb"
        assert body["id"]

    def test_create_webhook_bad_url_422(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings") as mock_cfg:
            mock_cfg.return_value.reports_dir = tmp_path
            resp = client.post("/api/webhooks", json={
                "url": "ftp://bad-scheme.com",
                "events": ["chat.complete"],
            })
        assert resp.status_code == 422

    def test_create_webhook_unknown_event_422(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings") as mock_cfg:
            mock_cfg.return_value.reports_dir = tmp_path
            resp = client.post("/api/webhooks", json={
                "url": "http://ok.com",
                "events": ["bad.event"],
            })
        assert resp.status_code == 422

    def test_list_webhooks_200(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings") as mock_cfg:
            mock_cfg.return_value.reports_dir = tmp_path
            client.post("/api/webhooks", json={"url": "http://x.com", "events": ["tool.error"]})
            resp = client.get("/api/webhooks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_delete_webhook_204(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings") as mock_cfg:
            mock_cfg.return_value.reports_dir = tmp_path
            create_resp = client.post("/api/webhooks", json={
                "url": "http://del.com", "events": ["eval.complete"],
            })
            wid = create_resp.json()["id"]
            resp = client.delete(f"/api/webhooks/{wid}")
        assert resp.status_code == 204

    def test_delete_nonexistent_webhook_404(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings") as mock_cfg:
            mock_cfg.return_value.reports_dir = tmp_path
            resp = client.delete("/api/webhooks/nonexistent-id")
        assert resp.status_code == 404

    def test_get_deliveries_200(self, client, tmp_path):
        with patch("jarvis.api.server.get_settings") as mock_cfg:
            mock_cfg.return_value.reports_dir = tmp_path
            create_resp = client.post("/api/webhooks", json={
                "url": "http://hist.com", "events": ["chat.complete"],
            })
            wid = create_resp.json()["id"]
            resp = client.get(f"/api/webhooks/{wid}/deliveries")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

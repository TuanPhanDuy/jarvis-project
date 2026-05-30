"""Tests for the system notification center."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from jarvis.events.notifications import (
    clear_read,
    list_notifications,
    mark_read,
    push_notification,
    unread_count,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "jarvis.db"


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestPushNotification:
    def test_returns_record_with_id(self, db):
        rec = push_notification(db, "system.info", "Test alert")
        assert rec["id"]
        assert rec["title"] == "Test alert"
        assert rec["read"] is False

    def test_persisted_to_db(self, db):
        push_notification(db, "tool.error", "Tool failed", body="details")
        rows = list_notifications(db)
        assert len(rows) == 1
        assert rows[0]["body"] == "details"

    def test_invalid_severity_defaults_to_info(self, db):
        rec = push_notification(db, "system.info", "Hi", severity="critical")
        assert rec["severity"] == "info"

    def test_valid_severities(self, db):
        for sev in ("info", "warning", "error"):
            rec = push_notification(db, "system.info", f"msg-{sev}", severity=sev)
            assert rec["severity"] == sev


class TestListNotifications:
    def test_empty_db_returns_empty(self, db):
        assert list_notifications(db) == []

    def test_returns_newest_first(self, db):
        push_notification(db, "system.info", "first")
        time.sleep(0.01)
        push_notification(db, "system.info", "second")
        rows = list_notifications(db)
        assert rows[0]["title"] == "second"
        assert rows[1]["title"] == "first"

    def test_unread_only_filter(self, db):
        rec = push_notification(db, "system.info", "unread one")
        push_notification(db, "system.info", "read one")
        mark_read(db, list_notifications(db)[-1]["id"])
        unread = list_notifications(db, unread_only=True)
        assert all(not r["read"] for r in unread)

    def test_limit_and_offset(self, db):
        for i in range(5):
            push_notification(db, "system.info", f"msg-{i}")
        page1 = list_notifications(db, limit=2, offset=0)
        page2 = list_notifications(db, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0]["id"] != page2[0]["id"]


class TestMarkRead:
    def test_mark_existing_returns_true(self, db):
        rec = push_notification(db, "system.info", "mark me")
        assert mark_read(db, rec["id"]) is True

    def test_mark_updates_read_flag(self, db):
        rec = push_notification(db, "eval.complete", "done")
        mark_read(db, rec["id"])
        rows = list_notifications(db)
        assert rows[0]["read"] is True

    def test_mark_nonexistent_returns_false(self, db):
        assert mark_read(db, "no-such-id") is False


class TestClearRead:
    def test_clears_only_read_notifications(self, db):
        rec1 = push_notification(db, "system.info", "keep me")
        rec2 = push_notification(db, "system.info", "delete me")
        mark_read(db, rec2["id"])
        deleted = clear_read(db)
        assert deleted == 1
        remaining = list_notifications(db)
        assert len(remaining) == 1
        assert remaining[0]["id"] == rec1["id"]

    def test_returns_count(self, db):
        for i in range(3):
            rec = push_notification(db, "system.info", f"n{i}")
            mark_read(db, rec["id"])
        assert clear_read(db) == 3


class TestUnreadCount:
    def test_zero_when_empty(self, db):
        assert unread_count(db) == 0

    def test_counts_unread(self, db):
        push_notification(db, "system.info", "a")
        push_notification(db, "system.info", "b")
        assert unread_count(db) == 2

    def test_not_counting_read(self, db):
        rec = push_notification(db, "system.info", "c")
        mark_read(db, rec["id"])
        assert unread_count(db) == 0


# ── API endpoint tests ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_auth():
    import jarvis.api.server as _s
    _s._require_auth = None
    yield
    _s._require_auth = None


@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def patched_db(tmp_path):
    from unittest.mock import patch
    from jarvis.api.server import get_settings
    import jarvis.api.server as _s
    settings = type("S", (), {"reports_dir": tmp_path})()
    with patch("jarvis.api.server.get_settings", return_value=settings):
        yield tmp_path


class TestNotificationApiEndpoints:
    def test_list_returns_200(self, client, patched_db):
        resp = client.get("/api/notifications")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_201(self, client, patched_db):
        resp = client.post("/api/notifications",
                           json={"title": "Hello", "event": "system.info"})
        assert resp.status_code == 201
        assert resp.json()["title"] == "Hello"

    def test_create_empty_title_422(self, client, patched_db):
        resp = client.post("/api/notifications", json={"title": ""})
        assert resp.status_code == 422

    def test_mark_read_200(self, client, patched_db):
        create = client.post("/api/notifications",
                              json={"title": "mark test", "event": "system.info"})
        nid = create.json()["id"]
        resp = client.patch(f"/api/notifications/{nid}/read")
        assert resp.status_code == 200
        assert resp.json()["read"] is True

    def test_mark_nonexistent_404(self, client, patched_db):
        resp = client.patch("/api/notifications/fake-id/read")
        assert resp.status_code == 404

    def test_clear_read_200(self, client, patched_db):
        create = client.post("/api/notifications",
                              json={"title": "clear test", "event": "system.info"})
        nid = create.json()["id"]
        client.patch(f"/api/notifications/{nid}/read")
        resp = client.delete("/api/notifications")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1

    def test_unread_count_endpoint(self, client, patched_db):
        client.post("/api/notifications",
                    json={"title": "u1", "event": "system.info"})
        resp = client.get("/api/notifications/unread-count")
        assert resp.status_code == 200
        assert resp.json()["unread"] >= 1

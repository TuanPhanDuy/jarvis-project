"""Tests for POST /api/config/reload."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


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


def _fake_settings(**overrides):
    from jarvis.config import Settings
    defaults = {
        "auth_enabled": False,
        "jwt_secret": "test-secret",
        "chat_rate_limit": "30/minute",
        "rate_limit_enabled": False,
        "rate_limit_per_user": False,
        "reports_dir": MagicMock(),
    }
    defaults.update(overrides)
    s = MagicMock(spec=Settings)
    s.model_fields = {}
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class TestConfigReloadEndpoint:
    def test_returns_200(self, client):
        with patch("jarvis.api.server.get_settings", return_value=_fake_settings()), \
             patch("jarvis.api.server.make_auth_dependency", return_value=None, create=True):
            resp = client.post("/api/config/reload")
        assert resp.status_code == 200

    def test_response_has_changed_and_reloaded_at(self, client):
        with patch("jarvis.api.server.get_settings", return_value=_fake_settings()):
            resp = client.post("/api/config/reload")
        body = resp.json()
        assert "changed" in body
        assert "reloaded_at" in body

    def test_changed_is_dict(self, client):
        with patch("jarvis.api.server.get_settings", return_value=_fake_settings()):
            resp = client.post("/api/config/reload")
        assert isinstance(resp.json()["changed"], dict)

    def test_reloaded_at_is_float(self, client):
        with patch("jarvis.api.server.get_settings", return_value=_fake_settings()):
            resp = client.post("/api/config/reload")
        assert isinstance(resp.json()["reloaded_at"], float)

    def test_detects_changed_settings(self, client):
        from jarvis.config import Settings
        old = MagicMock(spec=Settings)
        old.model_fields = {"chat_rate_limit": None}
        old.chat_rate_limit = "10/minute"
        old.auth_enabled = False
        old.jwt_secret = "s"
        old.reports_dir = MagicMock()

        new = MagicMock(spec=Settings)
        new.model_fields = {"chat_rate_limit": None}
        new.chat_rate_limit = "60/minute"
        new.auth_enabled = False
        new.jwt_secret = "s"
        new.reports_dir = MagicMock()

        call_count = [0]
        def side_effect():
            call_count[0] += 1
            return old if call_count[0] == 1 else new

        with patch("jarvis.api.server.get_settings", side_effect=side_effect):
            resp = client.post("/api/config/reload")

        body = resp.json()
        assert "chat_rate_limit" in body["changed"]
        assert body["changed"]["chat_rate_limit"]["old"] == "10/minute"
        assert body["changed"]["chat_rate_limit"]["new"] == "60/minute"

    def test_no_changes_returns_empty_changed(self, client):
        from jarvis.config import Settings
        s = MagicMock(spec=Settings)
        s.model_fields = {"chat_rate_limit": None}
        s.chat_rate_limit = "30/minute"
        s.auth_enabled = False
        s.jwt_secret = "secret"
        s.reports_dir = MagicMock()

        with patch("jarvis.api.server.get_settings", return_value=s):
            resp = client.post("/api/config/reload")

        assert resp.json()["changed"] == {}

    def test_auth_dependency_updated_on_reload(self, client):
        import jarvis.api.server as _s
        _s._require_auth = None
        fake = _fake_settings(auth_enabled=True, jwt_secret="new-secret")
        mock_dep = MagicMock()
        with patch("jarvis.api.server.get_settings", return_value=fake), \
             patch("jarvis.auth.core.make_auth_dependency", return_value=mock_dep):
            client.post("/api/config/reload")
        assert _s._require_auth is mock_dep

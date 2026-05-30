"""Tests for plugin hot-reload API endpoints and plugin_loader runtime management."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


# ── plugin_loader unit tests ──────────────────────────────────────────────────

class TestPluginLoaderRuntime:
    def setup_method(self):
        from jarvis.tools import plugin_loader
        plugin_loader._disabled.clear()

    def test_disable_known_plugin(self):
        from jarvis.tools.plugin_loader import disable_plugin, list_plugin_info
        info = list_plugin_info()
        if not info:
            pytest.skip("no plugins available")
        tool_name = info[0]["tool_name"]
        result = disable_plugin(tool_name)
        assert result is True

    def test_disable_unknown_plugin_returns_false(self):
        from jarvis.tools.plugin_loader import disable_plugin
        assert disable_plugin("nonexistent_tool_xyz") is False

    def test_enable_not_disabled_returns_false(self):
        from jarvis.tools.plugin_loader import enable_plugin
        assert enable_plugin("not_disabled_tool") is False

    def test_enable_after_disable(self):
        from jarvis.tools.plugin_loader import disable_plugin, enable_plugin, list_plugin_info
        info = list_plugin_info()
        if not info:
            pytest.skip("no plugins available")
        tool_name = info[0]["tool_name"]
        disable_plugin(tool_name)
        result = enable_plugin(tool_name)
        assert result is True

    def test_disabled_plugin_skipped_on_load(self):
        from jarvis.tools.plugin_loader import disable_plugin, load_plugins, list_plugin_info
        info = list_plugin_info()
        if not info:
            pytest.skip("no plugins available")
        tool_name = info[0]["tool_name"]
        disable_plugin(tool_name)
        schemas, registry = load_plugins()
        tool_names_loaded = [s["name"] for s in schemas]
        assert tool_name not in tool_names_loaded
        assert tool_name not in registry

    def test_list_plugin_info_returns_list_of_dicts(self):
        from jarvis.tools.plugin_loader import list_plugin_info
        info = list_plugin_info()
        assert isinstance(info, list)
        for entry in info:
            assert "module" in entry
            assert "enabled" in entry

    def test_reload_plugins_returns_schemas_and_registry(self):
        from jarvis.tools.plugin_loader import reload_plugins
        schemas, registry = reload_plugins()
        assert isinstance(schemas, list)
        assert isinstance(registry, dict)
        for schema in schemas:
            assert schema["name"] in registry


# ── API endpoint tests ────────────────────────────────────────────────────────

class TestPluginApiEndpoints:
    def setup_method(self):
        from jarvis.tools import plugin_loader
        plugin_loader._disabled.clear()

    def test_list_plugins_returns_200(self, client):
        resp = client.get("/api/plugins")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_plugins_contains_enabled_field(self, client):
        resp = client.get("/api/plugins")
        assert resp.status_code == 200
        for item in resp.json():
            assert "enabled" in item
            assert "module" in item

    def test_reload_returns_tool_list(self, client):
        resp = client.post("/api/plugins/reload")
        assert resp.status_code == 200
        body = resp.json()
        assert "reloaded" in body
        assert "tools" in body
        assert isinstance(body["tools"], list)

    def test_disable_known_plugin(self, client):
        resp_list = client.get("/api/plugins")
        plugins = [p for p in resp_list.json() if p.get("tool_name")]
        if not plugins:
            pytest.skip("no plugins available")
        tool_name = plugins[0]["tool_name"]
        resp = client.post(f"/api/plugins/{tool_name}/disable")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_disable_unknown_plugin_returns_404(self, client):
        resp = client.post("/api/plugins/totally_fake_tool_xyz/disable")
        assert resp.status_code == 404

    def test_enable_unknown_plugin_returns_404(self, client):
        resp = client.post("/api/plugins/totally_fake_tool_xyz/enable")
        assert resp.status_code == 404

    def test_enable_previously_disabled_plugin(self, client):
        resp_list = client.get("/api/plugins")
        plugins = [p for p in resp_list.json() if p.get("tool_name")]
        if not plugins:
            pytest.skip("no plugins available")
        tool_name = plugins[0]["tool_name"]
        client.post(f"/api/plugins/{tool_name}/disable")
        resp = client.post(f"/api/plugins/{tool_name}/enable")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True

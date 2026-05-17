"""Tests for the Digital Twin tools: schemas, handlers, and registry registration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.twin.main import (
    QUERY_SCHEMA,
    SNAPSHOT_SCHEMA,
    handle_query_system_twin,
    handle_snapshot_system,
)


# ── Schema structure ──────────────────────────────────────────────────────────


def test_snapshot_schema_has_required_fields() -> None:
    for key in ("name", "description", "input_schema"):
        assert key in SNAPSHOT_SCHEMA
    assert SNAPSHOT_SCHEMA["name"] == "snapshot_system"


def test_query_schema_has_required_fields() -> None:
    for key in ("name", "description", "input_schema"):
        assert key in QUERY_SCHEMA
    assert QUERY_SCHEMA["name"] == "query_system_twin"


# ── Handlers ──────────────────────────────────────────────────────────────────


def test_snapshot_system_handler_returns_string(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    with patch("jarvis.twin.main.take_snapshot", return_value=5):
        result = handle_snapshot_system({}, db)
    assert isinstance(result, str)
    assert "5" in result
    assert "snapshot" in result.lower()


def test_snapshot_system_handler_returns_error_on_failure(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    with patch("jarvis.twin.main.take_snapshot", side_effect=RuntimeError("psutil broken")):
        result = handle_snapshot_system({}, db)
    assert result.startswith("ERROR:")
    assert "psutil broken" in result


def test_query_system_twin_returns_string_when_no_data(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    # Empty DB → no entities → should mention running snapshot first
    result = handle_query_system_twin({}, db)
    assert isinstance(result, str)
    assert "snapshot" in result.lower() or "No" in result


def test_query_system_twin_uses_entity_param(tmp_path: Path) -> None:
    db = tmp_path / "jarvis.db"
    with patch("jarvis.memory.graph.handle_query_knowledge_graph", return_value="port info") as mock_q:
        handle_query_system_twin({"entity": "port:tcp:8000"}, db)
    mock_q.assert_called_once()
    assert mock_q.call_args[0][0]["entity"] == "port:tcp:8000"


# ── Registry registration ─────────────────────────────────────────────────────


def test_twin_tools_registered_in_build_registry(tmp_path: Path) -> None:
    from jarvis.tools.registry import build_registry

    schemas, registry = build_registry(reports_dir=tmp_path)
    schema_names = {s["name"] for s in schemas}
    assert "snapshot_system" in schema_names
    assert "query_system_twin" in schema_names
    assert "snapshot_system" in registry
    assert "query_system_twin" in registry

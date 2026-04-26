"""Auto-discover and load JARVIS tool plugins from tools/plugins/.

Plugin contract:
  - Each .py file in tools/plugins/ (not starting with _) is a plugin.
  - Must export: SCHEMA (dict) and handle(tool_input: dict) -> str
  - Plugin name is taken from SCHEMA["name"].
  - Plugins call get_settings() internally for any config they need.
  - Plugins must never raise exceptions — return "ERROR: ..." strings.

Example plugin layout:
    SCHEMA = {"name": "my_tool", "description": "...", "input_schema": {...}}
    def handle(tool_input: dict) -> str: ...
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import structlog

log = structlog.get_logger()

_PLUGINS_PACKAGE = "jarvis.tools.plugins"
_PLUGINS_DIR = Path(__file__).parent / "plugins"


def load_plugins() -> tuple[list[dict], dict[str, callable]]:
    """Discover all plugins in tools/plugins/ and return (schemas, dispatch_map).

    Silently skips any plugin that fails to import or is missing required exports.
    """
    schemas: list[dict] = []
    registry: dict[str, callable] = {}

    for module_info in pkgutil.iter_modules([str(_PLUGINS_DIR)]):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{_PLUGINS_PACKAGE}.{module_info.name}")
            if not hasattr(module, "SCHEMA") or not hasattr(module, "handle"):
                log.warning("plugin_skipped", plugin=module_info.name, reason="missing SCHEMA or handle")
                continue
            tool_name = module.SCHEMA["name"]
            schemas.append(module.SCHEMA)
            registry[tool_name] = module.handle
            log.info("plugin_loaded", plugin=module_info.name, tool=tool_name)
        except Exception as exc:
            log.warning("plugin_load_failed", plugin=module_info.name, error=str(exc))

    return schemas, registry

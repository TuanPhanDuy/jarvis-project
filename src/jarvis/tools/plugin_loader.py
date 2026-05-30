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

# Runtime state: tracks disabled plugin names
_disabled: set[str] = set()
# Maps tool_name → module_name for hot-reload
_tool_to_module: dict[str, str] = {}


def load_plugins() -> tuple[list[dict], dict[str, callable]]:
    """Discover all plugins in tools/plugins/ and return (schemas, dispatch_map).

    Silently skips any plugin that fails to import or is missing required exports.
    Respects the disabled set populated via disable_plugin().
    """
    schemas: list[dict] = []
    registry: dict[str, callable] = {}
    _tool_to_module.clear()

    for module_info in pkgutil.iter_modules([str(_PLUGINS_DIR)]):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{_PLUGINS_PACKAGE}.{module_info.name}")
            if not hasattr(module, "SCHEMA") or not hasattr(module, "handle"):
                log.warning("plugin_skipped", plugin=module_info.name, reason="missing SCHEMA or handle")
                continue
            tool_name = module.SCHEMA["name"]
            _required = {"name", "description", "input_schema"}
            missing = _required - set(module.SCHEMA.keys())
            if missing:
                log.warning("plugin_schema_invalid", plugin=module_info.name, missing=sorted(missing))
                continue
            if tool_name in registry:
                log.warning(
                    "plugin_name_collision",
                    plugin=module_info.name,
                    tool=tool_name,
                    reason="tool name already registered — skipping duplicate",
                )
                continue
            if tool_name in _disabled:
                log.info("plugin_disabled_skipped", plugin=module_info.name, tool=tool_name)
                continue
            _tool_to_module[tool_name] = module_info.name
            schemas.append(module.SCHEMA)
            registry[tool_name] = module.handle
            log.info("plugin_loaded", plugin=module_info.name, tool=tool_name)
        except Exception as exc:
            log.warning("plugin_load_failed", plugin=module_info.name, error=str(exc))

    return schemas, registry


def reload_plugins() -> tuple[list[dict], dict[str, callable]]:
    """Force-reimport every plugin module and reload the dispatch map."""
    for module_info in pkgutil.iter_modules([str(_PLUGINS_DIR)]):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{_PLUGINS_PACKAGE}.{module_info.name}"
        try:
            import sys
            if full_name in sys.modules:
                importlib.reload(sys.modules[full_name])
        except Exception as exc:
            log.warning("plugin_reload_failed", plugin=module_info.name, error=str(exc))
    log.info("plugins_reloaded")
    return load_plugins()


def list_plugin_info() -> list[dict]:
    """Return metadata for every discovered plugin (including disabled ones)."""
    info: list[dict] = []
    for module_info in pkgutil.iter_modules([str(_PLUGINS_DIR)]):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{_PLUGINS_PACKAGE}.{module_info.name}"
        try:
            import sys
            mod = sys.modules.get(full_name) or importlib.import_module(full_name)
            if not hasattr(mod, "SCHEMA") or not hasattr(mod, "handle"):
                continue
            tool_name = mod.SCHEMA["name"]
            info.append({
                "module": module_info.name,
                "tool_name": tool_name,
                "description": mod.SCHEMA.get("description", ""),
                "enabled": tool_name not in _disabled,
            })
        except Exception:
            info.append({"module": module_info.name, "tool_name": None, "enabled": False, "error": "load_failed"})
    return info


def disable_plugin(tool_name: str) -> bool:
    """Disable a plugin by tool name. Returns True if the plugin was found."""
    known = {i["tool_name"] for i in list_plugin_info() if i.get("tool_name")}
    if tool_name not in known:
        return False
    _disabled.add(tool_name)
    log.info("plugin_disabled", tool=tool_name)
    return True


def enable_plugin(tool_name: str) -> bool:
    """Re-enable a previously disabled plugin. Returns True if it was disabled."""
    if tool_name not in _disabled:
        return False
    _disabled.discard(tool_name)
    log.info("plugin_enabled", tool=tool_name)
    return True

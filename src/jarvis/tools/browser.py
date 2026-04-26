"""Tool: browse — automate a Chromium browser to navigate, read, click, and type.

Uses Playwright in synchronous mode. Chromium must be installed separately:
    uv run playwright install chromium

Actions supported:
  navigate  — go to a URL and return the page text
  get_text  — return visible text of the current page
  click     — click an element matching a CSS selector
  type      — type text into an input element
  screenshot — take a screenshot and return its file path
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_browser = None
_page = None


def _get_page():
    global _browser, _page
    if _page is not None:
        return _page
    from playwright.sync_api import sync_playwright

    _pw = sync_playwright().start()
    _browser = _pw.chromium.launch(headless=True)
    _page = _browser.new_page()
    return _page


def handle_browse(tool_input: dict, screenshots_dir: Path | None = None) -> str:
    try:
        action = tool_input.get("action", "navigate")
        page = _get_page()

        if action == "navigate":
            url = tool_input["url"]
            page.goto(url, timeout=20000)
            page.wait_for_load_state("domcontentloaded")
            text = page.inner_text("body")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            cleaned = "\n".join(lines)
            max_chars = int(tool_input.get("max_chars", 6000))
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars] + f"\n[...truncated at {max_chars} chars]"
            return cleaned or "(page had no readable text)"

        elif action == "get_text":
            text = page.inner_text("body")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return "\n".join(lines)[:6000]

        elif action == "click":
            selector = tool_input["selector"]
            page.click(selector, timeout=5000)
            return f"Clicked: {selector}"

        elif action == "type":
            selector = tool_input["selector"]
            text = tool_input["text"]
            page.fill(selector, text)
            return f"Typed into {selector}: {text!r}"

        elif action == "screenshot":
            if screenshots_dir is None:
                screenshots_dir = Path("reports")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            path = screenshots_dir / "screenshot.png"
            page.screenshot(path=str(path))
            return f"Screenshot saved to: {path}"

        else:
            return f"ERROR: unknown browser action '{action}'"

    except Exception as e:
        return f"ERROR: browse failed — {e}"


SCHEMA: dict = {
    "name": "browse",
    "description": (
        "Control a headless Chromium browser. Can navigate to URLs, read page text, "
        "click elements, type into forms, and take screenshots. "
        "Requires `playwright install chromium` to be run once before use."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "get_text", "click", "type", "screenshot"],
                "description": "Browser action to perform.",
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (required for 'navigate' action).",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for the target element (required for 'click' and 'type').",
            },
            "text": {
                "type": "string",
                "description": "Text to type into the element (required for 'type' action).",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters of page text to return for 'navigate'/'get_text' (default 6000).",
                "default": 6000,
            },
        },
        "required": ["action"],
    },
}

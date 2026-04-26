"""Plugin: execute_python — run Python code snippets in a sandboxed subprocess.

Security:
  - 10-second wall-clock timeout (configurable up to 30s)
  - Runs in a temporary directory (no access to project files by path)
  - stdout + stderr captured and truncated to 2000 chars
  - No network isolation (OS-level sandboxing would require Docker/nsjail)

Usage: only available when explicitly wired into the agent's tool registry.
Claude should only call this for computing, data transformation, or quick
experiments — never for side effects that could affect external systems.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

MAX_OUTPUT = 2000
MAX_TIMEOUT = 30


def handle(tool_input: dict) -> str:
    code = tool_input.get("code", "").strip()
    if not code:
        return "ERROR: no code provided."

    timeout = min(int(tool_input.get("timeout", 10)), MAX_TIMEOUT)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "script.py"
            script.write_text(code, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )

        output = result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr

        output = output[:MAX_OUTPUT]
        if not output:
            output = "(no output)"

        if result.returncode != 0:
            return f"Exit code {result.returncode}:\n{output}"
        return output

    except subprocess.TimeoutExpired:
        return f"ERROR: execution timed out after {timeout}s."
    except Exception as e:
        return f"ERROR: execute_python failed — {e}"


SCHEMA: dict = {
    "name": "execute_python",
    "description": (
        "Execute a Python code snippet in a sandboxed subprocess and return stdout/stderr. "
        "Use for calculations, data processing, or quick experiments. "
        "Do NOT use for side effects on external systems. Max timeout: 30s."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max execution time in seconds (default 10, max 30).",
                "default": 10,
            },
        },
        "required": ["code"],
    },
}

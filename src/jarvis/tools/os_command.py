"""Tool: run_command — execute an allowlisted shell command and return its output.

Security: only commands whose base name appears in the allowlist can run.
This prevents arbitrary command injection while still letting JARVIS interact
with the local system (list files, run scripts, check git status, etc.).
"""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass


@dataclass
class CommandInput:
    command: str
    timeout: int = 15


def handle_run_command(tool_input: dict, allowed_commands: list[str]) -> str:
    try:
        inp = CommandInput(
            command=tool_input["command"],
            timeout=int(tool_input.get("timeout", 15)),
        )

        # Parse command to extract the base executable name
        try:
            parts = shlex.split(inp.command)
        except ValueError as e:
            return f"ERROR: invalid command syntax — {e}"

        if not parts:
            return "ERROR: empty command"

        base = parts[0].split("/")[-1].split("\\")[-1]  # basename on both OS
        if base not in allowed_commands:
            return (
                f"ERROR: '{base}' is not in the allowed commands list. "
                f"Allowed: {', '.join(sorted(allowed_commands))}"
            )

        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=inp.timeout,
        )

        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return (
                f"Command exited with code {result.returncode}.\n"
                + (f"stdout:\n{output}\n" if output else "")
                + (f"stderr:\n{stderr}" if stderr else "")
            ).strip()

        return output or "(command produced no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {tool_input.get('timeout', 15)}s"
    except Exception as e:
        return f"ERROR: run_command failed — {e}"


SCHEMA: dict = {
    "name": "run_command",
    "description": (
        "Run an allowlisted shell command on the local system and return its output. "
        "Useful for listing files, checking git status, running Python scripts, or "
        "inspecting the environment. Only pre-approved commands are permitted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run, e.g. 'git log --oneline -5' or 'ls reports/'.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum seconds to wait for the command (default 15).",
                "default": 15,
            },
        },
        "required": ["command"],
    },
}

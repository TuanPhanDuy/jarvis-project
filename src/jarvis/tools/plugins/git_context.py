"""Plugin: git_context — query git history, status, and diffs for any repo.

Personal data connector: gives JARVIS awareness of your codebase state.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


_ALLOWED_OPS = {"log", "diff", "status", "show", "blame", "branch", "tag"}


def handle(tool_input: dict) -> str:
    try:
        operation = tool_input.get("operation", "status").strip().lower()
        repo_path = tool_input.get("repo_path", ".").strip()
        args = tool_input.get("args", "").strip()

        if operation not in _ALLOWED_OPS:
            return f"ERROR: operation must be one of: {', '.join(sorted(_ALLOWED_OPS))}."

        repo = Path(repo_path).expanduser().resolve()
        if not (repo / ".git").exists() and not (repo.parent / ".git").exists():
            return f"ERROR: '{repo}' does not appear to be a git repository."

        cmd = ["git", "-C", str(repo), operation]
        if args:
            cmd.extend(args.split())

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )

        output = (result.stdout + result.stderr).strip()
        if not output:
            return f"git {operation}: no output."
        return output[:4000]  # truncate very long diffs

    except subprocess.TimeoutExpired:
        return "ERROR: git command timed out."
    except FileNotFoundError:
        return "ERROR: git is not installed or not in PATH."
    except Exception as e:
        return f"ERROR: git_context failed — {e}"


SCHEMA: dict = {
    "name": "git_context",
    "description": (
        "Run git commands (log, diff, status, show, blame, branch, tag) on a repository. "
        "Use this to understand recent changes, history, or the current state of a codebase."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Git sub-command: log | diff | status | show | blame | branch | tag",
                "default": "status",
            },
            "repo_path": {
                "type": "string",
                "description": "Path to the git repository (default: current directory).",
                "default": ".",
            },
            "args": {
                "type": "string",
                "description": "Additional arguments, e.g. '--oneline -10' for log, 'HEAD~1' for diff.",
            },
        },
        "required": ["operation"],
    },
}

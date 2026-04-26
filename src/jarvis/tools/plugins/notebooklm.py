"""Locate JARVIS research reports for NotebookLM upload via the /notebooklm Claude Code skill."""
from pathlib import Path
import os

DRIVE_FOLDER_NAME = "JARVIS-NotebookLM"
DRIVE_FOLDER_ID = "1rVNDpi-y6vwxsZnv9UZ1hXN_z3iglDfK"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
NOTEBOOKLM_URL = "https://notebooklm.google.com"

SCHEMA = {
    "name": "push_to_notebooklm",
    "description": (
        "Find the latest (or a specific) JARVIS research report and return its path "
        "for upload to the JARVIS-NotebookLM Google Drive folder. "
        "The actual upload is handled by the /notebooklm Claude Code skill."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "report_path": {
                "type": "string",
                "description": "Path to a specific report. Omit to use the most recently modified .md in reports/.",
            },
            "list_all": {
                "type": "boolean",
                "description": "If true, list all available reports instead of picking the latest.",
            },
        },
        "required": [],
    },
}


def handle(tool_input: dict) -> str:
    try:
        reports_dir = Path(os.getenv("JARVIS_REPORTS_DIR", "reports"))

        if tool_input.get("list_all"):
            candidates = sorted(
                reports_dir.glob("*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return f"No reports found in {reports_dir.resolve()}."
            lines = "\n".join(f"  {p.name}" for p in candidates)
            return f"Available reports in {reports_dir.resolve()}:\n{lines}"

        if "report_path" in tool_input and tool_input["report_path"]:
            report = Path(tool_input["report_path"])
            if not report.exists():
                return f"ERROR: File not found: {report}"
        else:
            candidates = sorted(
                reports_dir.glob("*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return (
                    f"ERROR: No .md reports in {reports_dir.resolve()}. "
                    f"Run `uv run jarvis --topic \"...\"` first."
                )
            report = candidates[0]

        return (
            f"Report: {report.name}\n"
            f"Path: {report.resolve()}\n"
            f"Drive folder: {DRIVE_FOLDER_NAME} ({DRIVE_FOLDER_URL})\n"
            f"Action: Run `/notebooklm` in Claude Code to upload and sync with NotebookLM."
        )
    except Exception as e:
        return f"ERROR: {e}"

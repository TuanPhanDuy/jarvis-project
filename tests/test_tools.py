"""Unit tests for tool handlers. No API keys needed — tools are called directly."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jarvis.tools.report_writer import handle_save_report, handle_update_report
from jarvis.tools.web_search import SCHEMA as WEB_SEARCH_SCHEMA
from jarvis.tools.report_writer import SCHEMA as SAVE_REPORT_SCHEMA
from jarvis.tools.os_command import handle_run_command, SCHEMA as OS_COMMAND_SCHEMA


class TestReportWriter:
    def test_saves_file(self, tmp_path: Path) -> None:
        result = handle_save_report(
            {"title": "Test Report", "content": "## Intro\nHello world.", "topic": "test"},
            reports_dir=tmp_path,
        )
        assert "saved to" in result
        files = list(tmp_path.glob("*.md"))
        assert len(files) == 1
        text = files[0].read_text()
        assert "Test Report" in text
        assert "Hello world." in text

    def test_slug_in_filename(self, tmp_path: Path) -> None:
        handle_save_report(
            {"title": "RLHF Overview", "content": "content", "topic": "rlhf overview"},
            reports_dir=tmp_path,
        )
        files = list(tmp_path.glob("*.md"))
        assert any("rlhf-overview" in f.name for f in files)

    def test_creates_reports_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        result = handle_save_report(
            {"title": "T", "content": "c", "topic": "t"},
            reports_dir=nested,
        )
        assert "ERROR" not in result
        assert nested.exists()

    def test_returns_error_string_on_bad_input(self, tmp_path: Path) -> None:
        result = handle_save_report({}, reports_dir=tmp_path)
        assert result.startswith("ERROR:")


class TestUpdateReport:
    def test_appends_section_to_existing_report(self, tmp_path: Path) -> None:
        handle_save_report(
            {"title": "Original", "content": "Body.", "topic": "orig"},
            reports_dir=tmp_path,
        )
        fname = next(tmp_path.glob("*.md")).name
        result = handle_update_report(
            {"filename": fname, "section_title": "New Section", "section_content": "Extra content."},
            reports_dir=tmp_path,
        )
        assert "ERROR" not in result
        text = (tmp_path / fname).read_text()
        assert "New Section" in text
        assert "Extra content." in text

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        result = handle_update_report(
            {"filename": "nonexistent.md", "section_title": "S", "section_content": "C"},
            reports_dir=tmp_path,
        )
        assert result.startswith("ERROR:")

    def test_missing_keys_returns_error(self, tmp_path: Path) -> None:
        result = handle_update_report({}, reports_dir=tmp_path)
        assert result.startswith("ERROR:")


class TestOSCommand:
    _ALLOWED = ["echo", "python", "python3", "git"]

    def test_allowlisted_command_runs(self) -> None:
        result = handle_run_command({"command": "echo hello"}, self._ALLOWED)
        assert "hello" in result
        assert "ERROR" not in result

    def test_disallowed_command_blocked(self) -> None:
        result = handle_run_command({"command": "rm -rf /"}, self._ALLOWED)
        assert result.startswith("ERROR:")
        assert "not in the allowed commands list" in result

    def test_path_traversal_basename_blocked(self) -> None:
        # /usr/bin/rm → basename is "rm" which is not allowed
        result = handle_run_command({"command": "/usr/bin/rm -rf /"}, self._ALLOWED)
        assert result.startswith("ERROR:")

    def test_empty_command_returns_error(self) -> None:
        result = handle_run_command({"command": ""}, self._ALLOWED)
        assert result.startswith("ERROR:")

    def test_missing_command_key_returns_error(self) -> None:
        result = handle_run_command({}, self._ALLOWED)
        assert result.startswith("ERROR:")

    def test_python_echo_output(self) -> None:
        python_exe = "python" if sys.platform == "win32" else "python3"
        if python_exe not in self._ALLOWED:
            allowed = self._ALLOWED + [python_exe]
        else:
            allowed = self._ALLOWED
        result = handle_run_command(
            {"command": f'{python_exe} -c "print(42)"'},
            allowed,
        )
        assert "42" in result

    def test_os_command_schema_valid(self) -> None:
        assert OS_COMMAND_SCHEMA["name"] == "run_command"
        assert "command" in OS_COMMAND_SCHEMA["input_schema"]["required"]


class TestCodeExecutorPlugin:
    def test_basic_execution(self) -> None:
        from jarvis.tools.plugins.code_executor import handle
        result = handle({"code": "print(1 + 1)"})
        assert "2" in result

    def test_syntax_error_returns_error(self) -> None:
        from jarvis.tools.plugins.code_executor import handle
        result = handle({"code": "def broken(:"})
        assert "1" in result or "SyntaxError" in result or "error" in result.lower()

    def test_empty_code_returns_error(self) -> None:
        from jarvis.tools.plugins.code_executor import handle
        result = handle({"code": ""})
        assert "ERROR" in result

    def test_timeout_respected(self) -> None:
        from jarvis.tools.plugins.code_executor import handle
        result = handle({"code": "import time; time.sleep(999)", "timeout": 1})
        assert "timed out" in result.lower()

    def test_stdout_captured(self) -> None:
        from jarvis.tools.plugins.code_executor import handle
        result = handle({"code": "for i in range(3): print(i)"})
        assert "0" in result
        assert "1" in result
        assert "2" in result

    def test_schema_valid(self) -> None:
        from jarvis.tools.plugins.code_executor import SCHEMA
        assert SCHEMA["name"] == "execute_python"
        assert "code" in SCHEMA["input_schema"]["required"]


class TestToolSchemas:
    def test_web_search_schema_has_required_fields(self) -> None:
        assert WEB_SEARCH_SCHEMA["name"] == "web_search"
        assert "query" in WEB_SEARCH_SCHEMA["input_schema"]["required"]

    def test_save_report_schema_has_required_fields(self) -> None:
        assert SAVE_REPORT_SCHEMA["name"] == "save_report"
        required = SAVE_REPORT_SCHEMA["input_schema"]["required"]
        assert "title" in required
        assert "content" in required
        assert "topic" in required

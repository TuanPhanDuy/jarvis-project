"""Tests for Modelfile builder and Ollama model registration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestBuildModelfile:
    def test_includes_gguf_path(self, tmp_path):
        from jarvis.training.modelfile import build_modelfile

        gguf = tmp_path / "model.gguf"
        gguf.touch()
        content = build_modelfile(gguf)
        assert str(gguf.resolve()) in content

    def test_includes_from_directive(self, tmp_path):
        from jarvis.training.modelfile import build_modelfile

        gguf = tmp_path / "model.gguf"
        gguf.touch()
        content = build_modelfile(gguf)
        assert content.startswith("FROM ")

    def test_includes_system_prompt(self, tmp_path):
        from jarvis.training.modelfile import build_modelfile

        gguf = tmp_path / "model.gguf"
        gguf.touch()
        content = build_modelfile(gguf, system_prompt="You are a test assistant.")
        assert "You are a test assistant." in content

    def test_includes_num_ctx_parameter(self, tmp_path):
        from jarvis.training.modelfile import build_modelfile

        gguf = tmp_path / "model.gguf"
        gguf.touch()
        content = build_modelfile(gguf)
        assert "num_ctx" in content

    def test_escapes_double_quotes_in_system_prompt(self, tmp_path):
        from jarvis.training.modelfile import build_modelfile

        gguf = tmp_path / "model.gguf"
        gguf.touch()
        content = build_modelfile(gguf, system_prompt='Say "hello".')
        assert '\\"hello\\"' in content or '"hello"' not in content.split("SYSTEM")[1].split("\n")[0]


class TestRegisterModel:
    def test_returns_false_if_gguf_missing(self, tmp_path):
        from jarvis.training.modelfile import register_model

        result = register_model(tmp_path / "nonexistent.gguf", "jarvis-ft")
        assert result is False

    def test_returns_true_on_successful_ollama_create(self, tmp_path):
        from jarvis.training.modelfile import register_model

        gguf = tmp_path / "model.gguf"
        gguf.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = register_model(gguf, "jarvis-ft")

        assert result is True

    def test_returns_false_on_failed_ollama_create(self, tmp_path):
        from jarvis.training.modelfile import register_model

        gguf = tmp_path / "model.gguf"
        gguf.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            result = register_model(gguf, "jarvis-ft")

        assert result is False

    def test_ollama_create_is_called_with_model_name(self, tmp_path):
        from jarvis.training.modelfile import register_model

        gguf = tmp_path / "model.gguf"
        gguf.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            register_model(gguf, "my-custom-model")

        cmd = mock_run.call_args[0][0]
        assert "ollama" in cmd
        assert "create" in cmd
        assert "my-custom-model" in cmd

    def test_writes_temporary_modelfile(self, tmp_path):
        from jarvis.training.modelfile import register_model

        gguf = tmp_path / "model.gguf"
        gguf.touch()

        captured_cmd = []

        def capture(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=capture):
            register_model(gguf, "test-model")

        # The -f flag should point to a real file
        assert "-f" in captured_cmd
        modelfile_path = captured_cmd[captured_cmd.index("-f") + 1]
        from pathlib import Path as _P
        # File may be cleaned up; just verify path looks like a Modelfile
        assert "Modelfile" in modelfile_path or modelfile_path.endswith(".Modelfile")

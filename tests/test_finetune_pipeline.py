"""Tests for Finetuner (mlx-lm orchestration)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch


class TestFinetuner:
    def _finetuner(self, tmp_path: Path):
        from jarvis.training.finetune import Finetuner
        return Finetuner(
            base_model="mlx-community/TestModel-4bit",
            adapter_dir=tmp_path / "adapters",
        )

    def test_train_creates_adapter_dir(self, tmp_path):
        ft = self._finetuner(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ft.train(data_dir, epochs=1, lora_rank=8)

        assert (tmp_path / "adapters").exists()

    def test_train_invokes_mlx_lm_lora(self, tmp_path):
        ft = self._finetuner(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ft.train(data_dir, epochs=2, lora_rank=16)

        cmd = mock_run.call_args[0][0]
        assert "mlx_lm.lora" in " ".join(cmd)
        assert "--train" in cmd
        assert str(data_dir) in cmd
        assert "--lora-layers" in cmd
        assert "16" in cmd

    def test_train_raises_on_nonzero_exit(self, tmp_path):
        import pytest
        ft = self._finetuner(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            with pytest.raises(RuntimeError, match="mlx_lm.lora"):
                ft.train(data_dir, epochs=1)

    def test_export_gguf_invokes_mlx_lm_fuse(self, tmp_path):
        ft = self._finetuner(tmp_path)
        output_path = tmp_path / "model.gguf"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ft.export_gguf(output_path)

        cmd = mock_run.call_args[0][0]
        assert "mlx_lm.fuse" in " ".join(cmd)
        assert "--export-gguf" in cmd
        assert str(output_path) in cmd

    def test_export_gguf_creates_parent_dir(self, tmp_path):
        ft = self._finetuner(tmp_path)
        nested_output = tmp_path / "subdir" / "nested" / "model.gguf"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ft.export_gguf(nested_output)

        assert nested_output.parent.exists()

    def test_export_gguf_raises_on_nonzero_exit(self, tmp_path):
        import pytest
        ft = self._finetuner(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=2)
            with pytest.raises(RuntimeError, match="mlx_lm.fuse"):
                ft.export_gguf(tmp_path / "model.gguf")

    def test_adapter_dir_path_passed_to_train(self, tmp_path):
        ft = self._finetuner(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ft.train(data_dir, epochs=1)

        cmd = mock_run.call_args[0][0]
        assert str(tmp_path / "adapters") in cmd

"""Tests for pvtt.core.model_manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pvtt.core.model_manager import MODEL_REPO_MAP, ModelManager
from pvtt.exceptions import ModelDownloadError, ModelNotFoundError


class TestModelManagerListModels:
    """Tests for ModelManager.list_models."""

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        manager = ModelManager(models_dir=tmp_path)

        models = manager.list_models()

        assert models == []

    def test_finds_valid_model_directories(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "tiny.en"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"\x00" * 100)
        (model_dir / "config.json").write_text("{}")

        manager = ModelManager(models_dir=tmp_path)
        models = manager.list_models()

        assert len(models) == 1
        assert models[0].name == "tiny.en"
        assert models[0].path == model_dir
        assert models[0].size_bytes > 0

    def test_ignores_directories_without_model_bin(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-model").mkdir()
        (tmp_path / "not-a-model" / "readme.txt").write_text("hi")

        manager = ModelManager(models_dir=tmp_path)
        models = manager.list_models()

        assert models == []

    def test_multiple_models_sorted(self, tmp_path: Path) -> None:
        for name in ("base.en", "tiny.en"):
            d = tmp_path / name
            d.mkdir()
            (d / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        models = manager.list_models()

        assert len(models) == 2
        assert models[0].name == "base.en"
        assert models[1].name == "tiny.en"


class TestModelManagerGetModelPath:
    """Tests for ModelManager.get_model_path."""

    def test_finds_model_by_short_name(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "tiny.en"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        path = manager.get_model_path("tiny.en")

        assert path == model_dir

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        manager = ModelManager(models_dir=tmp_path)

        with pytest.raises(ModelNotFoundError, match="not found locally"):
            manager.get_model_path("tiny.en")

    def test_finds_model_with_slash_name(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "Systran--faster-whisper-tiny.en"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        path = manager.get_model_path("Systran/faster-whisper-tiny.en")

        assert path == model_dir

    def test_accepts_direct_path(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "custom_model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        path = manager.get_model_path(str(model_dir))

        assert path == model_dir


class TestModelManagerRemoveModel:
    """Tests for ModelManager.remove_model."""

    def test_removes_model_directory(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "tiny.en"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        manager.remove_model("tiny.en")

        assert not model_dir.exists()

    def test_remove_missing_model_raises(self, tmp_path: Path) -> None:
        manager = ModelManager(models_dir=tmp_path)

        with pytest.raises(ModelNotFoundError):
            manager.remove_model("nonexistent")


class TestModelManagerGetModelInfo:
    """Tests for ModelManager.get_model_info."""

    def test_returns_model_info(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "tiny.en"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"\x00" * 1024)

        manager = ModelManager(models_dir=tmp_path)
        info = manager.get_model_info("tiny.en")

        assert info.name == "tiny.en"
        assert info.path == model_dir
        assert info.size_bytes == 1024


class TestModelManagerDownload:
    """Tests for ModelManager.download."""

    def test_download_calls_snapshot_download(self, tmp_path: Path) -> None:
        manager = ModelManager(models_dir=tmp_path)
        expected_dir = tmp_path / "tiny.en"

        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(expected_dir)
            path = manager.download("tiny.en")

        mock_dl.assert_called_once_with(
            "Systran/faster-whisper-tiny.en",
            local_dir=str(expected_dir),
        )
        assert path == expected_dir

    def test_download_failure_raises(self, tmp_path: Path) -> None:
        manager = ModelManager(models_dir=tmp_path)

        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("Network error"),
        ), pytest.raises(ModelDownloadError, match="Network error"):
            manager.download("tiny.en")

    def test_repo_map_has_common_models(self) -> None:
        expected = {"tiny", "tiny.en", "base", "base.en", "small", "small.en",
                    "medium", "medium.en", "large-v3-turbo"}

        assert expected.issubset(set(MODEL_REPO_MAP.keys()))


class TestModelManagerResolveModelName:
    """Tests for ModelManager.resolve_model_name."""

    def test_prefers_large_v3_turbo_when_installed(self, tmp_path: Path) -> None:
        for name in ("tiny.en", "large-v3-turbo"):
            d = tmp_path / name
            d.mkdir()
            (d / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        result = manager.resolve_model_name()

        assert result == "large-v3-turbo"

    def test_falls_back_to_first_model_when_turbo_missing(self, tmp_path: Path) -> None:
        d = tmp_path / "tiny.en"
        d.mkdir()
        (d / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        result = manager.resolve_model_name()

        assert result == "tiny.en"

    def test_falls_back_to_first_sorted_model(self, tmp_path: Path) -> None:
        for name in ("small.en", "base.en"):
            d = tmp_path / name
            d.mkdir()
            (d / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        result = manager.resolve_model_name()

        assert result == "base.en"  # alphabetically first

    def test_no_models_installed_raises(self, tmp_path: Path) -> None:
        manager = ModelManager(models_dir=tmp_path)

        with pytest.raises(ModelNotFoundError, match="No models installed"):
            manager.resolve_model_name()

    def test_custom_preferred_model(self, tmp_path: Path) -> None:
        d = tmp_path / "medium.en"
        d.mkdir()
        (d / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        result = manager.resolve_model_name(preferred="medium.en")

        assert result == "medium.en"

    def test_ignores_invalid_dirs_during_fallback(self, tmp_path: Path) -> None:
        # Directory without model.bin should be ignored
        (tmp_path / "broken-model").mkdir()
        (tmp_path / "broken-model" / "readme.txt").write_text("hi")

        d = tmp_path / "tiny.en"
        d.mkdir()
        (d / "model.bin").write_bytes(b"\x00")

        manager = ModelManager(models_dir=tmp_path)
        result = manager.resolve_model_name()

        assert result == "tiny.en"

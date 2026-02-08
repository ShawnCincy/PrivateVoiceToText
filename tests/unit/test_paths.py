"""Tests for pvtt.config.paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pvtt.config.paths import (
    get_config_path,
    get_data_dir,
    get_models_dir,
    get_profiles_dir,
)


class TestGetDataDir:
    """Tests for get_data_dir."""

    def test_returns_path_object(self) -> None:
        result = get_data_dir()

        assert isinstance(result, Path)

    def test_directory_exists(self) -> None:
        result = get_data_dir()

        assert result.is_dir()

    def test_ends_with_pvtt(self) -> None:
        result = get_data_dir()

        assert result.name == "pvtt"

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        fake_dir = tmp_path / "test_pvtt"
        with patch("pvtt.config.paths.user_data_dir", return_value=str(fake_dir)):
            result = get_data_dir()

        assert result.is_dir()
        assert result == fake_dir


class TestGetConfigPath:
    """Tests for get_config_path."""

    def test_returns_toml_path(self) -> None:
        result = get_config_path()

        assert result.name == "config.toml"
        assert result.parent.name == "pvtt"


class TestGetModelsDir:
    """Tests for get_models_dir."""

    def test_returns_models_subdir(self) -> None:
        result = get_models_dir()

        assert result.name == "models"
        assert result.is_dir()

    def test_creates_directory(self, tmp_path: Path) -> None:
        fake_dir = tmp_path / "test_pvtt"
        with patch("pvtt.config.paths.user_data_dir", return_value=str(fake_dir)):
            result = get_models_dir()

        assert result.is_dir()
        assert result == fake_dir / "models"


class TestGetProfilesDir:
    """Tests for get_profiles_dir."""

    def test_returns_profiles_subdir(self) -> None:
        result = get_profiles_dir()

        assert result.name == "profiles"
        assert result.is_dir()

"""Tests for pvtt.config.loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pvtt.config.loader import deep_merge, load_config, load_env_overrides, load_toml
from pvtt.exceptions import ConfigError


class TestLoadToml:
    """Tests for load_toml."""

    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        result = load_toml(tmp_path / "nonexistent.toml")

        assert result == {}

    def test_valid_toml_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text('[model]\nname = "tiny.en"\n', encoding="utf-8")

        result = load_toml(toml_file)

        assert result == {"model": {"name": "tiny.en"}}

    def test_invalid_toml_raises_config_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "bad.toml"
        toml_file.write_bytes(b"[invalid\ngarbage")

        with pytest.raises(ConfigError, match="Failed to parse"):
            load_toml(toml_file)


class TestLoadEnvOverrides:
    """Tests for load_env_overrides."""

    def test_reads_pvtt_prefixed_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PVTT_MODEL_NAME", "tiny.en")

        result = load_env_overrides()

        assert result == {"model": {"name": "tiny.en"}}

    def test_ignores_non_pvtt_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTHER_VAR", "value")

        result = load_env_overrides()

        assert "other" not in result

    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PVTT_MODEL_DEVICE", "cpu")
        monkeypatch.setenv("PVTT_TRANSCRIPTION_LANGUAGE", "en")

        result = load_env_overrides()

        assert result["model"]["device"] == "cpu"
        assert result["transcription"]["language"] == "en"


class TestDeepMerge:
    """Tests for deep_merge."""

    def test_simple_override(self) -> None:
        base: dict[str, Any] = {"a": 1, "b": 2}
        override: dict[str, Any] = {"b": 3}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self) -> None:
        base: dict[str, Any] = {"model": {"name": "large-v3-turbo", "device": "auto"}}
        override: dict[str, Any] = {"model": {"name": "tiny.en"}}

        result = deep_merge(base, override)

        assert result == {"model": {"name": "tiny.en", "device": "auto"}}

    def test_new_keys_added(self) -> None:
        base: dict[str, Any] = {"a": 1}
        override: dict[str, Any] = {"b": 2}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self) -> None:
        base: dict[str, Any] = {"a": 1}
        override: dict[str, Any] = {"a": 2}

        deep_merge(base, override)

        assert base == {"a": 1}


class TestLoadConfig:
    """Tests for load_config."""

    def test_no_file_returns_defaults(self, tmp_path: Path) -> None:
        config = load_config(config_path=tmp_path / "nonexistent.toml")

        assert config.model.name == "auto"
        assert config.model.local_files_only is True

    def test_toml_overrides_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[model]\nname = "tiny.en"\ndevice = "cpu"\n',
            encoding="utf-8",
        )

        config = load_config(config_path=toml_file)

        assert config.model.name == "tiny.en"
        assert config.model.device == "cpu"
        # Unset fields keep defaults
        assert config.model.compute_type == "auto"

    def test_env_overrides_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text('[model]\nname = "tiny.en"\n', encoding="utf-8")
        monkeypatch.setenv("PVTT_MODEL_NAME", "base.en")

        config = load_config(config_path=toml_file)

        assert config.model.name == "base.en"

    def test_cli_overrides_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text('[model]\nname = "tiny.en"\n', encoding="utf-8")
        monkeypatch.setenv("PVTT_MODEL_NAME", "base.en")

        config = load_config(
            cli_overrides={"model": {"name": "small.en"}},
            config_path=toml_file,
        )

        assert config.model.name == "small.en"

    def test_invalid_config_raises_config_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[model]\nname = "not-a-real-model"\n',
            encoding="utf-8",
        )

        with pytest.raises(ConfigError, match="Configuration validation failed"):
            load_config(config_path=toml_file)

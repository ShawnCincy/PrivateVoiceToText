"""Tests for pvtt config set command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pvtt.cli.config_cmd import _clean_none_values, _coerce_value


class TestCoerceValue:
    """Tests for _coerce_value."""

    def test_true_values(self) -> None:
        assert _coerce_value("true") is True
        assert _coerce_value("True") is True
        assert _coerce_value("yes") is True
        assert _coerce_value("YES") is True

    def test_false_values(self) -> None:
        assert _coerce_value("false") is False
        assert _coerce_value("False") is False
        assert _coerce_value("no") is False
        assert _coerce_value("NO") is False

    def test_none_values(self) -> None:
        assert _coerce_value("none") is None
        assert _coerce_value("None") is None
        assert _coerce_value("null") is None
        assert _coerce_value("") is None

    def test_integer(self) -> None:
        assert _coerce_value("5") == 5
        assert _coerce_value("0") is False  # "0" matches boolean first
        assert _coerce_value("42") == 42
        assert _coerce_value("-1") == -1

    def test_float(self) -> None:
        assert _coerce_value("0.5") == 0.5
        assert _coerce_value("3.14") == 3.14
        assert _coerce_value("-0.1") == -0.1

    def test_string_fallback(self) -> None:
        assert _coerce_value("tiny.en") == "tiny.en"
        assert _coerce_value("hello world") == "hello world"
        assert _coerce_value("en") == "en"


class TestCleanNoneValues:
    """Tests for _clean_none_values."""

    def test_removes_none(self) -> None:
        data: dict[str, object] = {"a": 1, "b": None, "c": "hello"}

        result = _clean_none_values(data)

        assert result == {"a": 1, "c": "hello"}

    def test_recursive_cleaning(self) -> None:
        data: dict[str, object] = {
            "section": {"field": "value", "empty": None},
            "other": "ok",
        }

        result = _clean_none_values(data)

        assert result == {"section": {"field": "value"}, "other": "ok"}

    def test_removes_empty_sections(self) -> None:
        data: dict[str, object] = {"section": {"only": None}}

        result = _clean_none_values(data)

        assert result == {}

    def test_preserves_false_and_zero(self) -> None:
        data: dict[str, object] = {"a": False, "b": 0, "c": ""}

        result = _clean_none_values(data)

        assert result == {"a": False, "b": 0, "c": ""}


class TestConfigSetCommand:
    """Tests for the config set CLI command."""

    @patch("pvtt.cli.config_cmd._write_toml")
    @patch("pvtt.cli.config_cmd.load_toml", return_value={})
    @patch("pvtt.cli.config_cmd.get_config_path")
    def test_set_model_name(
        self,
        mock_path: MagicMock,
        mock_load: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_path.return_value = tmp_path / "config.toml"

        from typer.testing import CliRunner

        from pvtt.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["config", "set", "model.name", "tiny.en"])

        assert result.exit_code == 0
        mock_write.assert_called_once()
        written_data = mock_write.call_args[0][1]
        assert written_data["model"]["name"] == "tiny.en"

    @patch("pvtt.cli.config_cmd._write_toml")
    @patch("pvtt.cli.config_cmd.load_toml", return_value={})
    @patch("pvtt.cli.config_cmd.get_config_path")
    def test_set_beam_size_integer(
        self,
        mock_path: MagicMock,
        mock_load: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_path.return_value = tmp_path / "config.toml"

        from typer.testing import CliRunner

        from pvtt.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["config", "set", "transcription.beam_size", "3"])

        assert result.exit_code == 0
        mock_write.assert_called_once()
        written_data = mock_write.call_args[0][1]
        assert written_data["transcription"]["beam_size"] == 3

    @patch("pvtt.cli.config_cmd._write_toml")
    @patch("pvtt.cli.config_cmd.load_toml", return_value={})
    @patch("pvtt.cli.config_cmd.get_config_path")
    def test_set_boolean_value(
        self,
        mock_path: MagicMock,
        mock_load: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_path.return_value = tmp_path / "config.toml"

        from typer.testing import CliRunner

        from pvtt.cli.app import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["config", "set", "model.local_files_only", "true"]
        )

        assert result.exit_code == 0
        written_data = mock_write.call_args[0][1]
        assert written_data["model"]["local_files_only"] is True

    def test_invalid_key_format(self) -> None:
        from typer.testing import CliRunner

        from pvtt.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["config", "set", "badkey", "value"])

        assert result.exit_code == 1

    def test_invalid_value_for_field(self) -> None:
        from typer.testing import CliRunner

        from pvtt.cli.app import app

        runner = CliRunner()
        # beam_size max is 20, 99 exceeds it
        result = runner.invoke(
            app, ["config", "set", "transcription.beam_size", "99"]
        )

        assert result.exit_code == 1

    @patch("pvtt.cli.config_cmd._write_toml")
    @patch("pvtt.cli.config_cmd.load_toml", return_value={"model": {"device": "cpu"}})
    @patch("pvtt.cli.config_cmd.get_config_path")
    def test_merges_with_existing(
        self,
        mock_path: MagicMock,
        mock_load: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_path.return_value = tmp_path / "config.toml"

        from typer.testing import CliRunner

        from pvtt.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["config", "set", "model.name", "base.en"])

        assert result.exit_code == 0
        written_data = mock_write.call_args[0][1]
        # Existing value preserved
        assert written_data["model"]["device"] == "cpu"
        # New value added
        assert written_data["model"]["name"] == "base.en"

    @patch("pvtt.cli.config_cmd._write_toml")
    @patch("pvtt.cli.config_cmd.load_toml", return_value={})
    @patch("pvtt.cli.config_cmd.get_config_path")
    def test_set_float_value(
        self,
        mock_path: MagicMock,
        mock_load: MagicMock,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_path.return_value = tmp_path / "config.toml"

        from typer.testing import CliRunner

        from pvtt.cli.app import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["config", "set", "streaming.vad_threshold", "0.7"]
        )

        assert result.exit_code == 0
        written_data = mock_write.call_args[0][1]
        assert written_data["streaming"]["vad_threshold"] == 0.7

    def test_triple_dotted_key_fails(self) -> None:
        from typer.testing import CliRunner

        from pvtt.cli.app import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["config", "set", "a.b.c", "value"]
        )

        assert result.exit_code == 1

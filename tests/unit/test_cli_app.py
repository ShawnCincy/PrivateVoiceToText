"""Tests for pvtt CLI application."""

from __future__ import annotations

from typer.testing import CliRunner

from pvtt.cli.app import app

runner = CliRunner()


class TestCLIApp:
    """Tests for the root CLI app."""

    def test_help_shows_commands(self) -> None:
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "transcribe" in result.stdout
        assert "model" in result.stdout
        assert "config" in result.stdout

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])

        # Typer's no_args_is_help uses exit code 0
        assert result.exit_code in (0, 2)
        assert "transcribe" in result.stdout

    def test_transcribe_help(self) -> None:
        result = runner.invoke(app, ["transcribe", "--help"])

        assert result.exit_code == 0
        assert "file" in result.stdout

    def test_model_help(self) -> None:
        result = runner.invoke(app, ["model", "--help"])

        assert result.exit_code == 0
        assert "download" in result.stdout
        assert "list" in result.stdout
        assert "info" in result.stdout
        assert "remove" in result.stdout

    def test_config_help(self) -> None:
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        assert "show" in result.stdout
        assert "path" in result.stdout

    def test_config_path_runs(self) -> None:
        result = runner.invoke(app, ["config", "path"])

        assert result.exit_code == 0
        assert "pvtt" in result.stdout.lower() or "pvtt" in result.stderr if hasattr(result, 'stderr') else True

    def test_model_list_runs(self) -> None:
        result = runner.invoke(app, ["model", "list"])

        # Should succeed even with no models
        assert result.exit_code == 0

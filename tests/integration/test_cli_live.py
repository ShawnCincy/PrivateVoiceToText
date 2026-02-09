"""Integration tests for the pvtt transcribe live CLI command.

These tests use Typer's CliRunner and mock the streaming pipeline
to verify CLI behavior without actual audio hardware or models.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pvtt.cli.app import app
from pvtt.core.streaming import StreamingEvent, StreamingEventType
from pvtt.util.types import TranscriptionSegment

runner = CliRunner()


@pytest.mark.integration
class TestCLILiveCommand:
    """Integration tests for `pvtt transcribe live` via CliRunner."""

    def test_live_help_text(self) -> None:
        """Live command shows help with expected options."""
        result = runner.invoke(app, ["transcribe", "live", "--help"])

        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--audio-device" in result.stdout
        assert "--format" in result.stdout

    @patch("pvtt.core.streaming.StreamingPipeline")
    @patch("pvtt.engine.registry.get_engine")
    @patch("pvtt.core.transcriber.Transcriber._ensure_model_loaded")
    def test_live_starts_and_stops_pipeline(
        self,
        mock_ensure: MagicMock,
        mock_get_engine: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """Live command creates and starts a pipeline."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_get_engine.return_value = mock_engine

        mock_pipeline = MagicMock()
        mock_pipeline.is_running = False
        mock_pipeline_cls.return_value = mock_pipeline

        runner.invoke(app, ["transcribe", "live"])

        # Pipeline should have been constructed and started
        assert mock_pipeline.start.called
        assert mock_pipeline.stop.called

    @patch("pvtt.cli.transcribe.get_exporter")
    @patch("pvtt.core.streaming.StreamingPipeline")
    @patch("pvtt.engine.registry.get_engine")
    @patch("pvtt.core.transcriber.Transcriber._ensure_model_loaded")
    def test_live_writes_output_file(
        self,
        mock_ensure: MagicMock,
        mock_get_engine: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_get_exporter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Live command writes accumulated segments to output file."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_get_engine.return_value = mock_engine

        # Pipeline that injects segments via the callback
        def pipeline_init(
            config: object, engine: object, on_event: object,
        ) -> MagicMock:
            # Deliver a segment via callback
            seg = TranscriptionSegment(
                start=0.0, end=1.0, text="Test output."
            )
            event = StreamingEvent(
                type=StreamingEventType.SEGMENT, segment=seg
            )
            on_event(event)

            mock_pipeline = MagicMock()
            mock_pipeline.is_running = False
            return mock_pipeline

        mock_pipeline_cls.side_effect = pipeline_init

        mock_exporter = MagicMock()
        mock_get_exporter.return_value = mock_exporter

        output_file = tmp_path / "output.srt"
        runner.invoke(
            app,
            [
                "transcribe", "live",
                "--output", str(output_file),
                "--format", "srt",
            ],
        )

        # Exporter should have been called to write the file
        mock_get_exporter.assert_called()
        mock_exporter.write.assert_called_once()

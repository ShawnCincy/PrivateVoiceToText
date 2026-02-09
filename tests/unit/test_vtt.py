"""Tests for pvtt.export.vtt."""

from __future__ import annotations

from pathlib import Path

import pytest

from pvtt.exceptions import ExportError
from pvtt.export.vtt import VttExporter, _format_timestamp_vtt
from pvtt.util.types import TranscriptionResult, TranscriptionSegment


class TestFormatTimestampVtt:
    """Tests for the VTT timestamp formatter."""

    def test_zero_seconds(self) -> None:
        assert _format_timestamp_vtt(0.0) == "00:00:00.000"

    def test_uses_period_not_comma(self) -> None:
        result = _format_timestamp_vtt(1.5)

        assert "." in result
        assert "," not in result

    def test_minutes_and_millis(self) -> None:
        assert _format_timestamp_vtt(65.25) == "00:01:05.250"

    def test_hours(self) -> None:
        assert _format_timestamp_vtt(3723.1) == "01:02:03.100"


class TestVttExporter:
    """Tests for the VttExporter class."""

    def test_format_name(self) -> None:
        exporter = VttExporter()

        assert exporter.format_name == "vtt"

    def test_format_starts_with_webvtt_header(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        exporter = VttExporter()

        output = exporter.format(result)

        assert output.startswith("WEBVTT\n")

    def test_format_single_segment(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.5, text="Hello world.")],
        )
        exporter = VttExporter()

        output = exporter.format(result)

        assert "00:00:00.000 --> 00:00:01.500" in output
        assert "Hello world." in output

    def test_format_no_entry_numbers(self) -> None:
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text="First."),
                TranscriptionSegment(start=1.0, end=2.0, text="Second."),
            ],
        )
        exporter = VttExporter()

        output = exporter.format(result)

        # VTT should not have numbered entries like SRT
        assert "\n1\n" not in output
        assert "\n2\n" not in output

    def test_format_multiple_segments(self) -> None:
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.5, text="Hello."),
                TranscriptionSegment(start=1.5, end=3.0, text="World."),
            ],
        )
        exporter = VttExporter()

        output = exporter.format(result)

        assert "Hello." in output
        assert "World." in output
        assert output.count("-->") == 2

    def test_format_empty_result(self) -> None:
        result = TranscriptionResult(segments=[])
        exporter = VttExporter()

        output = exporter.format(result)

        assert output.startswith("WEBVTT\n")

    def test_format_strips_whitespace(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="  padded  ")],
        )
        exporter = VttExporter()

        output = exporter.format(result)

        assert "padded" in output
        assert "  padded  " not in output

    def test_write_creates_file(self, tmp_path: Path) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        output_file = tmp_path / "output.vtt"
        exporter = VttExporter()

        exporter.write(result, output_file)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT\n")

    def test_write_invalid_path_raises_export_error(self, tmp_path: Path) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        exporter = VttExporter()

        with pytest.raises(ExportError):
            exporter.write(result, tmp_path / "nonexistent" / "output.vtt")

"""Tests for pvtt.export.srt."""

from __future__ import annotations

from pathlib import Path

import pytest

from pvtt.exceptions import ExportError
from pvtt.export.srt import SrtExporter, _format_timestamp_srt
from pvtt.util.types import TranscriptionResult, TranscriptionSegment


class TestFormatTimestampSrt:
    """Tests for the SRT timestamp formatter."""

    def test_zero_seconds(self) -> None:
        assert _format_timestamp_srt(0.0) == "00:00:00,000"

    def test_simple_seconds(self) -> None:
        assert _format_timestamp_srt(1.5) == "00:00:01,500"

    def test_minutes(self) -> None:
        assert _format_timestamp_srt(65.25) == "00:01:05,250"

    def test_hours(self) -> None:
        assert _format_timestamp_srt(3723.1) == "01:02:03,100"

    def test_millisecond_precision(self) -> None:
        assert _format_timestamp_srt(0.001) == "00:00:00,001"

    def test_large_value(self) -> None:
        assert _format_timestamp_srt(86399.999) == "23:59:59,999"


class TestSrtExporter:
    """Tests for the SrtExporter class."""

    def test_format_name(self) -> None:
        exporter = SrtExporter()

        assert exporter.format_name == "srt"

    def test_format_single_segment(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.5, text="Hello world.")],
        )
        exporter = SrtExporter()

        output = exporter.format(result)

        assert "1\n" in output
        assert "00:00:00,000 --> 00:00:01,500" in output
        assert "Hello world." in output

    def test_format_multiple_segments(self) -> None:
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.5, text="Hello world."),
                TranscriptionSegment(start=1.5, end=3.0, text="This is a test."),
            ],
        )
        exporter = SrtExporter()

        output = exporter.format(result)

        assert output.startswith("1\n")
        assert "\n\n2\n" in output
        assert "00:00:01,500 --> 00:00:03,000" in output

    def test_format_empty_result(self) -> None:
        result = TranscriptionResult(segments=[])
        exporter = SrtExporter()

        output = exporter.format(result)

        assert output == ""

    def test_format_strips_whitespace(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="  padded  ")],
        )
        exporter = SrtExporter()

        output = exporter.format(result)

        assert "padded" in output
        assert "  padded  " not in output

    def test_write_creates_file(self, tmp_path: Path) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        output_file = tmp_path / "output.srt"
        exporter = SrtExporter()

        exporter.write(result, output_file)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:01,000" in content

    def test_write_invalid_path_raises_export_error(self, tmp_path: Path) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        exporter = SrtExporter()

        with pytest.raises(ExportError):
            exporter.write(result, tmp_path / "nonexistent" / "output.srt")

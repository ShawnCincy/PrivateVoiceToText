"""Tests for pvtt.export.plain_text."""

from __future__ import annotations

from pathlib import Path

import pytest

from pvtt.exceptions import ExportError
from pvtt.export.plain_text import PlainTextExporter
from pvtt.util.types import TranscriptionResult, TranscriptionSegment


class TestPlainTextExporter:
    """Tests for PlainTextExporter."""

    def test_format_name(self) -> None:
        exporter = PlainTextExporter()

        assert exporter.format_name == "text"

    def test_format_uses_text_field(self) -> None:
        result = TranscriptionResult(
            segments=[],
            text="  Hello world.  ",
        )
        exporter = PlainTextExporter()

        output = exporter.format(result)

        assert output == "Hello world."

    def test_format_joins_segments_when_no_text(self) -> None:
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text=" Hello. "),
                TranscriptionSegment(start=1.0, end=2.0, text=" World. "),
            ],
            text="",
        )
        exporter = PlainTextExporter()

        output = exporter.format(result)

        assert output == "Hello.\nWorld."

    def test_format_empty_result(self) -> None:
        result = TranscriptionResult(segments=[], text="")
        exporter = PlainTextExporter()

        output = exporter.format(result)

        assert output == ""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        result = TranscriptionResult(
            segments=[],
            text="Hello world.",
        )
        output_file = tmp_path / "output.txt"
        exporter = PlainTextExporter()

        exporter.write(result, output_file)

        assert output_file.read_text(encoding="utf-8") == "Hello world."

    def test_write_invalid_path_raises_export_error(self) -> None:
        result = TranscriptionResult(segments=[], text="Hello")
        exporter = PlainTextExporter()
        bad_path = Path("/nonexistent/directory/file.txt")

        with pytest.raises(ExportError, match="Failed to write"):
            exporter.write(result, bad_path)

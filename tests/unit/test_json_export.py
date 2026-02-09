"""Tests for pvtt.export.json_export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pvtt.exceptions import ExportError
from pvtt.export.json_export import JsonExporter
from pvtt.util.types import TranscriptionResult, TranscriptionSegment


class TestJsonExporter:
    """Tests for the JsonExporter class."""

    def test_format_name(self) -> None:
        exporter = JsonExporter()

        assert exporter.format_name == "json"

    def test_format_returns_valid_json(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        exporter = JsonExporter()

        output = exporter.format(result)
        data = json.loads(output)

        assert isinstance(data, dict)

    def test_format_includes_text_field(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Hello.")],
            text="Hello.",
        )
        exporter = JsonExporter()

        output = exporter.format(result)
        data = json.loads(output)

        assert data["text"] == "Hello."

    def test_format_includes_language_fields(self) -> None:
        result = TranscriptionResult(
            segments=[],
            language="fr",
            language_probability=0.95,
        )
        exporter = JsonExporter()

        output = exporter.format(result)
        data = json.loads(output)

        assert data["language"] == "fr"
        assert data["language_probability"] == 0.95

    def test_format_includes_duration(self) -> None:
        result = TranscriptionResult(
            segments=[],
            duration=42.5,
        )
        exporter = JsonExporter()

        output = exporter.format(result)
        data = json.loads(output)

        assert data["duration"] == 42.5

    def test_format_includes_segments_array(self) -> None:
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.5, text="First."),
                TranscriptionSegment(start=1.5, end=3.0, text="Second."),
            ],
        )
        exporter = JsonExporter()

        output = exporter.format(result)
        data = json.loads(output)

        assert len(data["segments"]) == 2
        assert data["segments"][0]["start"] == 0.0
        assert data["segments"][0]["end"] == 1.5
        assert data["segments"][0]["text"] == "First."
        assert data["segments"][1]["text"] == "Second."

    def test_format_segment_includes_logprob_fields(self) -> None:
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start=0.0, end=1.0, text="Hi.",
                    avg_logprob=-0.5, no_speech_prob=0.01,
                ),
            ],
        )
        exporter = JsonExporter()

        output = exporter.format(result)
        data = json.loads(output)

        seg = data["segments"][0]
        assert seg["avg_logprob"] == -0.5
        assert seg["no_speech_prob"] == 0.01

    def test_format_empty_result(self) -> None:
        result = TranscriptionResult(segments=[])
        exporter = JsonExporter()

        output = exporter.format(result)
        data = json.loads(output)

        assert data["segments"] == []
        assert data["text"] == ""

    def test_format_uses_indentation(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        exporter = JsonExporter()

        output = exporter.format(result)

        # json.dumps with indent=2 produces multi-line output
        assert "\n" in output
        assert "  " in output

    def test_format_handles_unicode(self) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Héllo wörld.")],
        )
        exporter = JsonExporter()

        output = exporter.format(result)

        # ensure_ascii=False means unicode chars are preserved
        assert "Héllo wörld." in output

    def test_write_creates_file(self, tmp_path: Path) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        output_file = tmp_path / "output.json"
        exporter = JsonExporter()

        exporter.write(result, output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert data["segments"][0]["text"] == "Test."

    def test_write_invalid_path_raises_export_error(self, tmp_path: Path) -> None:
        result = TranscriptionResult(
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Test.")],
        )
        exporter = JsonExporter()

        with pytest.raises(ExportError):
            exporter.write(result, tmp_path / "nonexistent" / "output.json")

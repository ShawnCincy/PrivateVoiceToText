"""Tests for pvtt.util.types."""

from __future__ import annotations

from pathlib import Path

from pvtt.util.types import (
    ModelInfo,
    TranscribeOptions,
    TranscriptionResult,
    TranscriptionSegment,
)


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_create_segment(self) -> None:
        seg = TranscriptionSegment(start=0.0, end=1.5, text="Hello")

        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "Hello"
        assert seg.avg_logprob == 0.0
        assert seg.no_speech_prob == 0.0

    def test_segment_is_frozen(self) -> None:
        seg = TranscriptionSegment(start=0.0, end=1.0, text="test")

        with __import__("pytest").raises(AttributeError):
            seg.text = "modified"  # type: ignore[misc]


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_create_result_with_defaults(self) -> None:
        segs = [TranscriptionSegment(start=0.0, end=1.0, text="Hi")]
        result = TranscriptionResult(segments=segs)

        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.language_probability == 0.0
        assert result.duration == 0.0
        assert result.text == ""

    def test_create_result_with_all_fields(self) -> None:
        segs = [TranscriptionSegment(start=0.0, end=1.0, text="Hi")]
        result = TranscriptionResult(
            segments=segs,
            language="fr",
            language_probability=0.95,
            duration=1.0,
            text="Hi",
        )

        assert result.language == "fr"
        assert result.language_probability == 0.95
        assert result.duration == 1.0
        assert result.text == "Hi"


class TestTranscribeOptions:
    """Tests for TranscribeOptions dataclass."""

    def test_defaults(self) -> None:
        opts = TranscribeOptions()

        assert opts.language is None
        assert opts.beam_size == 5
        assert opts.temperature == 0.0
        assert opts.initial_prompt is None
        assert opts.vad_filter is False
        assert opts.word_timestamps is False

    def test_custom_values(self) -> None:
        opts = TranscribeOptions(
            language="en",
            beam_size=3,
            temperature=(0.0, 0.2, 0.4),
            initial_prompt="Hello",
        )

        assert opts.language == "en"
        assert opts.beam_size == 3
        assert opts.temperature == (0.0, 0.2, 0.4)
        assert opts.initial_prompt == "Hello"


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_create_model_info(self) -> None:
        info = ModelInfo(name="tiny.en", path=Path("/models/tiny.en"))

        assert info.name == "tiny.en"
        assert info.path == Path("/models/tiny.en")
        assert info.size_bytes == 0
        assert info.compute_type == ""

"""Tests for pvtt.core.transcriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pvtt.config.schema import PvttConfig
from pvtt.core.transcriber import Transcriber
from pvtt.exceptions import AudioError
from pvtt.util.types import TranscriptionSegment


class TestTranscriber:
    """Tests for the Transcriber class."""

    def _make_transcriber(
        self,
        mock_engine: MagicMock,
        tmp_path: Path,
        config: PvttConfig | None = None,
    ) -> Transcriber:
        """Helper to create a Transcriber with mocked dependencies."""
        cfg = config or PvttConfig.model_validate({
            "model": {"name": "tiny.en", "device": "cpu", "compute_type": "int8"},
        })
        model_manager = MagicMock()
        model_manager.get_model_path.return_value = tmp_path / "models" / "tiny.en"
        return Transcriber(config=cfg, engine=mock_engine, model_manager=model_manager)

    def test_transcribe_file_returns_result(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        transcriber = self._make_transcriber(mock_engine, tmp_path)
        result = transcriber.transcribe_file(audio_file)

        assert len(result.segments) == 2
        assert "Hello world" in result.text
        assert "This is a test" in result.text

    def test_transcribe_file_loads_model_on_first_call(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        transcriber = self._make_transcriber(mock_engine, tmp_path)
        transcriber.transcribe_file(audio_file)

        mock_engine.load_model.assert_called_once()

    def test_transcribe_file_does_not_reload_model(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        # Simulate model already loaded after first transcribe
        mock_engine.is_loaded = True
        transcriber = self._make_transcriber(mock_engine, tmp_path)

        # Reset to track new calls after making fresh segments
        mock_engine.transcribe.return_value = iter([
            TranscriptionSegment(start=0.0, end=1.0, text="Again."),
        ])
        transcriber.transcribe_file(audio_file)

        mock_engine.load_model.assert_not_called()

    def test_transcribe_file_invalid_audio_raises(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        transcriber = self._make_transcriber(mock_engine, tmp_path)

        with pytest.raises(AudioError):
            transcriber.transcribe_file(tmp_path / "nonexistent.wav")

    def test_format_output_uses_config_format(
        self, mock_engine: MagicMock, tmp_path: Path, sample_result: object
    ) -> None:
        from pvtt.util.types import TranscriptionResult

        result = TranscriptionResult(
            segments=[],
            text="Hello world.",
        )
        transcriber = self._make_transcriber(mock_engine, tmp_path)

        output = transcriber.format_output(result)

        assert output == "Hello world."

    def test_format_output_with_explicit_format(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        from pvtt.util.types import TranscriptionResult

        result = TranscriptionResult(segments=[], text="Test.")
        transcriber = self._make_transcriber(mock_engine, tmp_path)

        output = transcriber.format_output(result, format_name="text")

        assert output == "Test."

    def test_auto_model_resolves_via_model_manager(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        cfg = PvttConfig.model_validate({
            "model": {"name": "auto", "device": "cpu", "compute_type": "int8"},
        })
        model_manager = MagicMock()
        model_manager.resolve_model_name.return_value = "tiny.en"
        model_manager.get_model_path.return_value = tmp_path / "models" / "tiny.en"

        transcriber = Transcriber(config=cfg, engine=mock_engine, model_manager=model_manager)
        transcriber.transcribe_file(audio_file)

        model_manager.resolve_model_name.assert_called_once()
        model_manager.get_model_path.assert_called_once_with("tiny.en")

    def test_explicit_model_does_not_call_resolve(
        self, mock_engine: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        cfg = PvttConfig.model_validate({
            "model": {"name": "tiny.en", "device": "cpu", "compute_type": "int8"},
        })
        model_manager = MagicMock()
        model_manager.get_model_path.return_value = tmp_path / "models" / "tiny.en"

        transcriber = Transcriber(config=cfg, engine=mock_engine, model_manager=model_manager)
        transcriber.transcribe_file(audio_file)

        model_manager.resolve_model_name.assert_not_called()
        model_manager.get_model_path.assert_called_once_with("tiny.en")

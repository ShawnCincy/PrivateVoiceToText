"""Tests for pvtt.config.schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pvtt.config.schema import (
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    PvttConfig,
    TranscriptionConfig,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_defaults(self) -> None:
        config = ModelConfig()

        assert config.name == "auto"
        assert config.device == "auto"
        assert config.compute_type == "auto"
        assert config.local_files_only is True

    def test_auto_model_name_accepted(self) -> None:
        config = ModelConfig(name="auto")

        assert config.name == "auto"

    def test_known_model_names_accepted(self) -> None:
        for name in ("tiny", "tiny.en", "base", "small.en", "medium", "large-v3-turbo"):
            config = ModelConfig(name=name)
            assert config.name == name

    def test_huggingface_repo_id_accepted(self) -> None:
        config = ModelConfig(name="Systran/faster-whisper-large-v3-turbo")

        assert config.name == "Systran/faster-whisper-large-v3-turbo"

    def test_unknown_model_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown model"):
            ModelConfig(name="nonexistent-model")

    def test_invalid_device_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(device="tpu")  # type: ignore[arg-type]

    def test_invalid_compute_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(compute_type="quantum")  # type: ignore[arg-type]


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig."""

    def test_defaults(self) -> None:
        config = TranscriptionConfig()

        assert config.language is None
        assert config.beam_size == 5
        assert config.temperature == 0.0
        assert config.vad_filter is False
        assert config.word_timestamps is False
        assert config.initial_prompt is None

    def test_beam_size_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionConfig(beam_size=0)

        with pytest.raises(ValidationError):
            TranscriptionConfig(beam_size=21)

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            TranscriptionConfig(temperature=1.1)


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_defaults(self) -> None:
        config = OutputConfig()

        assert config.format == "text"

    def test_valid_formats(self) -> None:
        for fmt in ("text", "srt", "vtt", "json"):
            config = OutputConfig(format=fmt)  # type: ignore[arg-type]
            assert config.format == fmt

    def test_invalid_format_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OutputConfig(format="csv")  # type: ignore[arg-type]


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_defaults(self) -> None:
        config = LoggingConfig()

        assert config.verbosity == 0

    def test_verbosity_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig(verbosity=-1)

        with pytest.raises(ValidationError):
            LoggingConfig(verbosity=3)


class TestPvttConfig:
    """Tests for the root PvttConfig."""

    def test_defaults(self) -> None:
        config = PvttConfig()

        assert config.model.name == "auto"
        assert config.transcription.beam_size == 5
        assert config.output.format == "text"
        assert config.logging.verbosity == 0

    def test_privacy_default(self) -> None:
        config = PvttConfig()

        assert config.model.local_files_only is True

    def test_from_dict(self) -> None:
        data = {
            "model": {"name": "tiny.en", "device": "cpu"},
            "transcription": {"language": "en"},
        }
        config = PvttConfig.model_validate(data)

        assert config.model.name == "tiny.en"
        assert config.model.device == "cpu"
        assert config.transcription.language == "en"

    def test_empty_dict_produces_defaults(self) -> None:
        config = PvttConfig.model_validate({})

        assert config.model.name == "auto"
        assert config.model.local_files_only is True

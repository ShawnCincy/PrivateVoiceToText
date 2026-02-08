"""Shared test fixtures for pvtt."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pvtt.config.schema import PvttConfig
from pvtt.util.types import TranscriptionResult, TranscriptionSegment


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary pvtt data directory with standard structure."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    return tmp_path


@pytest.fixture()
def sample_config() -> PvttConfig:
    """Provide a PvttConfig with test-friendly defaults."""
    return PvttConfig.model_validate({
        "model": {
            "name": "tiny.en",
            "device": "cpu",
            "compute_type": "int8",
        },
    })


@pytest.fixture()
def mock_engine() -> MagicMock:
    """Provide a mock InferenceEngine that returns canned segments."""
    engine = MagicMock()
    engine.is_loaded = False
    engine.model_name = None

    def fake_load(model_name_or_path: str, **kwargs: object) -> None:
        engine.is_loaded = True
        engine.model_name = model_name_or_path

    engine.load_model.side_effect = fake_load
    engine.transcribe.return_value = iter([
        TranscriptionSegment(start=0.0, end=1.5, text="Hello world."),
        TranscriptionSegment(start=1.5, end=3.0, text="This is a test."),
    ])
    return engine


@pytest.fixture()
def sample_segments() -> list[TranscriptionSegment]:
    """Provide sample transcription segments."""
    return [
        TranscriptionSegment(start=0.0, end=1.5, text="Hello world."),
        TranscriptionSegment(start=1.5, end=3.0, text="This is a test."),
        TranscriptionSegment(start=3.0, end=4.5, text="Final segment."),
    ]


@pytest.fixture()
def sample_result(sample_segments: list[TranscriptionSegment]) -> TranscriptionResult:
    """Provide a complete TranscriptionResult."""
    return TranscriptionResult(
        segments=sample_segments,
        text="Hello world. This is a test. Final segment.",
        language="en",
        language_probability=0.99,
        duration=4.5,
    )

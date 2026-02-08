"""Tests for pvtt.engine.base."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pvtt.engine.base import InferenceEngine
from pvtt.util.types import TranscribeOptions, TranscriptionSegment


class _ValidEngine:
    """A minimal engine that satisfies the Protocol."""

    def load_model(
        self,
        model_name_or_path: str,
        device: str,
        compute_type: str,
        *,
        local_files_only: bool = True,
    ) -> None:
        pass

    def transcribe(
        self,
        audio: Path | str,
        options: TranscribeOptions,
    ) -> Iterator[TranscriptionSegment]:
        yield TranscriptionSegment(start=0.0, end=1.0, text="test")

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def model_name(self) -> str | None:
        return None


class _InvalidEngine:
    """A class that does NOT satisfy the Protocol."""

    def load_model(self) -> None:  # Wrong signature
        pass


class TestInferenceEngineProtocol:
    """Tests for InferenceEngine Protocol."""

    def test_valid_engine_satisfies_protocol(self) -> None:
        engine = _ValidEngine()

        assert isinstance(engine, InferenceEngine)

    def test_invalid_engine_does_not_satisfy_protocol(self) -> None:
        engine = _InvalidEngine()

        # runtime_checkable only checks method existence, not signatures
        # but the missing methods should fail
        assert not isinstance(engine, InferenceEngine)

    def test_faster_whisper_engine_satisfies_protocol(self) -> None:
        from pvtt.engine.faster_whisper import FasterWhisperEngine

        engine = FasterWhisperEngine()

        assert isinstance(engine, InferenceEngine)

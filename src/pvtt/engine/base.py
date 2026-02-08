"""Inference engine protocol for pvtt.

All backends (Faster-Whisper, whisper.cpp, etc.) implement this Protocol.
New backends are registered via engine/registry.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from pvtt.util.types import TranscribeOptions, TranscriptionSegment


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol that all inference backends must satisfy."""

    def load_model(
        self,
        model_name_or_path: str,
        device: str,
        compute_type: str,
        *,
        local_files_only: bool = True,
    ) -> None:
        """Load a model into memory.

        Args:
            model_name_or_path: Model identifier or local path.
            device: 'cuda' or 'cpu'.
            compute_type: CTranslate2 compute type string.
            local_files_only: If True, never make network requests.

        Raises:
            ModelNotFoundError: If model not available locally.
            EngineError: If model loading fails.
        """
        ...

    def transcribe(
        self,
        audio: Path | str,
        options: TranscribeOptions,
    ) -> Iterator[TranscriptionSegment]:
        """Transcribe an audio file.

        Args:
            audio: Path to audio file.
            options: Transcription parameters.

        Yields:
            TranscriptionSegment for each detected segment.

        Raises:
            EngineError: If transcription fails.
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        ...

    @property
    def model_name(self) -> str | None:
        """Name of the currently loaded model, or None."""
        ...

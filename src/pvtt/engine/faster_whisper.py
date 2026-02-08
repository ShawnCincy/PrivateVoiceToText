"""Faster-Whisper inference engine for pvtt.

Uses CTranslate2 backend for 2-6x faster inference than PyTorch Whisper.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pvtt.exceptions import EngineError, ModelNotFoundError
from pvtt.util.logging import get_logger
from pvtt.util.types import TranscribeOptions, TranscriptionSegment

logger = get_logger(__name__)


class FasterWhisperEngine:
    """Inference engine using Faster-Whisper (CTranslate2 backend).

    Satisfies the InferenceEngine Protocol through structural typing.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None

    def load_model(
        self,
        model_name_or_path: str,
        device: str,
        compute_type: str,
        *,
        local_files_only: bool = True,
    ) -> None:
        """Load a Faster-Whisper model.

        Args:
            model_name_or_path: Model identifier or local path.
            device: 'cuda' or 'cpu'.
            compute_type: CTranslate2 compute type string.
            local_files_only: If True, never make network requests.

        Raises:
            EngineError: If faster-whisper is not installed or loading fails.
            ModelNotFoundError: If model is not found locally.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise EngineError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            ) from exc

        try:
            self._model = WhisperModel(
                model_name_or_path,
                device=device,
                compute_type=compute_type,
                local_files_only=local_files_only,
            )
            self._model_name = model_name_or_path
            logger.info(
                "Loaded model %s on %s (%s)",
                model_name_or_path,
                device,
                compute_type,
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if "not found" in error_msg or "no such file" in error_msg:
                raise ModelNotFoundError(
                    f"Model {model_name_or_path!r} not found locally. "
                    f"Download it first with: pvtt model download {model_name_or_path}"
                ) from exc
            raise EngineError(f"Failed to load model: {exc}") from exc

    def transcribe(
        self,
        audio: Path | str,
        options: TranscribeOptions,
    ) -> Iterator[TranscriptionSegment]:
        """Transcribe audio using Faster-Whisper.

        Args:
            audio: Path to audio file.
            options: Transcription parameters.

        Yields:
            TranscriptionSegment for each detected segment.

        Raises:
            EngineError: If no model is loaded or transcription fails.
        """
        if self._model is None:
            raise EngineError("No model loaded. Call load_model() first.")

        try:
            temperature = options.temperature
            if isinstance(temperature, (int, float)):
                temperature = [temperature]

            segments_iter, _info = self._model.transcribe(
                str(audio),
                language=options.language,
                beam_size=options.beam_size,
                temperature=temperature,
                initial_prompt=options.initial_prompt,
                vad_filter=options.vad_filter,
                word_timestamps=options.word_timestamps,
            )

            for seg in segments_iter:
                yield TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                )
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"Transcription failed: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        return self._model is not None

    @property
    def model_name(self) -> str | None:
        """Name of the currently loaded model, or None."""
        return self._model_name

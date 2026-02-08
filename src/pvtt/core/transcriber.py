"""Transcription orchestration for pvtt.

The Transcriber is the primary entry point for core transcription logic.
The CLI layer creates a Transcriber and calls its methods.
"""

from __future__ import annotations

from pathlib import Path

from pvtt.audio.file_reader import validate_audio_file
from pvtt.config.schema import PvttConfig
from pvtt.core.model_manager import ModelManager
from pvtt.engine.base import InferenceEngine
from pvtt.engine.registry import get_engine
from pvtt.export.registry import get_exporter
from pvtt.util.hardware import resolve_device_and_compute
from pvtt.util.logging import get_logger
from pvtt.util.types import TranscribeOptions, TranscriptionResult

logger = get_logger(__name__)


class Transcriber:
    """Orchestrates model loading, transcription, and output formatting.

    Args:
        config: Application configuration.
        engine: Optional engine instance (for testing).
        model_manager: Optional model manager (for testing).
    """

    def __init__(
        self,
        config: PvttConfig,
        engine: InferenceEngine | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self._config = config
        self._engine = engine or get_engine()
        self._model_manager = model_manager or ModelManager()

    def _ensure_model_loaded(self) -> None:
        """Load the configured model if not already loaded."""
        if self._engine.is_loaded:
            return

        model_cfg = self._config.model
        model_path = self._model_manager.get_model_path(model_cfg.name)

        device, compute_type = resolve_device_and_compute(
            model_cfg.device,
            model_cfg.compute_type,
        )

        logger.info(
            "Loading model %s on %s (%s)",
            model_cfg.name,
            device,
            compute_type,
        )

        self._engine.load_model(
            model_name_or_path=str(model_path),
            device=device,
            compute_type=compute_type,
            local_files_only=model_cfg.local_files_only,
        )

    def transcribe_file(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            TranscriptionResult with segments and full text.

        Raises:
            AudioError: If the file is invalid.
            ModelNotFoundError: If the model is not downloaded.
            EngineError: If transcription fails.
        """
        validated_path = validate_audio_file(audio_path)
        self._ensure_model_loaded()

        tx_cfg = self._config.transcription
        options = TranscribeOptions(
            language=tx_cfg.language,
            beam_size=tx_cfg.beam_size,
            temperature=tx_cfg.temperature,
            initial_prompt=tx_cfg.initial_prompt,
            vad_filter=tx_cfg.vad_filter,
            word_timestamps=tx_cfg.word_timestamps,
        )

        segments = list(self._engine.transcribe(validated_path, options))
        full_text = " ".join(seg.text.strip() for seg in segments)

        return TranscriptionResult(
            segments=segments,
            text=full_text,
        )

    def format_output(
        self,
        result: TranscriptionResult,
        format_name: str | None = None,
    ) -> str:
        """Format a transcription result as a string.

        Args:
            result: Transcription to format.
            format_name: Output format. Defaults to config value.

        Returns:
            Formatted string.
        """
        fmt = format_name or self._config.output.format
        exporter = get_exporter(fmt)
        return exporter.format(result)

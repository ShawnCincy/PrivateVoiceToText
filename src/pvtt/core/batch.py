"""Batch file transcription for pvtt."""

from __future__ import annotations

from pathlib import Path

from pvtt.core.transcriber import Transcriber
from pvtt.util.types import TranscriptionResult


def transcribe_files(
    transcriber: Transcriber,
    paths: list[Path],
) -> list[tuple[Path, TranscriptionResult]]:
    """Transcribe multiple audio files sequentially.

    Args:
        transcriber: Configured Transcriber instance.
        paths: List of audio file paths.

    Returns:
        List of (path, result) tuples.
    """
    results: list[tuple[Path, TranscriptionResult]] = []
    for path in paths:
        result = transcriber.transcribe_file(path)
        results.append((path, result))
    return results

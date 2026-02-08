"""Exporter protocol for pvtt output formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pvtt.util.types import TranscriptionResult


@runtime_checkable
class Exporter(Protocol):
    """Protocol for output formatters.

    Implementations: PlainTextExporter, SrtExporter (Phase 2),
    VttExporter (Phase 2), JsonExporter (Phase 2).
    """

    @property
    def format_name(self) -> str:
        """Short name of this format (e.g., 'text', 'srt')."""
        ...

    def format(self, result: TranscriptionResult) -> str:
        """Format a transcription result as a string.

        Args:
            result: The transcription to format.

        Returns:
            Formatted string representation.
        """
        ...

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write a formatted transcription to a file.

        Args:
            result: The transcription to format.
            path: Output file path.
        """
        ...

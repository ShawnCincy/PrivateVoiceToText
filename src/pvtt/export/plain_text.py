"""Plain text exporter for pvtt."""

from __future__ import annotations

from pathlib import Path

from pvtt.exceptions import ExportError
from pvtt.util.types import TranscriptionResult


class PlainTextExporter:
    """Exports transcription as plain text.

    Satisfies the Exporter Protocol.
    """

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "text"

    def format(self, result: TranscriptionResult) -> str:
        """Format transcription as plain text.

        Uses the pre-joined text field if available, otherwise joins
        segments with newlines.

        Args:
            result: The transcription to format.

        Returns:
            Plain text string.
        """
        if result.text:
            return result.text.strip()
        return "\n".join(seg.text.strip() for seg in result.segments).strip()

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write plain text transcription to file.

        Args:
            result: The transcription to format.
            path: Output file path.

        Raises:
            ExportError: If writing fails.
        """
        try:
            path.write_text(self.format(result), encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"Failed to write to {path}: {exc}") from exc

"""SRT subtitle exporter for pvtt."""

from __future__ import annotations

import math
from pathlib import Path

from pvtt.exceptions import ExportError
from pvtt.util.types import TranscriptionResult


def _format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SrtExporter:
    """Exports transcription as SRT (SubRip) subtitle format.

    SRT format example::

        1
        00:00:00,000 --> 00:00:01,500
        Hello world.

        2
        00:00:01,500 --> 00:00:03,000
        This is a test.

    Satisfies the Exporter Protocol.
    """

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "srt"

    def format(self, result: TranscriptionResult) -> str:
        """Format transcription as SRT subtitle text.

        Args:
            result: The transcription to format.

        Returns:
            SRT-formatted string.
        """
        if not result.segments:
            return ""

        blocks: list[str] = []
        for idx, seg in enumerate(result.segments, start=1):
            start_ts = _format_timestamp_srt(seg.start)
            end_ts = _format_timestamp_srt(seg.end)
            text = seg.text.strip()
            blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}")

        return "\n\n".join(blocks) + "\n"

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write SRT subtitle transcription to file.

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

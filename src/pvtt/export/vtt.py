"""WebVTT subtitle exporter for pvtt."""

from __future__ import annotations

import math
from pathlib import Path

from pvtt.exceptions import ExportError
from pvtt.util.types import TranscriptionResult


def _format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class VttExporter:
    """Exports transcription as WebVTT subtitle format.

    WebVTT format example::

        WEBVTT

        00:00:00.000 --> 00:00:01.500
        Hello world.

        00:00:01.500 --> 00:00:03.000
        This is a test.

    Satisfies the Exporter Protocol.
    """

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "vtt"

    def format(self, result: TranscriptionResult) -> str:
        """Format transcription as WebVTT subtitle text.

        Args:
            result: The transcription to format.

        Returns:
            WebVTT-formatted string.
        """
        lines: list[str] = ["WEBVTT", ""]

        for seg in result.segments:
            start_ts = _format_timestamp_vtt(seg.start)
            end_ts = _format_timestamp_vtt(seg.end)
            text = seg.text.strip()
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write WebVTT subtitle transcription to file.

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

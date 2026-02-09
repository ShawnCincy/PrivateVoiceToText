"""JSON exporter for pvtt."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pvtt.exceptions import ExportError
from pvtt.util.types import TranscriptionResult


class JsonExporter:
    """Exports transcription as structured JSON.

    Output structure::

        {
            "text": "Hello world. This is a test.",
            "language": "en",
            "language_probability": 0.99,
            "duration": 3.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hello world.",
                    "avg_logprob": 0.0,
                    "no_speech_prob": 0.0
                }
            ]
        }

    Satisfies the Exporter Protocol.
    """

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "json"

    def format(self, result: TranscriptionResult) -> str:
        """Format transcription as JSON string.

        Args:
            result: The transcription to format.

        Returns:
            JSON-formatted string.
        """
        data = {
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "segments": [asdict(seg) for seg in result.segments],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write JSON transcription to file.

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

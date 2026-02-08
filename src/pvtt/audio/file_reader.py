"""Audio file validation and metadata for pvtt."""

from __future__ import annotations

from pathlib import Path

from pvtt.exceptions import AudioError
from pvtt.util.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".opus",
    ".webm",
    ".mp4",
})


def validate_audio_file(path: Path) -> Path:
    """Validate that the given path is a readable audio file.

    Args:
        path: Path to the audio file.

    Returns:
        The validated Path.

    Raises:
        AudioError: If the file does not exist, is not a file,
                    or has an unsupported extension.
    """
    if not path.exists():
        raise AudioError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioError(f"Not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"Unsupported audio format: {path.suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def get_audio_duration(path: Path) -> float | None:
    """Get the duration of an audio file in seconds, or None if unavailable.

    Uses PyAV (bundled with faster-whisper) for format-agnostic duration.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds, or None if it cannot be determined.
    """
    try:
        import av  # type: ignore[import-untyped]

        with av.open(str(path)) as container:
            if container.duration is not None:
                return float(container.duration) / 1_000_000
    except Exception:
        logger.debug("Could not determine duration for %s", path)
    return None

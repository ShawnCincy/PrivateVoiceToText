"""Domain-specific exceptions for pvtt."""

from __future__ import annotations


class PvttError(Exception):
    """Base exception for all pvtt errors."""


class ConfigError(PvttError):
    """Configuration loading or validation failed."""


class ModelNotFoundError(PvttError):
    """Requested model is not downloaded locally."""


class ModelDownloadError(PvttError):
    """Model download failed."""


class EngineError(PvttError):
    """Inference engine error."""


class EngineNotFoundError(EngineError):
    """Requested engine backend is not available."""


class AudioError(PvttError):
    """Audio file reading or processing error."""


class AudioCaptureError(AudioError):
    """Audio capture (microphone) error."""


class ExportError(PvttError):
    """Output formatting/export error."""


class HardwareError(PvttError):
    """Hardware detection or capability error."""


class StreamingError(PvttError):
    """Streaming pipeline error."""

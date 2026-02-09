"""Tests for pvtt.exceptions."""

from __future__ import annotations

import pytest

from pvtt.exceptions import (
    AudioCaptureError,
    AudioError,
    ConfigError,
    EngineError,
    EngineNotFoundError,
    ExportError,
    HardwareError,
    ModelDownloadError,
    ModelNotFoundError,
    PvttError,
    StreamingError,
)


class TestExceptionHierarchy:
    """Verify all exceptions inherit from PvttError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigError,
            ModelNotFoundError,
            ModelDownloadError,
            EngineError,
            EngineNotFoundError,
            AudioError,
            AudioCaptureError,
            ExportError,
            HardwareError,
            StreamingError,
        ],
    )
    def test_inherits_from_pvtt_error(
        self, exc_class: type[PvttError]
    ) -> None:
        exc = exc_class("test message")

        assert isinstance(exc, PvttError)
        assert str(exc) == "test message"

    def test_engine_not_found_inherits_from_engine_error(self) -> None:
        exc = EngineNotFoundError("missing")

        assert isinstance(exc, EngineError)
        assert isinstance(exc, PvttError)

    def test_audio_capture_error_inherits_from_audio_error(self) -> None:
        exc = AudioCaptureError("no mic")

        assert isinstance(exc, AudioError)
        assert isinstance(exc, PvttError)

    def test_pvtt_error_is_exception(self) -> None:
        assert issubclass(PvttError, Exception)

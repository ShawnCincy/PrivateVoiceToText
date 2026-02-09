"""Tests for pvtt.audio.vad."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pvtt.audio.vad import (
    VAD_WINDOW_SIZE,
    SpeechSegment,
    VoiceActivityDetector,
)

_PATCH_TARGET = "faster_whisper.vad.get_vad_model"


def _make_frame(value: float = 0.0) -> np.ndarray:
    """Create a VAD-sized frame filled with a constant value."""
    return np.full(VAD_WINDOW_SIZE, value, dtype=np.float32)


def _make_detector(
    threshold: float = 0.5,
    min_silence_duration_ms: int = 100,
    min_speech_duration_ms: int = 0,
    max_speech_duration_s: float = 30.0,
) -> VoiceActivityDetector:
    """Create a VoiceActivityDetector with short silence thresholds for testing."""
    return VoiceActivityDetector(
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
    )


@pytest.fixture()
def mock_vad_model() -> MagicMock:
    """Mock the Silero VAD model callable."""
    model = MagicMock()
    # Default: return 0.0 (silence)
    model.return_value = 0.0
    return model


class TestSpeechSegment:
    """Tests for the SpeechSegment dataclass."""

    def test_create_segment(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        seg = SpeechSegment(audio=audio, start_time=0.0, end_time=0.0625)

        assert len(seg.audio) == 1000
        assert seg.start_time == 0.0
        assert seg.end_time == 0.0625

    def test_segment_is_frozen(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        seg = SpeechSegment(audio=audio, start_time=0.0, end_time=1.0)

        with pytest.raises(AttributeError):
            seg.start_time = 2.0  # type: ignore[misc]


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector."""

    def test_default_parameters(self) -> None:
        vad = VoiceActivityDetector()

        assert vad.threshold == 0.5
        assert vad.min_silence_duration_ms == 500
        assert vad.min_speech_duration_ms == 250
        assert vad.max_speech_duration_s == 30.0
        assert vad.sample_rate == 16000

    def test_wrong_frame_size_raises(self, mock_vad_model: MagicMock) -> None:
        vad = _make_detector()

        with pytest.raises(ValueError, match="512 samples"):
            vad.process_frame(np.zeros(100, dtype=np.float32))

    @patch(_PATCH_TARGET)
    def test_silence_frames_return_none(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model
        mock_vad_model.return_value = 0.1  # Below threshold

        vad = _make_detector()
        frame = _make_frame()

        result = vad.process_frame(frame)

        assert result is None

    @patch(_PATCH_TARGET)
    def test_speech_then_silence_emits_segment(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model

        # min_silence_duration_ms=100 at 16kHz with 512 samples/frame
        # = 32ms/frame → need ~4 silence frames (4*32=128 > 100)
        vad = _make_detector(min_silence_duration_ms=100, min_speech_duration_ms=0)
        frame = _make_frame(0.5)

        # Feed 3 speech frames
        mock_vad_model.return_value = 0.8
        for _ in range(3):
            result = vad.process_frame(frame)
            assert result is None

        # Feed silence frames until a segment is emitted
        mock_vad_model.return_value = 0.1
        segment = None
        for _ in range(10):
            segment = vad.process_frame(frame)
            if segment is not None:
                break

        assert segment is not None
        assert isinstance(segment, SpeechSegment)
        assert len(segment.audio) > 0

    @patch(_PATCH_TARGET)
    def test_segment_start_end_times(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model

        # Use min_silence_duration_ms=100 (~4 frames) for reliable detection
        vad = _make_detector(
            min_silence_duration_ms=100, min_speech_duration_ms=0,
        )
        frame = _make_frame()

        # 3 speech frames
        mock_vad_model.return_value = 0.9
        for _ in range(3):
            vad.process_frame(frame)

        # Feed enough silence frames to trigger segment emission
        mock_vad_model.return_value = 0.1
        segment = None
        for _ in range(10):
            segment = vad.process_frame(frame)
            if segment is not None:
                break

        assert segment is not None
        assert segment.start_time >= 0.0
        assert segment.end_time > segment.start_time

    @patch(_PATCH_TARGET)
    def test_min_speech_duration_filters_short_segments(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model

        # min_speech_duration_ms=200 → need ~7 frames (7*32=224ms)
        vad = _make_detector(
            min_silence_duration_ms=32,
            min_speech_duration_ms=200,
        )
        frame = _make_frame()

        # Only 1 speech frame (32ms < 200ms minimum)
        mock_vad_model.return_value = 0.9
        vad.process_frame(frame)

        # Then enough silence to trigger end
        mock_vad_model.return_value = 0.1
        segment = None
        for _ in range(5):
            segment = vad.process_frame(frame)
            if segment is not None:
                break

        # Should be discarded because too short
        assert segment is None

    @patch(_PATCH_TARGET)
    def test_max_duration_forced_split(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model
        mock_vad_model.return_value = 0.9  # Always speech

        # max 0.5s = ~16 frames (16*32ms = 512ms)
        vad = _make_detector(
            max_speech_duration_s=0.5,
            min_speech_duration_ms=0,
        )
        frame = _make_frame()

        segment = None
        for _ in range(50):  # Feed enough frames to trigger forced split
            segment = vad.process_frame(frame)
            if segment is not None:
                break

        assert segment is not None
        assert len(segment.audio) > 0

    @patch(_PATCH_TARGET)
    def test_flush_emits_pending_speech(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model
        mock_vad_model.return_value = 0.9

        vad = _make_detector(min_speech_duration_ms=0)
        frame = _make_frame()

        # Feed a few speech frames without ending
        for _ in range(3):
            vad.process_frame(frame)

        # Flush should emit the accumulated speech
        segment = vad.flush()

        assert segment is not None
        assert len(segment.audio) == 3 * VAD_WINDOW_SIZE

    @patch(_PATCH_TARGET)
    def test_flush_returns_none_when_silent(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model
        mock_vad_model.return_value = 0.1

        vad = _make_detector()
        frame = _make_frame()

        # Feed silence frames
        for _ in range(3):
            vad.process_frame(frame)

        segment = vad.flush()

        assert segment is None

    @patch(_PATCH_TARGET)
    def test_reset_clears_state(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model
        mock_vad_model.return_value = 0.9

        vad = _make_detector(min_speech_duration_ms=0)
        frame = _make_frame()

        # Accumulate some speech
        for _ in range(3):
            vad.process_frame(frame)

        vad.reset()

        # After reset, flush should return nothing
        segment = vad.flush()
        assert segment is None

    @patch(_PATCH_TARGET)
    def test_model_loaded_once(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model
        mock_vad_model.return_value = 0.1

        vad = _make_detector()
        frame = _make_frame()

        vad.process_frame(frame)
        vad.process_frame(frame)
        vad.process_frame(frame)

        # get_vad_model should be called only once (lazy loaded)
        mock_get_model.assert_called_once()

    @patch(_PATCH_TARGET)
    def test_speech_segment_audio_is_copy(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model

        vad = _make_detector(min_silence_duration_ms=32, min_speech_duration_ms=0)
        frame = _make_frame(0.42)

        # Speech
        mock_vad_model.return_value = 0.9
        vad.process_frame(frame)

        # Silence to trigger emit
        mock_vad_model.return_value = 0.1
        segment = None
        for _ in range(5):
            segment = vad.process_frame(frame)
            if segment is not None:
                break

        if segment is not None:
            # Modifying the original frame should not affect the segment audio
            frame[:] = 999.0
            assert not np.any(segment.audio == 999.0)

    @patch(_PATCH_TARGET)
    def test_multiple_segments_in_sequence(
        self, mock_get_model: MagicMock, mock_vad_model: MagicMock,
    ) -> None:
        mock_get_model.return_value = mock_vad_model

        vad = _make_detector(min_silence_duration_ms=32, min_speech_duration_ms=0)
        frame = _make_frame()
        segments: list[SpeechSegment] = []

        # First speech burst
        mock_vad_model.return_value = 0.9
        for _ in range(3):
            result = vad.process_frame(frame)
            if result:
                segments.append(result)

        # Silence
        mock_vad_model.return_value = 0.1
        for _ in range(5):
            result = vad.process_frame(frame)
            if result:
                segments.append(result)

        # Second speech burst
        mock_vad_model.return_value = 0.9
        for _ in range(3):
            result = vad.process_frame(frame)
            if result:
                segments.append(result)

        # More silence
        mock_vad_model.return_value = 0.1
        for _ in range(5):
            result = vad.process_frame(frame)
            if result:
                segments.append(result)

        # Should have detected at least 2 segments
        assert len(segments) >= 2
        # Second segment should start after first
        assert segments[1].start_time > segments[0].start_time

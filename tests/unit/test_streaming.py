"""Tests for pvtt.core.streaming."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pvtt.audio.capture import AudioFrame
from pvtt.audio.vad import VAD_WINDOW_SIZE, SpeechSegment
from pvtt.config.schema import PvttConfig
from pvtt.core.streaming import (
    StreamingEvent,
    StreamingEventType,
    StreamingPipeline,
)
from pvtt.exceptions import StreamingError
from pvtt.util.types import TranscriptionSegment


def _make_config() -> PvttConfig:
    """Create a test config."""
    return PvttConfig.model_validate({
        "model": {"name": "tiny.en", "device": "cpu", "compute_type": "int8"},
        "streaming": {
            "vad_threshold": 0.5,
            "min_silence_duration_ms": 100,
            "min_speech_duration_ms": 0,
            "max_speech_duration_s": 30.0,
        },
    })


def _make_mock_engine(loaded: bool = True) -> MagicMock:
    """Create a mock engine with is_loaded property."""
    engine = MagicMock()
    engine.is_loaded = loaded
    engine.transcribe_audio.return_value = iter([
        TranscriptionSegment(start=0.0, end=1.0, text="Hello."),
    ])
    return engine


class TestStreamingEvent:
    """Tests for StreamingEvent dataclass."""

    def test_create_segment_event(self) -> None:
        seg = TranscriptionSegment(start=0.0, end=1.0, text="Hi.")
        event = StreamingEvent(
            type=StreamingEventType.SEGMENT,
            segment=seg,
        )

        assert event.type == StreamingEventType.SEGMENT
        assert event.segment is not None
        assert event.segment.text == "Hi."

    def test_create_status_event(self) -> None:
        event = StreamingEvent(
            type=StreamingEventType.STATUS,
            text="Listening...",
        )

        assert event.type == StreamingEventType.STATUS
        assert event.text == "Listening..."
        assert event.segment is None

    def test_event_is_frozen(self) -> None:
        event = StreamingEvent(type=StreamingEventType.STATUS, text="Test")

        with pytest.raises(AttributeError):
            event.text = "changed"  # type: ignore[misc]


class TestStreamingEventType:
    """Tests for StreamingEventType enum."""

    def test_all_types_exist(self) -> None:
        assert StreamingEventType.SEGMENT.value == "segment"
        assert StreamingEventType.PARTIAL.value == "partial"
        assert StreamingEventType.STATUS.value == "status"
        assert StreamingEventType.ERROR.value == "error"


class TestStreamingPipeline:
    """Tests for StreamingPipeline."""

    def test_not_running_initially(self) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        callback = MagicMock()

        pipeline = StreamingPipeline(config, engine, callback)

        assert pipeline.is_running is False

    def test_start_raises_if_engine_not_loaded(self) -> None:
        config = _make_config()
        engine = _make_mock_engine(loaded=False)
        callback = MagicMock()

        pipeline = StreamingPipeline(config, engine, callback)

        with pytest.raises(StreamingError, match="no model loaded"):
            pipeline.start()

    def test_stop_when_not_running_is_noop(self) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        callback = MagicMock()

        pipeline = StreamingPipeline(config, engine, callback)

        pipeline.stop()  # Should not raise

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_start_creates_capture_and_vad(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        callback = MagicMock()

        mock_capture = MagicMock()
        mock_capture.get_frame.return_value = None
        mock_capture_cls.return_value = mock_capture

        mock_vad = MagicMock()
        mock_vad.process_frame.return_value = None
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()

        # Let it run briefly
        time.sleep(0.1)
        pipeline.stop()

        mock_capture_cls.assert_called_once()
        mock_capture.start.assert_called_once()
        mock_vad_cls.assert_called_once()

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_start_twice_raises(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        callback = MagicMock()

        mock_capture = MagicMock()
        mock_capture.get_frame.return_value = None
        mock_capture_cls.return_value = mock_capture

        mock_vad = MagicMock()
        mock_vad.process_frame.return_value = None
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()

        try:
            with pytest.raises(StreamingError, match="already running"):
                pipeline.start()
        finally:
            pipeline.stop()

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_emits_status_events(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        events: list[StreamingEvent] = []
        def callback(e: StreamingEvent) -> None:
            events.append(e)

        mock_capture = MagicMock()
        mock_capture.get_frame.return_value = None
        mock_capture_cls.return_value = mock_capture

        mock_vad = MagicMock()
        mock_vad.process_frame.return_value = None
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()

        status_events = [e for e in events if e.type == StreamingEventType.STATUS]
        assert len(status_events) >= 2  # "started" and "stopped"
        assert "started" in status_events[0].text.lower()

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_transcribes_speech_segments(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        events: list[StreamingEvent] = []
        def callback(e: StreamingEvent) -> None:
            events.append(e)

        # Set up capture to return a few frames then None
        frame_data = np.zeros(VAD_WINDOW_SIZE, dtype=np.float32)
        frame = AudioFrame(data=frame_data, sample_rate=16000)
        call_count = 0

        def get_frame_side_effect(timeout: float = 1.0) -> AudioFrame | None:
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return frame
            return None

        mock_capture = MagicMock()
        mock_capture.get_frame.side_effect = get_frame_side_effect
        mock_capture_cls.return_value = mock_capture

        # Set up VAD to emit a speech segment on the 3rd frame
        speech_audio = np.zeros(VAD_WINDOW_SIZE * 3, dtype=np.float32)
        speech_segment = SpeechSegment(
            audio=speech_audio, start_time=0.0, end_time=0.096,
        )
        vad_call_count = 0

        def vad_side_effect(frame_arg: object) -> SpeechSegment | None:
            nonlocal vad_call_count
            vad_call_count += 1
            if vad_call_count == 3:
                return speech_segment
            return None

        mock_vad = MagicMock()
        mock_vad.process_frame.side_effect = vad_side_effect
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()

        segment_events = [e for e in events if e.type == StreamingEventType.SEGMENT]
        assert len(segment_events) >= 1
        assert segment_events[0].segment is not None
        assert "Hello" in segment_events[0].segment.text

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_flushes_on_stop(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        events: list[StreamingEvent] = []
        def callback(e: StreamingEvent) -> None:
            events.append(e)

        mock_capture = MagicMock()
        mock_capture.get_frame.return_value = None
        mock_capture_cls.return_value = mock_capture

        # VAD flush returns a segment
        speech_audio = np.zeros(VAD_WINDOW_SIZE * 2, dtype=np.float32)
        flush_segment = SpeechSegment(
            audio=speech_audio, start_time=0.0, end_time=0.064,
        )
        mock_vad = MagicMock()
        mock_vad.process_frame.return_value = None
        mock_vad.flush.return_value = flush_segment
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()

        segment_events = [e for e in events if e.type == StreamingEventType.SEGMENT]
        assert len(segment_events) >= 1

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_stop_cleans_up_resources(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        callback = MagicMock()

        mock_capture = MagicMock()
        mock_capture.get_frame.return_value = None
        mock_capture_cls.return_value = mock_capture

        mock_vad = MagicMock()
        mock_vad.process_frame.return_value = None
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()

        mock_capture.stop.assert_called()
        assert pipeline.is_running is False

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_error_in_pipeline_emits_error_event(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        config = _make_config()
        engine = _make_mock_engine()
        engine.transcribe_audio.side_effect = RuntimeError("boom")
        events: list[StreamingEvent] = []
        def callback(e: StreamingEvent) -> None:
            events.append(e)

        # Capture returns one frame
        frame = AudioFrame(
            data=np.zeros(VAD_WINDOW_SIZE, dtype=np.float32),
            sample_rate=16000,
        )
        mock_capture = MagicMock()
        mock_capture.get_frame.side_effect = [frame, None, None, None, None]
        mock_capture_cls.return_value = mock_capture

        # VAD returns speech on first frame
        speech = SpeechSegment(
            audio=np.zeros(VAD_WINDOW_SIZE, dtype=np.float32),
            start_time=0.0,
            end_time=0.032,
        )
        mock_vad = MagicMock()
        mock_vad.process_frame.return_value = speech
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()

        error_events = [e for e in events if e.type == StreamingEventType.ERROR]
        assert len(error_events) >= 1

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_callback_exception_does_not_crash_pipeline(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        config = _make_config()
        engine = _make_mock_engine()

        def bad_callback(event: StreamingEvent) -> None:
            raise RuntimeError("Callback crashed")

        mock_capture = MagicMock()
        mock_capture.get_frame.return_value = None
        mock_capture_cls.return_value = mock_capture

        mock_vad = MagicMock()
        mock_vad.process_frame.return_value = None
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, bad_callback)

        # Should not raise even though callback raises
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()

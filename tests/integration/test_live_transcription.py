"""Integration tests for live transcription pipeline.

These tests verify the streaming pipeline works end-to-end with
mocked audio and engine components.
"""

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
from pvtt.util.types import TranscriptionSegment


def _make_config() -> PvttConfig:
    """Create a test config for streaming."""
    return PvttConfig.model_validate({
        "model": {"name": "tiny.en", "device": "cpu", "compute_type": "int8"},
        "streaming": {
            "vad_threshold": 0.5,
            "min_silence_duration_ms": 100,
            "min_speech_duration_ms": 0,
            "max_speech_duration_s": 30.0,
        },
    })


@pytest.mark.integration
class TestLiveTranscriptionPipeline:
    """Integration tests for the streaming pipeline."""

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_full_pipeline_start_stop(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        """Pipeline starts, runs, and stops cleanly."""
        config = _make_config()
        engine = MagicMock()
        engine.is_loaded = True
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
        assert pipeline.is_running is True

        time.sleep(0.1)
        pipeline.stop()
        assert pipeline.is_running is False

        # Should have start and stop status events
        status_events = [
            e for e in events if e.type == StreamingEventType.STATUS
        ]
        assert len(status_events) >= 2

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_segment_delivery_end_to_end(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        """Segments detected by VAD are transcribed and delivered."""
        config = _make_config()
        engine = MagicMock()
        engine.is_loaded = True
        engine.transcribe_audio.return_value = iter([
            TranscriptionSegment(
                start=0.0, end=1.5, text="Hello, how are you?"
            ),
        ])

        events: list[StreamingEvent] = []

        def callback(e: StreamingEvent) -> None:
            events.append(e)

        # Capture returns a few frames then None
        frame = AudioFrame(
            data=np.zeros(VAD_WINDOW_SIZE, dtype=np.float32),
            sample_rate=16000,
        )
        call_count = 0

        def get_frame_side_effect(timeout: float = 1.0) -> AudioFrame | None:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return frame
            return None

        mock_capture = MagicMock()
        mock_capture.get_frame.side_effect = get_frame_side_effect
        mock_capture_cls.return_value = mock_capture

        # VAD emits speech on 2nd frame
        speech = SpeechSegment(
            audio=np.zeros(VAD_WINDOW_SIZE * 2, dtype=np.float32),
            start_time=0.0,
            end_time=0.064,
        )
        vad_count = 0

        def vad_side_effect(f: object) -> SpeechSegment | None:
            nonlocal vad_count
            vad_count += 1
            if vad_count == 2:
                return speech
            return None

        mock_vad = MagicMock()
        mock_vad.process_frame.side_effect = vad_side_effect
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()

        segment_events = [
            e for e in events if e.type == StreamingEventType.SEGMENT
        ]
        assert len(segment_events) >= 1
        assert segment_events[0].segment is not None
        assert "Hello" in segment_events[0].segment.text

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_multiple_segments_accumulated(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        """Multiple speech segments are all delivered."""
        config = _make_config()
        engine = MagicMock()
        engine.is_loaded = True

        seg_texts = ["First segment.", "Second segment.", "Third segment."]
        call_idx = 0

        def transcribe_side_effect(
            audio: object, options: object,
        ) -> list[TranscriptionSegment]:
            nonlocal call_idx
            idx = min(call_idx, len(seg_texts) - 1)
            call_idx += 1
            return iter([
                TranscriptionSegment(
                    start=0.0, end=1.0, text=seg_texts[idx]
                ),
            ])

        engine.transcribe_audio.side_effect = transcribe_side_effect

        events: list[StreamingEvent] = []

        def callback(e: StreamingEvent) -> None:
            events.append(e)

        # Return frames for a while
        frame = AudioFrame(
            data=np.zeros(VAD_WINDOW_SIZE, dtype=np.float32),
            sample_rate=16000,
        )
        frame_count = 0

        def get_frame_side_effect(timeout: float = 1.0) -> AudioFrame | None:
            nonlocal frame_count
            frame_count += 1
            if frame_count <= 10:
                return frame
            return None

        mock_capture = MagicMock()
        mock_capture.get_frame.side_effect = get_frame_side_effect
        mock_capture_cls.return_value = mock_capture

        # VAD emits speech on every 3rd frame
        speech = SpeechSegment(
            audio=np.zeros(VAD_WINDOW_SIZE * 2, dtype=np.float32),
            start_time=0.0,
            end_time=0.064,
        )
        vad_count = 0

        def vad_side_effect(f: object) -> SpeechSegment | None:
            nonlocal vad_count
            vad_count += 1
            if vad_count % 3 == 0:
                return speech
            return None

        mock_vad = MagicMock()
        mock_vad.process_frame.side_effect = vad_side_effect
        mock_vad.flush.return_value = None
        mock_vad_cls.return_value = mock_vad

        pipeline = StreamingPipeline(config, engine, callback)
        pipeline.start()
        time.sleep(0.5)
        pipeline.stop()

        segment_events = [
            e for e in events if e.type == StreamingEventType.SEGMENT
        ]
        assert len(segment_events) >= 2
        texts = [e.segment.text for e in segment_events if e.segment]
        assert "First segment." in texts

    @patch("pvtt.core.streaming.AudioCapture")
    @patch("pvtt.core.streaming.VoiceActivityDetector")
    def test_error_recovery(
        self, mock_vad_cls: MagicMock, mock_capture_cls: MagicMock,
    ) -> None:
        """Pipeline continues after transcription errors."""
        config = _make_config()
        engine = MagicMock()
        engine.is_loaded = True
        engine.transcribe_audio.side_effect = RuntimeError("Engine failed")

        events: list[StreamingEvent] = []

        def callback(e: StreamingEvent) -> None:
            events.append(e)

        frame = AudioFrame(
            data=np.zeros(VAD_WINDOW_SIZE, dtype=np.float32),
            sample_rate=16000,
        )
        frame_count = 0

        def get_frame_side_effect(timeout: float = 1.0) -> AudioFrame | None:
            nonlocal frame_count
            frame_count += 1
            if frame_count <= 3:
                return frame
            return None

        mock_capture = MagicMock()
        mock_capture.get_frame.side_effect = get_frame_side_effect
        mock_capture_cls.return_value = mock_capture

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

        # Should have error events but pipeline still stopped cleanly
        error_events = [
            e for e in events if e.type == StreamingEventType.ERROR
        ]
        assert len(error_events) >= 1
        assert pipeline.is_running is False

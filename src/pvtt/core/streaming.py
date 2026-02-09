"""Streaming transcription pipeline for pvtt.

Three-thread architecture:
  - Main thread:     CLI / Rich Live display (caller)
  - Audio thread:    PortAudio callback → queue (managed by AudioCapture)
  - Pipeline thread: queue → VAD → engine.transcribe_audio() → on_event callback

The pipeline thread reads audio frames from AudioCapture, feeds them
through VAD to detect speech segments, then transcribes each segment
and delivers results via a callback.
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pvtt.audio.capture import AudioCapture
from pvtt.audio.preprocessing import WHISPER_SAMPLE_RATE, prepare_audio_chunk
from pvtt.audio.vad import VAD_WINDOW_SIZE, SpeechSegment, VoiceActivityDetector
from pvtt.config.schema import PvttConfig
from pvtt.engine.base import InferenceEngine
from pvtt.exceptions import StreamingError
from pvtt.util.logging import get_logger
from pvtt.util.types import TranscribeOptions, TranscriptionSegment

logger = get_logger(__name__)


class StreamingEventType(enum.Enum):
    """Types of streaming events."""

    SEGMENT = "segment"
    PARTIAL = "partial"
    STATUS = "status"
    ERROR = "error"


@dataclass(frozen=True)
class StreamingEvent:
    """An event emitted by the streaming pipeline.

    Attributes:
        type: The event type.
        segment: Transcription segment (for SEGMENT events).
        text: Status or error message (for STATUS/ERROR events).
    """

    type: StreamingEventType
    segment: TranscriptionSegment | None = None
    text: str = ""


class StreamingPipeline:
    """Orchestrates real-time audio capture → VAD → transcription.

    Args:
        config: Full pvtt configuration.
        engine: Loaded inference engine.
        on_event: Callback invoked for each streaming event.
    """

    def __init__(
        self,
        config: PvttConfig,
        engine: InferenceEngine,
        on_event: Callable[[StreamingEvent], None],
    ) -> None:
        self._config = config
        self._engine = engine
        self._on_event = on_event
        self._stop_event = threading.Event()
        self._pipeline_thread: threading.Thread | None = None
        self._capture: AudioCapture | None = None
        self._vad: VoiceActivityDetector | None = None
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is currently running."""
        return (
            self._pipeline_thread is not None
            and self._pipeline_thread.is_alive()
        )

    def start(self) -> None:
        """Start the streaming pipeline.

        Sets up audio capture, VAD, and starts the pipeline thread.

        Raises:
            StreamingError: If the pipeline is already running or setup fails.
        """
        if self.is_running:
            raise StreamingError("Streaming pipeline is already running")

        if not self._engine.is_loaded:
            raise StreamingError("Engine has no model loaded")

        try:
            streaming_cfg = self._config.streaming

            # Create audio capture
            self._capture = AudioCapture(
                device=streaming_cfg.audio_device,
                sample_rate=WHISPER_SAMPLE_RATE,
                channels=1,
                block_size=VAD_WINDOW_SIZE,
            )

            # Create VAD
            self._vad = VoiceActivityDetector(
                threshold=streaming_cfg.vad_threshold,
                min_silence_duration_ms=streaming_cfg.min_silence_duration_ms,
                min_speech_duration_ms=streaming_cfg.min_speech_duration_ms,
                max_speech_duration_s=streaming_cfg.max_speech_duration_s,
            )

            self._stop_event.clear()
            self._start_time = time.monotonic()

            # Start audio capture (this starts the audio thread)
            self._capture.start()

            # Start pipeline thread
            self._pipeline_thread = threading.Thread(
                target=self._pipeline_loop,
                name="pvtt-pipeline",
                daemon=True,
            )
            self._pipeline_thread.start()

            self._emit_event(StreamingEvent(
                type=StreamingEventType.STATUS,
                text="Streaming started — listening...",
            ))

            logger.info("Streaming pipeline started")

        except StreamingError:
            raise
        except Exception as exc:
            self._cleanup()
            raise StreamingError(
                f"Failed to start streaming pipeline: {exc}"
            ) from exc

    def stop(self) -> None:
        """Stop the streaming pipeline gracefully.

        Signals the pipeline thread to stop, flushes any pending
        speech, and cleans up resources.
        """
        if not self.is_running:
            return

        logger.info("Stopping streaming pipeline...")
        self._stop_event.set()

        # Wait for pipeline thread to finish
        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=5.0)

        self._cleanup()

        self._emit_event(StreamingEvent(
            type=StreamingEventType.STATUS,
            text="Streaming stopped.",
        ))

        logger.info("Streaming pipeline stopped")

    def _cleanup(self) -> None:
        """Release all resources."""
        if self._capture is not None:
            self._capture.stop()
            self._capture = None

        if self._vad is not None:
            self._vad.reset()
            self._vad = None

        self._pipeline_thread = None

    def _pipeline_loop(self) -> None:
        """Main loop for the pipeline thread.

        Reads frames from audio capture, processes through VAD,
        and transcribes detected speech segments.
        """
        assert self._capture is not None
        assert self._vad is not None

        try:
            while not self._stop_event.is_set():
                frame = self._capture.get_frame(timeout=0.1)
                if frame is None:
                    continue

                # Process frame through VAD
                speech_segment = self._vad.process_frame(frame.data)
                if speech_segment is not None:
                    self._transcribe_segment(speech_segment)

            # Flush any remaining speech
            remaining = self._vad.flush()
            if remaining is not None:
                self._transcribe_segment(remaining)

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            self._emit_event(StreamingEvent(
                type=StreamingEventType.ERROR,
                text=f"Pipeline error: {exc}",
            ))

    def _transcribe_segment(self, speech: SpeechSegment) -> None:
        """Transcribe a detected speech segment.

        Args:
            speech: The speech segment from VAD.
        """
        try:
            options = TranscribeOptions(
                language=self._config.transcription.language,
                beam_size=self._config.transcription.beam_size,
                temperature=self._config.transcription.temperature,
                initial_prompt=self._config.transcription.initial_prompt,
                vad_filter=False,  # We already did VAD
                word_timestamps=self._config.transcription.word_timestamps,
            )

            # Prepare audio (already mono 16kHz from capture, but ensure)
            audio = prepare_audio_chunk(
                speech.audio,
                orig_sr=WHISPER_SAMPLE_RATE,
                target_sr=WHISPER_SAMPLE_RATE,
                normalize=True,
            )

            for seg in self._engine.transcribe_audio(audio, options):
                # Adjust timestamps to be relative to stream start
                adjusted = TranscriptionSegment(
                    start=speech.start_time + seg.start,
                    end=speech.start_time + seg.end,
                    text=seg.text,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                )
                self._emit_event(StreamingEvent(
                    type=StreamingEventType.SEGMENT,
                    segment=adjusted,
                ))

        except Exception as exc:
            logger.error("Transcription error: %s", exc, exc_info=True)
            self._emit_event(StreamingEvent(
                type=StreamingEventType.ERROR,
                text=f"Transcription error: {exc}",
            ))

    def _emit_event(self, event: StreamingEvent) -> None:
        """Deliver an event to the callback."""
        try:
            self._on_event(event)
        except Exception:
            logger.debug("Error in event callback", exc_info=True)

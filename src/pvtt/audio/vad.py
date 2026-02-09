"""Voice Activity Detection wrapper for pvtt.

Wraps the Silero VAD model (bundled with faster-whisper as ONNX) for
frame-by-frame speech detection in the streaming pipeline.  No network
calls — the model is loaded from the faster-whisper package.

State machine: SILENCE → SPEECH → SILENCE.  Accumulates speech frames
and emits ``SpeechSegment`` objects when silence exceeds the configured
threshold or when max duration is hit.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pvtt.audio.preprocessing import WHISPER_SAMPLE_RATE
from pvtt.util.logging import get_logger

logger = get_logger(__name__)

# Silero VAD expects 512-sample windows at 16 kHz (32 ms per frame)
VAD_WINDOW_SIZE: int = 512
"""Number of samples per VAD window (512 @ 16 kHz = 32 ms)."""


@dataclass(frozen=True)
class SpeechSegment:
    """A contiguous segment of detected speech.

    Attributes:
        audio: Speech audio as float32 mono at 16 kHz.
        start_time: Start time in seconds relative to stream start.
        end_time: End time in seconds relative to stream start.
    """

    audio: npt.NDArray[np.float32]
    start_time: float
    end_time: float


class _VadState(enum.Enum):
    """Internal state for the VAD state machine."""

    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class VoiceActivityDetector:
    """Frame-by-frame voice activity detector using Silero VAD.

    Processes audio in 512-sample windows and tracks speech/silence
    state.  When a speech segment ends (due to silence or max
    duration), it is emitted via :meth:`process_frame`.

    Args:
        threshold: Speech probability threshold (0.0-1.0).
        min_silence_duration_ms: Minimum silence duration (ms) to end a
            speech segment.
        min_speech_duration_ms: Minimum speech duration (ms) to accept
            a speech segment.
        max_speech_duration_s: Maximum speech duration (s) before forced
            split.
        sample_rate: Expected sample rate of input audio.
    """

    threshold: float = 0.5
    min_silence_duration_ms: int = 500
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 30.0
    sample_rate: int = WHISPER_SAMPLE_RATE

    # Internal state (not constructor args)
    _state: _VadState = field(default=_VadState.SILENCE, init=False, repr=False)
    _speech_frames: list[npt.NDArray[np.float32]] = field(
        default_factory=list, init=False, repr=False,
    )
    _speech_start_time: float = field(default=0.0, init=False, repr=False)
    _silence_frames_count: int = field(default=0, init=False, repr=False)
    _total_frames_processed: int = field(default=0, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)

    @property
    def _min_silence_frames(self) -> int:
        """Number of consecutive silence frames to trigger end of speech."""
        frame_duration_ms = (VAD_WINDOW_SIZE / self.sample_rate) * 1000
        return max(1, int(self.min_silence_duration_ms / frame_duration_ms))

    @property
    def _min_speech_frames(self) -> int:
        """Minimum number of speech frames to accept a segment."""
        frame_duration_ms = (VAD_WINDOW_SIZE / self.sample_rate) * 1000
        return max(0, int(self.min_speech_duration_ms / frame_duration_ms))

    @property
    def _max_speech_frames(self) -> int:
        """Maximum number of speech frames before forced split."""
        frame_duration_ms = (VAD_WINDOW_SIZE / self.sample_rate) * 1000
        return int((self.max_speech_duration_s * 1000) / frame_duration_ms)

    def _ensure_model(self) -> None:
        """Lazily load the Silero VAD ONNX model."""
        if self._model is not None:
            return

        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()
        logger.debug("Loaded Silero VAD ONNX model")

    def _get_speech_prob(self, frame: npt.NDArray[np.float32]) -> float:
        """Get speech probability for a single frame.

        Args:
            frame: Audio frame of exactly ``VAD_WINDOW_SIZE`` samples.

        Returns:
            Speech probability between 0.0 and 1.0.
        """
        self._ensure_model()
        # faster-whisper's SileroVADModel.__call__(audio, num_samples=512)
        # Returns ndarray of probabilities, one per num_samples chunk.
        result = self._model(frame, num_samples=VAD_WINDOW_SIZE)  # type: ignore[misc]
        # Result is ndarray for real model; extract last probability value
        if isinstance(result, np.ndarray):
            return float(result.flatten()[-1])
        return float(result)

    def _current_time(self) -> float:
        """Current time in seconds based on frames processed."""
        return (self._total_frames_processed * VAD_WINDOW_SIZE) / self.sample_rate

    def process_frame(
        self, frame: npt.NDArray[np.float32],
    ) -> SpeechSegment | None:
        """Process a single audio frame and return a speech segment if ready.

        Args:
            frame: Audio frame of exactly ``VAD_WINDOW_SIZE`` float32
                samples at ``sample_rate``.

        Returns:
            A ``SpeechSegment`` if a complete speech segment was
            detected, otherwise ``None``.
        """
        if len(frame) != VAD_WINDOW_SIZE:
            raise ValueError(
                f"Frame must be {VAD_WINDOW_SIZE} samples, got {len(frame)}"
            )

        prob = self._get_speech_prob(frame)
        is_speech = prob >= self.threshold
        segment: SpeechSegment | None = None

        if self._state == _VadState.SILENCE:
            if is_speech:
                # Transition: SILENCE → SPEECH
                self._state = _VadState.SPEECH
                self._speech_start_time = self._current_time()
                self._speech_frames = [frame.copy()]
                self._silence_frames_count = 0
                logger.debug(
                    "Speech started at %.2fs (prob=%.3f)",
                    self._speech_start_time, prob,
                )
        else:
            # Currently in SPEECH state
            self._speech_frames.append(frame.copy())

            if not is_speech:
                self._silence_frames_count += 1

                if self._silence_frames_count >= self._min_silence_frames:
                    # Transition: SPEECH → SILENCE (enough silence)
                    segment = self._emit_segment()
            else:
                self._silence_frames_count = 0

            # Check max duration forced split
            if (
                segment is None
                and len(self._speech_frames) >= self._max_speech_frames
            ):
                logger.debug(
                    "Forced split at %.2fs (max duration)",
                    self._current_time(),
                )
                segment = self._emit_segment()

        self._total_frames_processed += 1
        return segment

    def _emit_segment(self) -> SpeechSegment | None:
        """Emit accumulated speech frames as a SpeechSegment.

        Returns:
            A SpeechSegment if enough speech was accumulated, else None.
        """
        n_frames = len(self._speech_frames)

        # Reset state
        self._state = _VadState.SILENCE
        self._silence_frames_count = 0

        if n_frames < self._min_speech_frames:
            logger.debug(
                "Discarding short speech segment (%d frames < %d min)",
                n_frames, self._min_speech_frames,
            )
            self._speech_frames = []
            return None

        audio = np.concatenate(self._speech_frames)
        end_time = self._speech_start_time + (len(audio) / self.sample_rate)

        segment = SpeechSegment(
            audio=audio,
            start_time=self._speech_start_time,
            end_time=end_time,
        )
        self._speech_frames = []

        logger.debug(
            "Speech segment: %.2fs - %.2fs (%d samples)",
            segment.start_time, segment.end_time, len(audio),
        )
        return segment

    def flush(self) -> SpeechSegment | None:
        """Flush any remaining speech frames as a segment.

        Call this when the audio stream ends to emit any pending speech.

        Returns:
            A SpeechSegment if speech was in progress, else None.
        """
        if self._state == _VadState.SPEECH and self._speech_frames:
            return self._emit_segment()
        return None

    def reset(self) -> None:
        """Reset internal state for a new stream."""
        self._state = _VadState.SILENCE
        self._speech_frames = []
        self._speech_start_time = 0.0
        self._silence_frames_count = 0
        self._total_frames_processed = 0

"""Audio capture via sounddevice (PortAudio) for pvtt.

Provides microphone input capture for the streaming transcription
pipeline.  The PortAudio callback does minimal work — it copies the
audio frame into a queue for the pipeline thread to consume.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pvtt.audio.preprocessing import WHISPER_SAMPLE_RATE
from pvtt.exceptions import AudioCaptureError
from pvtt.util.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AudioFrame:
    """A captured audio frame from the microphone.

    Attributes:
        data: Audio samples as float32 mono.
        sample_rate: Sample rate of the captured audio.
    """

    data: npt.NDArray[np.float32]
    sample_rate: int


class AudioCapture:
    """Captures audio from an input device using sounddevice.

    Uses a PortAudio input stream with a callback that pushes
    ``AudioFrame`` objects into a queue.  The streaming pipeline
    thread can read frames via :meth:`get_frame`.

    Args:
        device: Audio device index, or None for system default.
        sample_rate: Desired sample rate. Defaults to 16000 (Whisper).
        channels: Number of channels. Defaults to 1 (mono).
        block_size: Number of samples per callback. Defaults to 512
            (matches VAD window size).
    """

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        channels: int = 1,
        block_size: int = 512,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=100)
        self._stream: object | None = None
        self._running = threading.Event()

    @property
    def sample_rate(self) -> int:
        """Return the configured sample rate."""
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        """Return True if the audio stream is currently capturing."""
        return self._running.is_set()

    def start(self) -> None:
        """Start capturing audio from the input device.

        Raises:
            AudioCaptureError: If the audio device cannot be opened.
        """
        if self._running.is_set():
            logger.warning("Audio capture already running")
            return

        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                device=self._device,
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._block_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()  # type: ignore[union-attr]
            self._running.set()

            device_info = self._device if self._device is not None else "default"
            logger.info(
                "Audio capture started: device=%s, rate=%d, channels=%d",
                device_info, self._sample_rate, self._channels,
            )
        except Exception as exc:
            raise AudioCaptureError(
                f"Failed to open audio device: {exc}"
            ) from exc

    def stop(self) -> None:
        """Stop capturing audio and close the stream."""
        if not self._running.is_set():
            return

        self._running.clear()

        if self._stream is not None:
            try:
                self._stream.stop()  # type: ignore[union-attr]
                self._stream.close()  # type: ignore[union-attr]
            except Exception:
                logger.debug("Error closing audio stream", exc_info=True)
            finally:
                self._stream = None

        logger.info("Audio capture stopped")

    def get_frame(self, timeout: float = 1.0) -> AudioFrame | None:
        """Get the next audio frame from the capture queue.

        Args:
            timeout: Maximum seconds to wait for a frame.

        Returns:
            An AudioFrame, or None if no frame is available.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _audio_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """PortAudio callback — called from the audio thread.

        Minimal work: copy data and enqueue. No processing here.
        """
        if status:
            logger.debug("Audio callback status: %s", status)

        try:
            # Squeeze to 1D if mono (sounddevice gives (frames, channels))
            audio_data = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
            frame = AudioFrame(data=audio_data, sample_rate=self._sample_rate)
            self._queue.put_nowait(frame)
        except queue.Full:
            logger.debug("Audio frame dropped — queue full")


def list_audio_devices() -> list[dict[str, object]]:
    """List available audio input devices.

    Returns:
        List of device info dicts with keys: index, name,
        max_input_channels, default_samplerate.

    Raises:
        AudioCaptureError: If sounddevice cannot query devices.
    """
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):  # type: ignore[arg-type]
            if dev["max_input_channels"] > 0:  # type: ignore[index]
                input_devices.append({
                    "index": i,
                    "name": dev["name"],  # type: ignore[index]
                    "max_input_channels": dev["max_input_channels"],  # type: ignore[index]
                    "default_samplerate": dev["default_samplerate"],  # type: ignore[index]
                })
        return input_devices
    except Exception as exc:
        raise AudioCaptureError(
            f"Failed to query audio devices: {exc}"
        ) from exc


def get_default_input_device() -> dict[str, object] | None:
    """Get the default audio input device.

    Returns:
        Device info dict, or None if no input device is available.

    Raises:
        AudioCaptureError: If sounddevice cannot query devices.
    """
    try:
        import sounddevice as sd

        device_id = sd.default.device[0]  # type: ignore[index]
        if device_id is None or device_id < 0:
            return None

        dev = sd.query_devices(device_id)
        if dev["max_input_channels"] <= 0:  # type: ignore[index]
            return None

        return {
            "index": device_id,
            "name": dev["name"],  # type: ignore[index]
            "max_input_channels": dev["max_input_channels"],  # type: ignore[index]
            "default_samplerate": dev["default_samplerate"],  # type: ignore[index]
        }
    except Exception as exc:
        raise AudioCaptureError(
            f"Failed to query default input device: {exc}"
        ) from exc

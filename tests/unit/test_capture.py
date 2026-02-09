"""Tests for pvtt.audio.capture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pvtt.audio.capture import (
    AudioCapture,
    AudioFrame,
    get_default_input_device,
    list_audio_devices,
)
from pvtt.exceptions import AudioCaptureError


class TestAudioFrame:
    """Tests for the AudioFrame dataclass."""

    def test_create_frame(self) -> None:
        data = np.zeros(512, dtype=np.float32)
        frame = AudioFrame(data=data, sample_rate=16000)

        assert len(frame.data) == 512
        assert frame.sample_rate == 16000

    def test_frame_is_frozen(self) -> None:
        data = np.zeros(512, dtype=np.float32)
        frame = AudioFrame(data=data, sample_rate=16000)

        with pytest.raises(AttributeError):
            frame.sample_rate = 44100  # type: ignore[misc]


class TestAudioCapture:
    """Tests for AudioCapture."""

    def test_default_parameters(self) -> None:
        capture = AudioCapture()

        assert capture.sample_rate == 16000
        assert capture.is_running is False

    def test_custom_parameters(self) -> None:
        capture = AudioCapture(device=2, sample_rate=44100, channels=2)

        assert capture.sample_rate == 44100

    @patch("sounddevice.InputStream")
    def test_start_opens_stream(self, mock_input_stream: MagicMock) -> None:
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture = AudioCapture()
        capture.start()

        mock_input_stream.assert_called_once()
        mock_stream.start.assert_called_once()
        assert capture.is_running is True

    @patch("sounddevice.InputStream")
    def test_stop_closes_stream(self, mock_input_stream: MagicMock) -> None:
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture = AudioCapture()
        capture.start()
        capture.stop()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert capture.is_running is False

    @patch("sounddevice.InputStream")
    def test_start_when_already_running_is_noop(
        self, mock_input_stream: MagicMock,
    ) -> None:
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture = AudioCapture()
        capture.start()
        capture.start()  # Second call should be a no-op

        mock_input_stream.assert_called_once()

    def test_stop_when_not_running_is_noop(self) -> None:
        capture = AudioCapture()

        capture.stop()  # Should not raise

        assert capture.is_running is False

    @patch("sounddevice.InputStream")
    def test_start_failure_raises_audio_capture_error(
        self, mock_input_stream: MagicMock,
    ) -> None:
        mock_input_stream.side_effect = RuntimeError("No device")

        capture = AudioCapture()

        with pytest.raises(AudioCaptureError, match="Failed to open"):
            capture.start()

    def test_get_frame_returns_none_on_timeout(self) -> None:
        capture = AudioCapture()

        frame = capture.get_frame(timeout=0.01)

        assert frame is None

    def test_audio_callback_enqueues_frame(self) -> None:
        capture = AudioCapture()

        # Simulate callback with mono data (frames, 1)
        indata = np.ones((512, 1), dtype=np.float32) * 0.5
        capture._audio_callback(indata, 512, None, None)

        frame = capture.get_frame(timeout=0.1)

        assert frame is not None
        assert len(frame.data) == 512
        assert frame.data[0] == pytest.approx(0.5)

    def test_audio_callback_copies_data(self) -> None:
        capture = AudioCapture()

        indata = np.ones((512, 1), dtype=np.float32) * 0.5
        capture._audio_callback(indata, 512, None, None)

        # Modify original — should not affect enqueued frame
        indata[:] = 999.0

        frame = capture.get_frame(timeout=0.1)
        assert frame is not None
        assert frame.data[0] == pytest.approx(0.5)


class TestListAudioDevices:
    """Tests for list_audio_devices."""

    @patch("sounddevice.query_devices")
    def test_returns_input_devices(self, mock_query: MagicMock) -> None:
        mock_query.return_value = [
            {"name": "Mic", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000},
        ]

        devices = list_audio_devices()

        assert len(devices) == 1
        assert devices[0]["name"] == "Mic"
        assert devices[0]["index"] == 0

    @patch("sounddevice.query_devices")
    def test_empty_when_no_input_devices(self, mock_query: MagicMock) -> None:
        mock_query.return_value = [
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000},
        ]

        devices = list_audio_devices()

        assert len(devices) == 0

    @patch("sounddevice.query_devices")
    def test_failure_raises_audio_capture_error(self, mock_query: MagicMock) -> None:
        mock_query.side_effect = RuntimeError("PortAudio error")

        with pytest.raises(AudioCaptureError, match="Failed to query"):
            list_audio_devices()


class TestGetDefaultInputDevice:
    """Tests for get_default_input_device."""

    @patch("sounddevice.query_devices")
    @patch("sounddevice.default")
    def test_returns_default_device(
        self, mock_default: MagicMock, mock_query: MagicMock,
    ) -> None:
        mock_default.device = [0, 1]
        mock_query.return_value = {
            "name": "Default Mic",
            "max_input_channels": 2,
            "default_samplerate": 44100,
        }

        device = get_default_input_device()

        assert device is not None
        assert device["name"] == "Default Mic"

    @patch("sounddevice.default")
    def test_returns_none_when_no_default(self, mock_default: MagicMock) -> None:
        mock_default.device = [None, 1]

        device = get_default_input_device()

        assert device is None

    @patch("sounddevice.query_devices")
    @patch("sounddevice.default")
    def test_failure_raises_audio_capture_error(
        self, mock_default: MagicMock, mock_query: MagicMock,
    ) -> None:
        mock_default.device = [0, 1]
        mock_query.side_effect = RuntimeError("PortAudio error")

        with pytest.raises(AudioCaptureError, match="Failed to query"):
            get_default_input_device()

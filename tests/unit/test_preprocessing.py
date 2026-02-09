"""Tests for pvtt.audio.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from pvtt.audio.preprocessing import (
    WHISPER_SAMPLE_RATE,
    normalize_audio,
    prepare_audio_chunk,
    resample_audio,
    to_mono,
)


class TestToMono:
    """Tests for to_mono."""

    def test_mono_input_unchanged(self) -> None:
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = to_mono(audio)

        np.testing.assert_array_equal(result, audio)

    def test_stereo_samples_channels_layout(self) -> None:
        # Shape (4, 2) — samples × channels
        audio = np.array(
            [[0.2, 0.4], [0.6, 0.8], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        result = to_mono(audio)

        assert result.ndim == 1
        assert len(result) == 4
        np.testing.assert_allclose(result, [0.3, 0.7, 0.5, 0.5])

    def test_stereo_channels_samples_layout(self) -> None:
        # Shape (2, 4) — channels × samples
        audio = np.array(
            [[0.2, 0.6, 1.0, 0.0], [0.4, 0.8, 0.0, 1.0]],
            dtype=np.float32,
        )

        result = to_mono(audio)

        assert result.ndim == 1
        assert len(result) == 4
        np.testing.assert_allclose(result, [0.3, 0.7, 0.5, 0.5])

    def test_returns_float32(self) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float64)

        result = to_mono(audio)

        assert result.dtype == np.float32

    def test_3d_array_raises(self) -> None:
        audio = np.zeros((2, 3, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="1D or 2D"):
            to_mono(audio)


class TestNormalizeAudio:
    """Tests for normalize_audio."""

    def test_silent_audio_unchanged(self) -> None:
        audio = np.zeros(100, dtype=np.float32)

        result = normalize_audio(audio)

        np.testing.assert_array_equal(result, audio)

    def test_peak_reaches_target(self) -> None:
        audio = np.array([0.1, -0.5, 0.3], dtype=np.float32)

        result = normalize_audio(audio, target_dbfs=-3.0)

        # -3 dBFS = 10^(-3/20) ≈ 0.7079
        expected_peak = 10.0 ** (-3.0 / 20.0)
        actual_peak = np.max(np.abs(result))
        np.testing.assert_allclose(actual_peak, expected_peak, rtol=1e-5)

    def test_zero_dbfs_normalizes_to_one(self) -> None:
        audio = np.array([0.1, -0.5, 0.3], dtype=np.float32)

        result = normalize_audio(audio, target_dbfs=0.0)

        np.testing.assert_allclose(np.max(np.abs(result)), 1.0, rtol=1e-5)

    def test_returns_float32(self) -> None:
        audio = np.array([0.5, -0.5], dtype=np.float64)

        result = normalize_audio(audio)

        assert result.dtype == np.float32

    def test_preserves_relative_amplitudes(self) -> None:
        audio = np.array([0.2, 0.4, 0.8], dtype=np.float32)

        result = normalize_audio(audio)

        # Ratios should be preserved
        np.testing.assert_allclose(result[0] / result[2], 0.25, rtol=1e-5)
        np.testing.assert_allclose(result[1] / result[2], 0.5, rtol=1e-5)


class TestResampleAudio:
    """Tests for resample_audio."""

    def test_same_rate_returns_input(self) -> None:
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = resample_audio(audio, orig_sr=16000, target_sr=16000)

        np.testing.assert_array_equal(result, audio)

    def test_downsample_halves_length(self) -> None:
        audio = np.ones(1000, dtype=np.float32)

        result = resample_audio(audio, orig_sr=32000, target_sr=16000)

        assert len(result) == 500

    def test_upsample_doubles_length(self) -> None:
        audio = np.ones(500, dtype=np.float32)

        result = resample_audio(audio, orig_sr=16000, target_sr=32000)

        assert len(result) == 1000

    def test_returns_float32(self) -> None:
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float64)

        result = resample_audio(audio, orig_sr=44100, target_sr=16000)

        assert result.dtype == np.float32

    def test_invalid_sample_rate_raises(self) -> None:
        audio = np.ones(100, dtype=np.float32)

        with pytest.raises(ValueError, match="positive"):
            resample_audio(audio, orig_sr=0, target_sr=16000)

        with pytest.raises(ValueError, match="positive"):
            resample_audio(audio, orig_sr=16000, target_sr=-1)

    def test_2d_array_raises(self) -> None:
        audio = np.ones((100, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="1D"):
            resample_audio(audio, orig_sr=44100, target_sr=16000)

    def test_default_target_is_whisper_rate(self) -> None:
        audio = np.ones(44100, dtype=np.float32)

        result = resample_audio(audio, orig_sr=44100)

        assert len(result) == WHISPER_SAMPLE_RATE

    def test_empty_audio_returns_empty(self) -> None:
        audio = np.array([], dtype=np.float32)

        result = resample_audio(audio, orig_sr=44100, target_sr=16000)

        assert len(result) == 0


class TestPrepareAudioChunk:
    """Tests for prepare_audio_chunk."""

    def test_full_pipeline_mono_16k(self) -> None:
        # Already mono at 16 kHz — should pass through with normalization
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        result = prepare_audio_chunk(audio, orig_sr=16000)

        assert result.ndim == 1
        assert result.dtype == np.float32

    def test_stereo_44100_to_mono_16000(self) -> None:
        # Stereo at 44.1 kHz
        n_samples = 44100  # 1 second
        left = np.ones(n_samples, dtype=np.float32) * 0.5
        right = np.ones(n_samples, dtype=np.float32) * 0.3
        audio = np.column_stack([left, right])

        result = prepare_audio_chunk(audio, orig_sr=44100, target_sr=16000)

        assert result.ndim == 1
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_no_normalize_skips_normalization(self) -> None:
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = prepare_audio_chunk(
            audio, orig_sr=16000, target_sr=16000, normalize=False,
        )

        # Without normalization, values should be unchanged
        np.testing.assert_array_almost_equal(result, audio)

    def test_with_normalize_changes_peak(self) -> None:
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = prepare_audio_chunk(
            audio, orig_sr=16000, target_sr=16000, normalize=True,
        )

        # Peak should be at -3 dBFS (default)
        expected_peak = 10.0 ** (-3.0 / 20.0)
        np.testing.assert_allclose(
            np.max(np.abs(result)), expected_peak, rtol=1e-5,
        )

    def test_accepts_int16_input(self) -> None:
        # sounddevice can return int16 data
        audio = np.array([1000, -2000, 3000], dtype=np.int16)

        result = prepare_audio_chunk(audio, orig_sr=16000)

        assert result.dtype == np.float32

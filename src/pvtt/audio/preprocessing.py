"""Audio preprocessing for pvtt.

Provides resampling, normalization, and mono conversion for audio data.
Used by the streaming pipeline to prepare audio chunks before inference.
All operations work on numpy float32 arrays in [-1.0, 1.0] range.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

WHISPER_SAMPLE_RATE: int = 16_000
"""Sample rate expected by Whisper models (16 kHz)."""


def to_mono(audio: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Convert multi-channel audio to mono by averaging channels.

    Args:
        audio: Audio array. Shape ``(samples,)`` for mono or
            ``(samples, channels)`` / ``(channels, samples)`` for
            multi-channel.

    Returns:
        Mono audio as float32 with shape ``(samples,)``.
    """
    audio = audio.astype(np.float32, copy=False)

    if audio.ndim == 1:
        return audio

    if audio.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got {audio.ndim}D")

    # Heuristic: if shape is (channels, samples), channels dim is the
    # smaller one.  Standard WAV layout is (samples, channels).
    if audio.shape[0] < audio.shape[1]:
        # (channels, samples) layout — average along axis 0
        return audio.mean(axis=0).astype(np.float32)

    # (samples, channels) layout — average along axis 1
    return audio.mean(axis=1).astype(np.float32)


def normalize_audio(
    audio: npt.NDArray[np.floating],
    target_dbfs: float = -3.0,
) -> npt.NDArray[np.float32]:
    """Normalize audio to a target dBFS level.

    Peak-normalizes the audio so the loudest sample reaches
    ``target_dbfs`` decibels below full scale.

    Args:
        audio: Audio array (float values, ideally in [-1.0, 1.0]).
        target_dbfs: Target peak level in dBFS (must be <= 0).

    Returns:
        Normalized audio as float32.
    """
    audio = audio.astype(np.float32, copy=False)

    peak = np.max(np.abs(audio))
    if peak == 0.0:
        return audio

    target_linear = 10.0 ** (target_dbfs / 20.0)
    gain = target_linear / peak
    return (audio * gain).astype(np.float32)


def resample_audio(
    audio: npt.NDArray[np.floating],
    orig_sr: int,
    target_sr: int = WHISPER_SAMPLE_RATE,
) -> npt.NDArray[np.float32]:
    """Resample audio to a target sample rate using linear interpolation.

    Uses numpy-only linear interpolation. This is sufficient for
    speech audio where the frequency content is well below the
    Nyquist limit of 8 kHz.

    Args:
        audio: 1D audio array.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz. Defaults to 16000
            (Whisper's expected rate).

    Returns:
        Resampled audio as float32.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"Sample rates must be positive: orig_sr={orig_sr}, "
            f"target_sr={target_sr}"
        )

    if audio.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got {audio.ndim}D")

    duration = len(audio) / orig_sr
    target_length = int(round(duration * target_sr))

    if target_length == 0:
        return np.array([], dtype=np.float32)

    # Linear interpolation via numpy
    orig_indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(orig_indices, np.arange(len(audio)), audio)
    return resampled.astype(np.float32)


def prepare_audio_chunk(
    audio: npt.NDArray[np.floating],
    orig_sr: int,
    target_sr: int = WHISPER_SAMPLE_RATE,
    normalize: bool = True,
) -> npt.NDArray[np.float32]:
    """Full preprocessing pipeline for an audio chunk.

    Applies mono conversion, resampling, and optional normalization
    in the correct order.

    Args:
        audio: Raw audio array (any dtype, mono or multi-channel).
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz. Defaults to 16000.
        normalize: Whether to apply peak normalization.

    Returns:
        Preprocessed audio as float32 mono at the target sample rate.
    """
    audio = audio.astype(np.float32, copy=False)

    # Step 1: Convert to mono if multi-channel
    audio = to_mono(audio)

    # Step 2: Resample to target rate
    audio = resample_audio(audio, orig_sr, target_sr)

    # Step 3: Normalize
    if normalize:
        audio = normalize_audio(audio)

    return audio

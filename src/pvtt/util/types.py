"""Shared type definitions for pvtt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ComputeType = Literal[
    "auto",
    "default",
    "float32",
    "float16",
    "bfloat16",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
]

DeviceType = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True)
class TranscriptionSegment:
    """A single transcription segment with timing information."""

    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete transcription output."""

    segments: list[TranscriptionSegment]
    language: str = "en"
    language_probability: float = 0.0
    duration: float = 0.0
    text: str = ""


@dataclass(frozen=True)
class TranscribeOptions:
    """Options passed to the inference engine."""

    language: str | None = None
    beam_size: int = 5
    temperature: float | tuple[float, ...] = 0.0
    initial_prompt: str | None = None
    vad_filter: bool = False
    word_timestamps: bool = False


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about a downloaded model."""

    name: str
    path: Path
    size_bytes: int = 0
    compute_type: str = ""

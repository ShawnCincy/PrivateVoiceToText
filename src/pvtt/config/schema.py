"""Pydantic v2 configuration models for pvtt."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

KNOWN_MODELS: frozenset[str] = frozenset({
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v2",
    "distil-large-v3",
    "distil-medium.en",
    "distil-small.en",
})


class ModelConfig(BaseModel):
    """Configuration for model selection and inference."""

    name: str = Field(
        default="auto",
        description=(
            "Whisper model name, HuggingFace repo ID, or 'auto' to select "
            "the best installed model (prefers large-v3-turbo)"
        ),
    )
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: Literal[
        "auto",
        "default",
        "float32",
        "float16",
        "int8",
        "int8_float16",
        "int8_float32",
    ] = "auto"
    local_files_only: bool = Field(
        default=True,
        description="Never make network calls during inference",
    )

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Accept 'auto', known model names, and HuggingFace repo IDs."""
        if v == "auto" or v in KNOWN_MODELS or "/" in v:
            return v
        raise ValueError(
            f"Unknown model: {v!r}. "
            f"Known models: auto, {', '.join(sorted(KNOWN_MODELS))}"
        )


class TranscriptionConfig(BaseModel):
    """Configuration for transcription behavior."""

    language: str | None = Field(
        default=None,
        description="Force language code (e.g., 'en')",
    )
    beam_size: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    vad_filter: bool = False
    word_timestamps: bool = False
    initial_prompt: str | None = None


class OutputConfig(BaseModel):
    """Configuration for output formatting."""

    format: Literal["text", "srt", "vtt", "json"] = "text"


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    verbosity: int = Field(default=0, ge=0, le=2)


class PvttConfig(BaseModel):
    """Root configuration model.

    All sections are optional with sensible defaults.
    A completely empty TOML file produces a valid config.
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

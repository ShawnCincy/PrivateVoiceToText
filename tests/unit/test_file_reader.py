"""Tests for pvtt.audio.file_reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from pvtt.audio.file_reader import SUPPORTED_EXTENSIONS, validate_audio_file
from pvtt.exceptions import AudioError


class TestValidateAudioFile:
    """Tests for validate_audio_file."""

    def test_valid_wav_file(self, tmp_path: Path) -> None:
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

        result = validate_audio_file(wav_file)

        assert result == wav_file

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AudioError, match="not found"):
            validate_audio_file(tmp_path / "missing.wav")

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AudioError, match="Not a file"):
            validate_audio_file(tmp_path)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not audio")

        with pytest.raises(AudioError, match="Unsupported audio format"):
            validate_audio_file(txt_file)

    @pytest.mark.parametrize("ext", sorted(SUPPORTED_EXTENSIONS))
    def test_all_supported_extensions_accepted(
        self, tmp_path: Path, ext: str
    ) -> None:
        audio_file = tmp_path / f"test{ext}"
        audio_file.write_bytes(b"\x00" * 10)

        result = validate_audio_file(audio_file)

        assert result == audio_file

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        wav_file = tmp_path / "test.WAV"
        wav_file.write_bytes(b"\x00" * 10)

        result = validate_audio_file(wav_file)

        assert result == wav_file

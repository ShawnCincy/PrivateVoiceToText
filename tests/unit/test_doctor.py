"""Tests for pvtt.cli.doctor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pvtt.cli.doctor import (
    _check_audio_devices,
    _check_config,
    _check_faster_whisper,
    _check_gpu,
    _check_installed_models,
    _status,
    doctor_command,
)
from pvtt.util.hardware import GpuInfo


class TestStatusHelper:
    """Tests for _status helper."""

    def test_ok_returns_green_check(self) -> None:
        result = _status(True)

        assert "green" in result
        assert "✓" in result

    def test_fail_returns_red_x(self) -> None:
        result = _status(False)

        assert "red" in result
        assert "✗" in result


class TestCheckFasterWhisper:
    """Tests for faster-whisper import check."""

    @patch.dict("sys.modules", {"faster_whisper": MagicMock(__version__="1.0.0")})
    def test_importable_returns_ok(self) -> None:
        ok, detail = _check_faster_whisper()

        assert ok is True
        assert "1.0.0" in detail

    @patch.dict("sys.modules", {"faster_whisper": None})
    def test_not_installed_returns_fail(self) -> None:
        # When a module is mapped to None in sys.modules, import raises ImportError
        ok, detail = _check_faster_whisper()

        assert ok is False
        assert "Not installed" in detail


class TestCheckGpu:
    """Tests for GPU check."""

    @patch(
        "pvtt.util.hardware.detect_gpu",
        return_value=GpuInfo(
            name="RTX 3060", vram_mb=12288, cuda_available=True,
        ),
    )
    def test_gpu_detected(self, mock_detect: MagicMock) -> None:
        ok, detail = _check_gpu()

        assert ok is True
        assert "RTX 3060" in detail
        assert "12288" in detail

    @patch("pvtt.util.hardware.detect_gpu", return_value=None)
    def test_no_gpu(self, mock_detect: MagicMock) -> None:
        ok, detail = _check_gpu()

        assert ok is False
        assert "No NVIDIA GPU" in detail


class TestCheckAudioDevices:
    """Tests for audio devices check."""

    @patch("pvtt.audio.capture.list_audio_devices")
    @patch("pvtt.audio.capture.get_default_input_device")
    def test_devices_found(
        self, mock_default: MagicMock, mock_list: MagicMock,
    ) -> None:
        mock_list.return_value = [{"name": "Mic1"}, {"name": "Mic2"}]
        mock_default.return_value = {"name": "Mic1"}

        ok, detail = _check_audio_devices()

        assert ok is True
        assert "2 device(s)" in detail
        assert "Mic1" in detail

    @patch("pvtt.audio.capture.list_audio_devices")
    def test_no_devices(self, mock_list: MagicMock) -> None:
        mock_list.return_value = []

        ok, detail = _check_audio_devices()

        assert ok is False
        assert "No input devices" in detail


class TestCheckInstalledModels:
    """Tests for installed models check."""

    @patch("pvtt.core.model_manager.ModelManager.list_models")
    @patch("pvtt.config.paths.get_models_dir")
    def test_models_found(
        self, mock_dir: MagicMock, mock_list: MagicMock, tmp_path: object,
    ) -> None:
        mock_dir.return_value = tmp_path
        model1 = MagicMock()
        model1.name = "tiny.en"
        model2 = MagicMock()
        model2.name = "base.en"
        mock_list.return_value = [model1, model2]

        ok, detail = _check_installed_models()

        assert ok is True
        assert "tiny.en" in detail

    @patch("pvtt.core.model_manager.ModelManager.list_models")
    @patch("pvtt.config.paths.get_models_dir")
    def test_no_models(
        self, mock_dir: MagicMock, mock_list: MagicMock, tmp_path: object,
    ) -> None:
        mock_dir.return_value = tmp_path
        mock_list.return_value = []

        ok, detail = _check_installed_models()

        assert ok is False
        assert "None" in detail


class TestCheckConfig:
    """Tests for config check."""

    @patch("pvtt.config.loader.load_config")
    def test_valid_config(self, mock_load: MagicMock) -> None:
        from pvtt.config.schema import PvttConfig

        mock_load.return_value = PvttConfig()

        ok, detail = _check_config()

        assert ok is True
        assert "model=" in detail

    @patch("pvtt.config.loader.load_config", side_effect=Exception("bad toml"))
    def test_invalid_config(self, mock_load: MagicMock) -> None:
        ok, detail = _check_config()

        assert ok is False
        assert "Invalid" in detail


class TestDoctorCommand:
    """Tests for the doctor_command function."""

    @patch("pvtt.cli.doctor._check_config", return_value=(True, "ok"))
    @patch("pvtt.cli.doctor._check_installed_models", return_value=(True, "tiny.en"))
    @patch("pvtt.cli.doctor._check_audio_devices", return_value=(True, "1 device"))
    @patch("pvtt.cli.doctor._check_gpu", return_value=(True, "RTX 3060"))
    @patch("pvtt.cli.doctor._check_faster_whisper", return_value=(True, "v1.0"))
    def test_doctor_runs_without_error(
        self,
        mock_fw: MagicMock,
        mock_gpu: MagicMock,
        mock_audio: MagicMock,
        mock_models: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        # Should not raise
        doctor_command()

        mock_fw.assert_called_once()
        mock_gpu.assert_called_once()
        mock_audio.assert_called_once()
        mock_models.assert_called_once()
        mock_config.assert_called_once()

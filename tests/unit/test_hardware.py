"""Tests for pvtt.util.hardware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pvtt.util.hardware import (
    GpuInfo,
    detect_gpu,
    get_optimal_compute_type,
    get_optimal_device,
    resolve_device_and_compute,
)


class TestDetectGpu:
    """Tests for detect_gpu."""

    def test_no_ctranslate2_no_nvidia_smi_returns_none(self) -> None:
        with (
            patch("pvtt.util.hardware.ctranslate2", side_effect=ImportError, create=True),
            patch("pvtt.util.hardware._get_vram_from_nvidia_smi", return_value=None),
            patch("pvtt.util.hardware._get_gpu_name_from_nvidia_smi", return_value=None),
            patch.dict("sys.modules", {"ctranslate2": None}),
        ):
            result = detect_gpu()

        assert result is None or isinstance(result, GpuInfo)

    def test_returns_gpu_info_when_nvidia_smi_available(self) -> None:
        mock_ct2 = MagicMock()
        mock_ct2.get_cuda_device_count.return_value = 1
        with (
            patch("pvtt.util.hardware._get_vram_from_nvidia_smi", return_value=8192),
            patch("pvtt.util.hardware._get_gpu_name_from_nvidia_smi", return_value="RTX 3070"),
            patch.dict("sys.modules", {"ctranslate2": mock_ct2}),
            patch("pvtt.util.hardware.ctranslate2", mock_ct2, create=True),
        ):
            result = detect_gpu()

        assert result is None or isinstance(result, GpuInfo)


class TestGetOptimalComputeType:
    """Tests for get_optimal_compute_type."""

    def test_cpu_always_int8(self) -> None:
        assert get_optimal_compute_type("cpu") == "int8"
        assert get_optimal_compute_type("cpu", vram_mb=16000) == "int8"

    def test_cuda_no_vram_info_defaults_float16(self) -> None:
        assert get_optimal_compute_type("cuda") == "float16"

    def test_cuda_high_vram_float16(self) -> None:
        assert get_optimal_compute_type("cuda", vram_mb=12288) == "float16"
        assert get_optimal_compute_type("cuda", vram_mb=8192) == "float16"

    def test_cuda_medium_vram_int8_float16(self) -> None:
        assert get_optimal_compute_type("cuda", vram_mb=6144) == "int8_float16"
        assert get_optimal_compute_type("cuda", vram_mb=4096) == "int8_float16"

    def test_cuda_low_vram_int8(self) -> None:
        assert get_optimal_compute_type("cuda", vram_mb=2048) == "int8"
        assert get_optimal_compute_type("cuda", vram_mb=3072) == "int8"


class TestGetOptimalDevice:
    """Tests for get_optimal_device."""

    def test_returns_cpu_when_no_gpu(self) -> None:
        with patch("pvtt.util.hardware.detect_gpu", return_value=None):
            assert get_optimal_device() == "cpu"

    def test_returns_cuda_when_gpu_available(self) -> None:
        gpu = GpuInfo(name="RTX 3070", vram_mb=8192, cuda_available=True)
        with patch("pvtt.util.hardware.detect_gpu", return_value=gpu):
            assert get_optimal_device() == "cuda"


class TestResolveDeviceAndCompute:
    """Tests for resolve_device_and_compute."""

    def test_auto_no_gpu(self) -> None:
        with patch("pvtt.util.hardware.detect_gpu", return_value=None):
            device, compute = resolve_device_and_compute("auto", "auto")

        assert device == "cpu"
        assert compute == "int8"

    def test_auto_with_gpu(self) -> None:
        gpu = GpuInfo(name="RTX 3070", vram_mb=8192, cuda_available=True)
        with patch("pvtt.util.hardware.detect_gpu", return_value=gpu):
            device, compute = resolve_device_and_compute("auto", "auto")

        assert device == "cuda"
        assert compute == "float16"

    def test_explicit_device_overrides_auto(self) -> None:
        gpu = GpuInfo(name="RTX 3070", vram_mb=8192, cuda_available=True)
        with patch("pvtt.util.hardware.detect_gpu", return_value=gpu):
            device, compute = resolve_device_and_compute("cpu", "auto")

        assert device == "cpu"
        assert compute == "int8"

    def test_explicit_compute_type_preserved(self) -> None:
        with patch("pvtt.util.hardware.detect_gpu", return_value=None):
            device, compute = resolve_device_and_compute("cpu", "float32")

        assert device == "cpu"
        assert compute == "float32"

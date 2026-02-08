"""GPU/VRAM detection and compute type selection for pvtt."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from pvtt.util.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    """Information about a detected GPU."""

    name: str
    vram_mb: int
    cuda_available: bool
    device_index: int = 0


def detect_gpu() -> GpuInfo | None:
    """Detect NVIDIA GPU and return its info, or None if unavailable.

    Tries ctranslate2 first (bundled with faster-whisper), then falls
    back to parsing nvidia-smi output.
    """
    # Try ctranslate2 first
    try:
        import ctranslate2

        device_count = ctranslate2.get_cuda_device_count()
        if device_count > 0:
            # ctranslate2 doesn't expose VRAM directly; try nvidia-smi
            vram = _get_vram_from_nvidia_smi()
            name = _get_gpu_name_from_nvidia_smi()
            return GpuInfo(
                name=name or "NVIDIA GPU",
                vram_mb=vram or 0,
                cuda_available=True,
                device_index=0,
            )
    except (ImportError, Exception):
        logger.debug("ctranslate2 GPU detection failed, trying nvidia-smi")

    # Fallback to nvidia-smi
    try:
        vram = _get_vram_from_nvidia_smi()
        name = _get_gpu_name_from_nvidia_smi()
        if vram is not None and vram > 0:
            return GpuInfo(
                name=name or "NVIDIA GPU",
                vram_mb=vram,
                cuda_available=True,
                device_index=0,
            )
    except Exception:
        logger.debug("nvidia-smi detection failed")

    return None


def _get_vram_from_nvidia_smi() -> int | None:
    """Query total GPU memory in MB via nvidia-smi."""
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if output.returncode == 0 and output.stdout.strip():
            return int(output.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _get_gpu_name_from_nvidia_smi() -> str | None:
    """Query GPU name via nvidia-smi."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if output.returncode == 0 and output.stdout.strip():
            return output.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_optimal_device() -> str:
    """Return 'cuda' if a capable GPU is found, otherwise 'cpu'."""
    gpu = detect_gpu()
    if gpu is not None and gpu.cuda_available:
        return "cuda"
    return "cpu"


def get_optimal_compute_type(device: str, vram_mb: int | None = None) -> str:
    """Select the best compute type for the given device and VRAM.

    Args:
        device: 'cuda' or 'cpu'.
        vram_mb: Total GPU memory in MB, or None if unknown.

    Returns:
        CTranslate2 compute type string.
    """
    if device == "cpu":
        return "int8"
    if vram_mb is None:
        return "float16"
    if vram_mb >= 8192:
        return "float16"
    if vram_mb >= 4096:
        return "int8_float16"
    return "int8"


def resolve_device_and_compute(
    device: str = "auto",
    compute_type: str = "auto",
) -> tuple[str, str]:
    """Resolve 'auto' values to concrete device and compute type.

    Args:
        device: 'auto', 'cuda', or 'cpu'.
        compute_type: 'auto' or a specific CTranslate2 compute type.

    Returns:
        Tuple of (resolved_device, resolved_compute_type).
    """
    gpu = detect_gpu()

    if device == "auto":
        resolved_device = "cuda" if (gpu and gpu.cuda_available) else "cpu"
    else:
        resolved_device = device

    if compute_type == "auto":
        vram = gpu.vram_mb if gpu else None
        resolved_compute_type = get_optimal_compute_type(resolved_device, vram)
    else:
        resolved_compute_type = compute_type

    return resolved_device, resolved_compute_type

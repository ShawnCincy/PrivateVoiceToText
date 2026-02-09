"""CLI command for system diagnostics: pvtt doctor."""

from __future__ import annotations

import sys

from rich.table import Table

from pvtt.cli.formatters import console


def doctor_command() -> None:
    """Run system diagnostics and check pvtt readiness."""
    console.print("[bold]pvtt doctor[/bold] — System diagnostics\n")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    # 1. Python version
    vi = sys.version_info
    py_version = f"{vi.major}.{vi.minor}.{vi.micro}"
    py_ok = sys.version_info >= (3, 10)
    table.add_row(
        "Python version",
        _status(py_ok),
        f"{py_version} ({'OK' if py_ok else 'Requires 3.10+'})",
    )

    # 2. faster-whisper import
    fw_ok, fw_detail = _check_faster_whisper()
    table.add_row("faster-whisper", _status(fw_ok), fw_detail)

    # 3. GPU / VRAM
    gpu_ok, gpu_detail = _check_gpu()
    table.add_row("GPU / CUDA", _status(gpu_ok), gpu_detail)

    # 4. Audio devices
    audio_ok, audio_detail = _check_audio_devices()
    table.add_row("Audio input", _status(audio_ok), audio_detail)

    # 5. Installed models
    models_ok, models_detail = _check_installed_models()
    table.add_row("Installed models", _status(models_ok), models_detail)

    # 6. Config validity
    config_ok, config_detail = _check_config()
    table.add_row("Configuration", _status(config_ok), config_detail)

    console.print(table)


def _status(ok: bool) -> str:
    """Return a Rich-formatted status indicator."""
    return "[green]✓[/green]" if ok else "[red]✗[/red]"


def _check_faster_whisper() -> tuple[bool, str]:
    """Check if faster-whisper is importable."""
    try:
        import faster_whisper

        version = getattr(faster_whisper, "__version__", "unknown")
        return True, f"v{version}"
    except ImportError:
        return False, "Not installed — pip install faster-whisper"


def _check_gpu() -> tuple[bool, str]:
    """Check GPU availability and VRAM."""
    try:
        from pvtt.util.hardware import detect_gpu

        gpu = detect_gpu()
        if gpu is None:
            return False, "No NVIDIA GPU detected — will use CPU"
        return True, f"{gpu.name} ({gpu.vram_mb} MB VRAM)"
    except Exception as exc:
        return False, f"Detection failed: {exc}"


def _check_audio_devices() -> tuple[bool, str]:
    """Check for available audio input devices."""
    try:
        from pvtt.audio.capture import get_default_input_device, list_audio_devices

        devices = list_audio_devices()
        if not devices:
            return False, "No input devices found"

        default = get_default_input_device()
        default_name = default["name"] if default else "none"
        return True, f"{len(devices)} device(s), default: {default_name}"
    except Exception as exc:
        return False, f"Detection failed: {exc}"


def _check_installed_models() -> tuple[bool, str]:
    """Check for locally installed models."""
    try:
        from pvtt.config.paths import get_models_dir
        from pvtt.core.model_manager import ModelManager

        manager = ModelManager(get_models_dir())
        models = manager.list_models()
        if not models:
            return False, "None — run: pvtt model download tiny.en"

        names = ", ".join(m.name for m in models[:5])
        suffix = f" (+{len(models) - 5} more)" if len(models) > 5 else ""
        return True, f"{names}{suffix}"
    except Exception as exc:
        return False, f"Check failed: {exc}"


def _check_config() -> tuple[bool, str]:
    """Check if configuration is valid."""
    try:
        from pvtt.config.loader import load_config

        config = load_config()
        return True, f"model={config.model.name}, format={config.output.format}"
    except Exception as exc:
        return False, f"Invalid: {exc}"

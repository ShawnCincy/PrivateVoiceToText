"""CLI commands for transcription: pvtt transcribe {file}."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from pvtt.cli.formatters import output_console, print_info
from pvtt.config.loader import load_config
from pvtt.core.transcriber import Transcriber
from pvtt.export.registry import get_exporter

transcribe_app = typer.Typer(no_args_is_help=True)


@transcribe_app.command("file")
def transcribe_file(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to audio file",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name (e.g., tiny.en)"
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Force language code (e.g., en)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    format_name: str = typer.Option(
        "text", "--format", "-f", help="Output format (text)"
    ),
    beam_size: Optional[int] = typer.Option(
        None, "--beam-size", help="Beam size for decoding"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: auto, cuda, or cpu"
    ),
    compute_type: Optional[str] = typer.Option(
        None, "--compute-type", help="Compute type (e.g., int8, float16)"
    ),
) -> None:
    """Transcribe an audio file."""
    # Build CLI overrides dict from provided flags
    cli_overrides: dict[str, dict[str, object]] = {}

    if model is not None:
        cli_overrides.setdefault("model", {})["name"] = model
    if device is not None:
        cli_overrides.setdefault("model", {})["device"] = device
    if compute_type is not None:
        cli_overrides.setdefault("model", {})["compute_type"] = compute_type
    if language is not None:
        cli_overrides.setdefault("transcription", {})["language"] = language
    if beam_size is not None:
        cli_overrides.setdefault("transcription", {})["beam_size"] = beam_size
    if format_name != "text":
        cli_overrides.setdefault("output", {})["format"] = format_name

    config = load_config(cli_overrides=cli_overrides)
    transcriber = Transcriber(config)

    print_info(f"Transcribing {input_path.name}...")

    result = transcriber.transcribe_file(input_path)
    formatted = transcriber.format_output(result)

    if output:
        exporter = get_exporter(config.output.format)
        exporter.write(result, output)
        print_info(f"Written to {output}")
    else:
        output_console.print(formatted)

"""CLI commands for transcription: pvtt transcribe {file,live}."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from pvtt.cli.formatters import console, output_console, print_error, print_info
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
        "text", "--format", "-f", help="Output format (text, srt, vtt, json)"
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


@transcribe_app.command("live")
def transcribe_live(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name (e.g., tiny.en)"
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Force language code (e.g., en)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (write on stop)"
    ),
    format_name: str = typer.Option(
        "text", "--format", "-f", help="Output format (text, srt, vtt, json)"
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
    audio_device: Optional[int] = typer.Option(
        None, "--audio-device", help="Audio input device index"
    ),
) -> None:
    """Transcribe from microphone in real-time.

    Press Ctrl+C to stop. Transcript is printed to stdout as segments
    are recognized. When --output is specified, the full transcript is
    written to the file on stop.
    """
    import time

    from rich.live import Live
    from rich.text import Text

    from pvtt.core.streaming import (
        StreamingEvent,
        StreamingEventType,
        StreamingPipeline,
    )
    from pvtt.engine.registry import get_engine
    from pvtt.util.types import TranscriptionResult, TranscriptionSegment

    # Build CLI overrides
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
    if audio_device is not None:
        cli_overrides.setdefault("streaming", {})["audio_device"] = audio_device

    config = load_config(cli_overrides=cli_overrides)

    # Load engine and model
    engine = get_engine()
    transcriber = Transcriber(config, engine=engine)
    transcriber._ensure_model_loaded()

    # Accumulate segments for output file
    all_segments: list[TranscriptionSegment] = []
    use_live_display = sys.stderr.isatty() and output is None

    # Build event callback
    live: Live | None = None
    status_text = Text("Listening...", style="dim")

    def on_event(event: StreamingEvent) -> None:
        nonlocal status_text
        if event.type == StreamingEventType.SEGMENT and event.segment is not None:
            all_segments.append(event.segment)
            output_console.print(event.segment.text.strip())
            if live is not None:
                status_text = Text(
                    f"[{len(all_segments)} segment(s)] Listening...",
                    style="dim",
                )
                live.update(status_text)
        elif event.type == StreamingEventType.STATUS:
            if live is not None:
                status_text = Text(event.text, style="dim")
                live.update(status_text)
        elif event.type == StreamingEventType.ERROR:
            print_error(event.text)

    pipeline = StreamingPipeline(config, engine, on_event)

    try:
        if use_live_display:
            with Live(
                status_text, console=console, refresh_per_second=4
            ) as live_ctx:
                live = live_ctx
                pipeline.start()
                print_info("Press Ctrl+C to stop.")
                # Block until interrupted
                while pipeline.is_running:
                    time.sleep(0.1)
        else:
            pipeline.start()
            print_info("Press Ctrl+C to stop.")
            while pipeline.is_running:
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

    # Write output file if requested
    if output and all_segments:
        full_text = " ".join(seg.text.strip() for seg in all_segments)
        result = TranscriptionResult(
            segments=all_segments,
            text=full_text,
        )
        exporter = get_exporter(config.output.format)
        exporter.write(result, output)
        print_info(f"Written {len(all_segments)} segment(s) to {output}")
    elif not all_segments:
        print_info("No speech detected.")

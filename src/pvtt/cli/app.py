"""Root CLI application for pvtt."""

from __future__ import annotations

import typer

from pvtt.cli.config_cmd import config_app
from pvtt.cli.formatters import console, print_error
from pvtt.cli.model import model_app
from pvtt.cli.transcribe import transcribe_app
from pvtt.exceptions import PvttError
from pvtt.util.logging import setup_logging

app = typer.Typer(
    name="pvtt",
    help="Private Voice To Text - local-only, privacy-first voice-to-text CLI.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def main_callback(
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
) -> None:
    """Private Voice To Text - local-only, privacy-first voice-to-text CLI."""
    setup_logging(verbose)


app.add_typer(transcribe_app, name="transcribe", help="Transcription commands")
app.add_typer(model_app, name="model", help="Model management commands")
app.add_typer(config_app, name="config", help="Configuration commands")


def main() -> None:
    """Entry point for the pvtt CLI."""
    try:
        app()
    except PvttError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        raise typer.Exit(code=130) from None

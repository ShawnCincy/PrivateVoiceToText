"""CLI commands for configuration: pvtt config {show,path}."""

from __future__ import annotations

import typer

from pvtt.cli.formatters import console
from pvtt.config.loader import load_config
from pvtt.config.paths import get_config_path, get_data_dir

config_app = typer.Typer(no_args_is_help=True)


@config_app.command("show")
def config_show() -> None:
    """Show the current resolved configuration."""
    config = load_config()
    console.print_json(config.model_dump_json(indent=2))


@config_app.command("path")
def config_path() -> None:
    """Show the configuration file path and data directory."""
    console.print(f"[bold]Config file:[/bold] {get_config_path()}")
    console.print(f"[bold]Data dir:[/bold]   {get_data_dir()}")

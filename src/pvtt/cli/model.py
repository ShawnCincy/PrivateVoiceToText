"""CLI commands for model management: pvtt model {download,list,info,remove}."""

from __future__ import annotations

import typer
from rich.table import Table

from pvtt.cli.formatters import console, format_file_size, print_success
from pvtt.core.model_manager import MODEL_REPO_MAP, ModelManager

model_app = typer.Typer(no_args_is_help=True)


@model_app.command("download")
def download_model(
    model_name: str = typer.Argument(
        ...,
        help="Model name (e.g., tiny.en, large-v3-turbo)",
    ),
) -> None:
    """Download a Whisper model from HuggingFace Hub."""
    manager = ModelManager()

    with console.status(f"Downloading {model_name}..."):
        path = manager.download(model_name)

    print_success(f"Model {model_name!r} downloaded to {path}")


@model_app.command("list")
def list_models() -> None:
    """List all locally downloaded models."""
    manager = ModelManager()
    models = manager.list_models()

    if not models:
        console.print("[dim]No models downloaded yet.[/dim]")
        console.print("[dim]Download one with: pvtt model download tiny.en[/dim]")
        return

    table = Table(title="Downloaded Models")
    table.add_column("Name", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Path", style="dim")

    for model in models:
        table.add_row(
            model.name,
            format_file_size(model.size_bytes),
            str(model.path),
        )

    console.print(table)


@model_app.command("info")
def model_info(
    model_name: str = typer.Argument(..., help="Model name"),
) -> None:
    """Show details about a downloaded model."""
    manager = ModelManager()
    info = manager.get_model_info(model_name)

    console.print(f"[bold]Model:[/bold] {info.name}")
    console.print(f"[bold]Path:[/bold]  {info.path}")
    console.print(f"[bold]Size:[/bold]  {format_file_size(info.size_bytes)}")


@model_app.command("remove")
def remove_model(
    model_name: str = typer.Argument(..., help="Model name to remove"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation prompt"
    ),
) -> None:
    """Remove a downloaded model."""
    manager = ModelManager()

    # Verify it exists first
    manager.get_model_info(model_name)

    if not force:
        confirm = typer.confirm(f"Remove model {model_name!r}?")
        if not confirm:
            raise typer.Abort()

    manager.remove_model(model_name)
    print_success(f"Model {model_name!r} removed.")


@model_app.command("available")
def available_models() -> None:
    """List all known model names that can be downloaded."""
    table = Table(title="Available Models")
    table.add_column("Name", style="bold")
    table.add_column("HuggingFace Repo", style="dim")

    for name, repo in sorted(MODEL_REPO_MAP.items()):
        table.add_row(name, repo)

    console.print(table)

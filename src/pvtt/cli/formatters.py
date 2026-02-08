"""Rich console output helpers for pvtt CLI."""

from __future__ import annotations

from rich.console import Console

# Status/progress messages go to stderr (don't pollute piped output)
console = Console(stderr=True)

# Transcription output goes to stdout (supports piping/redirection)
output_console = Console()


def print_error(message: str) -> None:
    """Print an error message to stderr."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    """Print a success message to stderr."""
    console.print(f"[bold green]Success:[/bold green] {message}")


def print_info(message: str) -> None:
    """Print an info message to stderr."""
    console.print(f"[dim]{message}[/dim]")


def format_file_size(size_bytes: int) -> str:
    """Format byte count as human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., '1.5 GB').
    """
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

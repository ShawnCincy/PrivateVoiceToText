"""CLI commands for configuration: pvtt config {show,path,set}."""

from __future__ import annotations

import typer

from pvtt.cli.formatters import console, print_error, print_success
from pvtt.config.loader import load_config, load_toml
from pvtt.config.paths import get_config_path, get_data_dir
from pvtt.config.schema import PvttConfig
from pvtt.exceptions import ConfigError

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


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        ...,
        help="Dotted config key (e.g., model.name, transcription.beam_size)",
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set",
    ),
) -> None:
    """Set a configuration value in the global config.toml.

    Examples:
        pvtt config set model.name tiny.en
        pvtt config set transcription.beam_size 3
        pvtt config set model.local_files_only true
    """
    parts = key.split(".")
    if len(parts) != 2:
        print_error(
            f"Key must be in 'section.field' format (e.g., model.name), "
            f"got: {key!r}"
        )
        raise typer.Exit(code=1)

    section, field = parts

    # Coerce value to appropriate Python type
    coerced = _coerce_value(value)

    # Validate by building a full config with the override
    try:
        override = {section: {field: coerced}}
        PvttConfig.model_validate(override)
    except Exception as exc:
        print_error(f"Invalid value for {key}: {exc}")
        raise typer.Exit(code=1) from exc

    # Read existing TOML, update, and write back
    config_file = get_config_path()
    existing = load_toml(config_file)

    existing.setdefault(section, {})[field] = coerced

    _write_toml(config_file, existing)
    print_success(f"{key} = {coerced!r}")


def _coerce_value(value: str) -> object:
    """Coerce a string value to the appropriate Python type.

    Tries: bool, int, float, then falls back to string.
    None/null becomes None.

    Args:
        value: Raw string from CLI.

    Returns:
        Coerced value.
    """
    # Booleans
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # None/null
    if value.lower() in ("none", "null", ""):
        return None

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # String
    return value


def _write_toml(path: object, data: dict[str, object]) -> None:
    """Write a dict to a TOML file using tomli-w.

    Args:
        path: Path to the TOML file.
        data: Dict to serialize.

    Raises:
        ConfigError: If tomli-w is not installed or write fails.
    """
    from pathlib import Path

    file_path = Path(str(path))

    try:
        import tomli_w
    except ImportError:
        raise ConfigError(
            "tomli-w package required for writing TOML. "
            "Install with: pip install tomli-w"
        ) from None

    # Filter out None values (TOML doesn't support null)
    cleaned = _clean_none_values(data)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, "wb") as f:
            tomli_w.dump(cleaned, f)
    except Exception as exc:
        raise ConfigError(f"Failed to write {file_path}: {exc}") from exc


def _clean_none_values(data: dict[str, object]) -> dict[str, object]:
    """Recursively remove None values from a dict (TOML doesn't support null).

    Args:
        data: Dict potentially containing None values.

    Returns:
        Cleaned dict with None values removed.
    """
    cleaned: dict[str, object] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            sub = _clean_none_values(value)
            if sub:  # Only include non-empty sections
                cleaned[key] = sub
        else:
            cleaned[key] = value
    return cleaned

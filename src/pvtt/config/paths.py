"""Platform-aware data directory resolution for pvtt."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "pvtt"


def get_data_dir() -> Path:
    """Return the platform-specific pvtt data directory.

    Windows:  %LOCALAPPDATA%\\pvtt
    Linux:    ~/.local/share/pvtt
    macOS:    ~/Library/Application Support/pvtt

    Creates the directory if it does not exist.
    """
    data_dir = Path(user_data_dir(APP_NAME, appauthor=False))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_config_path() -> Path:
    """Return path to the global config.toml file."""
    return get_data_dir() / "config.toml"


def get_models_dir() -> Path:
    """Return path to the models directory. Creates it if missing."""
    models_dir = get_data_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_profiles_dir() -> Path:
    """Return path to the profiles directory. Creates it if missing."""
    profiles_dir = get_data_dir() / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir

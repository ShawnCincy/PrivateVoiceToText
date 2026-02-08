"""Layered configuration loading for pvtt.

Resolution order (later wins):
1. Pydantic defaults (schema.py)
2. Global TOML file
3. Environment variables (PVTT_*)
4. CLI argument overrides
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pvtt.config.paths import get_config_path
from pvtt.config.schema import PvttConfig
from pvtt.exceptions import ConfigError
from pvtt.util.logging import get_logger

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

logger = get_logger(__name__)

ENV_PREFIX = "PVTT_"


def load_toml(path: Path) -> dict[str, Any]:
    """Load and parse a TOML file.

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed dict, or empty dict if file does not exist.

    Raises:
        ConfigError: If the file exists but cannot be parsed.
    """
    if not path.is_file():
        logger.debug("Config file not found: %s", path)
        return {}

    if tomllib is None:
        raise ConfigError(
            "tomli package required for Python <3.11. "
            "Install with: pip install tomli"
        )

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        raise ConfigError(f"Failed to parse {path}: {exc}") from exc


def load_env_overrides() -> dict[str, Any]:
    """Read PVTT_* environment variables as nested config dicts.

    Convention: PVTT_SECTION_FIELD -> {"section": {"field": value}}.
    Example: PVTT_MODEL_NAME=tiny.en -> {"model": {"name": "tiny.en"}}.
    """
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX) :].lower().split("_", maxsplit=1)
        if len(parts) == 2:
            section, field = parts
            overrides.setdefault(section, {})[field] = value
    return overrides


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override values win.

    Args:
        base: The base dictionary.
        override: Values to merge on top.

    Returns:
        New merged dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    cli_overrides: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> PvttConfig:
    """Load configuration with layered resolution.

    Args:
        cli_overrides: Dict of overrides from CLI flags.
        config_path: Optional path to config file. Defaults to platform path.

    Returns:
        Validated PvttConfig instance.

    Raises:
        ConfigError: If config file is malformed or validation fails.
    """
    path = config_path or get_config_path()

    # Layer 2: global TOML (Layer 1 = Pydantic defaults)
    toml_data = load_toml(path)

    # Layer 3: environment variables
    env_data = load_env_overrides()
    merged = deep_merge(toml_data, env_data)

    # Layer 4: CLI overrides
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)

    try:
        return PvttConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigError(f"Configuration validation failed: {exc}") from exc

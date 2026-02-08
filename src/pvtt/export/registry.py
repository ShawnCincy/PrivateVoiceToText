"""Export format registry for pvtt."""

from __future__ import annotations

from typing import Any

from pvtt.exceptions import ExportError
from pvtt.export.base import Exporter
from pvtt.util.logging import get_logger

logger = get_logger(__name__)

_EXPORTER_REGISTRY: dict[str, type[Any]] = {}


def register_exporter(format_name: str, exporter_class: type[Any]) -> None:
    """Register an exporter class for a format name.

    Args:
        format_name: Short name (e.g., 'text', 'srt').
        exporter_class: Class that satisfies the Exporter Protocol.
    """
    _EXPORTER_REGISTRY[format_name] = exporter_class
    logger.debug("Registered exporter: %s", format_name)


def get_exporter(format_name: str) -> Exporter:
    """Get an exporter instance by format name.

    Args:
        format_name: The output format to use.

    Returns:
        An Exporter instance.

    Raises:
        ExportError: If format is not registered.
    """
    cls = _EXPORTER_REGISTRY.get(format_name)
    if cls is None:
        available = ", ".join(sorted(_EXPORTER_REGISTRY.keys())) or "(none)"
        raise ExportError(
            f"Unknown output format: {format_name!r}. Available: {available}"
        )
    return cls()  # type: ignore[return-value]


def list_formats() -> list[str]:
    """Return all registered format names."""
    return sorted(_EXPORTER_REGISTRY.keys())


def _register_builtins() -> None:
    """Register built-in exporters."""
    from pvtt.export.plain_text import PlainTextExporter

    register_exporter("text", PlainTextExporter)


_register_builtins()

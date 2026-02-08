"""Logging configuration for pvtt."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbosity: int = 0) -> None:
    """Configure logging for pvtt.

    Args:
        verbosity: 0 = WARNING, 1 = INFO, 2+ = DEBUG.
    """
    level_map = {0: logging.WARNING, 1: logging.INFO}
    level = level_map.get(verbosity, logging.DEBUG)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )

    root_logger = logging.getLogger("pvtt")
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the pvtt namespace.

    Args:
        name: Module name, typically __name__.

    Returns:
        A logger instance.
    """
    return logging.getLogger(f"pvtt.{name}")

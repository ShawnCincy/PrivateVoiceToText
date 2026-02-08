"""Tests for pvtt.util.logging."""

from __future__ import annotations

import logging

from pvtt.util.logging import get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_default_verbosity_sets_warning(self) -> None:
        setup_logging(0)

        logger = logging.getLogger("pvtt")
        assert logger.level == logging.WARNING

    def test_verbosity_one_sets_info(self) -> None:
        setup_logging(1)

        logger = logging.getLogger("pvtt")
        assert logger.level == logging.INFO

    def test_verbosity_two_sets_debug(self) -> None:
        setup_logging(2)

        logger = logging.getLogger("pvtt")
        assert logger.level == logging.DEBUG

    def test_clears_existing_handlers(self) -> None:
        setup_logging(0)
        setup_logging(0)

        logger = logging.getLogger("pvtt")
        assert len(logger.handlers) == 1


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_namespaced_logger(self) -> None:
        logger = get_logger("test_module")

        assert logger.name == "pvtt.test_module"

    def test_logger_is_child_of_pvtt(self) -> None:
        logger = get_logger("child")

        assert logger.parent is not None
        assert logger.parent.name == "pvtt"

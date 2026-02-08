"""Tests for pvtt.export.registry."""

from __future__ import annotations

import pytest

from pvtt.exceptions import ExportError
from pvtt.export.base import Exporter
from pvtt.export.registry import get_exporter, list_formats, register_exporter


class TestExportRegistry:
    """Tests for export registry functions."""

    def test_text_format_registered(self) -> None:
        exporter = get_exporter("text")

        assert isinstance(exporter, Exporter)
        assert exporter.format_name == "text"

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ExportError, match="Unknown output format"):
            get_exporter("csv")

    def test_list_formats_includes_text(self) -> None:
        formats = list_formats()

        assert "text" in formats

    def test_register_custom_exporter(self) -> None:
        class DummyExporter:
            @property
            def format_name(self) -> str:
                return "dummy"

            def format(self, result: object) -> str:
                return "dummy"

            def write(self, result: object, path: object) -> None:
                pass

        register_exporter("dummy", DummyExporter)

        exporter = get_exporter("dummy")
        assert exporter.format_name == "dummy"
        assert "dummy" in list_formats()

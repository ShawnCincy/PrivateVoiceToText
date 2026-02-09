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

    def test_srt_format_registered(self) -> None:
        exporter = get_exporter("srt")

        assert isinstance(exporter, Exporter)
        assert exporter.format_name == "srt"

    def test_vtt_format_registered(self) -> None:
        exporter = get_exporter("vtt")

        assert isinstance(exporter, Exporter)
        assert exporter.format_name == "vtt"

    def test_json_format_registered(self) -> None:
        exporter = get_exporter("json")

        assert isinstance(exporter, Exporter)
        assert exporter.format_name == "json"

    def test_list_formats_includes_all_builtins(self) -> None:
        formats = list_formats()

        assert "text" in formats
        assert "srt" in formats
        assert "vtt" in formats
        assert "json" in formats

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

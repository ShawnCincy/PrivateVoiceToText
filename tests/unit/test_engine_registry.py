"""Tests for pvtt.engine.registry."""

from __future__ import annotations

import pytest

from pvtt.engine.base import InferenceEngine
from pvtt.engine.registry import get_engine, list_engines, register_engine
from pvtt.exceptions import EngineNotFoundError


class TestEngineRegistry:
    """Tests for engine registry functions."""

    def test_default_engine_is_faster_whisper(self) -> None:
        engine = get_engine()

        assert isinstance(engine, InferenceEngine)

    def test_get_engine_by_name(self) -> None:
        engine = get_engine("faster-whisper")

        assert isinstance(engine, InferenceEngine)

    def test_get_engine_unknown_raises(self) -> None:
        with pytest.raises(EngineNotFoundError, match="not-real"):
            get_engine("not-real")

    def test_list_engines_includes_faster_whisper(self) -> None:
        engines = list_engines()

        assert "faster-whisper" in engines

    def test_register_custom_engine(self) -> None:
        class DummyEngine:
            def load_model(self, *a: object, **kw: object) -> None:
                pass

            def transcribe(self, *a: object, **kw: object) -> list[object]:
                return []

            @property
            def is_loaded(self) -> bool:
                return False

            @property
            def model_name(self) -> str | None:
                return None

        register_engine("dummy", DummyEngine)

        engine = get_engine("dummy")
        assert engine is not None
        assert "dummy" in list_engines()

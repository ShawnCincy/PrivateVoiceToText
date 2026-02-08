"""Engine factory and registration for pvtt."""

from __future__ import annotations

from collections.abc import Callable

from pvtt.engine.base import InferenceEngine
from pvtt.exceptions import EngineNotFoundError
from pvtt.util.logging import get_logger

logger = get_logger(__name__)

_ENGINE_REGISTRY: dict[str, Callable[[], InferenceEngine]] = {}

DEFAULT_ENGINE = "faster-whisper"


def register_engine(name: str, factory: Callable[[], InferenceEngine]) -> None:
    """Register an inference engine factory.

    Args:
        name: Engine name (e.g., 'faster-whisper').
        factory: Callable that returns a new InferenceEngine instance.
    """
    _ENGINE_REGISTRY[name] = factory
    logger.debug("Registered engine: %s", name)


def get_engine(name: str | None = None) -> InferenceEngine:
    """Create an inference engine instance by name.

    Args:
        name: Engine name. None defaults to DEFAULT_ENGINE.

    Returns:
        A new InferenceEngine instance.

    Raises:
        EngineNotFoundError: If the engine name is not registered.
    """
    engine_name = name or DEFAULT_ENGINE
    factory = _ENGINE_REGISTRY.get(engine_name)
    if factory is None:
        available = ", ".join(sorted(_ENGINE_REGISTRY.keys())) or "(none)"
        raise EngineNotFoundError(
            f"Engine {engine_name!r} not found. Available: {available}"
        )
    return factory()


def list_engines() -> list[str]:
    """Return names of all registered engines."""
    return sorted(_ENGINE_REGISTRY.keys())


def _register_builtins() -> None:
    """Register built-in engine backends."""
    from pvtt.engine.faster_whisper import FasterWhisperEngine

    register_engine("faster-whisper", FasterWhisperEngine)


_register_builtins()

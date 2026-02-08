"""Model download, listing, and management for pvtt.

The download() method is the ONLY place in pvtt that makes network requests.
All other model access is strictly local.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from pvtt.config.paths import get_models_dir
from pvtt.exceptions import ModelDownloadError, ModelNotFoundError
from pvtt.util.logging import get_logger
from pvtt.util.types import ModelInfo

logger = get_logger(__name__)

PREFERRED_MODEL = "large-v3-turbo"

MODEL_REPO_MAP: dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
}


class ModelManager:
    """Manages Whisper model downloads, listing, and removal.

    Models are stored in the platform data directory under models/.
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        self._models_dir = models_dir or get_models_dir()

    @property
    def models_dir(self) -> Path:
        """Return the models directory path."""
        return self._models_dir

    def download(
        self,
        model_name: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download a model from HuggingFace Hub.

        This is the ONLY place in pvtt that makes network requests.

        Args:
            model_name: Short name (e.g., 'tiny.en') or HuggingFace repo ID.
            progress_callback: Optional callback(downloaded_bytes, total_bytes).

        Returns:
            Path to the downloaded model directory.

        Raises:
            ModelDownloadError: If download fails.
        """
        repo_id = MODEL_REPO_MAP.get(model_name, model_name)
        local_name = model_name.replace("/", "--")
        output_dir = self._models_dir / local_name

        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id,
                local_dir=str(output_dir),
            )
            logger.info("Downloaded model %s to %s", model_name, model_path)
            return Path(model_path)
        except Exception as exc:
            raise ModelDownloadError(
                f"Failed to download model {model_name!r}: {exc}"
            ) from exc

    def list_models(self) -> list[ModelInfo]:
        """List all locally downloaded models.

        Returns:
            List of ModelInfo for each valid model directory.
        """
        models: list[ModelInfo] = []
        if not self._models_dir.exists():
            return models

        for entry in sorted(self._models_dir.iterdir()):
            if entry.is_dir() and self._is_valid_model_dir(entry):
                size = sum(
                    f.stat().st_size for f in entry.rglob("*") if f.is_file()
                )
                models.append(
                    ModelInfo(
                        name=entry.name,
                        path=entry,
                        size_bytes=size,
                    )
                )
        return models

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model, verifying it exists.

        Args:
            model_name: Short name, repo ID, or local path.

        Returns:
            Path to the model directory.

        Raises:
            ModelNotFoundError: If model is not downloaded.
        """
        # Check if the name is already a valid path
        candidate = Path(model_name)
        if candidate.is_dir() and self._is_valid_model_dir(candidate):
            return candidate

        # Check in models directory
        local_name = model_name.replace("/", "--")
        model_dir = self._models_dir / local_name
        if model_dir.is_dir() and self._is_valid_model_dir(model_dir):
            return model_dir

        raise ModelNotFoundError(
            f"Model {model_name!r} not found locally. "
            f"Download it with: pvtt model download {model_name}"
        )

    def remove_model(self, model_name: str) -> None:
        """Remove a downloaded model.

        Args:
            model_name: Model name to remove.

        Raises:
            ModelNotFoundError: If model is not downloaded.
        """
        model_path = self.get_model_path(model_name)
        shutil.rmtree(model_path)
        logger.info("Removed model %s from %s", model_name, model_path)

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get detailed info about a downloaded model.

        Args:
            model_name: Model name to query.

        Returns:
            ModelInfo with name, path, and size.

        Raises:
            ModelNotFoundError: If model is not downloaded.
        """
        model_path = self.get_model_path(model_name)
        size = sum(
            f.stat().st_size for f in model_path.rglob("*") if f.is_file()
        )
        return ModelInfo(
            name=model_name,
            path=model_path,
            size_bytes=size,
        )

    def resolve_model_name(self, preferred: str = PREFERRED_MODEL) -> str:
        """Resolve 'auto' to the best locally installed model.

        Selection logic:
        1. If the preferred model (large-v3-turbo) is installed, use it.
        2. Otherwise, use the first installed model found.
        3. If no models are installed, raise ModelNotFoundError.

        Args:
            preferred: Preferred model name to check first.

        Returns:
            Name of the selected model.

        Raises:
            ModelNotFoundError: If no models are installed.
        """
        # Check preferred model first
        local_name = preferred.replace("/", "--")
        preferred_dir = self._models_dir / local_name
        if preferred_dir.is_dir() and self._is_valid_model_dir(preferred_dir):
            logger.info("Auto-selected preferred model: %s", preferred)
            return preferred

        # Fall back to first available model
        models = self.list_models()
        if models:
            selected = models[0].name
            logger.info("Auto-selected first available model: %s", selected)
            return selected

        raise ModelNotFoundError(
            "No models installed. Download one with: pvtt model download large-v3-turbo"
        )

    @staticmethod
    def _is_valid_model_dir(path: Path) -> bool:
        """Check if a directory contains a valid CTranslate2 model."""
        return (path / "model.bin").exists()

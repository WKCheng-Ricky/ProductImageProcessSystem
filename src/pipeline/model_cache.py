"""
Model Cache Module

Singleton cache for ML models to prevent duplicate loading.
"""

import logging
import threading
from typing import Any, Dict, Optional, Tuple

from .clip_loader import default_device, load_clip_model
from .rembg_loader import load_rembg_session

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe singleton cache for ML models."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models: Dict[str, Any] = {}
                    cls._instance._model_lock = threading.Lock()
        return cls._instance

    def get_clip_model(
        self,
        model_name: str,
        pretrained: Optional[str] = None,
        device: str = "auto",
    ) -> Tuple[Any, Any, Any, int]:
        """
        Get or load a CLIP model.

        Args:
            model_name: Model architecture name (e.g., 'ViT-B-32', 'hf-hub:laion/...')
            pretrained: Pretrained weights name
            device: Device to load model on ('cpu', 'cuda', 'auto')

        Returns:
            Tuple of (model, preprocess, tokenizer, dimension)
        """
        # Determine actual device
        if device == "auto":
            device = default_device()

        # Create cache key
        cache_key = f"clip:{model_name}:{pretrained}:{device}"

        with self._model_lock:
            if cache_key in self._models:
                logger.info(f"Using cached model: {cache_key}")
                return self._models[cache_key]

            logger.info(f"Loading model: {model_name} (pretrained={pretrained}, device={device})")

            model, preprocess, tokenizer, dimension = load_clip_model(model_name, pretrained, device)

            logger.info(f"Model loaded: {model_name}, dimension={dimension}, device={device}")

            # Cache and return
            result = (model, preprocess, tokenizer, dimension)
            self._models[cache_key] = result
            return result

    def get_rembg_session(self, model_name: str = "u2net", providers: tuple = None):
        """
        Get or load a rembg session.

        Args:
            model_name: rembg model name
            providers: ONNX providers tuple (must be hashable)

        Returns:
            rembg session
        """
        # Convert to list for rembg, but use tuple for cache key
        providers_list = list(providers) if providers else None
        cache_key = f"rembg:{model_name}:{providers}"

        with self._model_lock:
            if cache_key in self._models:
                logger.info(f"Using cached rembg session: {cache_key}")
                return self._models[cache_key]

            logger.info(f"Loading rembg session: {model_name}")
            session = load_rembg_session(model_name, providers_list)
            self._models[cache_key] = session
            return session

    def clear(self):
        """Clear all cached models."""
        with self._model_lock:
            self._models.clear()
            logger.info("Model cache cleared")

    def get_cached_models(self) -> list:
        """Get list of cached model keys."""
        with self._model_lock:
            return list(self._models.keys())


# Global singleton instance
_cache = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _cache
    if _cache is None:
        _cache = ModelCache()
    return _cache

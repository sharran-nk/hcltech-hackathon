"""Sentence-Transformers embeddings (all-MiniLM-L6-v2, 384 dimensions).

Vectors are L2-normalised, so FAISS inner product == cosine similarity.
The model is loaded once, lazily, and shared. First use downloads ~90 MB from Hugging Face
into the local cache; after that it works offline.
"""
import logging
import threading

import numpy as np

import config

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


class EmbeddingError(Exception):
    """Embedding generation failed (model missing, out of memory, ...)."""


class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name)
                except Exception as exc:
                    logger.error("Embedding model load failed: %s", type(exc).__name__)
                    raise EmbeddingError(f"Could not load embedding model '{self.model_name}': {type(exc).__name__}") from exc
        return self._model

    def embed_texts(self, texts: list) -> np.ndarray:
        """Return float32 array of shape (len(texts), 384), rows L2-normalised."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype="float32")
        model = self._load()
        try:
            vectors = model.encode(
                list(texts),
                batch_size=config.EMBEDDING_BATCH_SIZE,  # bounded memory on small EC2 instances
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.error("Embedding generation failed: %s", type(exc).__name__)
            raise EmbeddingError(f"Embedding generation failed: {type(exc).__name__}") from exc
        return np.ascontiguousarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


_default = None
_default_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _default
    with _default_lock:
        if _default is None:
            _default = Embedder()
        return _default

"""The embedding model adapter. Only `load_embedder` touches sentence-transformers, so tests can pass
any object with the same two methods and need no download.
"""

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE_VARIABLE = "SIE_DEVICE"
DEFAULT_DEVICE = "cpu"


class Embedder(Protocol):
    max_tokens: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """One embedding per text, as rows."""
        ...

    def count_tokens(self, text: str) -> int:
        """How many tokens the model sees in `text`, ignoring its limit."""
        ...


class CachingEmbedder:
    """Remembers each text's embedding, because Streamlit reruns the whole script on every click."""

    def __init__(self, model) -> None:
        self._model = model
        self._embeddings: dict[str, np.ndarray] = {}
        self.max_tokens: int = model.max_seq_length

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [text for text in dict.fromkeys(texts) if text not in self._embeddings]
        if missing:
            self._embeddings.update(zip(missing, self._model.encode(missing)))
        return np.array([self._embeddings[text] for text in texts])

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text, verbose=False)["input_ids"])


def load_embedder() -> Embedder:
    """Download (first run only) and load the model."""
    from sentence_transformers import SentenceTransformer

    device = os.environ.get(DEVICE_VARIABLE, DEFAULT_DEVICE)
    return CachingEmbedder(SentenceTransformer(MODEL_ID, device=device))

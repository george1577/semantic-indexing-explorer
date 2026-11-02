import hashlib

import numpy as np
import pytest


class FakeEmbedder:
    """Texts sharing words get similar vectors. No model, no download."""

    max_tokens = 256

    def encode(self, texts):
        return np.array([self._vector(text) for text in texts])

    def count_tokens(self, text):
        return len(text.split())

    def _vector(self, text):
        vector = np.zeros(32)
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode()).digest()
            vector[digest[0] % 32] += 1 if digest[1] % 2 else -1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


class TableEmbedder:
    """Returns the vector given for each text, for exact expected numbers."""

    max_tokens = 256

    def __init__(self, vectors):
        self._vectors = vectors

    def encode(self, texts):
        return np.array([self._vectors[text] for text in texts], dtype=float)

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()

"""Cosine similarity, written out so its zero-vector case is explicit."""

import numpy as np


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Similarity of every row in `left` with every row in `right`. A zero vector scores 0."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return _unit(left) @ _unit(right).T


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(cosine_matrix(np.atleast_2d(left), np.atleast_2d(right))[0, 0])


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)

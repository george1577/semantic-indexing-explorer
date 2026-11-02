"""Bag-of-words keyword search, using the notebook's CountVectorizer."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from semantic_lab.similarity import cosine_matrix


@dataclass(frozen=True)
class BagOfWords:
    chunk_vectors: np.ndarray
    query_vector: np.ndarray
    vocabulary: tuple[str, ...]
    query_words_used: tuple[str, ...]  # counted in the search
    query_words_unknown: tuple[str, ...]  # appear in no chunk, so they're ignored
    query_words_stopped: tuple[str, ...]  # removed as stopwords
    matched_words: tuple[tuple[str, ...], ...]  # per chunk, the query words it contains


def build_bag_of_words(chunks: Sequence[str], query: str, remove_stopwords: bool) -> BagOfWords:
    """Count words in the chunks and the query. The vocabulary comes from the chunks alone, as in the
    notebook, so a query word no chunk contains can't count.
    """
    vectorizer = CountVectorizer(stop_words="english" if remove_stopwords else None)
    try:
        chunk_vectors = vectorizer.fit_transform(chunks).toarray()
    except ValueError:  # every word in every chunk was removed, so there's no vocabulary
        chunk_vectors = np.zeros((len(chunks), 0), dtype=int)
        vocabulary: tuple[str, ...] = ()
        query_vector = np.zeros(0, dtype=int)
    else:
        vocabulary = tuple(vectorizer.get_feature_names_out())
        query_vector = vectorizer.transform([query]).toarray()[0]

    known = set(vocabulary)
    query_words = list(dict.fromkeys(_words(query)))
    stopped = [w for w in query_words if remove_stopwords and w in ENGLISH_STOP_WORDS]
    used = [w for w in query_words if w in known]
    unknown = [w for w in query_words if w not in known and w not in stopped]
    matched = tuple(
        tuple(word for word in used if row[vocabulary.index(word)] > 0) for row in chunk_vectors
    )
    return BagOfWords(
        chunk_vectors, query_vector, vocabulary, tuple(used), tuple(unknown), tuple(stopped), matched
    )


def query_similarities(bag: BagOfWords) -> np.ndarray:
    if not bag.vocabulary:
        return np.zeros(len(bag.chunk_vectors))
    return cosine_matrix(bag.query_vector[None, :], bag.chunk_vectors)[0]


def _words(text: str) -> list[str]:
    return CountVectorizer().build_tokenizer()(text.lower())

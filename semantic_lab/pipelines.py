"""Each tab's pipeline, returning plain results. The seam the tests use."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from semantic_lab.chunkers import (
    SemanticChunking,
    ends_with_cut_off_heading,
    fixed_size_chunks,
    notebook_structural_chunks,
    semantic_chunks,
    structural_chunks,
)
from semantic_lab.embedder import Embedder
from semantic_lab.keyword import BagOfWords, build_bag_of_words, query_similarities
from semantic_lab.presets import ChunkQuestion
from semantic_lab.similarity import cosine_matrix
from semantic_lab.tables import (
    HybridMatch,
    MetadataFilter,
    Product,
    TextMatch,
    rank_hybrid,
    rank_text,
)

FIXED = "Fixed-size"
NOTEBOOK_STRUCTURAL = "Structural (notebook)"
STRUCTURAL = "Structural (headings fixed)"
SEMANTIC = "Semantic"
METHODS = (FIXED, NOTEBOOK_STRUCTURAL, STRUCTURAL, SEMANTIC)


# Tab 1: keyword search vs embeddings


@dataclass(frozen=True)
class KeywordVsEmbeddings:
    chunks: tuple[str, ...]
    bag: BagOfWords
    keyword_similarities: np.ndarray  # query against each chunk
    embedding_similarities: np.ndarray
    keyword_matrix: np.ndarray  # chunk against chunk, the notebook's heatmaps
    embedding_matrix: np.ndarray


def run_keyword_vs_embeddings(
    chunks: Sequence[str], query: str, remove_stopwords: bool, embedder: Embedder
) -> KeywordVsEmbeddings:
    bag = build_bag_of_words(chunks, query, remove_stopwords)
    embeddings = embedder.encode(chunks)
    query_embedding = embedder.encode([query])
    keyword_matrix = (
        cosine_matrix(bag.chunk_vectors, bag.chunk_vectors)
        if bag.vocabulary
        else np.zeros((len(chunks), len(chunks)))
    )
    return KeywordVsEmbeddings(
        tuple(chunks),
        bag,
        query_similarities(bag),
        cosine_matrix(query_embedding, embeddings)[0],
        keyword_matrix,
        cosine_matrix(embeddings, embeddings),
    )


# Tab 2: chunking


@dataclass(frozen=True)
class ChunkingMethod:
    name: str
    chunks: tuple[str, ...]
    cut_off_headings: tuple[int, ...]  # indexes of chunks ending in one
    over_model_limit: tuple[int, ...]  # indexes of chunks the model only partly reads


@dataclass(frozen=True)
class ChunkingComparison:
    methods: tuple[ChunkingMethod, ...]
    semantic: SemanticChunking
    max_tokens: int

    def method(self, name: str) -> ChunkingMethod:
        return next(m for m in self.methods if m.name == name)


@dataclass(frozen=True)
class ChunkHit:
    chunk_index: int
    similarity: float


@dataclass(frozen=True)
class QuestionRanks:
    question: ChunkQuestion
    ranks: dict[str, int | None]  # method name -> answer chunk's rank, None if no chunk has it whole


def run_chunking(
    text: str, max_size: int, break_percentile: float, embedder: Embedder
) -> ChunkingComparison:
    semantic = semantic_chunks(text, embedder, break_percentile)
    candidates = {
        FIXED: fixed_size_chunks(text, max_size),
        NOTEBOOK_STRUCTURAL: notebook_structural_chunks(text, max_size),
        STRUCTURAL: structural_chunks(text, max_size),
        SEMANTIC: list(semantic.chunks),
    }
    methods = tuple(
        ChunkingMethod(
            name,
            tuple(chunks),
            tuple(i for i, c in enumerate(chunks) if ends_with_cut_off_heading(c)),
            tuple(
                i
                for i, c in enumerate(chunks)
                if embedder.count_tokens(c) > embedder.max_tokens
            ),
        )
        for name, chunks in candidates.items()
    )
    return ChunkingComparison(methods, semantic, embedder.max_tokens)


def search_chunks(
    chunks: Sequence[str], query: str, embedder: Embedder
) -> list[ChunkHit]:
    """Every chunk, best first. Ties go to the earlier chunk."""
    if not chunks:
        return []
    similarities = cosine_matrix(embedder.encode([query]), embedder.encode(chunks))[0]
    order = np.argsort(-similarities, kind="stable")
    return [ChunkHit(int(i), float(similarities[i])) for i in order]


def rank_questions(
    comparison: ChunkingComparison, questions: Sequence[ChunkQuestion], embedder: Embedder
) -> list[QuestionRanks]:
    return [
        QuestionRanks(
            question,
            {m.name: _answer_rank(m.chunks, question, embedder) for m in comparison.methods},
        )
        for question in questions
    ]


def _answer_rank(
    chunks: Sequence[str], question: ChunkQuestion, embedder: Embedder
) -> int | None:
    for rank, hit in enumerate(search_chunks(chunks, question.query, embedder), start=1):
        if question.answer in chunks[hit.chunk_index]:
            return rank
    return None


# Tab 3: tables


@dataclass(frozen=True)
class TableComparison:
    hybrid: list[HybridMatch]
    text: list[TextMatch]
    rows_searched: int
    rows_total: int


def run_table_search(
    products: Sequence[Product], query: str, filters: MetadataFilter, embedder: Embedder
) -> TableComparison:
    hybrid = rank_hybrid(products, query, embedder, filters)
    text = rank_text(products, query, embedder, filters)
    return TableComparison(hybrid, text, len(text), len(products))

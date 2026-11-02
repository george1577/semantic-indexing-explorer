import numpy as np
import pytest

from semantic_lab.pipelines import (
    FIXED,
    METHODS,
    NOTEBOOK_STRUCTURAL,
    STRUCTURAL,
    rank_questions,
    run_chunking,
    run_keyword_vs_embeddings,
    run_table_search,
    search_chunks,
)
from semantic_lab.presets import (
    CHUNK_QUESTIONS,
    NOTEBOOK_CHUNKS,
    NOTEBOOK_QUERY,
    ChunkQuestion,
    load_bach,
)
from semantic_lab.tables import MetadataFilter, load_products
from tests.conftest import TableEmbedder


def test_keyword_vs_embeddings_shapes(fake_embedder):
    result = run_keyword_vs_embeddings(NOTEBOOK_CHUNKS, NOTEBOOK_QUERY, False, fake_embedder)
    assert result.keyword_similarities.shape == (4,)
    assert result.embedding_similarities.shape == (4,)
    assert result.keyword_matrix.shape == (4, 4)
    assert np.allclose(np.diag(result.keyword_matrix), 1)
    assert np.allclose(np.diag(result.embedding_matrix), 1)


def test_chunking_runs_every_method_and_flags_cut_off_headings(fake_embedder):
    result = run_chunking(load_bach(), 1000, 20, fake_embedder)
    assert [m.name for m in result.methods] == list(METHODS)
    assert len(result.method(NOTEBOOK_STRUCTURAL).cut_off_headings) == 4
    assert result.method(STRUCTURAL).cut_off_headings == ()


def test_chunks_longer_than_the_model_reads_are_flagged(fake_embedder):
    text = "word " * 300
    result = run_chunking(text, 100_000, 20, fake_embedder)
    assert result.method(FIXED).over_model_limit == (0,)


def test_search_orders_best_first_and_ties_go_to_the_earlier_chunk():
    embedder = TableEmbedder({"q": [1, 0], "a": [0, 1], "b": [1, 0], "c": [1, 0]})
    hits = search_chunks(["a", "b", "c"], "q", embedder)
    assert [h.chunk_index for h in hits] == [1, 2, 0]
    assert hits[0].similarity == pytest.approx(1)


def test_question_rank_is_none_when_a_cut_splits_the_answer(fake_embedder):
    text = "x" * 95 + "answer phrase" + "y" * 95
    comparison = run_chunking(text, 100, 20, fake_embedder)
    ranks = rank_questions(comparison, [ChunkQuestion("q", "answer phrase")], fake_embedder)[0].ranks
    assert ranks[FIXED] is None
    assert ranks[STRUCTURAL] == 1


def test_every_preset_answer_is_in_the_document():
    text = load_bach()
    for question in CHUNK_QUESTIONS:
        assert question.answer in text


def test_table_search_counts_rows_and_returns_both_rankings(fake_embedder):
    products = load_products()
    result = run_table_search(products, "coffee", MetadataFilter(in_stock=False), fake_embedder)
    assert result.rows_total == 30
    assert result.rows_searched == len(result.hybrid) == len(result.text) == len(result.without_numbers)
    assert len(result.unfiltered) == 30
    assert 0 < result.rows_searched < 30

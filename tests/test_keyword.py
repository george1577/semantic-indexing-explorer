import pytest

from semantic_lab.keyword import build_bag_of_words, query_similarities
from semantic_lab.presets import NOTEBOOK_CHUNKS, NOTEBOOK_QUERY


def test_without_stopword_removal_the_and_was_outrank_film():
    bag = build_bag_of_words(NOTEBOOK_CHUNKS, NOTEBOOK_QUERY, remove_stopwords=False)
    assert query_similarities(bag) == pytest.approx([0.408, 0.577, 0.289, 0.516], abs=0.001)
    assert bag.matched_words[0] == ("the", "film")
    assert bag.matched_words[1] == ("the", "was")
    assert bag.query_words_unknown == ("great",)


def test_with_stopword_removal_only_film_counts():
    bag = build_bag_of_words(NOTEBOOK_CHUNKS, NOTEBOOK_QUERY, remove_stopwords=True)
    similarities = query_similarities(bag)
    assert similarities[1:] == pytest.approx([0, 0, 0])
    assert similarities[0] > 0
    assert bag.query_words_used == ("film",)
    assert bag.query_words_stopped == ("the", "was")


def test_query_of_only_stopwords_scores_zero_everywhere():
    bag = build_bag_of_words(NOTEBOOK_CHUNKS, "the was", remove_stopwords=True)
    assert query_similarities(bag) == pytest.approx([0, 0, 0, 0])


def test_chunks_of_only_stopwords_give_an_empty_vocabulary():
    bag = build_bag_of_words(["the was", "it is"], "film", remove_stopwords=True)
    assert bag.vocabulary == ()
    assert query_similarities(bag) == pytest.approx([0, 0])

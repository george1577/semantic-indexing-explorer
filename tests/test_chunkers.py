import re

import numpy as np

from semantic_lab.chunkers import (
    ends_with_cut_off_heading,
    fixed_size_chunks,
    notebook_structural_chunks,
    semantic_chunks,
    semantic_units,
    structural_chunks,
)
from semantic_lab.presets import load_bach
from tests.conftest import TableEmbedder


def _letters(text):
    return re.sub(r"\s+", "", text)


def test_fixed_size_chunks_are_exact_slices():
    chunks = fixed_size_chunks("abcdefgh", 3)
    assert chunks == ["abc", "def", "gh"]


def test_notebook_chunker_matches_the_notebooks_printed_sizes():
    chunks = notebook_structural_chunks(load_bach(), 1000)
    assert [len(c) for c in chunks] == [471, 915, 517, 547, 917, 744, 523, 814, 424, 503, 313, 191]


def test_notebook_chunker_cuts_off_four_headings_and_the_fix_cuts_off_none():
    text = load_bach()
    assert sum(map(ends_with_cut_off_heading, notebook_structural_chunks(text, 1000))) == 4
    assert not any(map(ends_with_cut_off_heading, structural_chunks(text, 1000)))


def test_structural_chunks_keep_all_the_text():
    text = load_bach()
    assert _letters("".join(structural_chunks(text, 400))) == _letters(text)


def test_stacked_headings_stay_with_their_text_even_past_the_limit():
    text = "## Career\n\n### Early\n\n" + "x" * 50
    assert structural_chunks(text, 10) == [text]


def test_a_heading_starts_a_new_chunk_at_any_level():
    chunks = structural_chunks("### One\n\nFirst.\n\n### Two\n\nSecond.", 1000)
    assert chunks == ["### One\n\nFirst.", "### Two\n\nSecond."]


def test_units_keep_lists_whole_and_join_headings_to_text():
    units = semantic_units("## Title\n\nOne here. Two here.\n\n- a\n- b")
    assert units == ["## Title\n\nOne here.", "Two here.", "- a\n- b"]


def test_units_do_not_split_after_abbreviations():
    assert semantic_units("He went to St. Michael's School. Then he left.") == [
        "He went to St. Michael's School.",
        "Then he left.",
    ]


def test_units_keep_a_trailing_heading():
    assert semantic_units("Some text.\n\n## End") == ["Some text.\n\n## End"]


def test_semantic_chunks_cut_at_the_similarity_drop():
    a, b = "A" * 30 + " sentence.", "B" * 30 + " sentence."
    text = f"{a} {a} {b} {b}"
    # Identical neighbours can't arise from different text, so give each unit its own vector.
    units = semantic_units(text)
    vectors = {units[0]: [1, 0], units[2]: [0, 1]}
    result = semantic_chunks(text, TableEmbedder(vectors), break_percentile=0, min_size=1)
    assert result.chunks == (f"{a} {a}", f"{b} {b}")
    assert result.breaks == (1,)
    assert np.allclose(result.neighbour_similarities, [1, 0, 1])


def test_semantic_chunks_never_leave_a_piece_under_the_minimum(fake_embedder):
    result = semantic_chunks(load_bach(), fake_embedder, break_percentile=50, min_size=200)
    assert len(result.chunks) > 1
    assert all(len(chunk) >= 200 for chunk in result.chunks)


def test_semantic_chunks_keep_all_the_text(fake_embedder):
    text = load_bach()
    result = semantic_chunks(text, fake_embedder)
    assert _letters("".join(result.chunks)) == _letters(text)


def test_a_single_sentence_is_one_chunk(fake_embedder):
    result = semantic_chunks("Just one sentence.", fake_embedder)
    assert result.chunks == ("Just one sentence.",)
    assert result.breaks == ()


def test_empty_text_has_no_chunks(fake_embedder):
    assert semantic_chunks("", fake_embedder).chunks == ()
    assert fixed_size_chunks("", 100) == []
    assert structural_chunks("", 100) == []
    assert notebook_structural_chunks("", 100) == []

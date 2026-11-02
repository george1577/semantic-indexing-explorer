"""The three ways of chunking text. Sizes are in characters, as in the notebook."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from semantic_lab.embedder import Embedder

MIN_SEMANTIC_CHUNK = 200
DEFAULT_BREAK_PERCENTILE = 20

_ABBREVIATIONS = ("St.", "O.S.", "Dr.", "Mr.", "Mrs.", "Ms.", "No.", "vs.", "e.g.", "i.e.")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")


def fixed_size_chunks(text: str, size: int) -> list[str]:
    """Cut every `size` characters, wherever that falls."""
    return [text[start : start + size] for start in range(0, len(text), size)]


def notebook_structural_chunks(text: str, max_size: int) -> list[str]:
    """The notebook's chunker, unchanged. Only `#` and `##` start a chunk, so a `###` heading can be
    left at the end of a chunk, cut off from the text it introduces.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for paragraph in _paragraphs(text):
        is_top_heading = paragraph.startswith(("# ", "## "))
        if (is_top_heading and current) or (current_size + len(paragraph) > max_size and current):
            chunks.append(current)
            current, current_size = [], 0
        current.append(paragraph)
        current_size += len(paragraph)
    if current:
        chunks.append(current)
    return ["\n\n".join(chunk) for chunk in chunks]


def structural_chunks(text: str, max_size: int) -> list[str]:
    """The notebook's chunker with its heading problem fixed: a heading of any level starts a chunk,
    and a run of headings stays with the text below it, even past the size limit.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for paragraph in _paragraphs(text):
        headings_only = all(_is_heading(part) for part in current)
        starts_section = _is_heading(paragraph) and not headings_only
        too_big = current_size + len(paragraph) > max_size and not headings_only
        if current and (starts_section or too_big):
            chunks.append(current)
            current, current_size = [], 0
        current.append(paragraph)
        current_size += len(paragraph)
    if current:
        chunks.append(current)
    return ["\n\n".join(chunk) for chunk in chunks]


def ends_with_cut_off_heading(chunk: str) -> bool:
    """True when a chunk's last paragraph is a heading, whose text is in the next chunk."""
    return _is_heading(chunk.rsplit("\n\n", 1)[-1].strip())


@dataclass(frozen=True)
class SemanticChunking:
    chunks: tuple[str, ...]
    units: tuple[str, ...]
    neighbour_similarities: tuple[float, ...]  # between each unit and the next
    threshold: float
    breaks: tuple[int, ...]  # index into neighbour_similarities where a cut was made


def semantic_units(text: str) -> list[str]:
    """The pieces whose meaning gets compared. Prose becomes sentences, a list stays whole, and a
    heading joins the text after it, so every part of the document ends up in exactly one unit.
    """
    units: list[str] = []
    headings: list[str] = []
    for paragraph in _paragraphs(text):
        if _is_heading(paragraph):
            headings.append(paragraph)
            continue
        pieces = [paragraph] if _is_list(paragraph) else _sentences(paragraph)
        if headings:
            pieces[0] = "\n\n".join([*headings, pieces[0]])
            headings = []
        units.extend(pieces)
    if headings:  # a document ending in a heading
        if units:
            units[-1] = "\n\n".join([units[-1], *headings])
        else:
            units = ["\n\n".join(headings)]
    return units


def semantic_chunks(
    text: str,
    embedder: Embedder,
    break_percentile: float = DEFAULT_BREAK_PERCENTILE,
    min_size: int = MIN_SEMANTIC_CHUNK,
) -> SemanticChunking:
    """Cut where neighbouring units are least alike: at the lowest `break_percentile` percent of
    similarities. Pieces under `min_size` characters then join the piece before them.
    """
    units = semantic_units(text)
    if len(units) < 2:
        return SemanticChunking(tuple(units), tuple(units), (), 0.0, ())
    embeddings = embedder.encode(units)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit_vectors = embeddings / np.where(norms > 0, norms, 1)
    similarities = (unit_vectors[:-1] * unit_vectors[1:]).sum(axis=1)
    threshold = float(np.percentile(similarities, break_percentile))
    # At percentile 0 the threshold is the lowest similarity, which is still worth cutting at.
    candidates = [i for i, s in enumerate(similarities) if s <= threshold]
    chunks, breaks = _merge_small(units, candidates, min_size)
    return SemanticChunking(
        tuple(chunks),
        tuple(units),
        tuple(float(s) for s in similarities),
        threshold,
        tuple(breaks),
    )


def _merge_small(
    units: Sequence[str], candidates: Sequence[int], min_size: int
) -> tuple[list[str], list[int]]:
    """Group units at the candidate cuts, dropping any cut that would leave a piece under `min_size`.
    The last piece is checked too, so a short ending joins the piece before it.
    """
    groups: list[list[str]] = [[units[0]]]
    kept: list[int] = []
    wanted = set(candidates)
    for index, unit in enumerate(units[1:]):
        if index in wanted and sum(map(len, groups[-1])) >= min_size:
            groups.append([])
            kept.append(index)
        groups[-1].append(unit)
    if len(groups) > 1 and sum(map(len, groups[-1])) < min_size:
        groups[-2].extend(groups.pop())
        kept.pop()
    return [_join_units(group) for group in groups], kept


def _join_units(units: Sequence[str]) -> str:
    joined = units[0]
    for unit in units[1:]:
        separator = "\n\n" if _is_list(unit) or unit.startswith("#") else " "
        joined += separator + unit
    return joined


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def _is_heading(paragraph: str) -> bool:
    return bool(re.match(r"#{1,6} ", paragraph))


def _is_list(paragraph: str) -> bool:
    return bool(_LIST_LINE.match(paragraph))


def _sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(paragraph):
        if sentences and sentences[-1].endswith(_ABBREVIATIONS):
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return sentences

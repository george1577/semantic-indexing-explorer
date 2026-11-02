"""Streamlit UI: displays pipeline results and holds no indexing logic. The only file importing Streamlit."""

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from semantic_lab.embedder import MODEL_NAME, Embedder, load_embedder
from semantic_lab.pipelines import (
    METHODS,
    SEMANTIC,
    STRUCTURAL,
    ChunkingComparison,
    ChunkingMethod,
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
    TABLE_QUERIES,
    TableQuery,
    load_bach,
)
from semantic_lab.tables import PRICE_RANGES, MetadataFilter, load_products

GUIDE = "docs/how-semantic-indexing-works.md"
RESULTS_SHOWN = 10
MAX_PASTED_CHARACTERS = 50_000
BACH = "The Bach article (notebook)"
PASTE = "Paste your own markdown"
ANY = "Any"
IN_STOCK = "In stock"
OUT_OF_STOCK = "Out of stock"
EVEN_STYLE = "background: rgba(66, 133, 244, 0.14);"
ODD_STYLE = "background: rgba(128, 128, 128, 0.14);"
PROBLEM_STYLE = "background: rgba(255, 171, 0, 0.45); font-weight: 600;"
SHARE_COLOURS = {"Text": "#4285f4", "Numbers": "#f4a742", "In stock": "#9aa0a6"}

st.set_page_config(page_title="Semantic Indexing Explorer", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner=f"Loading {MODEL_NAME}...")
def get_embedder() -> Embedder:
    return load_embedder()


def main() -> None:
    st.title("🧭 Semantic Indexing Explorer")
    st.caption(
        "See how the way you index content changes what a search finds. "
        f"Companion to `examples/07_semantic_indexing`. New to this? Read `{GUIDE}`."
    )
    embedder = get_embedder()
    first, second, third = st.tabs(
        ["① Keyword vs embeddings", "② Chunking", "③ Indexing tables"]
    )
    with first:
        keyword_tab(embedder)
    with second:
        chunking_tab(embedder)
    with third:
        tables_tab(embedder)


# Tab 1


def keyword_tab(embedder: Embedder) -> None:
    st.subheader("Do they share words, or do they share meaning?")
    st.markdown(
        "Keyword search compares the words a query and a chunk have in common. An embedding "
        "compares what they mean. Try the toggle: with stopwords kept, *the* and *was* do most of "
        "the keyword search's work."
    )
    left, right = st.columns([3, 2])
    chunk_text = left.text_area("Chunks, one per line", "\n".join(NOTEBOOK_CHUNKS), height=130)
    query = right.text_input("Query", NOTEBOOK_QUERY)
    remove_stopwords = right.toggle("Remove stopwords from the keyword search", value=False)

    chunks = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    if not chunks or not query.strip():
        st.info("Enter at least one chunk and a query.")
        return

    result = run_keyword_vs_embeddings(chunks, query, remove_stopwords, embedder)

    bag = result.bag
    st.markdown(
        f"**How the keyword search reads the query:** counted {_word_list(bag.query_words_used)} · "
        f"in no chunk, so ignored {_word_list(bag.query_words_unknown)} · "
        f"removed as stopwords {_word_list(bag.query_words_stopped)}"
    )

    left, right = st.columns(2)
    left.markdown("#### Keyword search")
    keyword_rows = pd.DataFrame(
        {
            "Similarity": result.keyword_similarities,
            "Shared words": [", ".join(words) or "none" for words in bag.matched_words],
            "Chunk": chunks,
        }
    ).sort_values("Similarity", ascending=False, kind="stable")
    left.dataframe(_ranked(keyword_rows), hide_index=True, width="stretch")

    right.markdown(f"#### Embeddings ({MODEL_NAME})")
    embedding_rows = pd.DataFrame(
        {"Similarity": result.embedding_similarities, "Chunk": chunks}
    ).sort_values("Similarity", ascending=False, kind="stable")
    right.dataframe(_ranked(embedding_rows), hide_index=True, width="stretch")

    st.caption(
        "The notebook prepares this query but never compares it with anything, because its scoring "
        "loop is commented out. These rankings add that step. Bag of words is the simplest keyword "
        "search; BM25, covered in the Basic RAG Explorer, gives rare words more weight."
    )

    st.markdown("#### Chunk against chunk (the notebook's heatmaps)")
    left, right = st.columns(2)
    left.altair_chart(_heatmap(chunks, result.keyword_matrix, "Keyword"), width="stretch")
    right.altair_chart(_heatmap(chunks, result.embedding_matrix, "Embeddings"), width="stretch")


def _word_list(words: tuple[str, ...]) -> str:
    return ", ".join(f"`{word}`" for word in words) or "none"


def _ranked(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.round({"Similarity": 3}).reset_index(drop=True)
    rows.insert(0, "Rank", range(1, len(rows) + 1))
    return rows


def _heatmap(chunks: list[str], matrix, title: str) -> alt.Chart:
    labels = [f"{i + 1}. {chunk[:28]}{'…' if len(chunk) > 28 else ''}" for i, chunk in enumerate(chunks)]
    cells = pd.DataFrame(
        [
            {"Row": labels[r], "Column": labels[c], "Similarity": round(float(matrix[r, c]), 2)}
            for r in range(len(chunks))
            for c in range(len(chunks))
        ]
    )
    base = alt.Chart(cells, title=title).encode(
        x=alt.X("Column:N", sort=labels, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=170)),
        y=alt.Y("Row:N", sort=labels, title=None, axis=alt.Axis(labelLimit=200)),
    )
    squares = base.mark_rect().encode(
        color=alt.Color(
            "Similarity:Q", scale=alt.Scale(scheme="blues", domain=[0, 1], clamp=True), legend=None
        ),
        tooltip=["Row", "Column", "Similarity"],
    )
    numbers = base.mark_text().encode(
        text=alt.Text("Similarity:Q", format=".2f"),
        color=alt.condition(alt.datum.Similarity > 0.6, alt.value("white"), alt.value("black")),
    )
    return (squares + numbers).properties(height=max(200, 45 * len(chunks)))


# Tab 2


def chunking_tab(embedder: Embedder) -> None:
    st.subheader("Where should a document be cut?")
    st.markdown(
        "Three ways to cut, from knowing nothing to measuring meaning: **fixed-size** counts "
        "characters, **structural** follows headings and paragraphs, and **semantic** cuts where "
        "neighbouring sentences stop being alike. The notebook's \"semantic chunking\" is really "
        "structural, so it's shown here as it was written and with its heading problem fixed."
    )
    source = st.radio("Document", (BACH, PASTE), horizontal=True)
    is_bach = source == BACH
    if is_bach:
        text = load_bach()
    else:
        text = st.text_area("Your markdown", height=200, max_chars=MAX_PASTED_CHARACTERS)
        if not text.strip():
            st.info("Paste some markdown to chunk it.")
            return

    left, right = st.columns(2)
    max_size = left.slider(
        "Chunk size in characters (fixed-size and structural)", 200, 1500, 1000, 100
    )
    percentile = right.slider(
        "Semantic cuts: the lowest % of neighbour similarities", 5, 50, 20, 5
    )

    comparison = run_chunking(text, max_size, percentile, embedder)
    _chunk_summary(comparison)

    view = st.radio("Show the chunks from", METHODS, horizontal=True, index=METHODS.index(STRUCTURAL))
    method = comparison.method(view)
    if view == SEMANTIC:
        _semantic_chart(comparison)
    with st.expander(f"The {len(method.chunks)} chunks", expanded=True):
        _chunk_boxes(method)

    st.markdown("#### Does it change what a search finds?")
    if is_bach:
        _rank_table(comparison, embedder)
    query = st.text_input("Search the chunks", CHUNK_QUESTIONS[0].query if is_bach else "")
    if query.strip():
        rows = []
        for candidate in comparison.methods:
            hits = search_chunks(candidate.chunks, query, embedder)
            if hits:
                rows.append(
                    {
                        "Method": candidate.name,
                        "Best similarity": round(hits[0].similarity, 3),
                        "Best chunk": candidate.chunks[hits[0].chunk_index],
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _chunk_summary(comparison: ChunkingComparison) -> None:
    rows = []
    for method in comparison.methods:
        sizes = [len(chunk) for chunk in method.chunks] or [0]
        rows.append(
            {
                "Method": method.name,
                "Chunks": len(method.chunks),
                "Smallest": min(sizes),
                "Largest": max(sizes),
                "Ending in a cut-off heading": len(method.cut_off_headings),
                "Too long for the model": len(method.over_model_limit),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if any(method.over_model_limit for method in comparison.methods):
        st.caption(
            f"The model reads at most {comparison.max_tokens} tokens of a text and ignores the rest, "
            "so part of a chunk that's too long has no effect on its embedding."
        )


def _chunk_boxes(method: ChunkingMethod) -> None:
    boxes = []
    for index, chunk in enumerate(method.chunks):
        style = EVEN_STYLE if index % 2 == 0 else ODD_STYLE
        body = escape(chunk)
        if index in method.cut_off_headings:
            head, _, tail = chunk.rpartition("\n\n")
            body = (
                f"{escape(head)}\n\n" if head else ""
            ) + f"<span style='{PROBLEM_STYLE}'>{escape(tail)} ← cut off from its text</span>"
        notes = f"{len(chunk)} characters"
        if index in method.over_model_limit:
            notes += " · too long: the model ignores the end"
        boxes.append(
            f"<div style='{style} padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; "
            "border-radius: 0.3rem; white-space: pre-wrap; font-size: 0.9rem;'>"
            f"<strong>Chunk {index + 1}</strong> <small>({notes})</small>\n{body}</div>"
        )
    st.html("".join(boxes))


def _semantic_chart(comparison: ChunkingComparison) -> None:
    semantic = comparison.semantic
    if not semantic.neighbour_similarities:
        st.caption("This text has only one sentence, so there's nothing to compare.")
        return
    breaks = set(semantic.breaks)
    points = pd.DataFrame(
        {
            "Between sentences": range(1, len(semantic.neighbour_similarities) + 1),
            "Similarity": semantic.neighbour_similarities,
            "Cut": ["Cut here" if i in breaks else "No cut" for i in range(len(semantic.neighbour_similarities))],
        }
    )
    line = alt.Chart(points).mark_line(color="#4285f4").encode(
        x="Between sentences:Q", y=alt.Y("Similarity:Q", scale=alt.Scale(zero=False))
    )
    dots = alt.Chart(points).mark_circle(size=70).encode(
        x="Between sentences:Q",
        y="Similarity:Q",
        color=alt.Color("Cut:N", scale=alt.Scale(domain=["Cut here", "No cut"], range=["#d93025", "#4285f4"]), title=None),
        tooltip=["Between sentences", alt.Tooltip("Similarity:Q", format=".3f"), "Cut"],
    )
    threshold = alt.Chart(pd.DataFrame({"y": [semantic.threshold]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((line + dots + threshold).properties(height=240), width="stretch")
    st.caption(
        f"{len(semantic.units)} sentences and lists. Dashed line: the cut-off, {semantic.threshold:.3f}. "
        "A low point that makes no red dot was skipped, because it would have left a piece under "
        "200 characters."
    )


def _rank_table(comparison: ChunkingComparison, embedder: Embedder) -> None:
    rows = []
    results = rank_questions(comparison, CHUNK_QUESTIONS, embedder)
    for result in results:
        row = {"Question": result.question.query, "Answer contains": result.question.answer}
        row.update({name: result.ranks[name] for name in METHODS})
        rows.append(row)
    table = pd.DataFrame(rows).astype({name: "Int64" for name in METHODS})
    st.dataframe(table, hide_index=True, width="stretch")
    firsts = " · ".join(
        f"{name}: {sum(result.ranks[name] == 1 for result in results)}/{len(results)}"
        for name in METHODS
    )
    st.caption(
        f"The rank of the first chunk holding the answer phrase, where 1 is best. Blank means a cut "
        f"split the phrase. Ranked first: {firsts}. This article has clean headings, which suits "
        "structural chunking, so this doesn't show which method is best in general."
    )


# Tab 3


def tables_tab(embedder: Embedder) -> None:
    st.subheader("How do you index a row that's part meaning, part numbers?")
    st.markdown(
        "The notebook joins each row's text embedding with its scaled numbers into one **hybrid "
        "vector**. The query has no numbers, so it borrows each column's median and claims to be in "
        "stock. Compare that with searching on text alone and leaving the numbers to **metadata "
        "filters**."
    )
    _set_table_defaults()
    labels = [preset.label for preset in TABLE_QUERIES]
    st.selectbox("Start from a preset", labels, key="table_preset", on_change=_apply_table_preset)
    query = st.text_input("Query", key="table_query")

    columns = st.columns(5)
    price = columns[0].selectbox("Price range", (ANY, *PRICE_RANGES), key="table_price")
    products = load_products(notebook_only=st.session_state.get("table_notebook_only", False))
    categories = sorted({product.category for product in load_products()})
    category = columns[1].selectbox("Category", (ANY, *categories), key="table_category")
    stock = columns[2].selectbox("Stock", (ANY, IN_STOCK, OUT_OF_STOCK), key="table_stock")
    rating = columns[3].slider("Minimum rating", 0.0, 5.0, step=0.1, key="table_rating")
    columns[4].checkbox("Notebook's 5 products only", key="table_notebook_only")

    if not query.strip():
        st.info("Enter a query.")
        return

    filters = MetadataFilter(
        price_range=None if price == ANY else price,
        category=None if category == ANY else category,
        in_stock=None if stock == ANY else stock == IN_STOCK,
        min_rating=rating or None,
    )
    result = run_table_search(products, query, filters, embedder)
    st.caption(f"{result.rows_searched} of {result.rows_total} rows pass the filters.")
    if not result.hybrid:
        st.warning("No rows match these filters.")
        return

    left, right = st.columns(2)
    left.markdown("#### Hybrid vector (notebook)")
    left.dataframe(
        _ranked(
            pd.DataFrame(
                {
                    "Product": [m.product.name for m in result.hybrid],
                    "Similarity": [m.similarity for m in result.hybrid],
                    "Text alone": [round(m.text_similarity, 3) for m in result.hybrid],
                    "Price": [m.product.price for m in result.hybrid],
                    "In stock": [m.product.in_stock for m in result.hybrid],
                }
            )
        ).head(RESULTS_SHOWN),
        hide_index=True,
        width="stretch",
    )
    right.markdown("#### Text embedding plus filters")
    right.dataframe(
        _ranked(
            pd.DataFrame(
                {
                    "Product": [m.product.name for m in result.text],
                    "Similarity": [m.similarity for m in result.text],
                    "Price": [m.product.price for m in result.text],
                    "In stock": [m.product.in_stock for m in result.text],
                }
            )
        ).head(RESULTS_SHOWN),
        hide_index=True,
        width="stretch",
    )
    st.caption("The notebook prints distance, which is 1 minus the similarity shown here.")

    st.markdown("#### What makes up each hybrid similarity")
    st.altair_chart(_share_chart(result.hybrid[:RESULTS_SHOWN]), width="stretch")
    st.caption(
        "Each bar's parts add up to the row's similarity. Change the query and watch: only the blue "
        "part moves. The orange and grey parts come from the row's numbers and the query's fixed "
        "medians, so they're the same whatever you type."
    )

    with st.expander("The product table"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Product": [p.name for p in products],
                    "Description": [p.description for p in products],
                    "Category": [p.category for p in products],
                    "Price": [p.price for p in products],
                    "Price range": [p.price_range for p in products],
                    "Rating": [p.rating for p in products],
                    "Reviews": [p.review_count for p in products],
                    "In stock": [p.in_stock for p in products],
                    "Source": ["Notebook" if p.from_notebook else "Added" for p in products],
                }
            ),
            hide_index=True,
            width="stretch",
        )


def _set_table_defaults() -> None:
    if "table_query" not in st.session_state:
        _load_table_query(TABLE_QUERIES[0])


def _apply_table_preset() -> None:
    label = st.session_state["table_preset"]
    _load_table_query(next(preset for preset in TABLE_QUERIES if preset.label == label))


def _load_table_query(preset: TableQuery) -> None:
    state = st.session_state
    state["table_query"] = preset.query
    state["table_price"] = preset.filters.price_range or ANY
    state["table_category"] = preset.filters.category or ANY
    if preset.filters.in_stock is None:
        state["table_stock"] = ANY
    else:
        state["table_stock"] = IN_STOCK if preset.filters.in_stock else OUT_OF_STOCK
    state["table_rating"] = preset.filters.min_rating or 0.0
    state["table_notebook_only"] = preset.notebook_only


def _share_chart(matches) -> alt.Chart:
    names = [match.product.name for match in matches]
    rows = []
    for match in matches:
        rows += [
            {"Product": match.product.name, "Part": "Text", "Adds": match.text_share},
            {"Product": match.product.name, "Part": "Numbers", "Adds": match.numeric_share},
            {"Product": match.product.name, "Part": "In stock", "Adds": match.in_stock_share},
        ]
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X("Adds:Q", title="Contribution to similarity"),
            y=alt.Y("Product:N", sort=names, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color(
                "Part:N",
                scale=alt.Scale(domain=list(SHARE_COLOURS), range=list(SHARE_COLOURS.values())),
                title=None,
            ),
            tooltip=["Product", "Part", alt.Tooltip("Adds:Q", format="+.3f")],
        )
        .properties(height=max(160, 30 * len(names)))
    )


main()

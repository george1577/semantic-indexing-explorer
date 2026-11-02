"""Prints every number the guide quotes, from the real model. Needs the model; the tests don't."""

from semantic_lab.embedder import load_embedder
from semantic_lab.pipelines import METHODS, rank_questions, run_chunking, run_keyword_vs_embeddings, run_table_search
from semantic_lab.presets import CHUNK_QUESTIONS, NOTEBOOK_CHUNKS, NOTEBOOK_QUERY, TABLE_QUERIES, load_bach
from semantic_lab.tables import MetadataFilter, load_products

embedder = load_embedder()

print("== Tab 1")
for stop in (False, True):
    r = run_keyword_vs_embeddings(NOTEBOOK_CHUNKS, NOTEBOOK_QUERY, stop, embedder)
    print("stopwords removed:", stop, "| used", r.bag.query_words_used, "stopped", r.bag.query_words_stopped, "unknown", r.bag.query_words_unknown)
    for c, k, e, m in zip(r.chunks, r.keyword_similarities, r.embedding_similarities, r.bag.matched_words):
        print(f"  kw {k:.3f} emb {e:.3f} {m} {c}")

print("== Tab 2")
text = load_bach()
c = run_chunking(text, 1000, 20, embedder)
for m in c.methods:
    sizes = [len(x) for x in m.chunks]
    print(f"{m.name:30} chunks {len(sizes):3} sizes {min(sizes)}-{max(sizes)} cut-off {len(m.cut_off_headings)} over-limit {len(m.over_model_limit)}")
print("units", len(c.semantic.units), "threshold", round(c.semantic.threshold, 3))
rows = rank_questions(c, CHUNK_QUESTIONS, embedder)
print("question".ljust(52), *[n[:12].ljust(13) for n in METHODS])
for r in rows:
    print(r.question.query.ljust(52), *[str(r.ranks[n]).ljust(13) for n in METHODS])
for n in METHODS:
    print(n, "first:", sum(r.ranks[n] == 1 for r in rows), "of", len(rows), "split:", sum(r.ranks[n] is None for r in rows))

print("== Tab 3")
extra = ["affordable products", "cheap kitchen tools"]
for five in (True, False):
    products = load_products(notebook_only=five)
    print(f"--- {len(products)} products, no filters")
    for q in [t.query for t in TABLE_QUERIES[:4]] + extra:
        r = run_table_search(products, q, MetadataFilter(), embedder)
        print(f"Q {q}")
        print("  hybrid:", [f"{m.product.name[:22]} {m.similarity:.3f}" for m in r.hybrid[:3]])
        print("  text:  ", [f"{m.product.name[:22]} {m.similarity:.3f}" for m in r.text[:3]])
        m = r.hybrid[0]
        print(f"  top hybrid shares: text {m.text_share:+.3f} numbers {m.numeric_share:+.3f} stock {m.in_stock_share:+.3f}")
print("--- notebook presets")
for t in TABLE_QUERIES:
    if t.notebook_only:
        r = run_table_search(load_products(True), t.query, t.filters, embedder)
        print(t.label, [(m.product.name, round(1 - m.similarity, 3)) for m in r.hybrid])

print("== Follow-ups")
products = load_products()
r = run_table_search(products, TABLE_QUERIES[0].query, MetadataFilter(), embedder)
print("Gaming Laptop Pro hybrid rank of 30:", next(i for i, m in enumerate(r.hybrid, 1) if m.product.name == "Gaming Laptop Pro"))
m = next(m for m in r.hybrid if m.product.name == "Gaming Laptop Pro")
print(f"  similarity {m.similarity:.3f} text alone {m.text_similarity:.3f} shares {m.text_share:+.3f} {m.numeric_share:+.3f} {m.in_stock_share:+.3f}")
wins = sum(run_table_search(products, t.query, MetadataFilter(), embedder).hybrid[0].product.name == "Yoga Mat" for t in TABLE_QUERIES[:4])
print("Yoga Mat first on presets (30 rows):", wins, "of 4")
print("Semantic chunks: size / tokens / holds Mendelssohn / offset of Mendelssohn in tokens")
for chunk in c.method("Semantic").chunks:
    where = chunk.find("Mendelssohn")
    before = embedder.count_tokens(chunk[:where]) if where >= 0 else None
    print(" ", len(chunk), embedder.count_tokens(chunk), where >= 0, before)
from semantic_lab.pipelines import search_chunks
q = "Who revived interest in Bach in the 19th century?"
for name in ("Structural (notebook)", "Structural (headings fixed)", "Semantic"):
    chunks = c.method(name).chunks
    hits = search_chunks(chunks, q, embedder)
    print(name, [(round(h.similarity, 3), repr(chunks[h.chunk_index][:50])) for h in hits[:3]])
for label, q in (("heading fix effect", "Who was Bach's first wife?"), ("", "Where is Bach buried?")):
    for name in ("Structural (notebook)", "Structural (headings fixed)"):
        chunks = c.method(name).chunks
        h = search_chunks(chunks, q, embedder)[0]
        print(q, name, round(h.similarity, 3), len(chunks[h.chunk_index]))


print("== Tab 3 worked example (notebook's 5 products)")
from semantic_lab.tables import trace_vectors

five = load_products(notebook_only=True)
query = "Top Notebook for Gaming and Work"
t = trace_vectors(five, five[0], query, embedder)
print("sentence:", five[0].text)
print("size", t.vector_size, "means", t.scaling.means.round(3), "stds", t.scaling.deviations.round(3), "medians", t.scaling.medians.round(3))
print("row text", t.row_embedding[:4].round(3), "scaled", t.row_scaled.round(3), "stock", t.row_stock, "length", round(t.row_length, 3))
print("query text", t.query_embedding[:4].round(3), "scaled", t.query_scaled.round(3), "stock", t.query_stock, "length", round(t.query_length, 3))
m = t.match
d = t.row_length * t.query_length
print(f"divisor {d:.3f} raw text {m.text_share*d:.3f} numbers {m.numeric_share*d:.3f} stock {m.in_stock_share*d:.3f} total {m.similarity*d:.3f}")
print(f"shares text {m.text_share:.3f} numbers {m.numeric_share:.3f} stock {m.in_stock_share:.3f} similarity {m.similarity:.3f}")
for p in five:
    t = trace_vectors(five, p, query, embedder)
    print(f"  {p.name[:24]:25} length {t.row_length:.3f} shares {t.match.text_share:+.3f} {t.match.numeric_share:+.3f} {t.match.in_stock_share:+.3f} = {t.match.similarity:.3f} text alone {t.match.text_similarity:.3f}")

print("== Tab 3 with and without numeric columns")
queries = [x.query for x in TABLE_QUERIES[:4]]
for label, rows in (("5", five), ("30", products)):
    for q in queries:
        r = run_table_search(rows, q, MetadataFilter(), embedder)
        print(label, q)
        for n, ms in (("all", r.hybrid), ("no numbers", r.without_numbers), ("text", r.text)):
            print(f"   {n:11}", [f"{x.product.name[:20]} {x.similarity:.3f}" for x in ms[:5]])

print("== Tab 3 filters")
luxury = MetadataFilter(price_range="luxury", in_stock=True)
afford = MetadataFilter(price_range="affordable", in_stock=True)
for p in five:
    print(f"  {p.name[:24]:25} ${p.price} {p.price_range} stock={p.in_stock} luxury: {luxury.rejections(p) or 'keep'} afford: {afford.rejections(p) or 'keep'}")
for rows, q, f in ((five, "affordable products", afford), (products, "affordable products", MetadataFilter()), (products, "affordable products", MetadataFilter(price_range="affordable"))):
    r = run_table_search(rows, q, f, embedder)
    print(len(rows), q, f.price_range, r.rows_searched, "hybrid", [(x.product.name[:20], round(x.similarity, 3)) for x in r.hybrid[:4]])
    print("   text", [(x.product.name[:20], x.product.price, round(x.similarity, 3)) for x in r.text[:4]])
r = run_table_search(five, "affordable products", MetadataFilter(), embedder)
print("unfiltered", [(x.product.name[:20], round(x.similarity, 3)) for x in r.hybrid])
r = run_table_search(products, "something to keep my house safe", MetadataFilter(in_stock=True), embedder)
print("safe in-stock text", [(x.product.name[:20], round(x.similarity, 3)) for x in r.text[:3]])

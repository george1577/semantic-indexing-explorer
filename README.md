# Semantic Indexing Explorer

An interactive Streamlit app for seeing how the way you index content changes what a search finds. It runs on a laptop CPU and needs no API keys.

Companion to `examples/07_semantic_indexing` in [*Generative AI Design Patterns*](https://github.com/lakshmanok/generative-ai-design-patterns), covering all three notebooks.

> **New to this?** Read [How semantic indexing works](docs/how-semantic-indexing-works.md), a short guide built on real results from the app, including where they differ from the notebooks. Terms are defined in [GLOSSARY.md](GLOSSARY.md).

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/). You need internet the first time, to install the dependencies (about 1 GB including PyTorch) and download the embedding model.

```bash
uv sync
uv run streamlit run app.py
```

The app opens at http://localhost:8501. Once the model is downloaded, `HF_HUB_OFFLINE=1` runs everything with no network.

Run the tests, which need no network and no model:

```bash
uv run pytest
```

## Configuration

| Setting | How | Default |
|---|---|---|
| Chunk size | Slider on the Chunking tab, 200 to 1500 characters | 1000 |
| Semantic cuts | Slider on the Chunking tab, the lowest 5 to 50% of neighbour similarities | 20 |
| Device | `SIE_DEVICE` environment variable | `cpu` |

The embedding model is always `all-MiniLM-L6-v2`, the notebooks' model.

## How each tab works

The tabs are in learning order.

1. **Keyword vs embeddings** ranks your chunks against a query both ways, side by side. It lists the words behind each keyword score and has a stopword toggle. Below are the notebook's two chunk-against-chunk heatmaps. The ranking is an addition, because the notebook never uses its query.
2. **Chunking** cuts the Bach article, or markdown you paste, in four ways: fixed-size, the notebook's structural chunker, the same chunker with its heading problem fixed, and real semantic chunking. It flags cut-off headings and chunks too long for the model, and shows where semantic chunking cut. For the article, seven preset questions show the answer chunk's rank under each method. Any text can be searched freely.
3. **Indexing tables** searches 30 products with the notebook's hybrid vector and with a text embedding plus metadata filters, side by side. A chart splits each hybrid similarity into what its text, numbers and stock flag add. The filters start off. The notebook's two queries are presets, and they switch to its five products so its distances reproduce.

## Project structure

```
app.py                     Streamlit UI, the only file importing Streamlit
semantic_lab/pipelines.py  Each tab's pipeline, returning plain results. The test seam
semantic_lab/keyword.py    Bag-of-words search, showing which words matched
semantic_lab/chunkers.py   The chunking methods
semantic_lab/tables.py     Hybrid vectors, metadata filters and both table searches
semantic_lab/similarity.py Cosine similarity
semantic_lab/embedder.py   The embedding model adapter
semantic_lab/presets.py    The notebooks' inputs and the preset queries
data/                      The Bach article and the product table
docs/                      The beginner's guide and decisions
scripts/measure.py         Prints every number the guide quotes, from the real model
tests/                     Fakes only, no downloads
GLOSSARY.md                The project's terms
```

## Possible next step

`basic-rag` ends on BM25 failing to connect "ruptured", "broken" and "punctured" in the flame thrower manual. Running those same queries through embeddings here would show whether they close that gap. It's left out for now to keep the app small.

## Credits

`data/bach.md` and the first five products in `data/products.json` come from the book's [examples repository](https://github.com/lakshmanok/generative-ai-design-patterns), licensed under Apache 2.0. The other 25 products were written for this app and are marked as added in the app's product table.

# How semantic indexing works

> **The 30-second version.** A search can only find what its index lets it see. **Semantic indexing** represents content so a search can match it by meaning, not just shared words. Three choices decide how well that works: how you represent the text, where you cut it into **chunks**, and how you handle data that isn't text at all. This guide shows what each choice does, including some cases where the obvious answer loses.

The sections follow the app's tabs. Terms are defined in the [glossary](../GLOSSARY.md).

**Where the results come from.** Every number here comes from `scripts/measure.py`, which runs the app's real pipelines with `all-MiniLM-L6-v2` on the CPU. Run it yourself with `HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/measure.py`. These are small examples, so read them as illustrations, not benchmarks.

## 1. Keyword search vs embeddings

**Keyword search** compares words. An **embedding** turns text into a list of numbers, placed so that texts with similar meanings get similar numbers. Either way, **similarity** is the cosine of the angle between two lists of numbers, and higher means closer.

Here's the notebook's example, searching for "The film was great":

| Chunk | Keyword, all words | Keyword, no stopwords | Embedding |
|---|---:|---:|---:|
| The movie was excellent | 0.577 | 0.000 | 0.917 |
| I really enjoyed the film we watched last night | 0.408 | 0.447 | 0.691 |
| The cinematic experience was remarkable | 0.516 | 0.000 | 0.617 |
| I didn't like the documentary | 0.289 | 0.000 | 0.496 |

**Two methods can agree for different reasons.** Keyword search and embeddings both put "The movie was excellent" first. But keyword search only got there through `the` and `was`, and it ranks that chunk above the one containing the actual word `film`.

**Removing stopwords exposes the gap.** Without `the` and `was`, only `film` is left to match. `great` appears in no chunk, so it's ignored. "The movie was excellent" drops to zero, because keyword search has no idea a movie is a film. The embedding is unaffected and still ranks it first.

**A fair warning.** Bag of words is the simplest keyword search, so it's an easy opponent. BM25, covered in the Basic RAG Explorer, weighs rare words more heavily and does much better. Embeddings have weaknesses too: "I didn't like the documentary" still scores 0.496, because it's about the same topic, even though it says the opposite.

**A note on the notebook.** It prepares its query but never compares it with anything, because the scoring loop is commented out. Its heatmaps compare chunks with each other. The app keeps those heatmaps and adds the ranking.

## 2. Chunking: where to cut

Search works on pieces, so something has to choose the cuts. The notebook is called "semantic chunking", but its code never measures meaning. It follows the document's layout, so the app calls that **structural chunking**, and adds the other two methods for comparison:

- **Fixed-size chunking** cuts every N characters.
- **Structural chunking** cuts at headings and paragraph breaks.
- **Semantic chunking** embeds each sentence and cuts where neighbouring sentences are least alike.

On the Bach article, with a chunk size of 1000 and semantic cuts at the lowest 20%:

| Method | Chunks | Size range (characters) | Ending in a cut-off heading | Too long for the model | Answer chunk ranked first |
|---|---:|---:|---:|---:|---:|
| Fixed-size | 7 | 902 to 1000 | 1 | 0 | 4 of 7 |
| Structural (notebook) | 12 | 191 to 917 | 4 | 0 | 7 of 7 |
| Structural (headings fixed) | 18 | 191 to 559 | 0 | 0 | 7 of 7 |
| Semantic | 8 | 345 to 2091 | 0 | 2 | 6 of 7 |

The last column comes from seven preset questions. It counts a chunk as the answer if it contains a key phrase, which is a rough test.

**The notebook's heading problem.** Only `#` and `##` start a new chunk, so a `###` heading can land at the end of one chunk with its text in the next. Chunk 3, for example, ends in `### Arnstadt to Weimar (1708–1717)`. Fixing this didn't change a single rank here, because the old chunks already came first. It did tighten them: for "Who was Bach's first wife?", the top chunk shrinks from 503 characters to 289 and its similarity rises from 0.688 to 0.760. Smaller, more focused chunks mean less unrelated text sent on to whatever uses the results.

**Real semantic chunking didn't win here.** This article has clean headings, which is the best case for structural chunking, so this says nothing about messy text with no headings. Try pasting some into the app.

**Its one miss has a specific cause.** Asked "Who revived interest in Bach in the 19th century?", semantic chunking ranks the answer chunk third. That chunk is 444 tokens long, and "Mendelssohn" starts around token 269. The model reads only the first 256 tokens and ignores the rest, so it never saw the answer. A method that sets no size limit can produce chunks the model only partly reads, and nothing warns you. The app flags these chunks for that reason.

**Semantic chunking needs care with real documents.** Our first version split the numbered reference list into tiny pieces, some just `1.`, each an unreliable "sentence". So the app keeps lists whole, attaches headings to the text below them, and skips any cut that would leave a piece under 200 characters.

## 3. Indexing tables

A product row has text, numbers and a true/false flag. The notebook joins them into one **hybrid vector**: the row's text embedding, then its scaled price, rating and review count, then 1 or 0 for in stock. A query has no numbers, so it borrows each column's median and claims to be in stock.

**The numbers drown out the text.** A text embedding has length 1, but the structured part is longer: 1.2 to 2.6 on the notebook's five products. So it has a large say in the angle between vectors. With no filters on those five products, here's the top result for six different queries:

| Query | Hybrid vector | Text embedding |
|---|---|---|
| Top Notebook for Gaming and Work | Yoga Mat | Gaming Laptop Pro |
| morning caffeine | Yoga Mat | Organic Coffee Beans |
| something to keep my house safe | Yoga Mat | Smart Home Security Camera |
| exercise gear | Yoga Mat | Yoga Mat |
| affordable products | Yoga Mat | Wireless Noise-Cancelling Headphones |
| cheap kitchen tools | Yoga Mat | Organic Coffee Beans |

**Most of the score ignores the query.** The query's structured part is the same every time. So each row's numeric and in-stock contributions never change, whatever you type. For Yoga Mat those fixed parts add +0.127 and +0.403. Only the text contribution moves, and on the gaming query it adds just +0.029.

**The notebook's filters hide this.** Its first query filters to luxury items in stock, and only one product is left, so it can't be ranked wrong. But it does show the cost: Gaming Laptop Pro has a text similarity of 0.620 and a hybrid similarity of 0.189 (the notebook's distance of 0.811).

**More rows weaken the effect but don't remove it.** With all 30 products, Yoga Mat is first on 3 of the 6 queries, and second on the other 3. The gaming query's top hybrid result is still Yoga Mat, and Gaming Laptop Pro ranks 25th of 30, while the text embedding ranks it first.

**Text embeddings have their own blind spot.** On all 30 products, "affordable products" puts Running Shoes first, at $119. An embedding captures what a description is about, not whether a number is low. That's the job of a **metadata filter**, which removes rows before scoring and so never changes a similarity. Choosing "affordable" as a price range does what the words in the query can't.

**This isn't a verdict on hybrid vectors in general.** It's this recipe: unweighted numbers, and a query with no numeric intent of its own. Weighting the parts differently could change the picture, but the app doesn't explore that. The takeaway is to check what's actually driving your similarity before you trust it.

## What to take away

1. Inspect why a result ranked first, because the right answer can come from the wrong reason.
2. A method's name doesn't tell you what it does. "Semantic chunking" here is structural, and in a "hybrid embedding" the part that isn't an embedding can outweigh the part that is.
3. Structure is a strong signal when a document has it.
4. Check your chunks against your model's token limit, because anything past it is silently ignored.
5. Use embeddings for meaning and filters for facts.

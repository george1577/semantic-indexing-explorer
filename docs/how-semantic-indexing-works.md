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

So far everything has been text. A product table also has prices, ratings and true/false flags. This section follows one row and one query all the way through the notebook's method, using its five products. The app's third tab shows the same steps for any row and any query.

| Product | Category | Price | Rating | Reviews | In stock |
|---|---|---:|---:|---:|---|
| Gaming Laptop Pro | Electronics | $2,499.99 | 4.8 | 156 | Yes |
| Wireless Noise-Cancelling Headphones | Electronics | $349.99 | 4.6 | 243 | Yes |
| Smart Home Security Camera | Smart Home | $79.99 | 4.3 | 89 | No |
| Organic Coffee Beans | Food & Beverages | $24.99 | 4.7 | 312 | Yes |
| Yoga Mat | Sports & Fitness | $39.99 | 4.5 | 178 | Yes |

### Step 1: Every row is one chunk

Yes, each row is treated as one **chunk**. Nothing is split, so five rows make five searchable items, and a search returns whole rows.

### Step 2: Build the row's vector and the query's vector

#### The row

A **hybrid vector** is one long list of numbers standing for the whole row. Gaming Laptop Pro's is built in three parts.

**The text part.** The text columns are written out as one sentence, and the model turns that sentence into 384 numbers:

```
Product: Gaming Laptop Pro. Description: High-performance gaming laptop with
RTX 3080, 32GB RAM, and 1TB SSD. Category: Electronics

→ [-0.042, 0.060, -0.062, -0.044, … 380 more]
```

**The numeric part.** Raw prices would swamp ratings, so each number is scaled: (value − column mean) ÷ column standard deviation. The result says how far the value is from typical, and in which direction.

| Column | Laptop's value | Column mean | Standard deviation | Scaled |
|---|---:|---:|---:|---:|
| Price | 2,499.99 | 598.99 | 957.823 | **1.985** |
| Rating | 4.8 | 4.58 | 0.172 | **1.279** |
| Reviews | 156 | 195.6 | 76.207 | **−0.520** |

**The stock flag.** In stock is 1, and out of stock is 0. The laptop gets **1**.

**Join them end to end**, for 384 + 3 + 1 = 388 numbers:

```
Laptop:  [-0.042, 0.060, -0.062, … (384 text numbers) …,  1.985, 1.279, -0.520,  1]
          └───────────── text part ──────────────────┘    └── numeric ───────┘   └ stock
```

#### The query

The query must be the same shape as the rows to be compared with them. But "Top Notebook for Gaming and Work" has no price, rating or stock status, so the notebook fills those positions in:

| Part | Where it comes from | Value |
|---|---|---|
| Text (384 numbers) | The query, through the same model | `[-0.037, -0.033, -0.022, -0.041, …]` |
| Price | The column's median, $79.99, scaled | −0.542 |
| Rating | The column's median, 4.6, scaled | 0.116 |
| Reviews | The column's median, 178, scaled | −0.231 |
| In stock | Always assumed to be yes | 1 |

**The last four numbers are identical for every query.** Whatever you type, the query claims to want a median-priced, median-rated, in-stock product.

### Step 3: Compare the two vectors

**Similarity** multiplies the two vectors position by position, adds up the results, then divides by both vectors' lengths. Grouping the positions by part shows what each one adds. The laptop's vector has length 2.801 and the query's has length 1.536, so the divisor is 4.303.

| Part | Multiplied and added | ÷ 4.303 |
|---|---:|---:|
| Text | 0.620 | +0.144 |
| Price, rating, reviews | −0.807 | −0.187 |
| In stock | 1.000 | +0.232 |
| **Similarity** | **0.813** | **0.189** |

The notebook prints this as a distance, 1 − 0.189 = 0.811.

Those two numbers explain the rest of this section. Text similarity alone would be 0.620, which is a strong match, and two things shrink it:

- **The price disagrees with the query's invented median.** The laptop is expensive and the query "wants" $79.99, which costs −0.187.
- **Unusual numbers make a longer vector.** A longer vector means a bigger divisor, which shrinks everything the row scores.

Here is the same sum for all five rows. Only the Text column depends on what you typed:

| Product | Vector length | Text | Numeric | In stock | Similarity | Rank |
|---|---:|---:|---:|---:|---:|---:|
| Yoga Mat | 1.616 | +0.029 | +0.127 | +0.403 | 0.559 | 1 |
| Wireless Noise-Cancelling Headphones | 1.571 | +0.067 | +0.004 | +0.414 | 0.486 | 2 |
| Organic Coffee Beans | 2.276 | +0.015 | +0.015 | +0.286 | 0.316 | 3 |
| Gaming Laptop Pro | 2.801 | +0.144 | −0.187 | +0.232 | 0.189 | 4 |
| Smart Home Security Camera | 2.429 | +0.034 | +0.115 | +0.000 | 0.149 | 5 |

Yoga Mat wins a gaming search because its numbers sit close to the medians, which gives it a short vector and agreement with the query's fixed numbers. Its description hardly matters.

### Step 4: Metadata filters

So far every row has been ranked. A **metadata filter** decides which rows get ranked at all. It's a yes-or-no rule checked against a row's own columns. The notebook stores four columns for this: price range, category, rating and in stock.

Price range is a label worked out from the price:

| Price range | Price |
|---|---|
| budget | under $10 |
| affordable | $10 to under $50 |
| mid-range | $50 to under $200 |
| premium | $200 to under $1,000 |
| luxury | $1,000 and up |

**A filter runs first, and ranking only sees what's left.** Here's the notebook's second query, "affordable products", filtered to price range = affordable and in stock = yes:

| Product | Price range | In stock | Filter result | Similarity, no filter | Similarity, filtered |
|---|---|---|---|---:|---:|
| Yoga Mat | affordable | Yes | **Keep** | 0.646 | 0.646 |
| Organic Coffee Beans | affordable | Yes | **Keep** | 0.378 | 0.378 |
| Wireless Noise-Cancelling Headphones | premium | Yes | Removed: wrong price range | 0.553 | |
| Smart Home Security Camera | mid-range | No | Removed: wrong price range, out of stock | 0.169 | |
| Gaming Laptop Pro | luxury | Yes | Removed: wrong price range | 0.118 | |

Notice that the kept rows' similarities don't change. Headphones scored higher than the coffee beans, but it's still gone, because a filter is absolute.

**The notebook's filters hide the problem in Step 3.** Its first query filters to luxury and in stock, which leaves only Gaming Laptop Pro. A single row can't be ranked wrong, so its output looks fine, despite a similarity of just 0.189.

**Why filters and embeddings need each other.** Search all 30 products for "affordable products" using text only, and the top result is Running Shoes at $119, followed by Headphones at $349.99. The model knows what products are about, not whether a price is low. Add the affordable filter and the top result is Protein Snack Box at $36. The filter handled the fact, and the embedding handled the meaning.

| | A column in the vector | A metadata filter |
|---|---|---|
| **What it does** | Nudges a row's similarity up or down | Keeps or removes the row |
| **Effect on similarity** | Changes it, for every query | None |
| **Follows what you asked?** | No: a query's numbers are invented | Yes: you choose the rule |
| **Good for** | "Close to X", if the query has a real number (Step 6) | Hard facts such as price, stock and category |

### Step 5: Results as columns are removed

The same search runs on three indexes, each built from fewer columns. The filters are off.

| Index | Text (384 numbers) | Price, rating, reviews (3 numbers) | In stock (1 number) | Vector size |
|---|---|---|---|---:|
| All columns (the notebook's) | Yes | Yes | Yes | 388 |
| Text + in-stock flag | Yes | No | Yes | 385 |
| Text only | Yes | No | No | 384 |

So "text + in-stock flag" is the index with its numeric columns removed, and "text only" also removes the flag. Here's the top result for each:

| Query | All columns (notebook) | Text + in-stock flag | Text only |
|---|---|---|---|
| Top Notebook for Gaming and Work | Yoga Mat (0.559) | **Gaming Laptop Pro** (0.810) | **Gaming Laptop Pro** (0.620) |
| morning caffeine | Yoga Mat (0.571) | **Organic Coffee Beans** (0.633) | **Organic Coffee Beans** (0.265) |
| something to keep my house safe | Yoga Mat (0.607) | Headphones (0.595) | **Smart Home Security Camera** (0.353) |
| exercise gear | **Yoga Mat** (0.696) | **Yoga Mat** (0.705) | **Yoga Mat** (0.411) |

Bold marks the product you'd expect. Reading across:

- **Removing price, rating and reviews fixes most of it.** Going from the first column to the second, three of the four queries now find the right product. So the numeric columns are what override the meaning.
- **Removing the in-stock flag fixes the last one.** Going from the second column to the third, the house query finds the camera.

#### Text + in-stock flag vs text only

These two differ by a single number, so it's easiest to see on two rows. The query's flag is always 1, and the row's is 1 or 0. For a query and row whose text similarity is *s*:

| | In-stock row | Out-of-stock row |
|---|---|---|
| Text only | *s* | *s* |
| Text + in-stock flag | (*s* + 1) ÷ 2 | *s* ÷ 1.414 |

| Query | Row | In stock | Text only | Text + in-stock flag |
|---|---|---|---:|---:|
| Top Notebook for Gaming and Work | Gaming Laptop Pro | Yes | 0.620 | (0.620 + 1) ÷ 2 = **0.810** |
| something to keep my house safe | Smart Home Security Camera | No | 0.353 | 0.353 ÷ 1.414 = **0.250** |

- **Among in-stock rows, the order doesn't change.** Every one gets the same treatment, so the flag only lifts their similarities.
- **An out-of-stock row drops below almost all of them.** On the house query, the camera has the best text match but falls from first to last, even though nobody asked about stock.

Removing the flag and using a stock filter (Step 4) gives you the choice per search, instead of always counting stock.

#### With more rows

All 30 products tell the same story. For the gaming query, the notebook's method puts Yoga Mat first and Gaming Laptop Pro 25th. With text + the in-stock flag, Gaming Laptop Pro is first and the Ultralight Business Notebook second. With more rows Yoga Mat doesn't always win, though it's first or second on all four queries here.

### Step 6: Can numbers be indexed in a better way?

Step 5 makes numeric columns look useless. **That's a result of this recipe, not a general rule.** The notebook's recipe has two weaknesses: the query's numbers are invented, and the numeric part has no weight to control how much it counts. Two experiments test other approaches, on all 30 products with no filters.

#### Experiment A: write the numbers into the text

Instead of adding positions to the vector, you can add the numbers to the sentence before embedding it. The vector stays at 384 numbers. There are two ways to write them:

| Index | Text added to Gaming Laptop Pro's sentence |
|---|---|
| Digits in the text | `Price: $2499.99. Rating: 4.8 out of 5. Reviews: 156. In stock: yes` |
| Words in the text | `Price range: luxury. Rating: excellent. Popularity: few reviews. Availability: in stock` |

**It's safe for ordinary searches.** This avoids the Yoga Mat problem entirely, because no separate number can outweigh the meaning. The gaming, caffeine and exercise queries keep the same top result as text only. On the house query the same three products stay on top, in a different order.

**But it barely helps with questions about numbers.** Here's how many of the top 5 results satisfy each query, against the most that could:

| Query | Correct means | Best possible | Text only | Digits in the text | Words in the text | All columns (notebook) |
|---|---|---:|---:|---:|---:|---:|
| products under $10 | Under $10 | 4 | 1 | 2 | 1 | 1 |
| cheap products | Under $50 | 5 | 3 | 4 | 3 | 3 |
| cheap kitchen tools | Kitchen, under $50 | 3 | 2 | 2 | 2 | 1 |
| coffee under $30 | Coffee, under $30 | 2 | 2 | 2 | 2 | 1 |
| luxury items over $1000 | $1000 or more | 4 | 1 | 1 | 1 | 0 |
| expensive electronics | Electronics, $200 or more | 3 | 3 | 3 | 3 | 1 |
| products rated 4.8 or higher | Rated 4.8 or more | 4 | 1 | 0 | 1 | 0 |
| highly rated products | Rated 4.7 or more | 5 | 2 | 2 | 2 | 1 |
| very popular products with many reviews | 1000 reviews or more | 5 | 0 | 0 | 0 | 0 |
| out of stock items | Out of stock | 5 | 0 | 0 | 0 | 0 |
| products in stock | In stock | 5 | 4 | 4 | 4 | 5 |
| **Total** | | **45** | **19** | **20** | **19** | **13** |

Writing the numbers in moves the total from 19 to 20 at best. "Out of stock items" scores 0 even when the sentence literally says "out of stock". A metadata filter on the same rule would score the best possible every time.

**Why digits don't work.** The model reads a number as text, so it doesn't reliably know that 9000 is more than 100. Similarity between price sentences shows this:

| | $10 | $11 | $100 | $1000 | $9000 |
|---|---:|---:|---:|---:|---:|
| **$10** | 1.00 | 0.86 | 0.85 | 0.85 | 0.80 |
| **$11** | 0.86 | 1.00 | 0.73 | 0.70 | 0.70 |
| **$100** | 0.85 | 0.73 | 1.00 | 0.84 | 0.81 |

To this model, $10 is about as close to $1000 (0.85) as to $11 (0.86), and $11 is exactly as close to $1000 as to $9000 (0.70). Similarities like these can't express "under $10".

**Why words don't help much either.** Words suit the model better, but the query has to use similar language. "Cheap products" still puts the $1349 Ultralight Business Notebook first, even though every row now states its price range.

**There's a small cost.** Digits are extra text unrelated to meaning, so they dilute the similarity. "exercise gear" against Resistance Bands falls from 0.495 to 0.438. It didn't change the order here, but it did on the house query.

**One reason you might still do it, which this app doesn't measure:** if a chunk later goes to an LLM, as in RAG, having the price in the text lets the model read and repeat it.

#### Experiment B: give the query a real number

What if the query really states a number? Here the query says the price it wants, and a weight sets how much the price counts. Only price is used. Weight 0 is the same as text only.

| Weight | "smart home device", wanting $1000 | "exercise gear", wanting $2000 |
|---:|---|---|
| 0 | Video Doorbell $149, Smart Thermostat $219, Security Camera $80 | Resistance Bands $20, Yoga Mat $40, Dumbbell Set $329 |
| 0.1 | Video Doorbell $149, Smart Thermostat $219, Security Camera $80 | Resistance Bands $20, **Carbon Road Bike $3,299**, Yoga Mat $40 |
| 0.3 | Video Doorbell $149, Smart Thermostat $219, **Water Leak Kit $1,099** | **Carbon Road Bike $3,299**, Gaming Laptop $2,500, Resistance Bands $20 |
| 0.5 | **Water Leak Kit $1,099**, Smart Thermostat $219, Video Doorbell $149 | **Carbon Road Bike $3,299**, Gaming Laptop $2,500, Ultralight Notebook $1,349 |
| 1 | **Water Leak Kit $1,099**, Gaming Laptop $2,500, Ultralight Notebook $1,349 | **Carbon Road Bike $3,299**, Gaming Laptop $2,500, Ultralight Notebook $1,349 |
| 2 | **Water Leak Kit $1,099**, Gaming Laptop $2,500, Carbon Road Bike $3,299 | **Carbon Road Bike $3,299**, Gaming Laptop $2,500, Ultralight Notebook $1,349 |

Bold marks the product matching both the meaning and the price.

**So numeric columns can help.** At weight 0.5, "smart home device" around $1000 finds the one smart home product at that price, which text only misses, while its other results are still smart home products. Text only can't do this.

**But the weight is hard to get right.** At weight 1 and above, a Gaming Laptop appears for a smart home search. The weight that works differs by query: a low target needed far more. "Smart home device" wanting $15 only finds the $14.99 Smart LED Bulb at weight 2, and by then the price is dominating. You'd need to tune this on real queries, and you'd also need a way to get a number out of the query at all.

A filter or sort avoids this tuning when you know the rule, but a price target can express something a filter can't: "around $1000", where closer is better and no row is ruled out.

#### So when do numeric columns help?

| Approach | Ordinary searches | Queries about numbers | What it needs |
|---|---|---|---|
| **Numeric columns, invented query numbers** (the notebook) | Hurts | Hurts | Nothing extra |
| **Numbers written into the text** | Roughly neutral | Barely helps | Nothing extra |
| **Numeric columns, real query numbers and a weight** | Neutral at weight 0, hurts at high weights | Helps for "close to X" | A number from the query, and tuning |
| **Text embedding plus metadata filter** | Works | Exact | A rule from the person or the query |

**A caveat on all of this.** These are 30 products, 25 of them written for this app, with one small model and a few queries, so they show how each approach behaves and don't rank them in general. A larger model may read numbers better, and a different table may reward different choices. Run this kind of check on your own data before choosing.

## What to take away

1. Inspect why a result ranked first, because the right answer can come from the wrong reason.
2. A method's name doesn't tell you what it does. "Semantic chunking" here is structural, and in a "hybrid embedding" the part that isn't an embedding can outweigh the part that is.
3. Structure is a strong signal when a document has it.
4. Check your chunks against your model's token limit, because anything past it is silently ignored.
5. Use embeddings for meaning and filters for facts. Writing numbers into text doesn't make a model understand them.

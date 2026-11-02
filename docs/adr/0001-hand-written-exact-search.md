# Search is a hand-written exact cosine scan, not ChromaDB

The notebook stores vectors in ChromaDB. The app instead scores every row with exact cosine similarity and applies metadata filters in its own code, because a table of about 30 rows gains nothing from an approximate index, and the app needs to show how much each part of a vector contributes to a score, which ChromaDB returns only as a total. The accepted cost is that we own this code and the app does not demonstrate a real vector database.

Exact cosine reproduces the notebook's ChromaDB output on its five products: distance 0.811 for Gaming Laptop Pro in the first query, and 0.354 and 0.622 for Yoga Mat and Organic Coffee Beans in the second (measured 2026-11-02 with `all-MiniLM-L6-v2`, using mean pooling through `transformers`).

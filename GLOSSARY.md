# Semantic Indexing Explorer

A learning tool for seeing how content is turned into a searchable index, and how those choices change what a search finds.

## Language

### Searching

**Semantic indexing**:
Representing content so that a search can match it by meaning, not only by shared words.
_Avoid_: Vector search, embedding search

**Query**:
The search text the person types.
_Avoid_: Prompt, question

**Keyword search**:
Finding chunks by the words they share with the query, whatever those words mean.
_Avoid_: Lexical search, full-text search

**Bag of words**:
A representation of text as a count of each word, ignoring word order. It is the form of keyword search this app uses.
_Avoid_: Keyword vector, count vector

**Stopword**:
A very common word removed from queries and chunks before searching, such as "the".
_Avoid_: Ignored word

**Embedding**:
A list of numbers a model produces from a piece of text, placed so that texts with similar meanings get similar numbers.
_Avoid_: Vector (when the text embedding specifically is meant), encoding

**Similarity**:
How closely two representations point the same way, measured as the cosine of the angle between them. Higher means closer.
_Avoid_: Score, relevance, distance

### Chunking

**Chunk**:
A contiguous piece of content, and the unit that gets searched and retrieved.
_Avoid_: Passage, segment, node

**Fixed-size chunking**:
Cutting text every N characters, whatever the text says.
_Avoid_: Naive chunking

**Structural chunking**:
Cutting text where the document's own layout changes, at headings and paragraph breaks. It uses no knowledge of meaning, and is what the source notebook calls "semantic chunking".
_Avoid_: Semantic chunking (for this method), markdown chunking

**Semantic chunking**:
Cutting text where the meaning shifts, found by a drop in similarity between neighbouring sentences.
_Avoid_: Smart chunking, structural chunking

### Tables

**Hybrid vector**:
A row's text embedding joined with its scaled numbers and flags, so one list of numbers stands for the whole row.
_Avoid_: Hybrid embedding, hybrid search

**Text part**:
The section of a hybrid vector that holds the text embedding.
_Avoid_: Semantic part

**Structured part**:
The section of a hybrid vector that holds the scaled numbers and flags.
_Avoid_: Numeric part, metadata

**Metadata filter**:
A rule that removes rows before scoring, so it never changes a similarity value.
_Avoid_: Post-filter, where clause

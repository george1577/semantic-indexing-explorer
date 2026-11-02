"""Everything the app ships with for its tabs to start from."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from semantic_lab.tables import MetadataFilter, Product

BACH_FILE = Path(__file__).parent.parent / "data" / "bach.md"

NOTEBOOK_CHUNKS = (
    "I really enjoyed the film we watched last night",
    "The movie was excellent",
    "I didn't like the documentary",
    "The cinematic experience was remarkable",
)
NOTEBOOK_QUERY = "The film was great"


@dataclass(frozen=True)
class ChunkQuestion:
    """A question about the Bach article. A chunk holds the answer if it contains `answer` whole."""

    query: str
    answer: str


CHUNK_QUESTIONS = (
    ChunkQuestion("Where did Bach work from 1708 to 1717?", "returning to Weimar"),
    ChunkQuestion("Who was Bach's first wife?", "Maria Barbara"),
    ChunkQuestion("How many children did Bach have?", "20 children"),
    ChunkQuestion("What did Prince Leopold hire Bach to do?", "Kapellmeister"),
    ChunkQuestion("Who revived interest in Bach in the 19th century?", "Mendelssohn"),
    ChunkQuestion("What are the Brandenburg Concertos?", "best-known orchestral works"),
    ChunkQuestion("Where is Bach buried?", "Cemetery"),
)


@dataclass(frozen=True)
class TableQuery:
    label: str
    query: str
    filters: MetadataFilter = MetadataFilter()
    notebook_only: bool = False


TABLE_QUERIES = (
    TableQuery("Gaming notebook, no filters", "Top Notebook for Gaming and Work"),
    TableQuery("Caffeine, no filters", "morning caffeine"),
    TableQuery("Home safety, no filters", "something to keep my house safe"),
    TableQuery("Exercise gear, no filters", "exercise gear"),
    TableQuery(
        "Notebook's query 1 (luxury, in stock)",
        "Top Notebook for Gaming and Work",
        MetadataFilter(price_range="luxury", in_stock=True),
        notebook_only=True,
    ),
    TableQuery(
        "Notebook's query 2 (affordable, in stock)",
        "affordable products",
        MetadataFilter(price_range="affordable", in_stock=True),
        notebook_only=True,
    ),
)


@dataclass(frozen=True)
class NumericQuestion:
    """A query that asks about a number or flag, and how to check a row actually satisfies it."""

    query: str
    correct_means: str
    is_correct: Callable[[Product], bool]


NUMERIC_QUESTIONS = (
    NumericQuestion("products under $10", "Under $10", lambda p: p.price < 10),
    NumericQuestion("cheap products", "Under $50", lambda p: p.price < 50),
    NumericQuestion(
        "cheap kitchen tools", "Kitchen, under $50", lambda p: p.price < 50 and p.category == "Kitchen"
    ),
    NumericQuestion(
        "coffee under $30", "Coffee, under $30", lambda p: p.price < 30 and "offee" in p.name
    ),
    NumericQuestion("luxury items over $1000", "$1000 or more", lambda p: p.price >= 1000),
    NumericQuestion(
        "expensive electronics",
        "Electronics, $200 or more",
        lambda p: p.price >= 200 and p.category == "Electronics",
    ),
    NumericQuestion("products rated 4.8 or higher", "Rated 4.8 or more", lambda p: p.rating >= 4.8),
    NumericQuestion("highly rated products", "Rated 4.7 or more", lambda p: p.rating >= 4.7),
    NumericQuestion(
        "very popular products with many reviews",
        "1000 reviews or more",
        lambda p: p.review_count >= 1000,
    ),
    NumericQuestion("out of stock items", "Out of stock", lambda p: not p.in_stock),
    NumericQuestion("products in stock", "In stock", lambda p: p.in_stock),
)


def load_bach() -> str:
    return BACH_FILE.read_text(encoding="utf-8")

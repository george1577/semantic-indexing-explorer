"""Indexing a product table: the notebook's hybrid vector, and a text embedding plus metadata filters."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semantic_lab.embedder import Embedder

PRODUCTS_FILE = Path(__file__).parent.parent / "data" / "products.json"
NUMERIC_COLUMNS = ("price", "rating", "review_count")
PRICE_RANGES = ("budget", "affordable", "mid-range", "premium", "luxury")


@dataclass(frozen=True)
class Product:
    product_id: int
    name: str
    description: str
    category: str
    price: float
    rating: float
    review_count: int
    in_stock: bool
    from_notebook: bool

    @property
    def price_range(self) -> str:
        return price_range(self.price)

    @property
    def text(self) -> str:
        """The notebook's way of writing a row as one string."""
        return f"Product: {self.name}. Description: {self.description}. Category: {self.category}"


@dataclass(frozen=True)
class MetadataFilter:
    """Each field left as None lets every row through."""

    price_range: str | None = None
    category: str | None = None
    in_stock: bool | None = None
    min_rating: float | None = None

    def allows(self, product: Product) -> bool:
        return (
            (self.price_range is None or product.price_range == self.price_range)
            and (self.category is None or product.category == self.category)
            and (self.in_stock is None or product.in_stock == self.in_stock)
            and (self.min_rating is None or product.rating >= self.min_rating)
        )


@dataclass(frozen=True)
class HybridMatch:
    """A hybrid vector similarity, split into what each section adds. The three parts sum to it."""

    product: Product
    similarity: float
    text_share: float
    numeric_share: float
    in_stock_share: float
    text_similarity: float  # the same row's text part alone, for comparison


@dataclass(frozen=True)
class TextMatch:
    product: Product
    similarity: float


def price_range(price: float) -> str:
    """The notebook's price bands."""
    if price < 10:
        return "budget"
    if price < 50:
        return "affordable"
    if price < 200:
        return "mid-range"
    if price < 1000:
        return "premium"
    return "luxury"


def load_products(notebook_only: bool = False) -> list[Product]:
    records = json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))
    products = [Product(**record) for record in records]
    return [p for p in products if p.from_notebook] if notebook_only else products


def rank_hybrid(
    products: Sequence[Product], query: str, embedder: Embedder, filters: MetadataFilter
) -> list[HybridMatch]:
    """Search as the notebook does. Numbers are scaled across all `products`, and the query has no
    numbers of its own, so it borrows the median of each column and claims to be in stock.
    """
    numbers = _numeric_matrix(products)
    means, deviations = numbers.mean(axis=0), numbers.std(axis=0)
    deviations = np.where(deviations > 0, deviations, 1.0)
    scaled = (numbers - means) / deviations
    query_numbers = (np.median(numbers, axis=0) - means) / deviations
    query_stock = 1.0

    query_text = embedder.encode([query])[0]
    query_norm = np.sqrt(query_text @ query_text + query_numbers @ query_numbers + query_stock**2)
    embeddings = embedder.encode([p.text for p in products])

    matches = []
    for product, text, row_numbers in zip(products, embeddings, scaled):
        if not filters.allows(product):
            continue
        stock = float(product.in_stock)
        row_norm = np.sqrt(text @ text + row_numbers @ row_numbers + stock**2)
        denominator = query_norm * row_norm
        text_share = float(query_text @ text / denominator)
        numeric_share = float(query_numbers @ row_numbers / denominator)
        stock_share = query_stock * stock / denominator
        matches.append(
            HybridMatch(
                product,
                text_share + numeric_share + stock_share,
                text_share,
                numeric_share,
                stock_share,
                _text_similarity(query_text, text),
            )
        )
    return sorted(matches, key=lambda m: -m.similarity)


def rank_text(
    products: Sequence[Product], query: str, embedder: Embedder, filters: MetadataFilter
) -> list[TextMatch]:
    """The correction: compare meaning with meaning, and let filters handle numbers and flags."""
    query_text = embedder.encode([query])[0]
    embeddings = embedder.encode([p.text for p in products])
    matches = [
        TextMatch(product, _text_similarity(query_text, text))
        for product, text in zip(products, embeddings)
        if filters.allows(product)
    ]
    return sorted(matches, key=lambda m: -m.similarity)


def _numeric_matrix(products: Sequence[Product]) -> np.ndarray:
    return np.array(
        [[getattr(p, column) for column in NUMERIC_COLUMNS] for p in products], dtype=float
    )


def _text_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))

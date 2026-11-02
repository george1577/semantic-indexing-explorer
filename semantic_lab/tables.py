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
QUERY_STOCK = 1.0  # the notebook assumes a query wants products in stock


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

    def rejections(self, product: Product) -> tuple[str, ...]:
        """Why this filter removes a row, one reason per failed rule. Empty means the row stays."""
        reasons = []
        if self.price_range is not None and product.price_range != self.price_range:
            reasons.append(f"price range is {product.price_range}, not {self.price_range}")
        if self.category is not None and product.category != self.category:
            reasons.append(f"category is {product.category}, not {self.category}")
        if self.in_stock is not None and product.in_stock != self.in_stock:
            reasons.append("out of stock" if self.in_stock else "in stock")
        if self.min_rating is not None and product.rating < self.min_rating:
            reasons.append(f"rating {product.rating} is under {self.min_rating}")
        return tuple(reasons)

    def allows(self, product: Product) -> bool:
        return not self.rejections(product)


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
class RowMatch:
    product: Product
    similarity: float


@dataclass(frozen=True)
class Scaling:
    """How each numeric column was scaled: subtract its mean, divide by its standard deviation."""

    means: np.ndarray
    deviations: np.ndarray
    medians: np.ndarray

    def scale(self, numbers: np.ndarray) -> np.ndarray:
        return (numbers - self.means) / self.deviations


@dataclass(frozen=True)
class VectorTrace:
    """Every step in building one row's hybrid vector and the query's, and in comparing them."""

    match: HybridMatch
    row_embedding: np.ndarray
    row_numbers: np.ndarray
    row_scaled: np.ndarray
    row_stock: float
    row_length: float
    query_embedding: np.ndarray
    query_numbers: np.ndarray  # the column medians, since a query has no numbers
    query_scaled: np.ndarray
    query_stock: float
    query_length: float
    scaling: Scaling

    @property
    def vector_size(self) -> int:
        return len(self.row_embedding) + len(self.row_scaled) + 1


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


def fit_scaling(products: Sequence[Product]) -> Scaling:
    numbers = _numeric_matrix(products)
    deviations = numbers.std(axis=0)
    return Scaling(
        numbers.mean(axis=0), np.where(deviations > 0, deviations, 1.0), np.median(numbers, axis=0)
    )


def rank_hybrid(
    products: Sequence[Product], query: str, embedder: Embedder, filters: MetadataFilter
) -> list[HybridMatch]:
    """Search as the notebook does. Numbers are scaled across all `products`, including rows a filter
    later removes, so a filter never changes a similarity.
    """
    scaling = fit_scaling(products)
    query_embedding = embedder.encode([query])[0]
    embeddings = embedder.encode([p.text for p in products])
    scaled_rows = scaling.scale(_numeric_matrix(products))
    matches = [
        _trace(product, embedding, scaled, query_embedding, scaling).match
        for product, embedding, scaled in zip(products, embeddings, scaled_rows)
        if filters.allows(product)
    ]
    return sorted(matches, key=lambda m: -m.similarity)


def trace_vectors(
    products: Sequence[Product], product: Product, query: str, embedder: Embedder
) -> VectorTrace:
    """Build and compare the hybrid vectors for one row and the query, keeping every step."""
    scaling = fit_scaling(products)
    numbers = _numeric_matrix([product])[0]
    return _trace(
        product,
        embedder.encode([product.text])[0],
        scaling.scale(numbers),
        embedder.encode([query])[0],
        scaling,
    )


def rank_text_and_stock(
    products: Sequence[Product], query: str, embedder: Embedder, filters: MetadataFilter
) -> list[RowMatch]:
    """The hybrid vector with its numeric columns removed: the text part and the in-stock flag."""
    query_embedding = embedder.encode([query])[0]
    query_vector = np.append(query_embedding, QUERY_STOCK)
    matches = [
        RowMatch(product, _text_similarity(query_vector, np.append(embedding, float(product.in_stock))))
        for product, embedding in zip(products, embedder.encode([p.text for p in products]))
        if filters.allows(product)
    ]
    return sorted(matches, key=lambda m: -m.similarity)


def rank_text(
    products: Sequence[Product], query: str, embedder: Embedder, filters: MetadataFilter
) -> list[RowMatch]:
    """Meaning against meaning: only the text part, with filters handling numbers and flags."""
    query_embedding = embedder.encode([query])[0]
    embeddings = embedder.encode([p.text for p in products])
    matches = [
        RowMatch(product, _text_similarity(query_embedding, embedding))
        for product, embedding in zip(products, embeddings)
        if filters.allows(product)
    ]
    return sorted(matches, key=lambda m: -m.similarity)


def _trace(
    product: Product,
    embedding: np.ndarray,
    scaled: np.ndarray,
    query_embedding: np.ndarray,
    scaling: Scaling,
) -> VectorTrace:
    query_scaled = scaling.scale(scaling.medians)
    stock = float(product.in_stock)
    row_length = float(np.sqrt(embedding @ embedding + scaled @ scaled + stock**2))
    query_length = float(
        np.sqrt(query_embedding @ query_embedding + query_scaled @ query_scaled + QUERY_STOCK**2)
    )
    divisor = row_length * query_length
    text_share = float(query_embedding @ embedding / divisor)
    numeric_share = float(query_scaled @ scaled / divisor)
    stock_share = QUERY_STOCK * stock / divisor
    match = HybridMatch(
        product,
        text_share + numeric_share + stock_share,
        text_share,
        numeric_share,
        stock_share,
        _text_similarity(query_embedding, embedding),
    )
    numbers = scaled * scaling.deviations + scaling.means
    return VectorTrace(
        match, embedding, numbers, scaled, stock, row_length,
        query_embedding, scaling.medians, query_scaled, QUERY_STOCK, query_length, scaling,
    )


def _numeric_matrix(products: Sequence[Product]) -> np.ndarray:
    return np.array(
        [[getattr(p, column) for column in NUMERIC_COLUMNS] for p in products], dtype=float
    )


def _text_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))

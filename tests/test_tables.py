import numpy as np
import pytest

from semantic_lab.tables import (
    MetadataFilter,
    Product,
    load_products,
    price_range,
    rank_hybrid,
    rank_text,
)
from tests.conftest import TableEmbedder


def _product(product_id, name, price, rating, reviews, in_stock=True, category="A"):
    return Product(product_id, name, "d", category, price, rating, reviews, in_stock, True)


PRODUCTS = [
    _product(1, "one", 10.0, 4.0, 10),
    _product(2, "two", 20.0, 4.5, 20, in_stock=False),
    _product(3, "three", 90.0, 5.0, 60, category="B"),
]
VECTORS = {
    "q": [1, 0],
    PRODUCTS[0].text: [1, 0],
    PRODUCTS[1].text: [0, 1],
    PRODUCTS[2].text: [0.6, 0.8],
}
EMBEDDER = TableEmbedder(VECTORS)
EVERYTHING = MetadataFilter()


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (9.99, "budget"),
        (10, "affordable"),
        (49.99, "affordable"),
        (50, "mid-range"),
        (199.99, "mid-range"),
        (200, "premium"),
        (999.99, "premium"),
        (1000, "luxury"),
    ],
)
def test_price_bands(price, expected):
    assert price_range(price) == expected


def test_hybrid_shares_sum_to_the_full_cosine():
    numbers = np.array([[10, 4.0, 10], [20, 4.5, 20], [90, 5.0, 60]], dtype=float)
    scaled = (numbers - numbers.mean(0)) / numbers.std(0)
    query_numbers = (np.median(numbers, 0) - numbers.mean(0)) / numbers.std(0)
    query = np.concatenate([[1, 0], query_numbers, [1]])

    for match in rank_hybrid(PRODUCTS, "q", EMBEDDER, EVERYTHING):
        index = PRODUCTS.index(match.product)
        stock = [float(match.product.in_stock)]
        row = np.concatenate([VECTORS[match.product.text], scaled[index], stock])
        expected = query @ row / (np.linalg.norm(query) * np.linalg.norm(row))
        assert match.similarity == pytest.approx(expected)
        total = match.text_share + match.numeric_share + match.in_stock_share
        assert total == pytest.approx(match.similarity)


def test_a_row_out_of_stock_gets_no_stock_share():
    matches = rank_hybrid(PRODUCTS, "q", EMBEDDER, EVERYTHING)
    two = next(m for m in matches if m.product.name == "two")
    assert two.in_stock_share == 0


def test_a_filter_removes_rows_without_changing_any_similarity():
    everything = {m.product.name: m.similarity for m in rank_hybrid(PRODUCTS, "q", EMBEDDER, EVERYTHING)}
    filtered = rank_hybrid(PRODUCTS, "q", EMBEDDER, MetadataFilter(category="B"))
    assert [m.product.name for m in filtered] == ["three"]
    assert filtered[0].similarity == pytest.approx(everything["three"])


def test_text_ranking_follows_meaning_alone():
    matches = rank_text(PRODUCTS, "q", EMBEDDER, EVERYTHING)
    assert [m.product.name for m in matches] == ["one", "three", "two"]
    assert [m.similarity for m in matches] == pytest.approx([1, 0.6, 0])


@pytest.mark.parametrize(
    ("filters", "names"),
    [
        (MetadataFilter(in_stock=True), {"one", "three"}),
        (MetadataFilter(in_stock=False), {"two"}),
        (MetadataFilter(min_rating=4.5), {"two", "three"}),
        (MetadataFilter(price_range="mid-range"), {"three"}),
        (MetadataFilter(price_range="luxury"), set()),
    ],
)
def test_filters(filters, names):
    assert {m.product.name for m in rank_text(PRODUCTS, "q", EMBEDDER, filters)} == names


def test_identical_rows_do_not_divide_by_zero():
    same = [_product(i, f"p{i}", 5.0, 4.0, 1) for i in range(2)]
    vectors = {"q": [1, 0], same[0].text: [1, 0], same[1].text: [0, 1]}
    assert len(rank_hybrid(same, "q", TableEmbedder(vectors), EVERYTHING)) == 2


def test_bundled_products():
    products = load_products()
    assert len(products) == 30
    assert len({p.product_id for p in products}) == 30
    assert {p.price_range for p in products} == {
        "budget", "affordable", "mid-range", "premium", "luxury"
    }
    notebook = load_products(notebook_only=True)
    assert [p.name for p in notebook][:2] == ["Gaming Laptop Pro", "Wireless Noise-Cancelling Headphones"]
    assert len(notebook) == 5

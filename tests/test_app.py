from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import FakeEmbedder

APP = Path(__file__).parent.parent / "app.py"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("semantic_lab.embedder.load_embedder", FakeEmbedder)
    import streamlit as st

    st.cache_resource.clear()
    return AppTest.from_file(str(APP), default_timeout=60).run()


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_every_tab_renders(app):
    assert not app.exception
    assert len(app.tabs) == 3


def test_stopword_toggle_reruns(app):
    _widget(app.toggle, "Remove stopwords from the keyword search").set_value(True).run()
    assert not app.exception


def test_every_chunk_view_renders(app):
    radio = _widget(app.radio, "Show the chunks from")
    for option in radio.options:
        radio.set_value(option).run()
        assert not app.exception


def test_empty_paste_asks_for_text(app):
    _widget(app.radio, "Document").set_value("Paste your own markdown").run()
    assert not app.exception
    assert any("Paste some markdown" in info.value for info in app.info)


def test_notebook_preset_sets_query_filters_and_rows(app):
    _widget(app.selectbox, "Start from a preset").set_value("Notebook's query 2 (affordable, in stock)").run()
    assert not app.exception
    assert _widget(app.text_input, "Query") is not None
    assert app.session_state["table_query"] == "affordable products"
    assert app.session_state["table_notebook_only"] is True
    assert app.session_state["table_price"] == "affordable"


def test_filters_that_match_nothing_warn(app):
    _widget(app.selectbox, "Category").set_value("Kitchen").run()
    _widget(app.checkbox, "Notebook's 5 products only").set_value(True).run()
    assert not app.exception
    assert any("No rows match" in warning.value for warning in app.warning)


def test_tracing_an_added_product_survives_the_switch_to_notebook_rows(app):
    _widget(app.selectbox, "Row to follow").set_value("Wooden Spoon Set").run()
    assert not app.exception
    _widget(app.checkbox, "Notebook's 5 products only").set_value(True).run()
    assert not app.exception


def test_an_empty_table_query_asks_for_one(app):
    next(w for w in app.text_input if w.key == "table_query").set_value(" ").run()
    assert not app.exception
    assert any("Enter a query" in info.value for info in app.info)
    next(w for w in app.text_input if w.key == "table_query").set_value("coffee").run()
    assert not app.exception


def test_step_six_still_shows_when_filters_match_nothing(app):
    _widget(app.selectbox, "Category").set_value("Kitchen").run()
    _widget(app.checkbox, "Notebook's 5 products only").set_value(True).run()
    assert any("Step 6" in markdown.value for markdown in app.markdown)


def test_step_six_controls_rerun(app):
    _widget(app.slider, "Weight on price").set_value(0.0).run()
    assert not app.exception
    next(w for w in app.text_input if w.key == "written_in_query").set_value("").run()
    assert not app.exception

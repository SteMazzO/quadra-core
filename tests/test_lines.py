"""Line and column reconstruction tests.

Fixtures are real Tesseract 5.3.4 output for a synthetic receipt, captured clean
and badly degraded, so these tests run without Tesseract or Pillow installed.
"""

from __future__ import annotations

import re

import pytest

from quadra_core.pipeline.lines import (
    Word,
    cluster_1d,
    discover_price_column,
    drop_speckle,
    group_lines,
    load_tsv,
    median_glyph_height,
    text_block,
)
from quadra_core.testdata import OCR as FIXTURES

PRICE = re.compile(r"\d{1,3},\d{2}")

# The eight products and the total, as printed on the synthetic receipt.
ITEM_PRICES = ["2,98", "2,30", "4,95", "3,09", "3,45", "3,79", "7,90", "3,56"]


def load(name: str):
    return load_tsv((FIXTURES / f"{name}.tsv").read_text())


@pytest.fixture(params=["synthetic_clean", "synthetic_faded"])
def variant(request):
    return request.param


def test_tsv_parses_and_drops_structural_rows(variant):
    words = load(variant)
    assert words, "fixture yielded no words"
    assert all(w.conf > -1 and w.text.strip() for w in words)


def test_speckle_filter_uses_height_not_confidence():
    """Regression guard for a measured trap.

    On the degraded fixture the noise artefact '-' scores 90.7 confidence while the
    genuine token '4X125.' scores 0.0. Anything that filters on confidence would keep
    the garbage and delete the data, so assert the filter is geometric.
    """
    words = load("synthetic_faded")
    kept = drop_speckle(words)
    dropped = [w for w in words if w not in kept]

    assert dropped, "degraded fixture should contain speckle"
    median = median_glyph_height(kept)
    assert all(w.height < median * 0.4 for w in dropped)
    # The high-confidence artefact must still be gone.
    assert any(w.conf > 70 for w in dropped), "expected a high-confidence artefact"
    # And low-confidence real text must survive.
    assert any(w.conf < 10 for w in kept), "low-confidence real text was deleted"


def test_line_count_is_stable_across_image_quality():
    """Clean and degraded images must reconstruct to the same line structure."""
    counts = {
        name: len(group_lines(drop_speckle(load(name))))
        for name in ("synthetic_clean", "synthetic_faded")
    }
    assert counts["synthetic_clean"] == counts["synthetic_faded"] == 13


def test_every_item_price_shares_a_line_with_its_description(variant):
    """The core requirement: a price must not be separated from its description."""
    lines = group_lines(drop_speckle(load(variant)))
    for price in ITEM_PRICES:
        hosts = [l for l in lines if price in l.text]
        assert len(hosts) >= 1, f"price {price} vanished"
        # Something alphabetic has to be on the line too, i.e. a description.
        assert any(c.isalpha() for c in hosts[0].text), f"price {price} is orphaned"


def test_price_column_is_right_anchored_and_tight(variant):
    """Right edges cluster even when the description-to-price gap fills with garbage."""
    lines = group_lines(drop_speckle(load(variant)))
    column = discover_price_column(lines, PRICE)
    assert column is not None
    low, high = column
    assert high > 0.9, "price column should sit at the right edge of the block"
    assert high - low < 0.05, f"column spread {high - low:.3f} too wide to be a column"


def test_grouping_tolerance_scales_with_resolution():
    """Doubling every coordinate must not change the reconstruction."""
    words = drop_speckle(load("synthetic_faded"))
    doubled = [
        Word(w.text, w.left * 2, w.top * 2, w.width * 2, w.height * 2, w.conf)
        for w in words
    ]
    assert len(group_lines(doubled)) == len(group_lines(words))


def test_cluster_1d_splits_on_gap_and_is_order_stable():
    values = [0.10, 0.12, 0.11, 0.90, 0.91]
    clusters = cluster_1d(values, gap=0.08)
    assert [len(c) for c in clusters] == [3, 2]
    assert cluster_1d(list(reversed(values)), gap=0.08) == clusters


def test_degenerate_inputs_do_not_raise():
    assert load_tsv("") == []
    assert load_tsv("level\tconf\ttext\n1\t-1\t\n") == []
    assert group_lines([]) == []
    assert drop_speckle([]) == []
    assert median_glyph_height([]) == 0.0
    assert text_block([]).width == 1
    assert discover_price_column([], PRICE) is None

"""Choosing between two readings of the same photo, and reading TSV in."""

from __future__ import annotations

from quadra_core.parse import _closer, _read_larger, read_input
from quadra_core.testdata import OCR as FIXTURES


def reading(balanced, delta, printed=100, items=1):
    return {
        "totals": {
            "balanced": balanced,
            "delta_minor": delta,
            "printed_total_minor": printed,
        },
        "line_items": [{}] * items,
    }


def test_a_balanced_reading_beats_an_unbalanced_one():
    assert _closer(reading(True, 0), reading(False, -5))
    assert not _closer(reading(False, -5), reading(True, 0))


def test_the_smaller_delta_wins_when_neither_balances():
    assert _closer(reading(False, -5), reading(False, 20))
    assert not _closer(reading(False, 20), reading(False, -5))


def test_a_reading_that_found_a_total_beats_one_that_did_not():
    with_total = reading(False, None, printed=100)
    without = reading(False, None, printed=None)
    assert _closer(with_total, without)
    assert not _closer(without, with_total)


def test_more_items_win_when_neither_found_a_total():
    more = reading(False, None, printed=None, items=8)
    fewer = reading(False, None, printed=None, items=3)
    assert _closer(more, fewer)
    assert not _closer(fewer, more)


def test_a_photo_whose_text_is_big_enough_is_not_read_again():
    """The enlarged second reading is for small text, and costs another OCR pass."""
    tsv = (FIXTURES / "esselunga_photo.tsv").read_text()
    assert _read_larger(object(), tsv, profile_id="esselunga") is None


def test_reading_a_tsv_file_needs_no_image_support():
    tsv, ocr_meta = read_input(FIXTURES / "esselunga_b.tsv")
    assert tsv.splitlines()[0].startswith("level")
    assert ocr_meta["engine"] == "tesseract"
    assert "_prepared" not in ocr_meta

"""Choosing between readings of the same photo, and the parse_tsv contract."""

from __future__ import annotations

import pytest

from quadra_core import ProfileError, parse_tsv
from quadra_core.parse import _rank, _settled
from quadra_core.testdata import OCR as FIXTURES


def reading(balanced, delta, *, status="review", items=1, reasons=1):
    return {
        "status": status,
        "review": [{}] * reasons,
        "totals": {"balanced": balanced, "delta_minor": delta},
        "line_items": [{}] * items,
    }


def test_a_balanced_reading_beats_an_unbalanced_one():
    assert _rank(reading(True, 0)) > _rank(reading(False, -5))


def test_a_clean_reading_beats_one_that_needs_review():
    ok = reading(True, 0, status="ok", reasons=0)
    assert _rank(ok) > _rank(reading(True, 0))


def test_the_smaller_delta_wins_when_neither_balances():
    assert _rank(reading(False, -5)) > _rank(reading(False, 20))


def test_a_reading_that_found_a_total_beats_one_that_did_not():
    assert _rank(reading(False, 900)) > _rank(reading(False, None, items=30))


def test_more_items_win_when_neither_found_a_total():
    assert _rank(reading(False, None, items=8)) > _rank(reading(False, None, items=3))


def test_a_reading_that_balanced_on_suggestions_is_read_again():
    """Balanced only by unconfirmed prices: another reading might balance cleanly."""
    guessed = reading(True, 0)
    guessed["review"] = [{"field": "price"}]
    assert not _settled(guessed)


def test_a_reading_whose_money_is_certain_is_final():
    """Only a doubtful name is left, and reading again will not fix money."""
    named = reading(True, 0)
    named["review"] = [{"field": "description"}]
    assert _settled(named)


def test_the_result_carries_the_ocr_for_archiving():
    tsv = (FIXTURES / "esselunga_b.tsv").read_text()
    result = parse_tsv(tsv, profile="esselunga")
    assert result.tsv == tsv
    assert result.image is None
    assert result.status == result.document["status"] == "ok"


def test_a_typed_in_total_replaces_the_warning_that_none_was_found():
    tsv = (FIXTURES / "esselunga_b.tsv").read_text()
    document = parse_tsv(tsv, profile="esselunga", total=2644).document
    assert document["totals"]["total_source"] == "supplied"
    assert "printed_total_not_found" not in document["warnings"]


@pytest.mark.parametrize("total", [0, -100])
def test_a_typed_in_total_must_be_positive(total):
    tsv = (FIXTURES / "esselunga_b.tsv").read_text()
    with pytest.raises(ValueError, match="positive"):
        parse_tsv(tsv, profile="esselunga", total=total)


def test_an_unknown_shop_is_a_clear_error():
    tsv = (FIXTURES / "synthetic_clean.tsv").read_text()
    lines = [row for row in tsv.splitlines() if "SUPERMERCATO" not in row]
    with pytest.raises(ProfileError, match="which shop"):
        parse_tsv("\n".join(lines))

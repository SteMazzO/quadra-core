"""End-to-end parsing tests over OCR of a synthetic receipt."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from quadra_core import parse_tsv
from quadra_core.pipeline.extract import extract
from quadra_core.pipeline.lines import drop_speckle, group_lines, load_tsv
from quadra_core.pipeline.review import reasons, status
from quadra_core.pipeline.validate import validate
from quadra_core.profiles import loader
from quadra_core.testdata import OCR as FIXTURES

EXPECTED_TOTALS = [298, 230, 495, 309, 345, 379, 790, 356]
PRINTED_TOTAL = 3202


def profile():
    return next(p for p in loader.available() if p.id == "synthetic")


def run(name: str):
    lines = group_lines(drop_speckle(load_tsv((FIXTURES / f"{name}.tsv").read_text())))
    ex = extract(lines, profile())
    return ex, validate(ex, profile())


def verdict(ex, v) -> str:
    return status(ex, reasons(ex, v, profile()))


def parse(name: str):
    return parse_tsv((FIXTURES / f"{name}.tsv").read_text(), profile="synthetic")


@pytest.mark.parametrize("name", ["synthetic_clean", "synthetic_faded"])
def test_all_line_totals_recovered_regardless_of_image_quality(name):
    """Prices survive degradation that wrecks descriptions."""
    ex, _ = run(name)
    assert [i.line_total_minor for i in ex.items] == EXPECTED_TOTALS


@pytest.mark.parametrize("name", ["synthetic_clean", "synthetic_faded"])
def test_receipt_balances(name):
    _, v = run(name)
    assert v.balanced and v.delta_minor == 0
    assert v.items_subtotal_minor == PRINTED_TOTAL


def test_clean_receipt_parses_descriptions_and_quantities_exactly():
    ex, v = run("synthetic_clean")
    assert verdict(ex, v) == "ok"
    assert [i.description for i in ex.items] == [
        "LATTE INTERO 1L",
        "PANE INTEGRALE",
        "PROSC. CRUDO 100G",
        "MELE GOLDEN KG",
        "YOGURT MAGRO 4X125",
        "CAFFE MACINATO 250G",
        "OLIO EXTRAV. 1L",
        "PASTA PENNE 500G",
    ]
    assert all(i.quantity_source == "parsed" for i in ex.items)


def test_unbalanced_receipt_is_caught_not_silently_accepted():
    ex, v = run("synthetic_unbalanced")
    assert not v.balanced
    assert v.delta_minor == 100
    assert "total_mismatch" in [r.code for r in reasons(ex, v, profile())]


def test_degraded_receipt_goes_to_review_rather_than_lying():
    """It balances, but must not claim every item was read cleanly."""
    ex, v = run("synthetic_faded")
    assert verdict(ex, v) == "review"
    assert any(i.flags for i in ex.items), "no item admitted uncertainty"


def test_uncorroborated_fractional_quantities_are_flagged_for_review():
    """Real and misread fractional quantities look alike, so both are flagged."""
    ex, _ = run("synthetic_faded")
    fractional = [
        i
        for i in ex.items
        if i.quantity is not None and i.quantity != i.quantity.to_integral_value()
    ]
    assert len(fractional) >= 2, "fixture should contain fractional reconstructions"
    assert all("quantity_unverified" in i.flags for i in fractional)


def test_directly_read_fractional_quantity_is_not_flagged():
    """A weighed quantity read straight off the receipt needs no second-guessing."""
    ex, _ = run("synthetic_clean")
    mele = next(i for i in ex.items if "MELE" in i.description)
    assert mele.quantity_source == "parsed"
    assert "quantity_unverified" not in mele.flags


def test_status_distinguishes_ok_from_review():
    ok = parse("synthetic_clean").status
    review = parse("synthetic_unbalanced").status
    assert (ok, review) == ("ok", "review")


def test_profile_fingerprint_matches_from_degraded_ocr():
    lines = [
        l.text
        for l in group_lines(
            drop_speckle(load_tsv((FIXTURES / "synthetic_faded.tsv").read_text()))
        )
    ]
    selected = loader.select(lines)
    assert selected is not None
    assert selected[0].id == "synthetic"


def test_uncalibrated_profile_warns():
    """An uncalibrated profile adds a warning."""
    ex, _ = run("synthetic_clean")
    base = profile()
    scaffold = replace(base, id="scaffold", calibrated=False)
    assert "profile_not_calibrated" in validate(ex, scaffold).warnings
    assert "profile_not_calibrated" not in validate(ex, base).warnings


def test_output_is_json_serialisable_and_uses_minor_units():
    doc = parse("synthetic_clean").document
    json.dumps(doc)  # must not raise
    for item in doc["line_items"]:
        assert isinstance(item["line_total_minor"], int)
        assert isinstance(item["quantity"], str)


def test_the_same_ocr_always_gives_the_same_document():
    """No random ids or timestamps: output depends on the input only."""
    assert parse("synthetic_clean").document == parse("synthetic_clean").document


# --- regressions ------------------------------------------------------------


def _extraction(items_total: int, printed: int, payments: list[int]):
    """One item plus a footer, so only the three totals are under test."""
    tsv = (FIXTURES / "synthetic_clean.tsv").read_text()
    ex = extract(group_lines(drop_speckle(load_tsv(tsv))), profile())
    ex.items = ex.items[:1]
    ex.items[0].line_total_minor = items_total
    ex.items[0].unit_price_minor = items_total
    ex.adjustments = []
    ex.printed_total_minor = printed
    ex.payments_minor = payments
    ex.change_minor = 0
    return ex


def test_a_misread_printed_total_loses_to_items_and_payments_agreeing():
    """Items and payments agreeing on 101,29 beat a total misread as 101,24."""
    ex = _extraction(items_total=10129, printed=10124, payments=[10129])
    result = validate(ex, profile())
    assert result.balanced
    assert result.total_source == "payments"
    assert result.printed_total_minor == 10129
    assert result.total_confirmed
    assert "printed_total_misread:10124" in result.warnings


def test_printed_total_still_wins_when_nothing_corroborates_it():
    """Without corroboration the printed total stands."""
    ex = _extraction(items_total=9000, printed=10124, payments=[10129])
    result = validate(ex, profile())
    assert not result.balanced
    assert result.total_source == "printed"
    assert result.delta_minor == 9000 - 10124
    assert result.total_disputed and not result.total_confirmed


def test_validating_twice_does_not_stack_flags():
    """parse_tsv validates again after a repair, on the same extraction."""
    ex, _ = run("synthetic_clean")
    ex.items[0].quantity = None  # make one check fail
    first = validate(ex, profile())
    second = validate(ex, profile())
    assert ex.items[0].flags.count("quantity_unverified") == 1
    assert first.warnings == second.warnings


def test_a_repaired_item_can_come_back_clean():
    """A fixed item loses its stale flag."""
    ex, _ = run("synthetic_clean")
    broken = ex.items[0].line_total_minor
    ex.items[0].line_total_minor = broken + 500
    validate(ex, profile())
    assert "quantity_unverified" in ex.items[0].flags
    ex.items[0].line_total_minor = broken
    validate(ex, profile())
    assert "quantity_unverified" not in ex.items[0].flags


def test_a_receipt_that_does_not_balance_is_sent_for_review():
    document = parse("synthetic_unbalanced").document
    assert document["status"] == "review"
    [total] = document["review"]
    assert total["code"] == "total_mismatch"
    assert total["message"] == "The lines add up to 32,02, but the total is 31,02."


def test_a_clean_receipt_is_not_sent_for_review():
    document = parse("synthetic_clean").document
    assert document["status"] == "ok"
    assert document["review"] == []
    assert not any(i["needs_review"] for i in document["line_items"])


def test_a_typed_in_total_is_never_overruled_by_the_payment_line():
    """A total typed in with --total is never overruled."""
    ex = _extraction(items_total=10129, printed=9999, payments=[10129])
    ex.printed_total_supplied = True
    result = validate(ex, profile())
    assert result.total_source == "supplied"
    assert result.printed_total_minor == 9999
    assert not any("printed_total_misread" in w for w in result.warnings)

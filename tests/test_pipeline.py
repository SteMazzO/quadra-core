"""End-to-end parsing tests over real Tesseract output.

synthetic_clean and synthetic_faded are the same receipt; the faded one was
captured at 35% contrast with blur and salt-and-pepper noise (mean word
confidence ~55). synthetic_unbalanced prints a total that disagrees with its own
line items by 1,00, to check that the arithmetic actually fails when it should.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from quadra_core import parse_tsv
from quadra_core.pipeline.extract import extract
from quadra_core.pipeline.lines import drop_speckle, group_lines, load_tsv
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


@pytest.mark.parametrize("name", ["synthetic_clean", "synthetic_faded"])
def test_all_line_totals_recovered_regardless_of_image_quality(name):
    """The headline claim: prices survive degradation that wrecks descriptions."""
    ex, _ = run(name)
    assert [i.line_total_minor for i in ex.items] == EXPECTED_TOTALS


@pytest.mark.parametrize("name", ["synthetic_clean", "synthetic_faded"])
def test_receipt_balances(name):
    _, v = run(name)
    assert v.balanced and v.delta_minor == 0
    assert v.items_subtotal_minor == PRINTED_TOTAL


def test_clean_receipt_parses_descriptions_and_quantities_exactly():
    ex, v = run("synthetic_clean")
    assert v.status == "ok"
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
    _, v = run("synthetic_unbalanced")
    assert not v.balanced
    assert v.delta_minor == 100
    assert any(c.name == "receipt_total" and not c.passed for c in v.checks)


def test_degraded_receipt_degrades_to_partial_rather_than_lying():
    """It balances, but must not claim every item was read cleanly."""
    ex, v = run("synthetic_faded")
    assert v.status == "partial"
    assert any(i.flags for i in ex.items), "no item admitted uncertainty"


def test_uncorroborated_fractional_quantities_are_flagged_for_review():
    """Arithmetic cannot judge whether a fractional quantity is real.

    The faded fixture contains one genuine weighed item (MELE GOLDEN KG, 1.241) and
    one misread (YOGURT, 0.111, because '1,15' was read as '31,15'). Both are
    non-integer reconstructions and nothing in the numbers distinguishes them, so the
    parser must flag both rather than pretend to tell them apart.
    """
    ex, _ = run("synthetic_faded")
    fractional = [
        i
        for i in ex.items
        if i.quantity is not None and i.quantity != i.quantity.to_integral_value()
    ]
    assert len(fractional) >= 2, "fixture should contain fractional reconstructions"
    assert all("quantity_verified" in i.flags for i in fractional)


def test_directly_read_fractional_quantity_is_not_flagged():
    """A weighed quantity read straight off the receipt needs no second-guessing."""
    ex, _ = run("synthetic_clean")
    mele = next(i for i in ex.items if "MELE" in i.description)
    assert mele.quantity_source == "parsed"
    assert "quantity_verified" not in mele.flags


def test_exit_codes_distinguish_ok_from_partial():
    _, ok = parse_tsv(
        (FIXTURES / "synthetic_clean.tsv").read_text(),
        profile_id="synthetic",
    )
    _, partial = parse_tsv(
        (FIXTURES / "synthetic_unbalanced.tsv").read_text(),
        profile_id="synthetic",
    )
    assert (ok, partial) == ("ok", "partial")


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
    """A scaffold profile must not produce confident output.

    Built in-memory rather than from disk: the shipped Esselunga profile is now
    calibrated from real receipts, so the warning path needs its own fixture.
    """
    ex, _ = run("synthetic_clean")
    base = profile()
    scaffold = replace(base, id="scaffold", calibrated=False)
    assert "profile_not_calibrated" in validate(ex, scaffold).warnings
    assert "profile_not_calibrated" not in validate(ex, base).warnings


def test_output_is_json_serialisable_and_uses_minor_units():
    doc, _ = parse_tsv(
        (FIXTURES / "synthetic_clean.tsv").read_text(),
        profile_id="synthetic",
        receipt_id="x",
    )
    json.dumps(doc)  # must not raise
    for item in doc["line_items"]:
        assert isinstance(item["line_total_minor"], int)
        # Quantity is a string, so fractional weights survive JSON round-tripping.
        assert isinstance(item["quantity"]["value"], str)


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
    """The real case: OCR read 'TOTALE EURO 101,29' as 101,24 while the items and
    the payment line both came to 101,29."""
    ex = _extraction(items_total=10129, printed=10124, payments=[10129])
    result = validate(ex, profile())
    assert result.balanced
    assert result.total_source == "payments"
    assert result.printed_total_minor == 10129
    assert "printed_total_misread:10124!=10129" in result.warnings


def test_printed_total_still_wins_when_nothing_corroborates_it():
    """Only two figures, and they disagree: no grounds to overrule the paper."""
    ex = _extraction(items_total=9000, printed=10124, payments=[10129])
    result = validate(ex, profile())
    assert not result.balanced
    assert result.total_source == "printed"
    assert result.delta_minor == 9000 - 10124


def test_validating_twice_does_not_stack_flags():
    """parse_tsv validates again after a repair, on the same extraction."""
    ex, _ = run("synthetic_clean")
    ex.items[0].quantity = None  # make one check fail
    first = validate(ex, profile())
    second = validate(ex, profile())
    assert ex.items[0].flags.count("line_arithmetic") == 1
    assert first.warnings == second.warnings


def test_a_repaired_item_can_come_back_clean():
    """A stale flag from the first pass used to keep a fixed receipt dirty."""
    ex, _ = run("synthetic_clean")
    broken = ex.items[0].line_total_minor
    ex.items[0].line_total_minor = broken + 500
    validate(ex, profile())
    assert "line_arithmetic" in ex.items[0].flags
    ex.items[0].line_total_minor = broken
    validate(ex, profile())
    assert "line_arithmetic" not in ex.items[0].flags


def test_a_receipt_that_does_not_balance_is_sent_for_review():
    """Filing an unbalanced receipt unseen is the one thing review is for."""
    document, status = parse_tsv(
        (FIXTURES / "synthetic_unbalanced.tsv").read_text(), profile_id="synthetic"
    )
    assert status == "partial"
    assert document["review"]["required"] is True
    assert "totals_do_not_balance" in document["review"]["reason"]


def test_a_clean_receipt_is_not_sent_for_review():
    document, status = parse_tsv(
        (FIXTURES / "synthetic_clean.tsv").read_text(), profile_id="synthetic"
    )
    assert status == "ok"
    assert document["review"]["required"] is False


def test_the_caller_keeps_the_last_word_on_review():
    mine = {"required": False, "reason": "sampled elsewhere", "sampled": True}
    document, _ = parse_tsv(
        (FIXTURES / "synthetic_unbalanced.tsv").read_text(),
        profile_id="synthetic",
        review=mine,
    )
    assert document["review"]["required"] is False
    assert document["review"]["reason"] == "sampled elsewhere"


def test_a_typed_in_total_is_never_overruled_by_the_payment_line():
    """--total is for a person reading the paper when OCR could not. Nothing
    downstream gets to second-guess that, corroboration included."""
    ex = _extraction(items_total=10129, printed=9999, payments=[10129])
    ex.printed_total_supplied = True
    result = validate(ex, profile())
    assert result.total_source == "printed"
    assert result.printed_total_minor == 9999
    assert not any("printed_total_misread" in w for w in result.warnings)

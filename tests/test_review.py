"""The rules for review: 'ok' must mean every figure is backed by the arithmetic."""

from __future__ import annotations

import pytest

from quadra_core.pipeline import recheck
from quadra_core.pipeline.document import build
from quadra_core.pipeline.extract import extract
from quadra_core.pipeline.lines import Word, group_lines
from quadra_core.pipeline.review import reasons, status
from quadra_core.pipeline.validate import validate
from quadra_core.profiles import loader

GLYPH = 24
ROW = 40


def esselunga():
    return next(p for p in loader.available() if p.id == "esselunga")


def receipt(*rows: str):
    """Lay out receipt rows in fixed columns: name, VAT bracket, price.

    A row is 'NAME | *c | 1,29'; a single cell sits at the left margin. A row
    starting with '^' is printed double height, like the total.
    """
    words = []
    for number, row in enumerate(rows):
        height = GLYPH * 2 if row.startswith("^") else GLYPH
        cells = [c.strip() for c in row.lstrip("^").split("|")]
        starts = [40] if len(cells) == 1 else [40, 600, 760][-len(cells) :]
        if len(cells) == 2:
            starts = [40, 760]
        for cell, left in zip(cells, starts, strict=True):
            x = left
            for text in cell.split():
                width = 14 * len(text)
                words.append(Word(text, x, number * ROW, width, height, 95.0))
                x += width + 14
    return group_lines(words)


HEADER = ("* Esselunga S.p.A. *", "P.I. 04916380159", "IVA | EURO")


def parsed(*rows: str):
    lines = receipt(*HEADER, *rows)
    ex = extract(lines, esselunga())
    return lines, ex, validate(ex, esselunga())


def codes(ex, v):
    return [r.code for r in reasons(ex, v, esselunga())]


# --- what the first reading gets right on its own ---------------------------


def test_a_discount_that_lost_its_minus_sign_still_takes_money_off():
    _, ex, v = parsed(
        "PESTO GENOVESE | *c | 2,59",
        "SCONTO FIDATY ORO | 0,26",
        "TOTALE EURO | 2,33",
    )
    assert [a.amount_minor for a in ex.adjustments] == [-26]
    assert v.balanced


def test_the_vat_summary_is_not_an_item_when_the_total_is_unreadable():
    _, ex, _ = parsed(
        "SFILATINO MORBIDO | *a | 1,29",
        "di cui IVA | 0,05",
        "*a = IVA 04% | 1,29",
    )
    assert [i.description for i in ex.items] == ["SFILATINO MORBIDO"]


def test_a_total_too_mangled_for_any_rule_is_still_not_an_item():
    """'TOTALE EURO' read as 'TIME XURO' is still printed double height."""
    _, ex, _ = parsed("SFILATINO MORBIDO | *a | 1,29", "^TIME XURO | 1,29")
    assert [i.description for i in ex.items] == ["SFILATINO MORBIDO"]


def test_a_discount_belongs_to_the_item_printed_above_it():
    lines, ex, v = parsed(
        "LATTE | *a | 1,00",
        "PESTO GENOVESE | *c | 2,59",
        "SCONTO FIDATY ORO | 0,26-S",
        "TOTALE EURO | 3,33",
    )
    document = build(
        lines=lines, extraction=ex, validation=v, profile=esselunga(), ocr_meta={}
    )
    assert document["adjustments"][0]["applies_to"] == 1
    assert [i["discount_minor"] for i in document["line_items"]] == [0, -26]


# --- totals ------------------------------------------------------------------


def test_a_receipt_that_adds_up_needs_no_review():
    _, ex, v = parsed("LATTE | *a | 1,00", "PANE | *a | 2,00", "TOTALE EURO | 3,00")
    assert reasons(ex, v, esselunga()) == []
    assert status(ex, []) == "ok"


def test_a_missing_total_is_reviewed():
    _, ex, v = parsed("LATTE | *a | 1,00")
    assert codes(ex, v) == ["total_missing"]


def test_the_lines_without_a_price_are_named_when_money_is_missing():
    lines, ex, v = parsed(
        "LATTE | *a | 1,00",
        "PANE | *a | 1,00",
        "NESCAFE' DG ROMA | *d | xx",
        "TOTALE EURO | 6,99",
    )
    found = reasons(ex, v, esselunga(), lines)
    assert [r.code for r in found] == ["total_mismatch", "price_missing"]
    assert found[1].message == "No price read for “NESCAFE' DG ROMA”."


def test_a_line_without_a_price_is_not_reviewed_when_everything_adds_up():
    _, ex, v = parsed("LATTE | *a | 1,00", "OFFERTA SPECIALE", "TOTALE EURO | 1,00")
    assert codes(ex, v) == []


def test_one_line_costing_more_than_the_receipt_is_reviewed():
    _, ex, v = parsed("BIRRA PERONI | *d | 80,99", "TOTALE EURO | 0,99")
    assert "price_implausible" in codes(ex, v)


# --- the re-read: the arithmetic proposes, the review confirms ---------------


def rereads(monkeypatch, readings):
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: readings)


def repaired(monkeypatch, readings, *rows):
    lines, ex, v = parsed(*rows)
    rereads(monkeypatch, readings)
    result = recheck.repair(ex, v, object(), esselunga(), lines)
    if result is not None:
        recheck.apply(ex, result, lines, esselunga())
    return ex, validate(ex, esselunga())


CUKI = ("CUKI | *d | 1,95", "DISARONNO | *d | 9,59")


def test_a_correction_every_reading_agrees_on_is_certain(monkeypatch):
    """Total confirmed by the payment, and all three re-reads say 1,95."""
    ex, v = repaired(
        monkeypatch,
        {3: [195, 195, 195]},
        "CUKI | *d | 1,98",
        "DISARONNO | *d | 9,59",
        "TOTALE EURO | 11,54",
        "PAGAMENTO CARTA | 11,54",
    )
    assert ex.items[0].line_total_minor == 195
    assert "price_reread" in ex.items[0].flags
    assert codes(ex, v) == []


def test_a_correction_the_readings_split_on_is_only_a_suggestion(monkeypatch):
    ex, v = repaired(
        monkeypatch,
        {3: [195, 198, 195]},
        "CUKI | *d | 1,98",
        "DISARONNO | *d | 9,59",
        "TOTALE EURO | 11,54",
        "PAGAMENTO CARTA | 11,54",
    )
    assert ex.items[0].line_total_minor == 195
    found = reasons(ex, v, esselunga())
    assert [(r.code, r.item) for r in found] == [("price_unconfirmed", 0)]
    assert found[0].message == "Read as 1,98; 1,95 would make the receipt add up."


def test_a_total_read_once_is_reviewed_with_what_was_bent_to_it(monkeypatch):
    """No payment line to confirm 11,54, so neither it nor the fix is trusted."""
    ex, v = repaired(
        monkeypatch,
        {3: [195, 195, 195]},
        "CUKI | *d | 1,98",
        "DISARONNO | *d | 9,59",
        "TOTALE EURO | 11,54",
    )
    assert codes(ex, v) == ["total_unconfirmed", "price_unconfirmed"]


def test_prices_are_never_bent_towards_a_disputed_total(monkeypatch):
    """The total reads 20,68 and the payment 20,65: aim at neither."""
    ex, v = repaired(
        monkeypatch,
        {3: [198, 198, 198]},
        "CUKI | *d | 1,95",
        "DISARONNO | *d | 9,59",
        "TOTALE EURO | 11,57",
        "PAGAMENTO CARTA | 11,56",
    )
    assert [i.line_total_minor for i in ex.items] == [195, 959]
    assert codes(ex, v) == ["total_disputed"]


def test_every_item_named_in_the_review_is_marked_for_it(monkeypatch):
    lines, ex, v = parsed("CUKI | *d | 1,95", "X9 | *d | 80,99", "TOTALE EURO | 1,95")
    document = build(
        lines=lines, extraction=ex, validation=v, profile=esselunga(), ocr_meta={}
    )
    named = {r["item"] for r in document["review"] if r["item"] is not None}
    marked = {i["index"] for i in document["line_items"] if i["needs_review"]}
    assert named == marked == {1}
    assert document["status"] == "review"


# --- reading the price column on its own -------------------------------------


@pytest.mark.parametrize(
    "text,cents",
    [
        ("2,96", 296),
        ("»,98", 198),  # the thin leading 1 lost
        ("ì,75", 175),
        ("1}20", 120),  # a mangled separator
        ("449", 449),  # the comma lost
        (".499", 499),
        ("120=S", -120),  # the minus read as '='
        ("1,20-S", -120),
    ],
)
def test_the_column_reread_reads_through_thermal_print_misreads(text, cents):
    assert recheck.read_cell(text, esselunga()) == cents


@pytest.mark.parametrize("text", ["1449", "0,3546", "bi", "1,865", "3,7"])
def test_the_column_reread_refuses_what_it_cannot_tell(text):
    assert recheck.read_cell(text, esselunga()) is None

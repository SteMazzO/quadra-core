"""Reading the price column a second time when a receipt does not add up.

The risk is not that the repair fails, but that it succeeds wrongly and writes a
number nobody checked. Most of these tests cover when it has to refuse.
"""

from __future__ import annotations

from decimal import Decimal

from quadra_core.pipeline import recheck
from quadra_core.pipeline.extract import Extraction, LineItem
from quadra_core.pipeline.lines import Line, Word
from quadra_core.pipeline.validate import Validation


def item(index: int, total: int, quantity_source: str = "implicit") -> LineItem:
    return LineItem(
        description_raw=f"THING {index}",
        description=f"THING {index}",
        quantity=Decimal(1),
        quantity_source=quantity_source,
        unit_price_minor=total,
        line_total_minor=total,
        vat_code=None,
        confidence=0.9,
        line_index=index,
        bbox=(0, index * 10, 100, 10),
    )


def extraction(totals: list[int]) -> Extraction:
    return Extraction(
        items=[item(i, t) for i, t in enumerate(totals)],
        adjustments=[],
        payments_minor=[],
        change_minor=0,
        printed_total_minor=None,
        item_region=(0, len(totals)),
        price_column=None,
    )


def validation(printed: int | None, delta: int) -> Validation:
    return Validation(status="partial", printed_total_minor=printed, delta_minor=delta)


def repair_with(totals, printed, second, monkeypatch):
    """Run repair with a stubbed second reading, so no image or Tesseract is needed."""
    ex = extraction(totals)
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: second)
    val = validation(printed, sum(totals) - printed)
    return ex, recheck.repair(ex, val, object(), object(), [])


# --- when it should act -----------------------------------------------------


def test_a_single_balancing_reading_is_taken(monkeypatch):
    """The real case: one price read 70 euro high, the column read it correctly."""
    ex, got = repair_with([7239, 299, 239], 777, {0: 239}, monkeypatch)
    assert got is not None
    assert got.changes == {0: (7239, 239)}

    recheck.apply(ex, got, [])
    assert ex.items[0].line_total_minor == 239
    assert "price_reread" in ex.items[0].flags


def test_a_repaired_line_carries_its_unit_price_with_it(monkeypatch):
    """Moving the total alone leaves the per-line arithmetic failing by the fix."""
    ex, got = repair_with([7239, 299, 239], 777, {0: 239}, monkeypatch)
    recheck.apply(ex, got, [])
    assert ex.items[0].unit_price_minor == 239


def test_a_quantity_from_a_modifier_keeps_its_unit_price(monkeypatch):
    """There the unit price was read off the receipt, so it is not ours to rewrite."""
    ex = extraction([7239])
    ex.items[0].quantity_source = "modifier"
    ex.items[0].unit_price_minor = 100
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: {0: 239})
    got = recheck.repair(ex, validation(239, 7000), object(), object(), [])
    recheck.apply(ex, got, [])
    assert ex.items[0].unit_price_minor == 100


# --- when it must refuse ----------------------------------------------------


def test_nothing_happens_when_two_readings_both_balance(monkeypatch):
    """Two items, each misread by the same amount in opposite directions.

    Swapping either one balances the receipt, so the arithmetic cannot say which
    was wrong. Picking one would be a guess with a proof's confidence.
    """
    _ex, got = repair_with([100, 300], 300, {0: 100 - 100, 1: 300 - 100}, monkeypatch)
    assert got is None


def test_a_receipt_that_already_adds_up_is_left_alone(monkeypatch):
    calls = []

    def spy(*_args, **_kwargs):
        calls.append(1)
        return {}

    monkeypatch.setattr(recheck, "second_opinion", spy)
    ex = extraction([100, 200])
    assert recheck.repair(ex, validation(300, 0), object(), object(), []) is None
    assert not calls, "a balanced receipt must not pay for a second reading"


def test_nothing_happens_without_a_total_to_balance_against(monkeypatch):
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: {0: 1})
    ex = extraction([100])
    assert recheck.repair(ex, validation(None, 5), object(), object(), []) is None


def test_nothing_happens_when_no_reading_balances(monkeypatch):
    _ex, got = repair_with([100, 200], 999, {0: 150}, monkeypatch)
    assert got is None


def test_agreement_is_not_a_change(monkeypatch):
    """The second reading agreeing with the first is not a repair."""
    _ex, got = repair_with([100, 200], 300, {0: 100, 1: 200}, monkeypatch)
    assert got is None


def test_too_many_disagreements_is_a_refusal_not_a_search(monkeypatch):
    """Past the cap the receipt is too badly read for arithmetic to single one out."""
    n = recheck.MAX_DISPUTED + 1
    totals = [100] * n
    second = dict.fromkeys(range(n), 200)
    _ex, got = repair_with(totals, 100 * n, second, monkeypatch)
    assert got is None


def test_a_recovered_line_does_not_shift_the_other_corrections():
    """apply() inserts recovered items and re-sorts, which used to invalidate the
    positional keys and write a correction onto the wrong item."""
    ex = extraction([100, 250])
    ex.items[0].line_index, ex.items[1].line_index = 5, 9
    ex.skipped_lines = [7]
    lines = [
        Line(words=[Word("APPLES", 0, 50, 60, 10, 90.0)], index=5),
        Line(words=[Word("MILK", 0, 70, 40, 10, 90.0)], index=7),
        Line(words=[Word("BREAD", 0, 90, 50, 10, 90.0)], index=9),
    ]
    result = recheck.Repair(changes={9: (250, 999)}, recovered={7: 50}, disputed=2)
    recheck.apply(ex, result, lines)

    by_line = {i.line_index: i.line_total_minor for i in ex.items}
    assert by_line == {5: 100, 7: 50, 9: 999}

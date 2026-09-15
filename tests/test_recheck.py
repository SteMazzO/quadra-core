"""Price-column re-read tests. Most cover when the repair must refuse."""

from __future__ import annotations

from decimal import Decimal

from quadra_core.pipeline import recheck
from quadra_core.pipeline.extract import Extraction, LineItem
from quadra_core.pipeline.lines import Line, Word
from quadra_core.pipeline.validate import Validation
from quadra_core.profiles import loader


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
    """Run repair with a stubbed second reading."""
    ex = extraction(totals)
    readings = {index: [amount] for index, amount in second.items()}
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: readings)
    val = validation(printed, sum(totals) - printed)
    return ex, recheck.repair(ex, val, object(), object(), [])


# --- when it should act -----------------------------------------------------


def test_a_single_balancing_reading_is_taken(monkeypatch):
    """A price read 70 euro high is corrected from the column."""
    ex, got = repair_with([7239, 299, 239], 777, {0: 239}, monkeypatch)
    assert got is not None
    assert got.changes == {0: (7239, 239)}

    recheck.apply(ex, got, [])
    assert ex.items[0].line_total_minor == 239
    assert "price_reread" in ex.items[0].flags


def test_a_repaired_line_carries_its_unit_price_with_it(monkeypatch):
    """The unit price moves with the corrected total."""
    ex, got = repair_with([7239, 299, 239], 777, {0: 239}, monkeypatch)
    recheck.apply(ex, got, [])
    assert ex.items[0].unit_price_minor == 239


def test_a_quantity_from_a_modifier_keeps_its_unit_price(monkeypatch):
    """A unit price read from a modifier line is left alone."""
    ex = extraction([7239])
    ex.items[0].quantity_source = "modifier"
    ex.items[0].unit_price_minor = 100
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: {0: [239]})
    got = recheck.repair(ex, validation(239, 7000), object(), object(), [])
    recheck.apply(ex, got, [])
    assert ex.items[0].unit_price_minor == 100


# --- when it must refuse ----------------------------------------------------


def test_nothing_happens_when_two_readings_both_balance(monkeypatch):
    """Either correction balances, so neither is made."""
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
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: {0: [1]})
    ex = extraction([100])
    assert recheck.repair(ex, validation(None, 5), object(), object(), []) is None


def test_nothing_happens_when_no_reading_balances(monkeypatch):
    _ex, got = repair_with([100, 200], 999, {0: 150}, monkeypatch)
    assert got is None


def test_agreement_is_not_a_change(monkeypatch):
    _ex, got = repair_with([100, 200], 300, {0: 100, 1: 200}, monkeypatch)
    assert got is None


def test_too_many_disagreements_is_a_refusal_not_a_search(monkeypatch):
    n = recheck.MAX_DISPUTED + 1
    totals = [100] * n
    second = dict.fromkeys(range(n), 200)
    _ex, got = repair_with(totals, 100 * n, second, monkeypatch)
    assert got is None


def test_a_recovered_line_does_not_shift_the_other_corrections():
    """Recovered lines don't shift corrections onto the wrong item."""
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


def esselunga():
    return next(p for p in loader.available() if p.id == "esselunga")


# --- several readings -------------------------------------------------------


def test_the_answer_most_readings_agree_on_wins_when_two_balance(monkeypatch):
    """When two answers balance, the one more readings back wins."""
    ex = extraction([120, 300])
    monkeypatch.setattr(
        recheck, "second_opinion", lambda *a, **k: {0: [20, 20], 1: [200]}
    )
    got = recheck.repair(ex, validation(320, 100), object(), object(), [])
    assert got is not None
    assert got.changes == {0: (120, 20)}


def test_votes_rank_answers_that_balance_but_never_make_one_balance(monkeypatch):
    """An outvoted reading is taken if it is the only one that balances."""
    ex = extraction([7239, 299])
    monkeypatch.setattr(
        recheck, "second_opinion", lambda *a, **k: {0: [239, 7239, 7239]}
    )
    got = recheck.repair(ex, validation(538, 7000), object(), object(), [])
    assert got is not None
    assert got.changes == {0: (7239, 239)}


def test_a_dropped_discount_comes_back_as_a_discount(monkeypatch):
    """A dropped discount comes back negative, even if a reading lost the sign."""
    ex = extraction([169])
    ex.skipped_lines = [1]
    lines = [
        Line(
            words=[
                Word("YOGA", 0, 0, 40, 10, 90.0),
                Word("1,69", 200, 0, 40, 10, 90.0),
            ],
            index=0,
        ),
        Line(
            words=[
                Word("SCONTO", 20, 20, 60, 10, 90.0),
                Word("30%", 90, 20, 30, 10, 90.0),
                Word("sai", 205, 16, 30, 25, 20.0),
            ],
            index=1,
        ),
    ]
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: {1: [51, -51, -51]})
    got = recheck.repair(ex, validation(118, 51), object(), esselunga(), lines)
    assert got is not None
    assert got.recovered == {1: -51}

    recheck.apply(ex, got, lines, esselunga())
    assert [(a.label, a.amount_minor) for a in ex.adjustments] == [("SCONTO 30%", -51)]
    assert ex.skipped_lines == [], "a recovered line still reported as priceless"


def test_a_recovered_line_leaves_the_failed_price_reading_out_of_its_name():
    """A recovered line's name leaves out the garbled price and the VAT bracket."""
    ex = extraction([])
    lines = [
        Line(
            words=[
                Word("YOGA", 0, 0, 40, 10, 93.0),
                Word("NETT.ALBICOCCA1L", 50, 0, 120, 10, 92.0),
                Word("*d", 180, 0, 15, 10, 32.0),
                Word("Di,", 210, -5, 30, 25, 45.0),
            ],
            index=4,
        )
    ]
    result = recheck.Repair(recovered={4: 169}, disputed=1, column_left=200)
    recheck.apply(ex, result, lines, esselunga())

    [got] = ex.items
    assert got.description == "YOGA NETT.ALBICOCCA1L"
    assert got.vat_code == "d"
    assert got.line_total_minor == 169
    assert "line_recovered" in got.flags


def test_a_row_between_two_lines_is_given_to_one_of_them_not_both():
    """A strip row goes to one line only."""
    assert recheck._assign([(15.0, "1,69")], {0: 10.0, 1: 22.0}, 10) == {0: "1,69"}
    # And a line offered two rows keeps the nearer.
    assert recheck._assign([(10.0, "a"), (13.0, "b")], {0: 10.0}, 10) == {0: "a"}


def test_the_total_may_not_overrule_the_readings_on_more_than_one_line(monkeypatch):
    """Only overruling two clear majorities would balance, so nothing changes."""
    ex = extraction([100, 200, 300])
    second = {0: [150, 150], 1: [250, 250], 2: [310, 310]}
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: second)
    assert recheck.repair(ex, validation(610, -10), object(), object(), []) is None


def test_a_stray_description_word_does_not_widen_the_column():
    """One stray word far to the left doesn't widen the price column."""
    ex = extraction([100, 200, 300, 400])
    lines = [
        Line(words=[Word("A", 0, 0, 20, 10, 90.0), Word("1,00", 800, 0, 40, 10, 90.0)]),
        Line(
            words=[Word("B", 0, 20, 20, 10, 90.0), Word("2,00", 805, 20, 40, 10, 90.0)]
        ),
        Line(
            words=[Word("C", 0, 40, 20, 10, 90.0), Word("3,00", 798, 40, 40, 10, 90.0)]
        ),
        Line(words=[Word("D", 102, 60, 20, 10, 90.0), Word("4", 810, 55, 30, 40, 9.0)]),
    ]
    for index, line in enumerate(lines):
        line.index = index
    assert recheck._column_left(lines, ex) == 798


def test_a_sign_in_the_column_marks_a_discount_the_label_hid(monkeypatch):
    """A minus sign in a reading marks a discount whose label OCR garbled."""
    ex = extraction([169])
    ex.skipped_lines = [1]
    lines = [
        Line(words=[Word("YOGA", 0, 0, 40, 10, 90.0)], index=0),
        Line(words=[Word("SCINTO", 20, 20, 60, 10, 90.0)], index=1),
    ]
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: {1: [51, -51]})
    got = recheck.repair(ex, validation(118, 51), object(), esselunga(), lines)
    assert got is not None
    assert got.recovered == {1: -51}


def test_a_nameless_price_repeating_its_neighbour_is_dropped(monkeypatch):
    """A nameless line repeating its neighbour's price is dropped."""
    ex = extraction([596])
    ex.items[0].description = ex.items[0].description_raw = "="
    ex.items[0].line_index = 1
    ex.skipped_lines = [0, 2]
    lines = [
        Line(words=[Word("PINSA", 0, 0, 50, 10, 90.0)], index=0),
        Line(
            words=[Word("=", 0, 12, 10, 10, 40.0), Word("5.96", 200, 12, 40, 10, 80.0)],
            index=1,
        ),
        Line(words=[Word("STAR", 0, 20, 40, 10, 90.0)], index=2),
    ]
    second = {0: [596, 596], 2: [233, 233]}
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: second)
    got = recheck.repair(ex, validation(829, -233), object(), esselunga(), lines)
    assert got is not None
    assert got.recovered == {0: 596, 2: 233}
    assert got.changes == {1: (596, 0)}

    recheck.apply(ex, got, lines, esselunga())
    assert [(i.description, i.line_total_minor) for i in ex.items] == [
        ("PINSA", 596),
        ("STAR", 233),
    ]

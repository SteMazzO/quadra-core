"""Money and quantity parsing tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quadra_core.pipeline.money import (
    digit_ratio,
    format_minor,
    parse_money,
    parse_quantity,
    reconstruct_quantity,
    repair_numeric,
)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("1,49", 149),
        ("31,02", 3102),
        ("0,89", 89),
        ("5", 500),
        (",50", 50),
        ("l,49", 149),  # l -> 1
        ("2,3O", 230),  # O -> 0
        ("O,99", 99),
        ("7.90", 790),  # '.' as decimal point despite a ',' profile
        ("1.234,50", 123450),
        ("1,234.50", 123450),
        ("abc", None),
        ("", None),
        ("--", None),
    ],
)
def test_parse_money(token, expected):
    assert parse_money(token) == expected


def test_confusion_repair_is_gated_on_looking_numeric():
    """Ungated, the confusion map turns 'abc' into money.

    'b' maps to '6' and 'c' is stripped, yielding 600 minor units from a pure word.
    The digit-ratio gate is what prevents descriptions being read as prices.
    """
    assert repair_numeric("abc") == "a6c"  # the map itself is indiscriminate
    assert digit_ratio("abc") == 0.0
    assert parse_money("abc") is None  # the gate stops it
    assert parse_money("2,3O") == 230  # but real numerics still get repaired


def test_parse_money_never_raises_on_hostile_input():
    for token in ("...", ",,,", "1,2,3", "-", "\x00", "9" * 400, ",", "."):
        assert parse_money(token) is None or isinstance(parse_money(token), int)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("2", Decimal(2)),
        ("1,240", Decimal("1.240")),
        ("1L", None),
        ("KG", None),
        ("", None),
    ],
)
def test_parse_quantity_rejects_descriptive_tokens(token, expected):
    """'1L' is part of a product name, not a quantity: digit ratio 0.5 rejects it."""
    assert parse_quantity(token) == expected


def test_reconstruct_quantity_from_prices():
    assert reconstruct_quantity(298, 149) == Decimal("2.000")
    assert reconstruct_quantity(356, 89) == Decimal("4.000")
    assert reconstruct_quantity(0, 149) is None
    assert reconstruct_quantity(298, 0) is None


def test_format_minor_round_trips():
    for minor in (0, 5, 99, 149, 3102, 123450):
        assert parse_money(format_minor(minor)) == minor
    assert format_minor(-50) == "-0,50"

"""How amounts are read, and how a receipt's number format is worked out."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quadra_core.pipeline.money import (
    Format,
    detect_format,
    format_minor,
    parse_amount,
    parse_quantity,
)

COMMA = Format(",", 2)
POINT = Format(".", 2)
WHOLE = Format(".", 0)


@pytest.mark.parametrize(
    "text,fmt,expected",
    [
        ("0,99", COMMA, 99),
        ("1.234,56", COMMA, 123456),
        ("1,234.56", POINT, 123456),
        ("7,20-S", COMMA, -720),
        ("0,74-8", COMMA, -74),
        ("-9,545", WHOLE, -9545),
        ("(5.00)", POINT, -500),
        ("(2,33", COMMA, 233),
        ("1;;19", COMMA, 119),
        ("2/76", COMMA, 276),
        ("2,49.", COMMA, 249),
        ("$3.99", POINT, 399),
        ("Rp58000.00", POINT, 5800000),
        ("2,99A", COMMA, 299),
        ("63,636", WHOLE, 63636),
        ("23.000", WHOLE, 23000),
        ("750", WHOLE, 750),
    ],
)
def test_amounts_are_read_in_the_receipts_own_format(text, fmt, expected):
    assert parse_amount(text, fmt) == expected


@pytest.mark.parametrize(
    "text,fmt",
    [
        ("1,5", COMMA),  # a size, as in '1,5 KG'
        ("0.000", COMMA),  # loyalty points
        ("10%", WHOLE),
        ("500G", WHOLE),
        ("BER2", WHOLE),
        ("4X125", COMMA),
        ("04916380159", COMMA),  # a VAT number
        ("5272.73", WHOLE),
        ("TOTALE", COMMA),
    ],
)
def test_what_is_not_money_is_refused(text, fmt):
    assert parse_amount(text, fmt) is None


@pytest.mark.parametrize(
    "words,expected",
    [
        ("0,99 0,61 5,25 0.000 0.000", COMMA),
        ("3.99 12.50 1,234.56", POINT),
        ("63,636 -9,545 5,409 4", Format(",", 0)),
        ("23.000 50.000 27.000", Format(".", 0)),
        ("58000.00 5272.73 *2", POINT),
        ("1.250 0.500 2.750", Format(".", 3)),
        ("1200 480 10%", Format(".", 0)),
    ],
)
def test_the_format_is_worked_out_from_the_numbers(words, expected):
    found = detect_format(words.split())
    assert found.places == expected.places
    if expected.places == 2:
        assert found.separator == expected.separator


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2", "2"),
        ("x2", "2"),
        ("3X", "3"),
        ("*2", "2"),
        ("1,240", "1.240"),
        ("0.456", "0.456"),
    ],
)
def test_quantities(text, expected):
    assert parse_quantity(text) == Decimal(expected)


@pytest.mark.parametrize("text", ["", "0", "KG", "12345", "1,5,2"])
def test_what_is_not_a_quantity(text):
    assert parse_quantity(text) is None


def test_money_is_written_the_way_the_receipt_prints_it():
    assert format_minor(10125, COMMA) == "101,25"
    assert format_minor(-5, POINT) == "-0.05"
    assert format_minor(23000, WHOLE) == "23000"

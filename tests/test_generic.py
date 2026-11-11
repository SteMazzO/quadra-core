"""Receipts from no particular shop: layouts, labels and sums the parser must take."""

from __future__ import annotations

import pytest

from quadra_core import parse_tsv
from quadra_core.pipeline.lexicon import is_heading, role

HEADER = "\t".join(
    [
        *("level", "page_num", "block_num", "par_num", "line_num", "word_num"),
        *("left", "top", "width", "height", "conf", "text"),
    ]
)


def receipt(*rows: str) -> dict:
    """Parse a receipt typed as 'left words | right-aligned amount' rows."""
    out = [HEADER]
    for number, row in enumerate(rows):
        left, _, right = row.partition("|")
        top = 40 * number
        x = 20
        for word in left.split():
            out.append(
                f"5\t1\t1\t1\t1\t1\t{x}\t{top}\t{14 * len(word)}\t24\t95\t{word}"
            )
            x += 14 * len(word) + 14
        right = right.strip()
        if right:
            width = 14 * len(right)
            out.append(
                f"5\t1\t1\t1\t1\t1\t{800 - width}\t{top}\t{width}\t24\t95\t{right}"
            )
    return parse_tsv("\n".join(out) + "\n").document


def items(document: dict) -> list[tuple[str, str, int]]:
    return [
        (i["description"], i["quantity"], i["line_total_minor"])
        for i in document["line_items"]
    ]


def test_tax_added_on_top_is_counted():
    document = receipt(
        "1 MINERAL 600 ML | 7,727",
        "1 BULGOGI RICE | 33,636",
        "Subtotal | 41,364",
        "Tax | 4,136",
        "Total | 45,500",
        "CASH | 50,000",
        "CHANGE | -4,500",
    )
    assert document["status"] == "ok"
    assert document["decimal_places"] == 0
    assert items(document) == [
        ("MINERAL 600 ML", "1", 7727),
        ("BULGOGI RICE", "1", 33636),
    ]
    assert document["totals"]["tax_minor"] == 4136
    assert [c["applied"] for c in document["charges"]] == [True]


def test_tax_already_in_the_prices_is_not_counted_twice():
    document = receipt(
        "*2 Cheese Tart | 58000.00",
        "Subtotal: | 58000.00",
        "Total Tax Included: | 5272.73",
        "Total: | 58000.00",
    )
    assert document["status"] == "ok"
    assert items(document) == [("Cheese Tart", "2", 5800000)]
    assert [c["applied"] for c in document["charges"]] == [False]


def test_service_charge_and_tax_are_both_added():
    document = receipt(
        "BLACK PEPPER MEATBALL | 76,500",
        "2 ORIGINAL BREWED TEA | 46,000",
        "SUBTTL | 122,500",
        "SVC CHG 6% | 7,350",
        "PB1 10% | 12,985",
        "TOTAL | 142,835",
    )
    assert document["status"] == "ok"
    assert [c["kind"] for c in document["charges"]] == ["service", "tax"]


def test_a_discount_printed_under_the_item_and_again_in_the_footer_counts_once():
    document = receipt(
        "4 Choco Devil | 63,636",
        "CP 360 Club Card | -9,545",
        "SubTotal: | 63,636",
        "Discount: | -9,545",
        "PB1: | 5,409",
        "Total: | 59,500",
    )
    assert document["status"] == "ok"
    assert [a["amount_minor"] for a in document["adjustments"]] == [-9545]
    assert [(c["kind"], c["applied"]) for c in document["charges"]] == [
        ("discount", False),
        ("tax", True),
    ]


def test_a_name_on_one_line_and_its_numbers_on_the_next():
    document = receipt(
        "Populaire Chocolate",
        "1 4,500 | 4,500",
        "Paddle Pop Choco Magma",
        "2 3,000 | 6,000",
        "Total | 10,500",
    )
    assert document["status"] == "ok"
    assert items(document) == [
        ("Populaire Chocolate", "1", 4500),
        ("Paddle Pop Choco Magma", "2", 6000),
    ]
    assert [i["unit_price_minor"] for i in document["line_items"]] == [4500, 3000]


def test_a_quantity_times_unit_price_row_is_an_item_not_a_quantity_line():
    document = receipt("Arem Arem", "2 x @ 12.000 | 24.000", "Total | 24.000")
    assert items(document) == [("Arem Arem", "2", 24000)]


def test_a_number_that_opens_a_name_is_not_a_quantity():
    document = receipt("7 UP | 1.50", "BREAD | 2.00", "COLA | 1.25", "TOTAL | 4.75")
    assert items(document)[0] == ("7 UP", "1", 150)


def test_a_footer_label_never_becomes_the_first_item():
    document = receipt("1x 52.000", "Subtotal | 52.000", "Total | 52.000")
    assert document["status"] == "failed"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("TOTALE EURO", "total"),
        ("T0TALE EURO", "total"),
        ("TOTA.E EURO", "total"),
        ("Grand Total", "total"),
        ("Total Bayar", "total"),
        ("Sub Total", "subtotal"),
        ("SubTTL", "subtotal"),
        ("Zwischensumme", "subtotal"),
        ("Total Tax Included:", "tax"),
        ("Pajak Resto", "tax"),
        ("di cui IVA", "tax"),
        ("MwSt 19%", "tax"),
        ("Service charge", "service"),
        ("SCONTO FIDATY 30%", "discount"),
        ("Kembali", "change"),
        ("Cash CHANGE", "change"),
        ("RESTO", "change"),
        ("Uang Bayar", "payment"),
        ("Pagamento Carta di Cre", "payment"),
        ("ITEMS: 13", "count"),
        ("POWERADE MOUNTAIN", None),
        ("VANILLA CHOCO HEART CAKE", None),
        ("SFILATINO MORBIDO", None),
    ],
)
def test_footer_labels(text, expected):
    assert role(text) == expected


def test_column_headings():
    assert is_heading("IVA EURO")
    assert is_heading("Qty Description Amount")
    assert not is_heading("PANE INTEGRALE")

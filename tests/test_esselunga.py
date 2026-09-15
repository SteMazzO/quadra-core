"""Esselunga profile tests, over OCR of rendered and photographed receipts."""

from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal

import pytest

from quadra_core.pipeline import recheck
from quadra_core.pipeline.extract import (
    _strip_leading_noise,
    extract,
    extract_item,
    find_item_region,
)
from quadra_core.pipeline.lines import (
    Line,
    Word,
    drop_speckle,
    group_lines,
    load_tsv,
    text_block,
)
from quadra_core.pipeline.money import normalize_separators, parse_money
from quadra_core.pipeline.validate import validate
from quadra_core.profiles import loader
from quadra_core.testdata import OCR as FIXTURES

# Esselunga loyalty numbers print as '040*******92'; OCR reads the zeros as letter O.
CARD_TOKEN = re.compile(r"\b[O0oQ]?4[O0oQ][A-Za-z0-9*]{5,}\b")

#   A: 0,99 + 0,61 + 0,01 + 2,98 + 0,65 + 0,01                    = 5,25
#   B: 1,19 + 0,99x4 + 1,29 + 1,79 + 1,49 + 23,92 - 7,20          = 26,44
RECEIPT_A_TOTAL = 525
RECEIPT_B_TOTAL = 2644
RECEIPT_B_ITEMS = [119, 99, 99, 99, 99, 129, 179, 149, 2392]


def esselunga():
    return next(p for p in loader.available() if p.id == "esselunga")


def run(name: str):
    lines = group_lines(drop_speckle(load_tsv((FIXTURES / f"{name}.tsv").read_text())))
    ex = extract(lines, esselunga())
    return ex, validate(ex, esselunga())


@pytest.mark.parametrize(
    "name,total",
    [
        ("esselunga_a", RECEIPT_A_TOTAL),
        ("esselunga_b", RECEIPT_B_TOTAL),
        ("esselunga_b_faded", RECEIPT_B_TOTAL),
    ],
)
def test_receipts_parse_and_balance(name, total):
    _, v = run(name)
    assert v.status == "ok", f"warnings: {v.warnings}"
    assert v.balanced and v.delta_minor == 0
    assert v.printed_total_minor == total


def test_item_prices_match_the_paper_receipt():
    ex, _ = run("esselunga_b")
    assert [i.line_total_minor for i in ex.items] == RECEIPT_B_ITEMS


def test_two_column_layout_yields_implicit_quantity_without_flagging():
    """Esselunga prints no quantity, so it is an unflagged implicit 1."""
    ex, _ = run("esselunga_a")
    assert len(ex.items) == 6
    assert all(i.quantity == Decimal(1) for i in ex.items)
    assert all(i.quantity_source == "implicit" for i in ex.items)
    assert not any(i.flags for i in ex.items)


@pytest.mark.parametrize("name", ["esselunga_b", "esselunga_b_faded"])
def test_preceding_quantity_modifier_is_attached_to_the_following_item(name):
    """'8 x 2,99' is printed above its item."""
    ex, _ = run(name)
    bondue = next(i for i in ex.items if "BONDUE" in i.description)
    assert bondue.quantity == Decimal(8)
    assert bondue.quantity_source == "modifier"
    assert bondue.unit_price_minor == 299
    assert bondue.line_total_minor == 2392
    # The modifier line itself must not also become an item.
    assert not any(i.line_total_minor == 299 for i in ex.items)


@pytest.mark.parametrize("name", ["esselunga_b", "esselunga_b_faded"])
def test_trailing_sign_discount_is_negative_and_kept_as_an_adjustment(name):
    """'SCONTO FIDATY 30%  7,20-S' is minus 7,20, kept as an adjustment."""
    ex, _ = run(name)
    assert len(ex.adjustments) == 1
    adjustment = ex.adjustments[0]
    assert adjustment.amount_minor == -720
    assert adjustment.kind == "discount"
    assert "SCONTO" in adjustment.label.upper()
    assert not any(i.line_total_minor == 720 for i in ex.items)


def test_loyalty_footer_never_reaches_the_totals():
    """Points are printed as '7.261' and must not be read as money."""
    ex, _ = run("esselunga_a")
    assert all(i.line_total_minor < 1000 for i in ex.items)
    assert not any("PUNTI" in i.description.upper() for i in ex.items)
    assert not any("FIDATY" in i.description.upper() for i in ex.items)


def test_payment_lines_are_not_items():
    ex, _ = run("esselunga_b")
    assert not any("PAGAMENTO" in i.description.upper() for i in ex.items)
    assert not any(i.line_total_minor == 1690 for i in ex.items)


def test_description_containing_a_decimal_is_not_split():
    """The '1,5' in 'CLEMENTINE IGP 1,5 KG' is not money."""
    ex, _ = run("esselunga_a")
    clementine = [i for i in ex.items if "IGP" in i.description]
    assert clementine, "item vanished"
    assert clementine[0].line_total_minor == 298
    assert "1,5" in clementine[0].description


def test_profile_fingerprint_uses_the_company_vat_number():
    """The VAT number is identical across stores, so it survives a shop change."""
    profile = esselunga()
    assert "04916380159" in profile.fingerprint
    lines = [
        l.text
        for l in group_lines(
            drop_speckle(load_tsv((FIXTURES / "esselunga_b_faded.tsv").read_text()))
        )
    ]
    selected = loader.select(lines)
    assert selected is not None and selected[0].id == "esselunga"


def test_profile_declares_the_observed_layout():
    profile = esselunga()
    assert profile.has_quantity_column is False
    assert profile.has_unit_price_column is False
    assert profile.modifier_position == "before"


@pytest.mark.parametrize(
    "token,expected",
    [("1;;19", 119), ("1:79", 179), ("2:99", 299), ("7,20-S", -720), ("23,92", 2392)],
)
def test_separator_confusion_is_normalised(token, expected):
    """On faded print OCR reads the decimal comma as ';' or ':'."""
    assert parse_money(normalize_separators(token)) == expected


def test_separator_normalisation_leaves_non_decimal_punctuation_alone():
    """Only separators between digits are decimal points."""
    untouched = "SACCHETTO COMPOST.LE 0,01"
    assert normalize_separators(untouched) == untouched
    assert normalize_separators("8x. 2:99") == "8x. 2,99"


# --- Real photograph -------------------------------------------------------------
# esselunga_photo: a low-resolution photo of receipt B, its total unreadable.


def test_real_photograph_parses_every_item():
    ex, _ = run("esselunga_photo")
    assert [i.line_total_minor for i in ex.items] == RECEIPT_B_ITEMS


def test_real_photograph_recovers_the_total_from_the_payment_lines():
    """The printed total is unreadable, but the payments add up to it."""
    ex, v = run("esselunga_photo")
    assert ex.printed_total_minor is None, "fixture no longer exercises the fallback"
    assert ex.payments_minor == [4, 950, 1690]
    assert v.total_source == "payments"
    assert v.printed_total_minor == RECEIPT_B_TOTAL
    assert v.balanced and v.delta_minor == 0


def test_fallback_is_recorded_and_still_routed_to_review():
    """A total taken from the payments still leaves the receipt partial."""
    _, v = run("esselunga_photo")
    assert v.status == "partial"
    assert "printed_total_unreadable_used_payments" in v.warnings


def test_change_is_subtracted_from_amounts_tendered():
    """On a cash sale the payments exceed the total by exactly the change given."""
    ex, _ = run("esselunga_photo")
    ex.change_minor = 500
    ex.payments_minor = [*ex.payments_minor, 500]
    v = validate(ex, esselunga())
    assert v.printed_total_minor == RECEIPT_B_TOTAL, "change was not netted off"


def test_printed_total_wins_when_both_are_readable():
    """The payments are a fallback, not a replacement."""
    ex, _ = run("esselunga_b")
    assert ex.printed_total_minor == RECEIPT_B_TOTAL
    v = validate(ex, esselunga())
    assert v.total_source == "printed"


def test_disagreement_between_the_two_totals_is_reported():
    ex, _ = run("esselunga_b")
    ex.payments_minor = [9999]
    v = validate(ex, esselunga())
    assert any("payments_disagree" in w for w in v.warnings)


def test_loyalty_card_is_scrubbed_from_the_committed_fixture():
    """The committed photo fixture carries no loyalty card number."""
    raw = (FIXTURES / "esselunga_photo.tsv").read_text()
    assert not CARD_TOKEN.search(raw)


# --- Real photographs -------------------------------------------------------
# Two low-resolution photos whose printed totals are unreadable.

PHOTO_CASES = [
    ("esselunga_photo_payments", RECEIPT_A_TOTAL, 6),
    ("esselunga_photo_modifier", RECEIPT_B_TOTAL, 9),
]


@pytest.mark.parametrize("name,total,items", PHOTO_CASES)
def test_real_photographs_balance(name, total, items):
    ex, v = run(name)
    assert len(ex.items) == items
    assert v.balanced, f"delta {v.delta_minor}: {v.warnings}"
    assert v.printed_total_minor == total


def test_merged_rows_are_separated_by_centre_clustering():
    """'ARANCE 0,65' must not be swallowed by the row beneath it."""
    ex, _ = run("esselunga_photo_payments")
    arance = [i for i in ex.items if i.description.strip().startswith("ARANCE")]
    assert len(arance) == 1, f"got {[i.description for i in ex.items]}"
    assert arance[0].line_total_minor == 65
    # And the row it was merged with survives separately.
    assert sum(1 for i in ex.items if i.line_total_minor == 1) == 2


def test_discount_survives_a_misread_vat_letter():
    """Tesseract read the trailing 'S' of '7,20-S' as '$'."""
    ex, _ = run("esselunga_photo_modifier")
    assert [a.amount_minor for a in ex.adjustments] == [-720]


def test_photograph_prices_are_exact_even_where_descriptions_are_not():
    """Prices come out exact even where descriptions are misread."""
    ex, _ = run("esselunga_photo_payments")
    assert sorted(i.line_total_minor for i in ex.items) == [1, 1, 61, 65, 99, 298]
    assert any(
        i.description != i.description_raw or "TL" in i.description for i in ex.items
    ), "expected OCR noise in descriptions"


# --- a long, curled receipt -------------------------------------------------
# 33 items and 7 discounts, photographed with the roll curled.


def curled() -> tuple:
    lines = group_lines(
        drop_speckle(load_tsv((FIXTURES / "esselunga_photo_curled.tsv").read_text()))
    )
    return lines, extract(lines, esselunga())


def test_a_wavy_receipt_keeps_each_price_with_its_own_description():
    """On a curled roll each price stays on its description's line."""
    lines, _ = curled()
    text = [l.text for l in lines]
    assert any(t.startswith("POMOIORO") and "2,38" in t for t in text)
    assert any("PATATE AL FORNO" in t and "2,62" in t for t in text)
    assert any("NATURAMA" in t and "2,61" in t for t in text)
    # No line may be a bare VAT code with a price and no product name.
    assert not [t for t in text if re.fullmatch(r"[a-z*«“]{1,3}\s+\d+,\d\d", t)]


def test_every_discount_on_the_receipt_is_read():
    """Discounts OCR split into '0,' and '48-S' are joined back together."""
    _, ex = curled()
    amounts = sorted(a.amount_minor for a in ex.adjustments)
    assert amounts == [-529, -78, -51, -51, -48, -48, -44]


def test_a_price_with_junk_in_front_is_still_a_price():
    """'(2,33' is still a price."""
    _, ex = curled()
    assert any(i.line_total_minor == 233 for i in ex.items)


def test_a_comma_printed_over_a_fold_is_still_a_comma():
    """'2/76' is still 2,76."""
    _, ex = curled()
    assert any(i.line_total_minor == 276 for i in ex.items)


def test_the_printed_total_is_read_rather_than_inferred():
    """A mangled 'TOTA.E EURO' still matches the total rule."""
    profile = esselunga()
    assert profile.rule_for("TOTA.E EURO 127,05 +") == "total"
    assert profile.rule_for("TOTALE EURO 26,44") == "total"


def test_the_whole_receipt_comes_out():
    _, ex = curled()
    assert len(ex.items) == 33
    assert len(ex.adjustments) == 7


# --- the IVA column ---------------------------------------------------------
# The VAT bracket, printed between the description and the price.


def test_the_vat_bracket_is_not_part_of_the_product_name():
    _, ex = curled()
    for item in ex.items:
        stray = re.search(r"[*«»#“x]\s*[a-z]\s*$", item.description)
        assert not stray, item.description


def test_every_line_on_this_receipt_has_its_bracket_read():
    _, ex = curled()
    assert all(i.vat_code for i in ex.items)


def test_the_bracket_is_the_letter_not_whatever_the_asterisk_became():
    """OCR mangles the asterisk; only the letter counts."""
    _, ex = curled()
    assert all(len(i.vat_code) == 1 and i.vat_code.islower() for i in ex.items)


def test_the_brackets_group_the_receipt_the_way_the_tax_does():
    """Staples share one bracket and non-food another, so the column was read."""
    _, ex = curled()
    by_name = {i.description: i.vat_code for i in ex.items}
    staples = [by_name[k] for k in by_name if k.startswith(("SFILATINO", "BANANE"))]
    nonfood = [by_name[k] for k in by_name if k.startswith(("DOVE", "POWERADE"))]
    assert len(set(staples)) == 1
    assert len(set(nonfood)) == 1
    assert set(staples) != set(nonfood)


def test_a_short_word_mid_description_is_not_mistaken_for_a_bracket():
    """In 'OLIVA FBERIO LT 0,75 *a 4,99' only '*a' is the bracket."""
    _, ex = curled()
    oliva = next(i for i in ex.items if i.description.startswith("OLIVA"))
    assert oliva.vat_code == "a"
    assert "LT" in oliva.description


def test_a_profile_without_that_column_is_unaffected():
    profile = esselunga()
    without = replace(profile, vat_code_re=None)
    lines, _ = curled()
    plain = extract(lines, without)
    assert all(i.vat_code is None for i in plain.items)
    # The marker stays in the description.
    assert any(re.search(r"[*«»#][a-z]", i.description) for i in plain.items)


# --- a receipt photographed on a wooden table -------------------------------
# 33 items, 8 discounts, 101,29, with desk grain in the crop and a tilted footer.


def table() -> tuple:
    lines = group_lines(
        drop_speckle(load_tsv((FIXTURES / "esselunga_photo_table.tsv").read_text()))
    )
    return lines, extract(lines, esselunga())


def test_the_whole_table_receipt_is_read():
    _, ex = table()
    assert len(ex.items) == 33
    assert len(ex.adjustments) == 8


def test_it_finds_its_own_total():
    """The total comes from the payment line, the printed one being unreadable."""
    _lines, ex = table()
    validation = validate(ex, esselunga())
    assert validation.printed_total_minor == 10129
    assert validation.delta_minor == 0


def test_a_payment_line_is_read_through_whatever_precedes_it():
    """OCR junk from the paper's edge before 'PAGAMENTO' doesn't hide the payment."""
    profile = esselunga()
    assert profile.rule_for("$ PAGAMENTO CARTA FIDATY ORO 101,29") == "payment"
    assert profile.rule_for("PAGAMENTO CARTA FIDATY ORO 101,29") == "payment"
    # The loyalty line itself is still noise: it carries no payment.
    assert profile.rule_for("N. CARTA FIDATY: 00XX*XX***00") == "noise"


def test_a_discount_whose_sign_letter_was_read_as_a_digit():
    """'0,74-S' read as '0,74-8' is still minus 0,74."""
    assert parse_money("0,74-8") == -74
    assert parse_money("0,74-S") == -74


def test_the_footer_lines_hold_together():
    """The wide, large-type footer lines keep their amounts on the same line."""
    lines, _ = table()
    tail = " | ".join(l.text for l in lines[-8:])
    assert re.search(r"PAGAMENTO[^|]*101,29", tail), tail


# --- the receipt whose column header did not survive --------------------------


def test_items_are_found_when_the_column_header_is_unreadable():
    """Without the 'EURO' header, items must not start after 'TOTALE EURO'."""
    extraction, _ = run("esselunga_photo_no_header")
    assert [i.line_total_minor for i in extraction.items] == [
        469,
        99,
        186,
        359,
        65,
        499,
        499,
        368,
    ]


def test_that_receipt_still_finds_its_total_through_the_payments():
    """The total comes from the card payment line."""
    _, validation = run("esselunga_photo_no_header")
    assert validation.printed_total_minor == 2554
    assert validation.total_source == "payments"
    # Short by a bag whose price only the price-column re-read can recover.
    assert validation.delta_minor == -10


def test_the_item_region_never_starts_after_the_total():
    """The end anchor keeps the line it matched, whatever else also matches it."""
    lines = group_lines(
        drop_speckle(load_tsv((FIXTURES / "esselunga_photo_no_header.tsv").read_text()))
    )
    total = next(i for i, ln in enumerate(lines) if "TOTALE" in ln.text)
    start, end = find_item_region(lines, esselunga())
    assert start < total, "items must not begin after the total"
    assert end <= total, "items must not run past the total"


# --- profile handling -------------------------------------------------------


def _line(index: int, words: list[str]):
    out, x = [], 0
    for text in words:
        out.append(Word(text, x, index * 30, len(text) * 10, 20, 95.0))
        x += len(text) * 10 + 15
    return Line(words=out, index=index)


def test_a_quantity_column_is_read_without_a_unit_price_column():
    """A shop printing quantity and total but no unit price."""
    base = next(p for p in loader.available() if p.id == "synthetic")
    shop = replace(base, has_quantity_column=True, has_unit_price_column=False)
    lines = [
        _line(0, ["SCONTRINO", "N", "1"]),
        _line(1, ["MELE", "3", "4,50"]),
        _line(2, ["TOTALE", "COMPLESSIVO", "4,50"]),
    ]
    (item,) = extract(lines, shop).items
    assert item.quantity == Decimal(3)
    assert item.quantity_source == "parsed"
    assert item.description == "MELE"
    assert item.unit_price_minor == 150


def test_a_modifier_printed_after_its_item_attaches_to_the_item_above():
    """Both items cost 23,92, so only the profile's position breaks the tie."""
    base = next(p for p in loader.available() if p.id == "esselunga")
    lines = [
        _line(0, ["EURO"]),
        _line(1, ["OLIVE", "23,92"]),
        _line(2, ["8", "x", "2,99"]),
        _line(3, ["PANE", "23,92"]),
        _line(4, ["TOTALE", "EURO", "47,84"]),
    ]
    after = extract(lines, replace(base, modifier_position="after")).items
    assert [(i.description, i.quantity_source) for i in after] == [
        ("OLIVE", "modifier"),
        ("PANE", "implicit"),
    ]

    before = extract(lines, replace(base, modifier_position="before")).items
    assert [(i.description, i.quantity_source) for i in before] == [
        ("OLIVE", "implicit"),
        ("PANE", "modifier"),
    ]


def test_an_unknown_modifier_position_is_refused_at_load_time(tmp_path):
    """A typo in modifier_position is an error, not silently ignored."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[profile]\nid="x"\nversion=1\n[format]\nmoney="\\\\d+"\n'
        '[layout]\nmodifier_position="Before"\n'
    )
    with pytest.raises(loader.ProfileError, match="modifier_position"):
        loader.load(bad)


def test_edge_noise_is_stripped_from_the_front_of_a_description():
    """A low-confidence one-letter word from the paper's edge is dropped."""
    line = Line(
        words=[
            Word("i", 0, 0, 10, 27, 12.5),
            Word("LOACKER", 30, 0, 90, 20, 92.2),
            Word("4,64", 200, 0, 45, 20, 95.3),
        ],
        index=1,
    )
    profile = next(p for p in loader.available() if p.id == "esselunga")
    item = extract_item(line, profile, text_block([line]), None)
    assert item.description == "LOACKER"
    assert item.confidence > 0.9


def test_a_description_is_never_stripped_away_entirely():
    """Stripping never removes the whole description."""
    words = [Word("i", 0, 0, 10, 27, 12.5), Word("s", 20, 0, 10, 20, 4.8)]
    assert _strip_leading_noise(words) == words


def test_a_confident_leading_word_is_kept():
    """'8 LOACKER CREAMKAKAO' really does start with an 8."""
    words = [Word("8", 0, 0, 10, 20, 87.0), Word("LOACKER", 30, 0, 90, 20, 92.2)]
    assert _strip_leading_noise(words) == words


# --- the same photograph, with its rows merged over the crease --------------
# The photo behind esselunga_photo_curled, read by Tesseract 5.3.4, which merges
# price rows over the crease. MERGED_ROWS_READINGS is what the column re-read saw.

MERGED_ROWS_READINGS = {
    7: [169, 169, 169],
    8: [-51, -51, -51],
    10: [213, 213, 213],
    12: [178, 178, 178],
    13: [169, 169, 169],
    14: [51, -51, -51],
    15: [89, 89, 89],
    16: [499, 499, 499],
    17: [74, 74, 74],
    18: [239, 7239, 7239],
    19: [-48, -48, -48],
    20: [299, 299, 299],
    21: [239, 239, 239],
    23: [529, 529, 529],
    24: [529, 529, 529],
    27: [518, 518, 518],
    31: [596, 596, 596],
    32: [233, 233, 233],
    33: [1996, 1996, 1996],
    34: [619, 619, 619],
    35: [1178, 1178, 1178],
    36: [399, 399, 399],
    37: [474, 474, 474],
    38: [269, 269, 269],
    39: [656, 555, 555],
    40: [488, 488, 488],
    41: [543, 643, 543],
    42: [238, 238, 238],
    43: [365, 365, 365],
    44: [82, 82, 82],
    45: [188, 188, 188],
    46: [276, 276],
    47: [262, 262, 262],
    48: [580, 580, 580],
    49: [208, 208, 208],
    50: [261, 261, 261],
}

CREASED_ITEMS = [
    169,
    213,
    178,
    169,
    89,
    499,
    74,
    239,
    299,
    239,
    529,
    529,
    518,
    596,
    233,
    1996,
    619,
    1178,
    399,
    474,
    269,
    555,
    488,
    543,
    238,
    365,
    82,
    188,
    276,
    262,
    580,
    208,
    261,
]
CREASED_DISCOUNTS = [-529, -78, -51, -51, -48, -48, -44]


def merged_rows() -> tuple:
    lines = group_lines(
        drop_speckle(
            load_tsv((FIXTURES / "esselunga_photo_merged_rows.tsv").read_text())
        )
    )
    return lines, extract(lines, esselunga())


def test_an_amount_in_the_description_is_not_taken_for_the_price():
    """'OLIVA FBERIO LT 0,75' is not priced at 0,75."""
    lines, ex = merged_rows()
    assert not [i for i in ex.items if i.description.startswith("OLIVA")]
    assert [i for i in ex.skipped_lines if lines[i].text.startswith("OLIVA")]


def test_a_number_split_at_its_comma_is_joined_across_a_wider_gap():
    """'0,' and '48-S' are joined even 15px apart."""
    _, ex = merged_rows()
    assert -48 in [a.amount_minor for a in ex.adjustments]


def test_merged_rows_are_recovered_from_the_price_column(monkeypatch):
    lines, ex = merged_rows()
    first = validate(ex, esselunga())
    assert not first.balanced, "fixture no longer exercises the recovery"

    readings = {int(k): v for k, v in MERGED_ROWS_READINGS.items()}
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: readings)
    result = recheck.repair(ex, first, object(), esselunga(), lines)
    assert result is not None
    recheck.apply(ex, result, lines, esselunga())

    v = validate(ex, esselunga())
    assert v.balanced, f"delta {v.delta_minor}"
    assert [i.line_total_minor for i in ex.items] == CREASED_ITEMS
    assert sorted(a.amount_minor for a in ex.adjustments) == CREASED_DISCOUNTS
    assert not ex.skipped_lines


def test_recovered_lines_are_named_without_the_price_that_failed(monkeypatch):
    lines, ex = merged_rows()
    readings = {int(k): v for k, v in MERGED_ROWS_READINGS.items()}
    monkeypatch.setattr(recheck, "second_opinion", lambda *a, **k: readings)
    result = recheck.repair(ex, validate(ex, esselunga()), object(), esselunga(), lines)
    recheck.apply(ex, result, lines, esselunga())

    recovered = [i for i in ex.items if "line_recovered" in i.flags]
    # The other three recovered lines are discounts.
    assert len(recovered) == 7
    assert "YOGA NETT.ALBICOCCAIL" in [i.description for i in recovered]
    assert "OLIVA FBERIO LT 0,75" in [i.description for i in recovered]
    assert all(i.vat_code for i in recovered)
    assert "SCONTO FIDATY 30%" in [a.label for a in ex.adjustments]

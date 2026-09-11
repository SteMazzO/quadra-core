"""Esselunga profile tests.

Fixtures are Tesseract output for renders transcribed from two real receipts.
They reproduce the layout - two columns, a quantity line above its item, a
discount signed on the right, a loyalty footer - but not the photography, so
passing here does not mean a real photo will parse.
"""

from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal

import pytest

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

# Transcribed by hand from the receipt photographs and arithmetically confirmed:
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
    """The finding that drove the layout model.

    Esselunga prints DESCRIPTION + PRICE only. Treating the absent quantity as an
    error would flag every line on every receipt and make the review queue useless,
    so quantity is implicit-1 and carries no flag.
    """
    ex, _ = run("esselunga_a")
    assert len(ex.items) == 6
    assert all(i.quantity == Decimal(1) for i in ex.items)
    assert all(i.quantity_source == "implicit" for i in ex.items)
    assert not any(i.flags for i in ex.items)


@pytest.mark.parametrize("name", ["esselunga_b", "esselunga_b_faded"])
def test_preceding_quantity_modifier_is_attached_to_the_following_item(name):
    """'8 x 2,99' sits ABOVE 'BONDUE. COCCOLE SPINA 23,92', not below it."""
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
    """'SCONTO FIDATY 30%  7,20-S' is minus 7,20.

    A leading-minus parser reads +7,20 and the receipt then misses by twice the
    discount. The discount stays an adjustment rather than being folded into an item
    price, so the receipt remains auditable.
    """
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
    """'CLEMENTINE IGP 1,5 KG' has a comma-number in the description.

    It is not money (one decimal digit, not two), so it must stay in the text.
    """
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
    """Measured: Tesseract reads the decimal comma as ';' or ':' on faded print."""
    assert parse_money(normalize_separators(token)) == expected


def test_separator_normalisation_leaves_non_decimal_punctuation_alone():
    """Only separators between digits are decimal points.

    'SACCHETTO COMPOST.LE' and the '8x.' of a modifier line must survive intact.
    """
    untouched = "SACCHETTO COMPOST.LE 0,01"
    assert normalize_separators(untouched) == untouched
    assert normalize_separators("8x. 2:99") == "8x. 2,99"


# --- Real photograph -------------------------------------------------------------
#
# esselunga_photo.tsv is Tesseract output for an actual photograph of receipt B
# (444x800, found online, upscaled 2.5x by preprocessing), not a render. It carries
# genuine photographic degradation: soft focus, bleed-through from the reverse of the
# paper, a site watermark, and a large bold total the engine cannot read. The loyalty
# card token has been scrubbed.


def test_real_photograph_parses_every_item():
    ex, _ = run("esselunga_photo")
    assert [i.line_total_minor for i in ex.items] == RECEIPT_B_ITEMS


def test_real_photograph_recovers_the_total_from_the_payment_lines():
    """The printed total is unreadable on the real photo; the payments are not.

    'TOTALE EURO 26,44' is set in a large bold face and came back as 'db, +' at every
    scale, PSM and threshold tried. The amounts tendered restate the total
    independently of the item lines (0,04 + 9,50 + 16,90 = 26,44), so the receipt
    can still be verified rather than merely accepted.
    """
    ex, v = run("esselunga_photo")
    assert ex.printed_total_minor is None, "fixture no longer exercises the fallback"
    assert ex.payments_minor == [4, 950, 1690]
    assert v.total_source == "payments"
    assert v.printed_total_minor == RECEIPT_B_TOTAL
    assert v.balanced and v.delta_minor == 0


def test_fallback_is_recorded_and_still_routed_to_review():
    """Recovering is not the same as being confident: the receipt must not read 'ok'."""
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
    """Guard the privacy rule the README states.

    This is the only fixture taken from a real photograph rather than a render.
    """
    raw = (FIXTURES / "esselunga_photo.tsv").read_text()
    assert not CARD_TOKEN.search(raw)


# --- Real photographs -------------------------------------------------------
#
# Tesseract output for the two receipt photographs in receipts/inbox/. These are
# low-resolution images found online (338x600 and 444x800), which makes them a
# harsher test than a phone photo: the printed TOTALE line is unreadable on both and
# the total has to be recovered from the payment lines.
#
# They are the reason two bugs got fixed. Extent-overlap line grouping merged
# 'ARANCE 0,65' into the row below it because Tesseract gave ARANCE a 40px box
# against a 22px median, and the parser then kept only the rightmost price and lost
# 0,65. And the trailing VAT letter of the discount came back as '$' rather than 'S',
# which a letter-only sign pattern dropped entirely.

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
    """Prices survive what descriptions do not.

    The same fixture yields 'TL RACC.' for 'IL RACC.' and 'CLEMENTINE TGP' for 'IGP',
    while every price on those lines is correct. No arithmetic check can catch a
    misread description; that is what review sampling is for.
    """
    ex, _ = run("esselunga_photo_payments")
    assert sorted(i.line_total_minor for i in ex.items) == [1, 1, 61, 65, 99, 298]
    assert any(
        i.description != i.description_raw or "TL" in i.description for i in ex.items
    ), "expected OCR noise in descriptions"


# --- a long, curled receipt -------------------------------------------------
#
# 33 items, 7 discounts, photographed on a table so the roll is wavy rather than
# flat. Every failure it exposed is pinned below, because each one lost real money
# quietly: a dropped item or a dropped discount changes the total and nothing in the
# output says a line went missing.


def curled() -> tuple:
    lines = group_lines(
        drop_speckle(load_tsv((FIXTURES / "esselunga_photo_curled.tsv").read_text()))
    )
    return lines, extract(lines, esselunga())


def test_a_wavy_receipt_keeps_each_price_with_its_own_description():
    """A curled roll is not tilted, it bends: the slope changes down the page.

    Where the description is short the gap before the price is wide, and the drop
    across that gap beat the old tolerance. The item split into a nameless price
    and a priceless name.
    """
    lines, _ = curled()
    text = [l.text for l in lines]
    assert any(t.startswith("POMOIORO") and "2,38" in t for t in text)
    assert any("PATATE AL FORNO" in t and "2,62" in t for t in text)
    assert any("NATURAMA" in t and "2,61" in t for t in text)
    # No line may be a bare VAT code with a price and no product name.
    assert not [t for t in text if re.fullmatch(r"[a-z*«“]{1,3}\s+\d+,\d\d", t)]


def test_every_discount_on_the_receipt_is_read():
    """OCR splits '0,48-S' into '0,' and '48-S'. Five discounts vanished that way.

    Worse than vanishing: '48-S' on its own parses as minus 48 euro, so the day the
    money pattern loosened, a 48 cent discount would have become a 48 euro one.
    """
    _, ex = curled()
    amounts = sorted(a.amount_minor for a in ex.adjustments)
    assert amounts == [-529, -78, -51, -51, -48, -48, -44]


def test_a_price_with_junk_in_front_is_still_a_price():
    """'(2,33' - the bracket is a crease. The whole item used to be dropped."""
    _, ex = curled()
    assert any(i.line_total_minor == 233 for i in ex.items)


def test_a_comma_printed_over_a_fold_is_still_a_comma():
    """'2/76'. Dropped as unparseable, so the item never existed."""
    _, ex = curled()
    assert any(i.line_total_minor == 276 for i in ex.items)


def test_the_printed_total_is_read_rather_than_inferred():
    """'TOTA.E EURO': the biggest print on the page, and the most creased."""
    profile = esselunga()
    assert profile.rule_for("TOTA.E EURO 127,05 +") == "total"
    assert profile.rule_for("TOTALE EURO 26,44") == "total"


def test_the_whole_receipt_comes_out():
    _, ex = curled()
    assert len(ex.items) == 33
    assert len(ex.adjustments) == 7


# --- the IVA column ---------------------------------------------------------
#
# Esselunga prints the VAT bracket between the description and the price, under a
# column its own header calls IVA. It was ending up in the product name, which is
# the field a budget groups by and the one the accuracy figure measures.


def test_the_vat_bracket_is_not_part_of_the_product_name():
    _, ex = curled()
    for item in ex.items:
        stray = re.search(r"[*«»#“x]\s*[a-z]\s*$", item.description)
        assert not stray, item.description


def test_every_line_on_this_receipt_has_its_bracket_read():
    _, ex = curled()
    assert all(i.vat_code for i in ex.items)


def test_the_bracket_is_the_letter_not_whatever_the_asterisk_became():
    """OCR renders the mark as * « » # x s i v " or nothing. Only the letter counts."""
    _, ex = curled()
    assert all(len(i.vat_code) == 1 and i.vat_code.islower() for i in ex.items)


def test_the_brackets_group_the_receipt_the_way_the_tax_does():
    """Bread, oil and fruit in one bracket; drinks and toiletries in another.

    Not a rule the parser knows, an observation that it read the column and not
    something else: if these were noise they would not sort themselves this way.
    """
    _, ex = curled()
    by_name = {i.description: i.vat_code for i in ex.items}
    staples = [by_name[k] for k in by_name if k.startswith(("SFILATINO", "BANANE"))]
    nonfood = [by_name[k] for k in by_name if k.startswith(("DOVE", "POWERADE"))]
    assert len(set(staples)) == 1
    assert len(set(nonfood)) == 1
    assert set(staples) != set(nonfood)


def test_a_short_word_mid_description_is_not_mistaken_for_a_bracket():
    """Only the word immediately left of the price is considered.

    'OLIVA FBERIO LT 0,75 *a 4,99' has a two letter word and a price in the middle
    of it, and neither is the bracket.
    """
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
    # And the descriptions keep the marker, because nothing claimed it.
    assert any(re.search(r"[*«»#][a-z]", i.description) for i in plain.items)


# --- a receipt photographed on a wooden table -------------------------------
#
# 33 items, 8 discounts, 101,29. The desk grain got into the crop, the text came
# out small, and the footer sat at a steeper angle than the middle of the page.
# Between them the receipt lost two lines and never found its own total, which was
# printed in the largest type on the paper.


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
    """It comes off the payment line, the printed one being unreadable here."""
    _lines, ex = table()
    validation = validate(ex, esselunga())
    assert validation.printed_total_minor == 10129
    assert validation.delta_minor == 0


def test_a_payment_line_is_read_through_whatever_precedes_it():
    """OCR puts the paper's edge in front of the text.

    Anchored to the start of the line, this fell through to the noise rule, which
    matches CARTA FIDATY, and the only total the receipt had left was discarded.
    """
    profile = esselunga()
    assert profile.rule_for("$ PAGAMENTO CARTA FIDATY ORO 101,29") == "payment"
    assert profile.rule_for("PAGAMENTO CARTA FIDATY ORO 101,29") == "payment"
    # The loyalty line itself is still noise: it carries no payment.
    assert profile.rule_for("N. CARTA FIDATY: 00XX*XX***00") == "noise"


def test_a_discount_whose_sign_letter_was_read_as_a_digit():
    """'0,74-S' came back as '0,74-8' and the discount was thrown away."""
    assert parse_money("0,74-8") == -74
    assert parse_money("0,74-S") == -74


def test_the_footer_lines_hold_together():
    """The total and the payment span the full width in very large type.

    Judged against a slope taken from the middle of the receipt they came apart,
    the label on one line and the amount on another.
    """
    lines, _ = table()
    tail = " | ".join(l.text for l in lines[-8:])
    assert re.search(r"PAGAMENTO[^|]*101,29", tail), tail


# --- the receipt whose column header did not survive --------------------------


def test_items_are_found_when_the_column_header_is_unreadable():
    """The start anchor matched the end anchor's own line, and the items vanished.

    Esselunga's items start after the word EURO, printed over the price column,
    and end before TOTALE EURO. On this photograph the column header came back as
    'sere cem |', so the first EURO on the page was the one in TOTALE EURO: the
    item region began after the total and eight readable products yielded none.
    """
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
    """TOTALE EURO reads as 'TOTALE EURO 584'. The card line is the way in."""
    _, validation = run("esselunga_photo_no_header")
    assert validation.printed_total_minor == 2554
    assert validation.total_source == "payments"
    # Short by the shopper bag, whose price the page could not read at all. The
    # second pass over the price column recovers it; that needs the photograph,
    # so it is tested in test_recheck.
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
    """A shop printing QUANTITY and TOTAL but no unit price used to have its
    quantity ignored and left sitting in the description."""
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
    """Both items cost 23,92, so the arithmetic fits either one and only the
    profile's stated position can break the tie."""
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
    """A capitalisation typo used to disable modifier handling in silence."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[profile]\nid="x"\nversion=1\n[format]\nmoney="\\\\d+"\n'
        '[layout]\nmodifier_position="Before"\n'
    )
    with pytest.raises(loader.ProfileError, match="modifier_position"):
        loader.load(bad)


def test_edge_noise_is_stripped_from_the_front_of_a_description():
    """The shadow along the paper edge comes back as a full-height word, so
    drop_speckle cannot see it; only its confidence gives it away."""
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
    """Stripping to nothing would drop the line, and take its money with it."""
    words = [Word("i", 0, 0, 10, 27, 12.5), Word("s", 20, 0, 10, 20, 4.8)]
    assert _strip_leading_noise(words) == words


def test_a_confident_leading_word_is_kept():
    """'8 LOACKER CREAMKAKAO' really does start with an 8."""
    words = [Word("8", 0, 0, 10, 20, 87.0), Word("LOACKER", 30, 0, 90, 20, 92.2)]
    assert _strip_leading_noise(words) == words

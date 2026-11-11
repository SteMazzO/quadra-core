"""Turn lines of words into receipt items and footer amounts.

Nothing here knows a shop. Prices are found by their column, quantities by
arithmetic, and the footer by the words every receipt uses for its totals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from quadra_core.pipeline.lexicon import is_heading, role
from quadra_core.pipeline.lines import (
    Block,
    Line,
    Word,
    discover_price_column,
    median_glyph_height,
    text_block,
)
from quadra_core.pipeline.money import (
    CURRENCY,
    Format,
    detect_format,
    parse_amount,
    parse_quantity,
    times,
)


@dataclass(slots=True)
class LineItem:
    """One product line off the receipt."""

    description: str
    quantity: Decimal
    quantity_source: str  # "parsed" | "reconstructed" | "modifier" | "implicit"
    unit_price_minor: int | None
    line_total_minor: int
    vat_code: str | None
    confidence: float
    line_index: int
    bbox: tuple[int, int, int, int]
    flags: list[str] = field(default_factory=list)
    # The price the first reading gave, when the re-read replaced it.
    first_read_minor: int | None = None
    # Whether the name was printed on the line beside the price, not on it.
    borrowed_name: bool = False


@dataclass(slots=True)
class Adjustment:
    """A discount printed among the items."""

    label: str
    amount_minor: int
    line_index: int
    # Set when the re-read suggested the amount but could not be sure of it.
    unconfirmed: bool = False


@dataclass(slots=True)
class Charge:
    """An amount printed between the items and the total: tax, service, rounding."""

    kind: str  # "tax" | "service" | "discount" | "rounding"
    label: str
    amount_minor: int
    line_index: int
    # Whether the total only adds up with this amount counted in. Tax printed on
    # a receipt whose prices already include it is not.
    applied: bool = False


@dataclass(slots=True)
class Extraction:
    """Everything pulled off one receipt."""

    fmt: Format = field(default_factory=Format)
    items: list[LineItem] = field(default_factory=list)
    adjustments: list[Adjustment] = field(default_factory=list)
    charges: list[Charge] = field(default_factory=list)
    # Tax printed below the total, which is already inside it.
    tax_breakdown_minor: int = 0
    subtotal_minor: int | None = None
    printed_total_minor: int | None = None
    # Set when a person typed the total in; nothing downstream may overrule it.
    printed_total_supplied: bool = False
    payments_minor: list[int] = field(default_factory=list)
    change_minor: int = 0
    warnings: list[str] = field(default_factory=list)
    # Lines that looked like products but had no readable price.
    skipped_lines: list[int] = field(default_factory=list)
    # Line index -> price of nameless lines the re-read dropped as a repeated price.
    removed: dict[int, int] = field(default_factory=dict)
    # Where the tax letters sit, as a fraction of the text width, if found.
    vat_column: float | None = None


# OCR often splits a price at the separator ('0,' '48-S'). Join halves closer than
# this many glyph heights, or the wider ratio when the split is at the separator.
_JOIN_GAP_RATIO = 0.6
_SEPARATOR_JOIN_GAP_RATIO = 1.0
_AT_SEPARATOR = re.compile(r"\d[,.;:]$|^[,.;:]\d")

# How close to the price column an amount must sit: tight first, then loose enough
# for a curled roll but never far enough to reach the description.
_COLUMN_MARGINS = (0.03, 0.2)

# Creases and shadows come back as punctuation or stray letters around the name.
_EDGE_NOISE_CONF = 80.0
_TRAILING_NOISE_CONF = 40.0

# A total is printed taller than the items, even when its label is misread.
_TALL_LINE_RATIO = 1.9

# A tax letter beside the price, such as '*c' or 'B'. OCR mangles the asterisk.
_VAT_CODE = re.compile(r"\W{0,2}[A-Za-z]?([A-Da-d])\W{0,2}")
# How far from the tax-letter column a word may sit and still be read as one.
_VAT_MARGIN = 0.06

# A line that is only a quantity and a unit price: '8 x 2,99', '2 PZ X E. 1,59',
# '0,456 kg x 12,90 EUR/kg'. Only a barcode may come before it.
_MODIFIER = re.compile(
    r"\W*(?:\d{5,}\W+)?(\d+(?:[.,]\d+)?)\s*[^\W\d_]{0,3}\s*[xX\u00d7*@]\W*(?:E\.?|EUR)?\W*"
    r"(\d\S*\d|\d)\W*(?:\s\D{1,6})?"
)

_CURRENCY_MARK = re.compile(rf"(?:{CURRENCY})\.?:?", re.IGNORECASE)

# A date as receipts print it: 01/06/2022, 1.6.22, 2022-06-01.
_DATE = re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}\b")

# The share of item lines that must show a quantity in one place for it to be a column.
_COLUMN_SHARE = 0.5
MAX_QUANTITY = 1000

# Footer lines that end the items. Payments only count once one of these is seen,
# since 'CARTA' and 'BAR' are also things people buy.
_FOOTER_START = frozenset({"subtotal", "total", "tax"})
_CHARGES = frozenset({"tax", "service", "discount", "rounding"})

Money = tuple[int, int, int]  # first word, last word, minor units


def _is_noise(word: Word) -> bool:
    """Whether a word at either end of a description is junk, not part of the name."""
    if not any(c.isalnum() for c in word.text):
        return True
    return (
        len(word.text) == 1 and not word.text.isdigit() and word.conf < _EDGE_NOISE_CONF
    )


def _is_trailing_noise(word: Word) -> bool:
    """Tell whether a short unsure word after a name is a speck, as in 'SHOPPER vi'."""
    return _is_noise(word) or (
        len(word.text) <= 2
        and not any(c.isdigit() for c in word.text)
        and word.conf < _TRAILING_NOISE_CONF
    )


def strip_noise(words: list[Word]) -> list[Word]:
    """Drop junk words from both ends of a description."""
    first, last = 0, len(words)
    while first < last and _is_noise(words[first]):
        first += 1
    while last > first + 1 and _is_trailing_noise(words[last - 1]):
        last -= 1
    return words[first:last] or words


def money_tokens(line: Line, fmt: Format) -> list[Money]:
    """Return every amount on the line, joining a number OCR split in two."""
    out: list[Money] = []
    glyph = median_glyph_height(line.words)
    i = 0
    while i < len(line.words):
        word = line.words[i]
        minor = parse_amount(word.text, fmt)
        if minor is None and i + 1 < len(line.words):
            nxt = line.words[i + 1]
            at_separator = _AT_SEPARATOR.search(word.text) or _AT_SEPARATOR.search(
                nxt.text
            )
            reach = _SEPARATOR_JOIN_GAP_RATIO if at_separator else _JOIN_GAP_RATIO
            if nxt.left - word.right <= glyph * reach and fmt.places:
                joined = parse_amount(word.text + nxt.text, fmt)
                if joined is not None:
                    out.append((i, i + 1, joined))
                    i += 2
                    continue
        if minor is not None:
            out.append((i, i, minor))
        i += 1
    return out


def price_of(
    line: Line, money: list[Money], block: Block, column: tuple[float, float] | None
) -> Money | None:
    """Return the line's price: its rightmost amount in the price column, or near it."""
    if column is None:
        return money[-1] if money else None
    low, high = column
    for margin in _COLUMN_MARGINS:
        near = [
            m
            for m in money
            if low - margin <= block.fraction(line.words[m[1]].right) <= high + margin
        ]
        if near:
            # A second column of zeros (a discount, a deposit) is not the price.
            return next((m for m in reversed(near) if m[2]), near[-1])
    return None


def read_vat_code(text: str) -> str | None:
    """Return the tax letter a word spells, or None if it is not one."""
    marker = _VAT_CODE.fullmatch(text)
    return marker.group(1).lower() if marker else None


def name_words(words: list[Word], block: Block, vat_column: float | None) -> list[Word]:
    """Return the words that name a line: left of the tax letters, junk trimmed."""
    if vat_column is not None:
        words = [w for w in words if block.fraction(w.cx) < vat_column - _VAT_MARGIN]
    return strip_noise(words) if words else []


def description_confidence(words: list[Word]) -> float:
    """How sure the OCR is of a name: the mean over its words, 0 to 1.

    Not the minimum, since OCR scores correct abbreviations like 'ESSEL.X6' low.
    """
    return round(sum(w.conf for w in words) / len(words) / 100, 3)


@dataclass(slots=True)
class _Row:
    """One line of the item area that carries a price."""

    line: Line
    money: list[Money]
    price: Money


@dataclass(slots=True)
class _Layout:
    """Where this receipt prints quantities and tax letters, judged over its rows."""

    leading_quantity: bool = False
    trailing_quantity: bool = False
    vat_column: float | None = None


def _leading(row: _Row) -> Decimal | None:
    """Return the whole number a row starts with, if anything follows it."""
    words = row.line.words
    if row.price[0] < 2 or not re.fullmatch(r"[x*]?\d{1,3}[xX*]?", words[0].text):
        return None
    return parse_quantity(words[0].text)


def _trailing(row: _Row) -> Decimal | None:
    """Return the whole number printed just left of the price, after the name."""
    index = row.price[0] - 1
    if index < 1 or not re.fullmatch(r"[x*]?\d{1,3}[xX*]?", row.line.words[index].text):
        return None
    return parse_quantity(row.line.words[index].text)


def _layout(rows: list[_Row], block: Block) -> _Layout:
    if not rows:
        return _Layout()
    needed = max(1.0, len(rows) * _COLUMN_SHARE)
    spots = [
        block.fraction(word.cx)
        for row in rows
        if row.price[0]
        and read_vat_code((word := row.line.words[row.price[0] - 1]).text) is not None
    ]
    return _Layout(
        leading_quantity=sum(_leading(r) is not None for r in rows) >= needed,
        trailing_quantity=sum(_trailing(r) is not None for r in rows) >= needed,
        # One letter-shaped word is not a column.
        vat_column=sorted(spots)[len(spots) // 2] if len(spots) >= 2 else None,
    )


def _quantity(row: _Row, layout: _Layout) -> tuple[Decimal, int | None, str, set[int]]:
    """Work out (quantity, unit price, source, words used) for one row.

    A quantity and a unit price printed on the row prove themselves by
    multiplying to its price. A bare number counts only where the receipt has a
    quantity column and it divides the price.
    """
    words = row.line.words
    total_index, _last, total = row.price
    units = [m for m in row.money if m[1] < total_index]
    for first, last, unit in reversed(units):
        if unit <= 0:
            continue
        for index in range(total_index):
            quantity = parse_quantity(words[index].text)
            if (
                quantity is not None
                and not first <= index <= last
                and quantity <= MAX_QUANTITY
                and abs(times(quantity, unit) - total) <= 1
            ):
                return quantity, unit, "parsed", {index, *range(first, last + 1)}

    # A unit price beside the total, its quantity unreadable: divide to get it.
    if units and units[-1][1] == total_index - 1 and units[-1][2] > 0 and total > 0:
        first, last, unit = units[-1]
        quantity = (Decimal(total) / unit).quantize(Decimal("0.001"))
        if quantity <= MAX_QUANTITY:
            return quantity, unit, "reconstructed", set(range(first, last + 1))

    for wanted, read, index in (
        (layout.leading_quantity, _leading, 0),
        (layout.trailing_quantity, _trailing, total_index - 1),
    ):
        quantity = read(row)
        # '2' is a quantity only in a quantity column; 'x2' says so itself.
        marked = quantity is not None and not words[index].text.isdigit()
        if quantity is not None and (wanted or marked) and total % int(quantity) == 0:
            return quantity, total // int(quantity), "parsed", {index}
    return Decimal(1), total, "implicit", set()


def _vat(row: _Row, block: Block, layout: _Layout) -> tuple[int, str | None]:
    """Return where the description ends, and the tax letter if one was read."""
    words, total_index = row.line.words, row.price[0]
    if layout.vat_column is None:
        # No column to go by: only a marked letter such as '*a', left of the price.
        text = words[total_index - 1].text if total_index else ""
        code = None if text.isalnum() else read_vat_code(text)
        return (total_index - 1 if code else total_index), code
    # From the tax-letter column rightwards there is only the letter and crease noise.
    end = next(
        (
            i
            for i, word in enumerate(words[:total_index])
            if block.fraction(word.cx) >= layout.vat_column - _VAT_MARGIN
        ),
        total_index,
    )
    for word in words[end:total_index]:
        code = read_vat_code(word.text)
        if code and abs(block.fraction(word.cx) - layout.vat_column) <= _VAT_MARGIN:
            return end, code
    return end, None


def _item(
    row: _Row, block: Block, layout: _Layout, beside: Line | None
) -> LineItem | None:
    """Build the item for one priced row, named by the line beside it if nameless."""
    line = row.line
    quantity, unit, source, used = _quantity(row, layout)
    end, vat_code = _vat(row, block, layout)
    named = strip_noise(
        [
            w
            for i, w in enumerate(line.words[:end])
            # The 'x' and '@' of '2 x @ 12.000' go with the numbers.
            if i not in used and not (used and w.text in {"x", "X", "@", "*"})
        ]
    )
    borrowed = (
        not any(c.isalpha() for w in named for c in w.text) and beside is not None
    )
    if borrowed:
        named = strip_noise(beside.words)
    if not named:
        return None
    return LineItem(
        description=" ".join(w.text for w in named),
        quantity=quantity,
        quantity_source=source,
        unit_price_minor=unit,
        line_total_minor=row.price[2],
        vat_code=vat_code,
        confidence=description_confidence(named),
        line_index=line.index,
        bbox=(line.left, line.top, line.right - line.left, line.bottom - line.top),
        borrowed_name=borrowed,
    )


def _modifier(line: Line, fmt: Format) -> tuple[Decimal, int] | None:
    """Read a quantity line such as '8 x 2,99': (quantity, unit price)."""
    found = _MODIFIER.fullmatch(line.text)
    if not found:
        return None
    quantity = parse_quantity(found.group(1))
    unit = parse_amount(found.group(2), fmt)
    if quantity is None or unit is None or unit <= 0:
        return None
    return quantity, unit


def _attach(
    modifiers: list[tuple[int, Decimal, int]],
    items: list[LineItem],
    warnings: list[str],
) -> None:
    """Give each quantity line to the neighbouring item its arithmetic fits.

    The item printed below is tried first, then the one above.
    """
    for line_index, quantity, unit in modifiers:
        below = next((i for i in items if i.line_index > line_index), None)
        above = next((i for i in reversed(items) if i.line_index < line_index), None)
        for item in (below, above):
            if (
                item is not None
                and abs(times(quantity, unit) - item.line_total_minor) <= 1
            ):
                item.quantity, item.unit_price_minor = quantity, unit
                item.quantity_source = "modifier"
                break
        else:
            warnings.append(f"modifier_unattached:line{line_index}")


def _footer(lines: list[Line], fmt: Format, out: Extraction) -> None:
    """Read the subtotal, the charges, the total, the payments and the change."""
    labelled = [
        (kind, money[-1][2], line)
        for line in lines
        if (kind := role(line.text)) and (money := money_tokens(line, fmt))
    ]
    # The total is the last one printed before the first payment.
    paid = next(
        (
            i
            for i, (kind, _a, _l) in enumerate(labelled)
            if kind in {"payment", "change"}
        ),
        len(labelled),
    )
    totals = [i for i, (kind, _a, _l) in enumerate(labelled[:paid]) if kind == "total"]
    final = totals[-1] if totals else None

    for index, (kind, amount, line) in enumerate(labelled):
        before_total = final is None or index < final
        if kind == "total" and index == final:
            out.printed_total_minor = abs(amount)
        elif (
            kind in {"subtotal", "total"}
            and before_total
            and out.subtotal_minor is None
        ):
            # 'TOTAL', then rounding, then 'TOTAL ROUNDED': the first is a subtotal.
            out.subtotal_minor = abs(amount)
        elif kind in _CHARGES and before_total:
            signed = -abs(amount) if kind == "discount" else amount
            out.charges.append(Charge(kind, line.text, signed, line.index))
        elif kind == "tax":
            # Tax printed under the total is a breakdown of it, never added.
            out.tax_breakdown_minor += abs(amount)
        elif kind == "payment" and index >= paid:
            out.payments_minor.append(abs(amount))
        elif kind == "change":
            out.change_minor = abs(amount)


def _without_currency(line: Line, fmt: Format) -> Line:
    """Drop a currency mark printed as its own word before an amount: 'Rp. 39.000'."""
    words = [
        word
        for word, after in zip(line.words, [*line.words[1:], None], strict=True)
        if not (
            after is not None
            and _CURRENCY_MARK.fullmatch(word.text)
            and parse_amount(after.text, fmt) is not None
        )
    ]
    return Line(words, line.index)


def _in_column(text: str, fmt: Format) -> bool:
    """Whether a word can mark the price column. Whole-number currencies print
    quantities that look like prices, so there only three digits or more count.
    """
    amount = parse_amount(text, fmt)
    return amount is not None and (fmt.places > 0 or abs(amount) >= 100)


def _split(
    lines: list[Line], fmt: Format, block: Block, column: tuple[float, float] | None
) -> tuple[list[_Row], list[Line], int]:
    """Split the receipt into priced rows, the lines between them, and the footer.

    The items run up to the first footer label, or the first tall line: a total
    is printed taller than the items even when its label is misread.
    """
    glyph = median_glyph_height([w for line in lines for w in line.words])
    rows: list[_Row] = []
    unpriced: list[Line] = []
    for position, line in enumerate(lines):
        # The time beside a date, '12:39', reads as a price with a misread comma.
        if _DATE.search(line.text):
            continue
        money = money_tokens(line, fmt)
        price = price_of(line, money, block, column)
        kind = role(line.text)
        tall = median_glyph_height(line.words) > glyph * _TALL_LINE_RATIO
        if (price and kind in _FOOTER_START) or (
            rows and (kind in _FOOTER_START or kind == "count" or (price and tall))
        ):
            return rows, unpriced, position
        if price is None or _modifier(line, fmt):
            unpriced.append(line)
        else:
            rows.append(_Row(line, money, price))
    return rows, unpriced, len(lines)


def _read_rows(
    rows: list[_Row], unpriced: list[Line], block: Block, out: Extraction
) -> None:
    """Turn the priced rows into items and discounts, and note the lines left over."""
    layout = _layout(rows, block)
    out.vat_column = layout.vat_column
    first_row = rows[0].line.index if rows else 0
    by_index = {line.index: line for line in unpriced}
    modifiers = [
        (line.index, *modifier)
        for line in unpriced
        if line.index >= first_row - 1 and (modifier := _modifier(line, out.fmt))
    ]
    taken = {index for index, _quantity, _unit in modifiers}

    for row in rows:
        line, amount = row.line, row.price[2]
        if amount < 0 or role(line.text) == "discount":
            label = name_words(line.words[: row.money[0][0]], block, layout.vat_column)
            # A discount whose minus sign was lost is still a discount.
            out.adjustments.append(
                Adjustment(" ".join(w.text for w in label), -abs(amount), line.index)
            )
            continue
        # 'Populaire Chocolate' on one line, '1 4,500 4,500' on the next, or the
        # other way round: a row with no name takes the free line beside it.
        beside = next(
            (
                by_index[i]
                for i in (line.index - 1, line.index + 1)
                if i in by_index and i not in taken
            ),
            None,
        )
        item = _item(row, block, layout, beside)
        if item is None:
            continue
        if item.borrowed_name:
            taken.add(beside.index)
        out.items.append(item)

    _attach(modifiers, out.items, out.warnings)
    # A line of column headings says where the items start, even when the first
    # of them lost their prices.
    start = max(
        (l.index for l in unpriced if l.index < first_row and is_heading(l.text)),
        default=first_row,
    )
    # Lines with a name and no price get a second reading, and a mention in review.
    out.skipped_lines = [
        line.index
        for line in unpriced
        if line.index > start and line.index not in taken and len(line.words) > 1
    ]


def extract(lines: list[Line]) -> Extraction:
    """Run extraction over a whole receipt."""
    fmt = detect_format(w.text for line in lines for w in line.words)
    out = Extraction(fmt)
    lines = [_without_currency(line, fmt) for line in lines]
    block = text_block(lines)
    column = discover_price_column(lines, lambda text: _in_column(text, fmt))

    rows, unpriced, end = _split(lines, fmt, block, column)
    _read_rows(rows, unpriced, block, out)
    _footer(lines[end:], fmt, out)

    if column is None:
        out.warnings.append("price_column_not_found")
    if not out.items:
        out.warnings.append("no_items_extracted")
    if out.printed_total_minor is None and not out.payments_minor:
        out.warnings.append("printed_total_not_found")
    return out

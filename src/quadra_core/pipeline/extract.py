"""Turn lines of words into receipt items, by column position rather than spacing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from quadra_core.pipeline.lines import (
    Block,
    Line,
    Word,
    discover_price_column,
    median_glyph_height,
    text_block,
)
from quadra_core.pipeline.money import (
    TRAILING_SIGN,
    normalize_separators,
    parse_money,
    parse_quantity,
    reconstruct_quantity,
    strip_sign,
)
from quadra_core.profiles.loader import Profile, fuzzy_contains


@dataclass(slots=True)
class LineItem:
    """One product line off the receipt."""

    description_raw: str
    description: str
    quantity: Decimal | None
    quantity_source: str  # "parsed" | "reconstructed" | "modifier" | "implicit"
    unit_price_minor: int | None
    line_total_minor: int | None
    vat_code: str | None
    confidence: float
    line_index: int
    bbox: tuple[int, int, int, int]
    flags: list[str] = field(default_factory=list)
    # The price the first reading gave, when the re-read replaced it.
    first_read_minor: int | None = None


ITEM_CANDIDATE_RULES = frozenset({"item", "unknown", "discount"})


@dataclass(slots=True)
class Adjustment:
    """A discount or similar amount, kept apart from the items."""

    kind: str
    label: str
    amount_minor: int
    line_index: int
    # Set when the re-read suggested the amount but could not be sure of it.
    unconfirmed: bool = False


@dataclass(slots=True)
class Modifier:
    """A quantity line like '8 x 2,99', matched to its item by arithmetic."""

    quantity: Decimal
    unit_price_minor: int
    line_index: int

    def extends_to(self, line_total_minor: int | None) -> bool:
        """Whether quantity times unit price equals this line total."""
        if line_total_minor is None:
            return False
        product = (self.quantity * Decimal(self.unit_price_minor)).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
        return abs(int(product) - line_total_minor) <= 1


@dataclass(slots=True)
class Extraction:
    """Everything pulled off one receipt."""

    items: list[LineItem]
    adjustments: list[Adjustment]
    payments_minor: list[int]
    change_minor: int
    printed_total_minor: int | None
    # Set when a person typed the total in; nothing downstream may overrule it.
    printed_total_supplied: bool = False
    warnings: list[str] = field(default_factory=list)
    # Lines that looked like products but had no readable price.
    skipped_lines: list[int] = field(default_factory=list)
    # Line index -> price of nameless lines the re-read dropped as a repeated price.
    removed: dict[int, int] = field(default_factory=dict)
    # Where the VAT brackets sit, as a fraction of the text width, if found.
    vat_column: float | None = None


def find_item_region(lines: list[Line], profile: Profile) -> tuple[int, int]:
    """Return the (start, end) line range between the profile's anchors."""
    # Find the end first: Esselunga's start anchor 'EURO' also matches 'TOTALE EURO'.
    end = len(lines)
    for i, line in enumerate(lines):
        if any(
            fuzzy_contains(line.text, a, profile.min_ratio)
            for a in profile.items_end_before
        ):
            end = i
            break

    start = 0
    for i in range(end):
        if any(
            fuzzy_contains(lines[i].text, a, profile.min_ratio)
            for a in profile.items_start_after
        ):
            start = i + 1
            break

    if start < end:
        return (start, end)
    # The anchors contradict each other: drop the start one, still exclude the footer.
    return (0, end) if end else (0, len(lines))


# OCR often splits a price at the comma ('0,' '48-S'). Join halves closer than this
# many glyph heights, or the wider ratio when the split is at the separator itself.
_JOIN_GAP_RATIO = 0.6
_SEPARATOR_JOIN_GAP_RATIO = 1.0
_ENDS_IN_SEPARATOR = re.compile(r"\d[,.;:]$")
_STARTS_WITH_SEPARATOR = re.compile(r"^[,.;:]\d")

# How close to the price column an amount must sit: tight first, then loose enough
# for a curled roll but never far enough to reach the description.
_COLUMN_MARGIN = 0.03
_COLUMN_DRIFT = 0.2

# Creases and shadows come back as punctuation or stray letters around the name.
_EDGE_NOISE_CONF = 80.0
_TRAILING_NOISE_CONF = 40.0

_TALL_LINE_RATIO = 1.9

# How far from the VAT column a word may sit and still be read as the bracket.
_VAT_MARGIN = 0.06


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


def _trim(text: str) -> str:
    """Drop OCR junk from both ends of a number, keeping its sign."""
    text = text.strip()
    sign = ""
    tail = TRAILING_SIGN.search(text)
    if tail:
        sign = tail.group(0)
        text = text[: tail.start()]
    lead = "-" if text.startswith("-") else ""
    core = re.sub(r"^\D+", "", re.sub(r"\D+$", "", text))
    return f"{lead}{core}{sign}"


def parse_amount(text: str, profile: Profile) -> int | None:
    """Parse one candidate token, or None if it is not money in this profile."""
    token = normalize_separators(_trim(text), profile.decimal_separator)
    # Profile money patterns don't include the sign.
    if not profile.money_re.fullmatch(strip_sign(token)):
        return None
    return parse_money(token, places=profile.decimal_places)


def _join_ratio(word: Word, nxt: Word) -> float:
    """How wide a gap may be bridged between these two halves of a number."""
    at_separator = _ENDS_IN_SEPARATOR.search(
        word.text
    ) or _STARTS_WITH_SEPARATOR.search(nxt.text)
    return _SEPARATOR_JOIN_GAP_RATIO if at_separator else _JOIN_GAP_RATIO


def _money_tokens(line: Line, profile: Profile) -> list[tuple[int, int, int]]:
    """Return (first_word, last_word, cents) for every amount on the line.

    First and last differ when a number was split across two words.
    """
    out: list[tuple[int, int, int]] = []
    glyph = median_glyph_height(line.words)
    i = 0
    while i < len(line.words):
        word = line.words[i]
        minor = parse_amount(word.text, profile)
        if minor is not None:
            out.append((i, i, minor))
            i += 1
            continue

        nxt = line.words[i + 1] if i + 1 < len(line.words) else None
        if nxt is not None and nxt.left - word.right <= glyph * _join_ratio(word, nxt):
            joined = parse_amount(word.text + nxt.text, profile)
            if joined is not None:
                out.append((i, i + 1, joined))
                i += 2
                continue
        i += 1
    return out


def parse_modifier(line: Line, profile: Profile) -> Modifier | None:
    """Read a 'N x UNIT_PRICE' line, e.g. '8 x 2,99'."""
    if profile.modifier_re is None:
        return None
    match = profile.modifier_re.search(
        normalize_separators(line.text, profile.decimal_separator)
    )
    if not match:
        return None
    try:
        qty = parse_quantity(match.group("qty"), separator=profile.decimal_separator)
        unit = parse_money(match.group("unit"), places=profile.decimal_places)
    except (IndexError, KeyError):
        return None
    if qty is None or unit is None or unit <= 0:
        return None
    return Modifier(qty, unit, line.index)


@dataclass(slots=True)
class _Quantity:
    """How many, at what unit price, and where that left the description."""

    value: Decimal
    unit_minor: int | None
    source: str
    desc_end: int
    flags: list[str] = field(default_factory=list)


def _resolve_quantity(
    line: Line,
    profile: Profile,
    money: list[tuple[int, int, int]],
    *,
    total_idx: int,
    total_minor: int,
) -> _Quantity:
    """Work out the quantity for one line, from whichever source the shop gives."""
    first_money = min(first for first, _last, _v in money)
    quantity: Decimal | None = None
    unit_minor: int | None = None
    source = "implicit"
    desc_end = first_money
    flags: list[str] = []

    # The quantity column sits just left of the first amount.
    if profile.has_quantity_column and first_money > 0:
        candidate = parse_quantity(
            line.words[first_money - 1].text, separator=profile.decimal_separator
        )
        if candidate is not None and candidate <= profile.max_quantity:
            quantity, source, desc_end = candidate, "parsed", first_money - 1

    if profile.has_unit_price_column:
        earlier = [v for f, _l, v in money if f < total_idx]
        unit_minor = earlier[-1] if earlier else None
        derived = reconstruct_quantity(total_minor, unit_minor) if unit_minor else None
        if derived is not None:
            if quantity is not None and abs(quantity - derived) > Decimal("0.01"):
                # Prices survive a bad photo better than the quantity column.
                flags.append("quantity_disagreement")
            if quantity is None or flags:
                quantity, source = derived, "reconstructed"
    elif quantity is not None:
        # Derive the unit price, or the per-line check would flag every line.
        unit_minor = int(
            (Decimal(total_minor) / quantity).quantize(
                Decimal(1), rounding=ROUND_HALF_UP
            )
        )

    if quantity is None:
        # Nothing said otherwise: one of whatever it is, at the line total.
        quantity = Decimal(1)
        if not profile.has_unit_price_column:
            unit_minor = total_minor

    return _Quantity(quantity, unit_minor, source, desc_end, flags)


def read_vat_code(text: str, profile: Profile) -> str | None:
    """Return the VAT bracket a word spells, or None if it is not one."""
    if profile.vat_code_re is None:
        return None
    marker = profile.vat_code_re.fullmatch(text)
    if not marker:
        return None
    # With a capture group the code is the group; the rest is the OCR'd asterisk.
    found = marker.group(1) if marker.groups() else marker.group(0)
    return found.strip().lower() or None


def price_candidates(
    line: Line,
    money: list[tuple[int, int, int]],
    block: Block,
    price_column: tuple[float, float] | None,
) -> list[tuple[int, int, int]]:
    """Return the amounts that can be the line's price: in the column, else near it."""
    if price_column is None:
        return money
    low, high = price_column
    for margin in (_COLUMN_MARGIN, _COLUMN_DRIFT):
        near = [
            (first, last, value)
            for first, last, value in money
            if low - margin <= block.fraction(line.words[last].right) <= high + margin
        ]
        if near:
            return near
    return []


def name_words(words: list[Word], block: Block, vat_column: float | None) -> list[Word]:
    """Return the words that name a line: left of the VAT column, junk trimmed."""
    if vat_column is not None:
        words = [w for w in words if block.fraction(w.cx) < vat_column - _VAT_MARGIN]
    return strip_noise(words) if words else []


def find_vat_column(
    lines: list[Line],
    profile: Profile,
    block: Block,
    price_column: tuple[float, float] | None,
) -> float | None:
    """Where the VAT brackets sit, as a fraction of the block width, or None."""
    if profile.vat_code_re is None:
        return None
    spots = []
    for line in lines:
        priced = price_candidates(
            line, _money_tokens(line, profile), block, price_column
        )
        if not priced or not priced[-1][0]:
            continue
        word = line.words[priced[-1][0] - 1]
        if read_vat_code(word.text, profile) is not None:
            spots.append(block.fraction(word.cx))
    # One bracket-shaped word is not a column.
    if len(spots) < 2:
        return None
    return sorted(spots)[len(spots) // 2]


def _split_vat(
    line: Line,
    profile: Profile,
    block: Block,
    total_idx: int,
    vat_column: float | None,
) -> tuple[int, str | None]:
    """Return where the description ends, and the VAT code if one was read."""
    if vat_column is None:
        # No column to go by: the bracket is the word just left of the price.
        vat_code = (
            read_vat_code(line.words[total_idx - 1].text, profile)
            if total_idx
            else None
        )
        return (total_idx - 1 if vat_code else total_idx), vat_code

    # From the VAT column rightwards there is only the bracket and crease noise.
    desc_end = total_idx
    for index, word in enumerate(line.words[:total_idx]):
        if block.fraction(word.cx) >= vat_column - _VAT_MARGIN:
            desc_end = index
            break
    for word in line.words[desc_end:total_idx]:
        if abs(block.fraction(word.cx) - vat_column) <= _VAT_MARGIN:
            vat_code = read_vat_code(word.text, profile)
            if vat_code is not None:
                return desc_end, vat_code
    return desc_end, None


def description_confidence(words: list[Word]) -> float:
    """How sure the OCR is of a name: the mean over its words, 0 to 1.

    Not the minimum, since Tesseract scores correct abbreviations like 'ESSEL.X6' low.
    """
    return round(sum(w.conf for w in words) / len(words) / 100, 3)


def extract_item(
    line: Line,
    profile: Profile,
    block: Block,
    price_column: tuple[float, float] | None,
    *,
    money: list[tuple[int, int, int]] | None = None,
    vat_column: float | None = None,
) -> LineItem | None:
    """Pull one item out of a line, or None if the line holds no price."""
    money = _money_tokens(line, profile) if money is None else money
    priced = price_candidates(line, money, block, price_column)
    if not priced:
        return None

    total_idx, _total_last, total_minor = priced[-1]
    resolved = _resolve_quantity(
        line, profile, money, total_idx=total_idx, total_minor=total_minor
    )
    vat_end, vat_code = _split_vat(line, profile, block, total_idx, vat_column)
    desc_end = min(resolved.desc_end, vat_end)

    desc_words = strip_noise(line.words[:desc_end])
    description_raw = " ".join(w.text for w in desc_words).strip()
    if not description_raw:
        return None

    return LineItem(
        description_raw=description_raw,
        description=" ".join(description_raw.split()),
        quantity=resolved.value,
        quantity_source=resolved.source,
        unit_price_minor=resolved.unit_minor,
        line_total_minor=total_minor,
        vat_code=vat_code,
        confidence=description_confidence(desc_words),
        line_index=line.index,
        bbox=(line.left, line.top, line.right - line.left, line.bottom - line.top),
        flags=resolved.flags,
    )


def _attach(
    modifier: Modifier, items: list[LineItem], profile: Profile, warnings: list[str]
) -> None:
    """Give a modifier's quantity to the neighbouring item its arithmetic fits.

    The profile's position only breaks a tie: on one receipt the same 23,92 sits
    on both sides of an '8 x 2,99'.
    """
    before = [i for i in items if i.line_index < modifier.line_index]
    after = [i for i in items if i.line_index > modifier.line_index]
    neighbours = [after[0] if after else None, before[-1] if before else None]
    if profile.modifier_position == "after":
        neighbours.reverse()

    for item in neighbours:
        if item is not None and modifier.extends_to(item.line_total_minor):
            item.quantity = modifier.quantity
            item.unit_price_minor = modifier.unit_price_minor
            item.quantity_source = "modifier"
            return
    warnings.append(f"modifier_unattached:line{modifier.line_index}")


def _scan_footer(
    lines: list[Line], profile: Profile
) -> tuple[int | None, list[int], int]:
    """Read the printed total, the amounts tendered, and any change given.

    The whole page is scanned: when the printed total is unreadable the item region
    runs to the last line, and the payments below it would be out of reach.
    """
    printed_total: int | None = None
    payments: list[int] = []
    change = 0

    for line in lines:
        rule = profile.rule_for(line.text)
        if rule not in {"total", "payment", "change"}:
            continue
        money = _money_tokens(line, profile)
        if not money:
            continue
        if rule == "total":
            printed_total = money[-1][2]
        elif rule == "payment":
            payments.append(money[-1][2])
        else:
            change = money[-1][2]

    return printed_total, payments, change


def _report(
    column: tuple[float, float] | None,
    items: list[LineItem],
    printed_total: int | None,
    payments: list[int],
) -> list[str]:
    """Say what the receipt did not yield, so nothing goes missing in silence."""
    warnings = []
    if column is None:
        warnings.append("price_column_not_found")
    if not items:
        warnings.append("no_items_extracted")
    if printed_total is None and not payments:
        warnings.append("printed_total_not_found")
    return warnings


def extract(lines: list[Line], profile: Profile) -> Extraction:
    """Run extraction over a whole receipt."""
    block = text_block(lines)
    column = discover_price_column(lines, profile.money_re)
    start, end = find_item_region(lines, profile)
    vat_column = find_vat_column(lines[start:end], profile, block, column)

    items: list[LineItem] = []
    adjustments: list[Adjustment] = []
    modifiers: list[Modifier] = []
    warnings: list[str] = []
    skipped: list[int] = []

    glyph = median_glyph_height([w for line in lines for w in line.words])
    for line in lines[start:end]:
        rule = profile.rule_for(line.text)

        # An allow list, so a new profile rule never turns its lines into items.
        if rule not in ITEM_CANDIDATE_RULES:
            continue
        # A misread 'TOTALE EURO' matches no rule, but is still printed tall.
        if median_glyph_height(line.words) > glyph * _TALL_LINE_RATIO:
            continue

        modifier = parse_modifier(line, profile)
        if modifier is not None:
            modifiers.append(modifier)
            continue

        money = _money_tokens(line, profile)
        priced = price_candidates(line, money, block, column)
        if not priced:
            # No readable price. Lines with a description get a second reading.
            if len(line.words) > 1:
                skipped.append(line.index)
            continue

        amount = priced[-1][2]
        if amount < 0 or rule == "discount":
            first = min(f for f, _l, _v in money)
            label = " ".join(
                w.text for w in name_words(line.words[:first], block, vat_column)
            )
            # A discount whose minus sign was lost is still a discount.
            adjustments.append(Adjustment("discount", label, -abs(amount), line.index))
            continue

        item = extract_item(
            line, profile, block, column, money=money, vat_column=vat_column
        )
        if item is not None:
            items.append(item)

    for modifier in modifiers:
        _attach(modifier, items, profile, warnings)

    printed_total, payments, change = _scan_footer(lines, profile)
    warnings += _report(column, items, printed_total, payments)

    return Extraction(
        items,
        adjustments,
        payments,
        change,
        printed_total,
        warnings=warnings,
        skipped_lines=skipped,
        vat_column=vat_column,
    )

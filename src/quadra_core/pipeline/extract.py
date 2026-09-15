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
    quantity_source: str  # "parsed" | "reconstructed" | "assumed" | "missing"
    unit_price_minor: int | None
    line_total_minor: int | None
    vat_code: str | None
    confidence: float
    line_index: int
    bbox: tuple[int, int, int, int]
    flags: list[str] = field(default_factory=list)


ITEM_CANDIDATE_RULES = frozenset({"item", "unknown", "discount"})


@dataclass(slots=True)
class Adjustment:
    """A discount or similar amount, kept apart from the items."""

    kind: str
    label: str
    amount_minor: int
    line_index: int


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
    item_region: tuple[int, int]
    price_column: tuple[float, float] | None
    # Set when a person typed the total in; nothing downstream may overrule it.
    printed_total_supplied: bool = False
    warnings: list[str] = field(default_factory=list)
    # Lines that looked like products but had no readable price.
    skipped_lines: list[int] = field(default_factory=list)


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


# OCR often splits a price at the comma ('0,' '48-S'). Join halves closer than
# this many glyph heights, or this many when the split is at the separator itself.
_JOIN_GAP_RATIO = 0.6
_SEPARATOR_JOIN_GAP_RATIO = 1.0
_ENDS_IN_SEPARATOR = re.compile(r"\d[,.;:]$")
_STARTS_WITH_SEPARATOR = re.compile(r"^[,.;:]\d")

# How far outside the price column, as a fraction of text width, a price may sit.
_COLUMN_DRIFT = 0.2

# Esselunga writes a negative as '7,20-S'.
_SIGN_TAIL = re.compile(r"-\s*[^\d\s]?\s*$")

# Shadows at the paper's edge come back as low-confidence one-letter words.
_EDGE_NOISE_CONF = 60.0


def _strip_leading_noise(words: list[Word]) -> list[Word]:
    """Drop single-character low-confidence words from the front of a description."""
    first = 0
    while (
        first < len(words)
        and len(words[first].text) == 1
        and words[first].conf < _EDGE_NOISE_CONF
    ):
        first += 1
    return words[first:] or words


def _trim(text: str) -> str:
    """Drop OCR junk from both ends of a number, keeping its sign."""
    text = text.strip()
    sign = ""
    tail = _SIGN_TAIL.search(text)
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


def _money_tokens(line: Line, profile: Profile) -> list[tuple[int, int, int]]:
    """Return (first_word, last_word, cents) for every amount on the line.

    First and last differ when a number was split across two words.
    """
    out: list[tuple[int, int, int]] = []
    limit = median_glyph_height(line.words) * _JOIN_GAP_RATIO
    i = 0
    while i < len(line.words):
        word = line.words[i]
        minor = parse_amount(word.text, profile)
        if minor is not None:
            out.append((i, i, minor))
            i += 1
            continue

        if i + 1 < len(line.words):
            nxt = line.words[i + 1]
            at_separator = _ENDS_IN_SEPARATOR.search(
                word.text
            ) or _STARTS_WITH_SEPARATOR.search(nxt.text)
            gap_limit = (
                median_glyph_height(line.words) * _SEPARATOR_JOIN_GAP_RATIO
                if at_separator
                else limit
            )
            if nxt.left - word.right <= gap_limit:
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
    modifier: Modifier | None,
) -> _Quantity:
    """Work out the quantity for one line, from whichever source the shop gives."""
    first_money = min(first for first, _last, _v in money)
    if modifier is not None:
        return _Quantity(
            modifier.quantity, modifier.unit_price_minor, "modifier", first_money
        )

    quantity: Decimal | None = None
    unit_minor: int | None = None
    source = "missing"
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
        earlier = [(f, v) for f, _l, v in money if f < total_idx]
        if earlier:
            _, unit_minor = earlier[-1]
        if unit_minor:
            derived = reconstruct_quantity(total_minor, unit_minor)
            if derived is not None:
                if quantity is None:
                    quantity, source = derived, "reconstructed"
                elif abs(quantity - derived) > Decimal("0.01"):
                    # Prices survive a bad photo better than the quantity column.
                    flags.append("quantity_disagreement")
                    quantity, source = derived, "reconstructed"
    elif quantity is not None and quantity > 0:
        # Derive the unit price, or the per-line check would flag every line.
        unit_minor = int(
            (Decimal(total_minor) / quantity).quantize(
                Decimal(1), rounding=ROUND_HALF_UP
            )
        )

    if quantity is None:
        quantity, source = Decimal(1), "implicit"
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

    def within(margin: float) -> list[tuple[int, int, int]]:
        return [
            (first, last, v)
            for first, last, v in money
            if low - margin <= block.fraction(line.words[last].right) <= high + margin
        ]

    return within(0.03) or within(_COLUMN_DRIFT)


def extract_item(
    line: Line,
    profile: Profile,
    block: Block,
    price_column: tuple[float, float] | None,
    modifier: Modifier | None = None,
    *,
    money: list[tuple[int, int, int]] | None = None,
) -> LineItem | None:
    """Pull one item out of a line, or None if the line holds no price."""
    money = _money_tokens(line, profile) if money is None else money
    in_column = price_candidates(line, money, block, price_column)
    if not in_column:
        return None

    total_idx, _total_last, total_minor = in_column[-1]
    vat_code: str | None = None

    resolved = _resolve_quantity(
        line,
        profile,
        money,
        total_idx=total_idx,
        total_minor=total_minor,
        modifier=modifier,
    )
    quantity = resolved.value
    unit_minor = resolved.unit_minor
    source = resolved.source
    desc_end = resolved.desc_end
    flags = resolved.flags

    # The VAT bracket sits just left of the price; keep it out of the description.
    if total_idx > 0:
        vat_code = read_vat_code(line.words[total_idx - 1].text, profile)
        if vat_code is not None:
            desc_end = min(desc_end, total_idx - 1)

    desc_words = _strip_leading_noise(line.words[:desc_end])
    description_raw = " ".join(w.text for w in desc_words).strip()
    if not description_raw:
        return None

    # One bad word makes the whole description suspect.
    confidence = round(min(w.conf for w in desc_words) / 100, 3) if desc_words else 0.0

    return LineItem(
        description_raw=description_raw,
        description=" ".join(description_raw.split()),
        quantity=quantity,
        quantity_source=source,
        unit_price_minor=unit_minor,
        line_total_minor=total_minor,
        vat_code=vat_code,
        confidence=confidence,
        line_index=line.index,
        bbox=(line.left, line.top, line.right - line.left, line.bottom - line.top),
        flags=flags,
    )


def _attach_backwards(
    modifier: Modifier, items: list[LineItem], warnings: list[str]
) -> None:
    """Apply a modifier to the previous item, or warn that it went unused."""
    if items and modifier.extends_to(items[-1].line_total_minor):
        previous = items[-1]
        previous.quantity = modifier.quantity
        previous.unit_price_minor = modifier.unit_price_minor
        previous.quantity_source = "modifier"
    else:
        warnings.append(f"modifier_unattached:line{modifier.line_index}")


def _scan_footer(
    lines: list[Line], profile: Profile
) -> tuple[int | None, list[int], int]:
    """Read the printed total, the amounts tendered, and any change given."""
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


def extract(lines: list[Line], profile: Profile) -> Extraction:
    """Run extraction over a whole receipt."""
    block = text_block(lines)
    column = discover_price_column(lines, profile.money_re)
    start, end = find_item_region(lines, profile)

    items: list[LineItem] = []
    adjustments: list[Adjustment] = []
    warnings: list[str] = []
    skipped: list[int] = []
    pending: Modifier | None = None

    for line in lines[start:end]:
        rule = profile.rule_for(line.text)

        # An allow list, so a new profile rule never turns its lines into items.
        if rule not in ITEM_CANDIDATE_RULES:
            continue

        modifier = parse_modifier(line, profile)
        if modifier is not None:
            if profile.modifier_position == "after":
                _attach_backwards(modifier, items, warnings)
            else:
                pending = modifier
            continue

        money = _money_tokens(line, profile)
        if not money or not price_candidates(line, money, block, column):
            # No readable price. Lines with a description get a second reading.
            if len(line.words) > 1:
                skipped.append(line.index)
            continue

        amount = money[-1][2]
        if amount < 0 or rule == "discount":
            first = min(f for f, _l, _v in money)
            label = " ".join(w.text for w in line.words[:first])
            adjustments.append(
                Adjustment("discount", label.strip(), amount, line.index)
            )
            if pending is not None:
                _attach_backwards(pending, items, warnings)
                pending = None
            continue

        attach = pending if pending is not None and pending.extends_to(amount) else None
        item = extract_item(line, profile, block, column, attach, money=money)

        if pending is not None and attach is None:
            _attach_backwards(pending, items, warnings)
        pending = None

        if item is not None:
            items.append(item)

    if pending is not None:
        _attach_backwards(pending, items, warnings)

    printed_total, payments, change = _scan_footer(lines, profile)

    if column is None:
        warnings.append("price_column_not_found")
    if not items:
        warnings.append("no_items_extracted")
    if printed_total is None and not payments:
        warnings.append("printed_total_not_found")

    return Extraction(
        items,
        adjustments,
        payments,
        change,
        printed_total,
        (start, end),
        column,
        warnings=warnings,
        skipped_lines=skipped,
    )

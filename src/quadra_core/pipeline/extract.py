"""Turn lines of words into receipt items.

Uses column position, not whitespace. On a bad photo the gap between description
and price fills with junk, so splitting on runs of spaces breaks down. The price
column holds up.
"""

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


# Rules whose lines can still turn into an item or an adjustment. Anything else the
# profile names is skipped.
ITEM_CANDIDATE_RULES = frozenset({"item", "unknown", "discount"})


@dataclass(slots=True)
class Adjustment:
    """A discount or similar amount that is not a product.

    Kept apart from the items rather than folded into a price, so the data still
    matches the paper. A discount is a fact about the transaction, not a cheaper jar
    of olives.
    """

    kind: str
    label: str
    amount_minor: int
    line_index: int


@dataclass(slots=True)
class Modifier:
    """A quantity line like '8 x 2,99'.

    Which item it belongs to is decided by arithmetic, not position. By eye it is
    ambiguous: on one receipt it sits between two products and could belong to
    either, but 8 x 2,99 = 23,92 settles it. The profile's `modifier_position` is
    only a tiebreaker for when the arithmetic cannot decide.
    """

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
    # True when a person typed the total in because OCR could not read it. That
    # is the best evidence there is, so nothing downstream may overrule it.
    printed_total_supplied: bool = False
    warnings: list[str] = field(default_factory=list)
    # Lines that looked like products but had no readable price. Kept so the
    # second reading can try again, since a dropped line takes its money with it.
    skipped_lines: list[int] = field(default_factory=list)


def find_item_region(lines: list[Line], profile: Profile) -> tuple[int, int]:
    """Find the block of items, between the anchors the profile names.

    Anchors are matched loosely because OCR mangles the all-caps header text they come
    from. If none match we fall back to the whole receipt: too wide a region gives
    items that fail classification, which is recoverable, while an empty one loses the
    receipt entirely.

    The end anchor is found first, and it keeps the line it matched. Esselunga's
    start anchor is the word EURO, printed over the price column, and its end
    anchor is TOTALE EURO - so the start anchor matches the end anchor's own line.
    That does no harm while the column header survives OCR, because the header
    comes first. On a receipt where it did not survive, the first EURO on the page
    was the one in TOTALE EURO, the items began after the total, and a receipt with
    eight perfectly readable products yielded none at all. Items cannot follow the
    total, so the search for where they start has no business past it.
    """
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
    # The anchors contradict each other. Keeping the footer out still beats
    # reading the whole page, so only the start anchor is dropped.
    return (0, end) if end else (0, len(lines))


# OCR sometimes splits one number in two, usually at the decimal comma, so
# '0,48-S' arrives as '0,' and '48-S'. On a real receipt the halves sat 10-14px
# apart while the narrowest genuine space was 20px, against a 25px median glyph
# height. Well under a space, so join them.
_JOIN_GAP_RATIO = 0.6

# A trailing sign, held aside while the number is cleaned up. Esselunga writes a
# negative as '7,20-S', so the letter after the dash belongs to the sign.
_SIGN_TAIL = re.compile(r"-\s*[^\d\s]?\s*$")


# OCR reads marks from the shadow along the paper's edge as words. They are full
# height, so drop_speckle cannot see them, but they come back far less confident
# than real text. Only stripped from the front of a description, which is where
# they land, and never so far that nothing is left.
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
    """Drop OCR's decoration from both ends of a number, keeping its sign.

    Listing the junk character by character does not work: a crease reads as a
    bracket, a speck as a degree sign, an edge as a backtick, and the list is never
    finished. A price starts and ends with a digit, so anything else on either end
    can go. On this receipt a stray ')' on '4,88)' was enough to drop the line and
    lose 4,88 from the total silently.
    """
    text = text.strip()
    sign = ""
    tail = _SIGN_TAIL.search(text)
    if tail:
        sign = tail.group(0)
        text = text[: tail.start()]
    # A leading minus is a real sign on receipts that write it that way.
    lead = "-" if text.startswith("-") else ""
    core = re.sub(r"^\D+", "", re.sub(r"\D+$", "", text))
    return f"{lead}{core}{sign}"


def parse_amount(text: str, profile: Profile) -> int | None:
    """Parse one candidate token, or None if it is not money in this profile."""
    token = normalize_separators(_trim(text), profile.decimal_separator)
    # Match on the unsigned core, then parse the signed token, so profiles do not
    # have to know about sign conventions.
    if not profile.money_re.fullmatch(strip_sign(token)):
        return None
    return parse_money(token, places=profile.decimal_places)


def _money_tokens(line: Line, profile: Profile) -> list[tuple[int, int, int]]:
    """(first_word, last_word, minor_units) for every amount on the line.

    First and last differ only where a number was split across two words. Both are
    needed: the description ends at the first, and whether the amount sits in the
    price column is judged from the right edge of the last.
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

        # Not money by itself. It may be half of one.
        if i + 1 < len(line.words):
            nxt = line.words[i + 1]
            if nxt.left - word.right <= limit:
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
        # A preceding 'N x UNIT' line is authoritative and checkable.
        return _Quantity(
            modifier.quantity, modifier.unit_price_minor, "modifier", first_money
        )

    quantity: Decimal | None = None
    unit_minor: int | None = None
    source = "missing"
    desc_end = first_money
    flags: list[str] = []

    # The quantity column sits immediately left of the first amount. Read it
    # whether or not the shop prints a unit price too: plenty print quantity and
    # line total only.
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
        # Quantity but no printed unit price, so derive it. Leaving it None would
        # make the per-line check flag every line on the receipt.
        unit_minor = int(
            (Decimal(total_minor) / quantity).quantize(
                Decimal(1), rounding=ROUND_HALF_UP
            )
        )

    if quantity is None:
        # Nothing said otherwise: one of whatever it is, at the line total.
        quantity, source = Decimal(1), "implicit"
        if not profile.has_unit_price_column:
            unit_minor = total_minor

    return _Quantity(quantity, unit_minor, source, desc_end, flags)


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
    if not money:
        return None

    # The line total is the rightmost money token in the price column. Falling
    # back to the rightmost token keeps single-column receipts working.
    in_column = money
    if price_column is not None:
        low, high = price_column
        margin = 0.03
        in_column = [
            (first, last, v)
            for first, last, v in money
            if low - margin <= block.fraction(line.words[last].right) <= high + margin
        ] or money

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

    # The VAT bracket sits in its own column just left of the price. Removing it
    # from the description matters more than recording it: the description is what
    # a budget groups by and what the accuracy figure measures, and every line was
    # dragging a stray '*c' or '«d' into both.
    if profile.vat_code_re is not None and total_idx > 0:
        marker = profile.vat_code_re.fullmatch(line.words[total_idx - 1].text)
        if marker:
            # If the profile captures a group, the code is that group. The rest
            # of the match is the mark in front, which OCR renders differently
            # every time and which is not worth storing.
            found = marker.group(1) if marker.groups() else marker.group(0)
            vat_code = found.strip().lower() or None
            desc_end = min(desc_end, total_idx - 1)

    desc_words = _strip_leading_noise(line.words[:desc_end])
    description_raw = " ".join(w.text for w in desc_words).strip()
    if not description_raw:
        return None

    # Lowest word confidence in the description. One bad glyph makes the whole
    # description suspect, so use the minimum rather than the mean.
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
    """Apply a modifier to the preceding item, or record that it went unused.

    Reached when a modifier did not extend the line that followed it. A store
    printing modifiers after their item would land here; so would a misread. Either
    way the modifier must not be silently dropped.
    """
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
    """Read the printed total, the amounts tendered, and any change given.

    These sit BELOW the item region, so they need their own pass. The payments are
    an independent statement of the total, which matters more than it sounds: on a
    real photograph 'TOTALE EURO 26,44' is set in a large bold face that Tesseract
    read as 'db, +' at every scale and threshold tried, while the small print of the
    payment lines came through cleanly and summed to exactly 26,44.
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

        # An allow list rather than a skip list. With a skip list any new rule a
        # profile adds falls through: a 'change' rule for 'RESTO 0,00' would have
        # become a zero-euro item.
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
        if not money:
            # Only worth revisiting if there is a description with it; a line of
            # pure noise is not a lost product.
            if len(line.words) > 1:
                skipped.append(line.index)
            continue

        # A negative amount is an adjustment, never a product. Esselunga signs these
        # with a trailing '-', so this depends on money parsing reading that suffix.
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

        # Attach the pending modifier only if the arithmetic actually works out.
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

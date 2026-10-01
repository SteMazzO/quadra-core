"""Re-read the price column when a receipt does not balance.

The arithmetic proposes; it never has the last word. A correction is taken as
certain only when the total was confirmed by two readings and every re-read of
the column agrees on the new price. Anything less is kept as a suggestion and the
line goes to review.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, NamedTuple

from quadra_core.pipeline.extract import (
    Adjustment,
    Extraction,
    LineItem,
    description_confidence,
    name_words,
    parse_amount,
    read_vat_code,
    strip_noise,
)
from quadra_core.pipeline.lines import (
    Block,
    Line,
    Word,
    cluster_1d,
    drop_speckle,
    group_lines,
    load_tsv,
    median_glyph_height,
    text_block,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..profiles.loader import Profile
    from .validate import Validation

# (scale, psm) per reading. Each misreads different prices, so they vote.
READINGS = ((3, 6), (2, 4), (1, 11))

# Margin around the column, so no digit is clipped.
PAD = 16

# Past this many disputed lines, too many combinations balance by coincidence.
MAX_DISPUTED = 16

# Lines on which the total may overrule a clear majority of readings. More than
# one lets a wrong answer balance by coincidence.
MAX_OVERRULED = 1

# A word taller than this many glyph heights is two rows read as one.
_MERGED_ROW_RATIO = 1.5

# A product name has at least two letters in a row.
_NAMED = re.compile(r"[^\W\d_]{2,}")

# How many re-reads must agree, and none disagree, for a correction to be certain.
CERTAIN_READINGS = 2

# Misreads seen in the price column on its own, where every row is a price.
# '1,20=S' for '1,20-S'; the dashes are en and em.
_SIGN_LIKE = re.compile("[-=~\u2013\u2014]" + r"\s*\S?\s*$")
_LOST_LEADING_ONE = re.compile(r"^[^\d,.]?(?=[,.]\d{2}$)")  # '»,98' for '1,98'
_ODD_SEPARATOR = re.compile(r"(?<=\d)\D(?=\d{2}$)")  # '1}20' for '1,20'
_NO_SEPARATOR = re.compile(r"\d{3}")  # '449' for '4,49'


def read_cell(text: str, profile: Profile) -> int | None:
    """Read one row of the price column, allowing for how thermal print misreads.

    Only used for the column re-read: every row there is a price, and the total
    decides between readings, so a wrong guess here is outvoted, not trusted.
    """
    amount = parse_amount(text, profile)
    if amount is not None or profile.decimal_places != 2:
        return amount

    negative = bool(_SIGN_LIKE.search(text))
    core = _SIGN_LIKE.sub("", text).strip()
    core = _LOST_LEADING_ONE.sub("1", core, count=1)
    core = _ODD_SEPARATOR.sub(profile.decimal_separator, core, count=1)
    core = core.lstrip(".,;:'\u2018\u2019")
    if _NO_SEPARATOR.fullmatch(core):
        core = f"{core[0]}{profile.decimal_separator}{core[1:]}"
    amount = parse_amount(core, profile)
    if amount is None:
        return None
    return -abs(amount) if negative else amount


@dataclass(slots=True)
class Repair:
    """What the second reading changed, and what it left alone."""

    # Keyed by line index, since apply() re-sorts the items.
    changes: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Line index -> amount for lines the first pass dropped; negative for a discount.
    recovered: dict[int, int] = field(default_factory=dict)
    # Line indexes whose new amount is certain; every other change needs review.
    certain: set[int] = field(default_factory=set)
    disputed: int = 0
    # Where a recovered line's description ends, in page pixels.
    column_left: int | None = None

    @property
    def applied(self) -> bool:
        """Whether the repair contains any changes."""
        return bool(self.changes) or bool(self.recovered)


class Column(NamedTuple):
    """Where the prices sit on the page, measured once per receipt."""

    # Line index -> the rightmost word of normal height, i.e. where its price is.
    prices: dict[int, Word]
    # Left edge of the price column in page pixels, or None if no line had a price.
    left: int | None
    glyph: float


def survey(lines: list[Line], extraction: Extraction) -> Column:
    """Locate the price of every item and dropped line, and the column they form."""
    by_index = {line.index: line for line in lines}
    glyph = median_glyph_height([w for line in lines for w in line.words])

    prices: dict[int, Word] = {}
    for index in [i.line_index for i in extraction.items] + extraction.skipped_lines:
        line = by_index.get(index)
        if not line or not line.words:
            continue
        # Words spanning two rows are skipped: a dropped line is usually dropped
        # because its price came back as one of those.
        sane = [w for w in line.words if w.height <= glyph * _MERGED_ROW_RATIO]
        prices[index] = max(sane or line.words, key=lambda w: w.right)

    # The left edge comes from the largest cluster, so one line offering only a
    # description word cannot widen the column to the whole page.
    lefts = [
        float(prices[i.line_index].left)
        for i in extraction.items
        if i.line_index in prices
    ]
    left = None
    if lefts:
        clusters = cluster_1d(lefts, gap=glyph or 1.0)
        left = int(min(max(clusters, key=lambda c: (len(c), c[0]))))
    return Column(prices, left, glyph)


def _column_rows(
    prepared: Any, left: int, right: int, scale: int, psm: int
) -> list[tuple[float, str]]:
    """One reading of the strip: (centre in page pixels, text) per row."""
    from PIL import Image  # noqa: PLC0415 - the parsing core runs without Pillow

    from .ocr import run  # noqa: PLC0415 - keeps Tesseract off the import path

    strip = prepared.crop((left, 0, right, prepared.height))
    if scale != 1:
        strip = strip.resize(
            (strip.width * scale, strip.height * scale), Image.Resampling.LANCZOS
        )
    rows = group_lines(drop_speckle(load_tsv(run(strip, psm=psm).tsv)))
    return [
        (
            sum(w.cy for w in row.words) / len(row.words) / scale,
            "".join(w.text for w in row.words),
        )
        for row in rows
    ]


def _assign(
    rows: list[tuple[float, str]], positions: dict[int, float], tolerance: float
) -> dict[int, str]:
    """Pair each strip row with its nearest line, keeping one row per line."""
    best: dict[int, tuple[float, str]] = {}
    for centre, text in rows:
        index, y = min(positions.items(), key=lambda p: abs(p[1] - centre))
        distance = abs(y - centre)
        if distance <= tolerance and (index not in best or distance < best[index][0]):
            best[index] = (distance, text)
    return {index: text for index, (_distance, text) in best.items()}


def second_opinion(
    prepared: Any, column: Column, profile: Profile
) -> dict[int, list[int]]:
    """Read the price column on its own, several ways: line index -> amounts."""
    if not column.prices:
        return {}

    left = max(0, (column.left or min(w.left for w in column.prices.values())) - PAD)
    right = min(prepared.width, max(w.right for w in column.prices.values()) + PAD)
    if right - left < 1:
        return {}

    positions = {index: word.cy for index, word in column.prices.items()}
    out: dict[int, list[int]] = {}
    for scale, psm in READINGS:
        rows = _column_rows(prepared, left, right, scale, psm)
        for index, text in _assign(rows, positions, column.glyph).items():
            amount = read_cell(text, profile)
            if amount is not None:
                out.setdefault(index, []).append(amount)
    return out


def _item_votes(item: LineItem, readings: list[int]) -> Counter[int]:
    """Return what this item's price might be: the first pass, plus each reading."""
    votes = Counter(readings)
    votes[item.line_total_minor or 0] += 1
    if not _NAMED.search(item.description):
        # Maybe a neighbour's price on a row of its own, so allow dropping it.
        votes.setdefault(0, 0)
    return votes


def _dropped_votes(
    line: Line | None, readings: list[int], profile: Profile
) -> Counter[int]:
    """Return what a dropped line might hold: nothing, or what the column read."""
    # A discount by its label, or by a minus sign in any reading.
    discount = any(amount < 0 for amount in readings) or (
        line is not None and profile.rule_for(line.text) == "discount"
    )
    votes = Counter(
        -abs(amount) if discount else amount
        for amount in readings
        if discount or amount > 0
    )
    if votes:
        votes.setdefault(0, 0)
    return votes


def _leader(votes: Counter[int]) -> int | None:
    """Return the amount with the most votes, or None when the top is shared."""
    top = max(votes.values())
    leaders = [amount for amount, count in votes.items() if count == top]
    return leaders[0] if len(leaders) == 1 else None


class _Cell(NamedTuple):
    """One running sum in _choose's table."""

    rank: tuple[int, int]  # (-lines overruled, votes), so higher is better
    ways: int  # how many ways reach that rank, counted up to 2
    previous: int  # the running sum this came from
    amount: int  # the amount taken to get here


def _choose(slots: list[Counter[int]], target: int) -> list[int] | None:
    """Pick one amount per slot summing to target, or None if none or tied.

    Prefers the fewest overruled majorities, then the most votes.
    """
    tables: list[dict[int, _Cell]] = []
    current = {0: _Cell((0, 0), 1, 0, 0)}
    for votes in slots:
        leader = _leader(votes)
        following: dict[int, _Cell] = {}
        for running, cell in current.items():
            for amount, count in votes.items():
                overruled = int(leader is not None and amount != leader)
                rank = (cell.rank[0] - overruled, cell.rank[1] + count)
                if -rank[0] > MAX_OVERRULED:
                    continue
                key = running + amount
                held = following.get(key)
                if held is None or rank > held.rank:
                    following[key] = _Cell(rank, cell.ways, running, amount)
                elif rank == held.rank:
                    following[key] = held._replace(ways=min(2, held.ways + cell.ways))
        tables.append(following)
        current = following

    end = current.get(target)
    if end is None or end.ways > 1:
        return None
    chosen = []
    running = target
    for table in reversed(tables):
        cell = table[running]
        chosen.append(cell.amount)
        running = cell.previous
    return chosen[::-1]


def _slots(
    extraction: Extraction,
    readings: dict[int, list[int]],
    lines: list[Line],
    profile: Profile,
) -> list[tuple[int, Counter[int]]]:
    """One slot of candidate amounts per item, and per line that lost its price."""
    by_index = {line.index: line for line in lines}
    slots = [
        (item.line_index, _item_votes(item, readings.get(item.line_index, [])))
        for item in extraction.items
    ]
    for index in extraction.skipped_lines:
        votes = _dropped_votes(by_index.get(index), readings.get(index, []), profile)
        if votes:
            slots.append((index, votes))
    return slots


def repair(
    extraction: Extraction,
    validation: Validation,
    prepared: Any,
    profile: Profile,
    lines: list[Line],
) -> Repair | None:
    """Reconcile the readings with the printed total, or None if unsure."""
    target = validation.printed_total_minor
    if target is None or validation.delta_minor == 0 or not extraction.items:
        return None
    # The printed total and the payments disagree: aiming at either could bend
    # correct prices to a misread total.
    if validation.total_disputed:
        return None

    column = survey(lines, extraction)
    readings = second_opinion(prepared, column, profile)
    if not readings:
        return None

    slots = _slots(extraction, readings, lines, profile)
    disputed = [(index, votes) for index, votes in slots if len(votes) > 1]
    if not disputed or len(disputed) > MAX_DISPUTED:
        return None

    settled = sum(next(iter(votes)) for _index, votes in slots if len(votes) == 1)
    adjustments = sum(a.amount_minor for a in extraction.adjustments)
    chosen = _choose(
        [votes for _index, votes in disputed], target - settled - adjustments
    )
    if chosen is None:
        return None

    result = Repair(disputed=len(disputed), column_left=column.left)
    _record(result, extraction, disputed, chosen, validation.total_confirmed)
    return result if result.applied else None


def _record(
    result: Repair,
    extraction: Extraction,
    disputed: list[tuple[int, Counter[int]]],
    chosen: list[int],
    total_confirmed: bool,
) -> None:
    """Write what the choice changed into the repair, and which of it is certain."""
    first = {item.line_index: item.line_total_minor for item in extraction.items}
    for (index, votes), amount in zip(disputed, chosen, strict=True):
        if index not in first:
            # A dropped line; zero means leave it dropped.
            if not amount:
                continue
            result.recovered[index] = amount
        elif amount != first[index]:
            result.changes[index] = (first[index], amount)
        else:
            continue
        if total_confirmed and _unanimous(votes, amount, index in first):
            result.certain.add(index)


def _unanimous(votes: Counter[int], amount: int, has_first_reading: bool) -> bool:
    """Whether every re-read of the column says `amount`, and enough of them do."""
    # Dropping a line is never read off the column, so it is never certain.
    if amount == 0:
        return False
    rereads = sum(votes.values()) - (1 if has_first_reading else 0)
    return votes[amount] >= CERTAIN_READINGS and votes[amount] == rereads


def _describing_words(
    line: Line,
    column_left: int | None,
    profile: Profile,
    block: Block,
    vat_column: float | None,
) -> tuple[list[Word], str | None]:
    """Return a recovered line's name, left of the price column, and its VAT code."""
    words = [
        w for w in line.words if column_left is None or w.cx < column_left
    ] or line.words
    if vat_column is not None:
        named = name_words(words, block, vat_column) or strip_noise(words)
        for word in words:
            if word not in named:
                vat_code = read_vat_code(word.text, profile)
                if vat_code is not None:
                    return named, vat_code
        return named, None
    if len(words) > 1:
        vat_code = read_vat_code(words[-1].text, profile)
        if vat_code is not None:
            return strip_noise(words[:-1]), vat_code
    return strip_noise(words), None


def _recovered_item(
    line: Line, words: list[Word], price: int, vat_code: str | None
) -> LineItem:
    """Build the item for a line the first pass dropped."""
    text = " ".join(w.text for w in words).strip()
    return LineItem(
        description_raw=text,
        description=" ".join(text.split()),
        quantity=Decimal(1),
        quantity_source="implicit",
        unit_price_minor=price,
        line_total_minor=price,
        vat_code=vat_code,
        confidence=description_confidence(words),
        line_index=line.index,
        bbox=(line.left, line.top, line.right - line.left, line.bottom - line.top),
        flags=["line_recovered"],
    )


def _recover(
    extraction: Extraction, result: Repair, lines: list[Line], profile: Profile
) -> None:
    """Add back the lines the first pass dropped, as items or as discounts."""
    by_index = {line.index: line for line in lines}
    block = text_block(lines)
    for line_index, amount in sorted(result.recovered.items()):
        line = by_index.get(line_index)
        if line is None:
            continue
        words, vat_code = _describing_words(
            line, result.column_left, profile, block, extraction.vat_column
        )
        if amount < 0:
            label = " ".join(" ".join(w.text for w in words).split())
            extraction.adjustments.append(
                Adjustment(
                    "discount",
                    label,
                    amount,
                    line_index,
                    unconfirmed=line_index not in result.certain,
                )
            )
        else:
            extraction.items.append(_recovered_item(line, words, amount, vat_code))

    extraction.items.sort(key=lambda i: i.line_index)
    extraction.adjustments.sort(key=lambda a: a.line_index)
    extraction.skipped_lines = [
        i for i in extraction.skipped_lines if i not in result.recovered
    ]


def apply(
    extraction: Extraction, result: Repair, lines: list[Line], profile: Profile
) -> None:
    """Write the agreed prices back, and flag everything that moved.

    A change that is not certain carries `price_unconfirmed`, which sends the line
    to review with the arithmetic's answer as a suggestion.
    """
    if result.recovered:
        _recover(extraction, result, lines, profile)
        for item in extraction.items:
            if item.line_index in result.recovered and (
                item.line_index not in result.certain
            ):
                item.flags.append("price_unconfirmed")

    by_item = {item.line_index: item for item in extraction.items}
    for line_index, (was, now) in result.changes.items():
        item = by_item.get(line_index)
        if item is None:
            continue
        if now == 0:
            # Only nameless lines are offered zero: a repeated price, not a product.
            extraction.items.remove(item)
            extraction.removed[line_index] = was
            continue
        item.line_total_minor = now
        item.first_read_minor = was
        # At quantity one the unit price is the line total, so move both.
        if item.quantity_source == "implicit":
            item.unit_price_minor = now
        if "price_reread" not in item.flags:
            item.flags.append("price_reread")
        if line_index not in result.certain and "price_unconfirmed" not in item.flags:
            item.flags.append("price_unconfirmed")

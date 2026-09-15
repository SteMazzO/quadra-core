"""Re-read the price column when a receipt does not balance."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quadra_core.pipeline.extract import (
    Adjustment,
    Extraction,
    LineItem,
    parse_amount,
    read_vat_code,
)
from quadra_core.pipeline.lines import (
    Line,
    Word,
    cluster_1d,
    drop_speckle,
    group_lines,
    load_tsv,
    median_glyph_height,
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


@dataclass(slots=True)
class Repair:
    """What the second reading changed, and what it left alone."""

    # Keyed by line index, since apply() re-sorts the items.
    changes: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Line index -> amount for lines the first pass dropped; negative for a discount.
    recovered: dict[int, int] = field(default_factory=dict)
    disputed: int = 0
    # Where a recovered line's description ends, in page pixels.
    column_left: int | None = None

    @property
    def applied(self) -> bool:
        """Whether the repair contains any changes."""
        return bool(self.changes) or bool(self.recovered)


def _price_words(lines: list[Line], extraction: Extraction) -> dict[int, Word]:
    """Return the rightmost normal-height word of each item and dropped line."""
    by_index = {line.index: line for line in lines}
    glyph = median_glyph_height([w for line in lines for w in line.words])
    wanted = [item.line_index for item in extraction.items]
    wanted += extraction.skipped_lines
    out = {}
    for index in wanted:
        line = by_index.get(index)
        if not line or not line.words:
            continue
        sane = [w for w in line.words if w.height <= glyph * _MERGED_ROW_RATIO]
        out[index] = max(sane or line.words, key=lambda w: w.right)
    return out


def _column_left(lines: list[Line], extraction: Extraction) -> int | None:
    """Return the price column's left edge in pixels, ignoring stray words."""
    words = _price_words(lines, extraction)
    lefts = [
        float(words[item.line_index].left)
        for item in extraction.items
        if item.line_index in words
    ]
    if not lefts:
        return None
    glyph = median_glyph_height([w for line in lines for w in line.words]) or 1.0
    largest = max(cluster_1d(lefts, gap=glyph), key=lambda c: (len(c), c[0]))
    return int(min(largest))


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
    prepared: Any, lines: list[Line], extraction: Extraction, profile: Profile
) -> dict[int, list[int]]:
    """Read the price column on its own, several ways: line index -> amounts."""
    words = _price_words(lines, extraction)
    if not words:
        return {}

    column_left = _column_left(lines, extraction)
    if column_left is None:
        column_left = min(w.left for w in words.values())
    left = max(0, column_left - PAD)
    right = min(prepared.width, max(w.right for w in words.values()) + PAD)
    if right - left < 1:
        return {}

    positions = {index: word.cy for index, word in words.items()}
    tolerance = median_glyph_height([w for line in lines for w in line.words])

    out: dict[int, list[int]] = {}
    for scale, psm in READINGS:
        rows = _column_rows(prepared, left, right, scale, psm)
        for index, text in _assign(rows, positions, tolerance).items():
            amount = parse_amount(text, profile)
            if amount is not None:
                out.setdefault(index, []).append(amount)
    return out


def _leader(votes: Counter[int]) -> int | None:
    """Return the amount with the most votes, or None when the top is shared."""
    top = max(votes.values())
    leaders = [amount for amount, count in votes.items() if count == top]
    return leaders[0] if len(leaders) == 1 else None


# A cell of _choose's table: rank, ways to reach it (capped at 2), previous sum,
# amount taken. Rank is (-lines overruled, votes), so higher is better.
_Entry = tuple[tuple[int, int], int, int, int]


def _choose(slots: list[Counter[int]], target: int) -> list[int] | None:
    """Pick one amount per slot summing to target, or None if none or tied.

    Prefers the fewest overruled majorities, then the most votes.
    """
    tables: list[dict[int, _Entry]] = []
    current: dict[int, _Entry] = {0: ((0, 0), 1, 0, 0)}
    for votes in slots:
        leader = _leader(votes)
        following: dict[int, _Entry] = {}
        for running, (rank, ways, _previous, _amount) in current.items():
            for amount, count in votes.items():
                overruled = int(leader is not None and amount != leader)
                candidate = (rank[0] - overruled, rank[1] + count)
                if -candidate[0] > MAX_OVERRULED:
                    continue
                key = running + amount
                held = following.get(key)
                if held is None or candidate > held[0]:
                    following[key] = (candidate, ways, running, amount)
                elif candidate == held[0]:
                    following[key] = (held[0], min(2, held[1] + ways), *held[2:])
        tables.append(following)
        current = following

    end = current.get(target)
    if end is None or end[1] > 1:
        return None
    chosen = []
    running = target
    for table in reversed(tables):
        _score, _ways, previous, amount = table[running]
        chosen.append(amount)
        running = previous
    return chosen[::-1]


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

    readings = second_opinion(prepared, lines, extraction, profile)
    if not readings:
        return None

    by_index = {line.index: line for line in lines}
    slots: list[tuple[int, Counter[int]]] = []
    for item in extraction.items:
        votes = Counter(readings.get(item.line_index, []))
        votes[item.line_total_minor or 0] += 1
        if not _NAMED.search(item.description):
            # Maybe a neighbour's price on a row of its own, so allow dropping it.
            votes.setdefault(0, 0)
        slots.append((item.line_index, votes))
    for line_index in extraction.skipped_lines:
        line = by_index.get(line_index)
        read = readings.get(line_index, [])
        # A discount by its label, or by a minus sign in any reading.
        discount = any(amount < 0 for amount in read) or (
            line is not None and profile.rule_for(line.text) == "discount"
        )
        votes = Counter(
            -abs(amount) if discount else amount
            for amount in read
            if discount or amount > 0
        )
        if votes:
            votes.setdefault(0, 0)
            slots.append((line_index, votes))

    disputed = [(index, votes) for index, votes in slots if len(votes) > 1]
    if not disputed or len(disputed) > MAX_DISPUTED:
        return None

    settled = sum(next(iter(votes)) for _index, votes in slots if len(votes) == 1)
    adjustments = sum(a.amount_minor for a in extraction.adjustments)
    remaining = target - settled - adjustments
    chosen = _choose([votes for _index, votes in disputed], remaining)
    if chosen is None:
        return None

    first = {item.line_index: item.line_total_minor for item in extraction.items}
    result = Repair(disputed=len(disputed), column_left=_column_left(lines, extraction))
    for (index, _votes), amount in zip(disputed, chosen, strict=True):
        if index not in first:
            # A dropped line; zero means leave it dropped.
            if amount:
                result.recovered[index] = amount
        elif amount != first[index]:
            result.changes[index] = (first[index], amount)
    return result if result.applied else None


def _describing_words(
    line: Line, column_left: int | None, profile: Profile | None
) -> tuple[list[Word], str | None]:
    """Return a recovered line's words left of the price column, and its VAT code."""
    words = [
        w for w in line.words if column_left is None or w.cx < column_left
    ] or line.words
    if profile is not None and len(words) > 1:
        vat_code = read_vat_code(words[-1].text, profile)
        if vat_code is not None:
            return words[:-1], vat_code
    return words, None


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
        confidence=round(min(w.conf for w in words) / 100, 3),
        line_index=line.index,
        bbox=(line.left, line.top, line.right - line.left, line.bottom - line.top),
        flags=["line_recovered"],
    )


def apply(
    extraction: Extraction,
    result: Repair,
    lines: list[Line],
    profile: Profile | None = None,
) -> None:
    """Write the agreed prices back, and flag everything that moved."""
    by_index = {line.index: line for line in lines}
    for line_index, amount in sorted(result.recovered.items()):
        line = by_index.get(line_index)
        if line is None:
            continue
        words, vat_code = _describing_words(line, result.column_left, profile)
        if amount < 0:
            label = " ".join(" ".join(w.text for w in words).split())
            extraction.adjustments.append(
                Adjustment("discount", label, amount, line_index)
            )
        else:
            extraction.items.append(_recovered_item(line, words, amount, vat_code))
    if result.recovered:
        extraction.items.sort(key=lambda i: i.line_index)
        extraction.adjustments.sort(key=lambda a: a.line_index)
        extraction.skipped_lines = [
            i for i in extraction.skipped_lines if i not in result.recovered
        ]

    by_item = {item.line_index: item for item in extraction.items}
    for line_index, (_was, now) in result.changes.items():
        item = by_item.get(line_index)
        if item is None:
            continue
        if now == 0:
            # Only nameless lines are offered zero: a repeated price, not a product.
            extraction.items.remove(item)
            continue
        item.line_total_minor = now
        # At quantity one the unit price is the line total, so move both.
        if item.quantity_source == "implicit":
            item.unit_price_minor = now
        if "price_reread" not in item.flags:
            item.flags.append("price_reread")

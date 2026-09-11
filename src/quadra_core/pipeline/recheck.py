"""Re-read the price column when a receipt does not balance."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quadra_core.pipeline.extract import Extraction, LineItem, parse_amount
from quadra_core.pipeline.lines import Line, drop_speckle, group_lines, load_tsv

if TYPE_CHECKING:  # pragma: no cover - import cost on a device that may not need it
    from ..profiles.loader import Profile
    from .validate import Validation

# Enlarging the strip is cheap, since the column is a fraction of the page, and
# gives Tesseract the glyph size it prefers.
SCALE = 3

# Slack around the column so no digit is clipped and the strip is not so tight
# that Tesseract loses the line.
PAD = 16

# 12 disagreements is 4096 combinations, which is fast enough. Past that the
# receipt is too badly read for the arithmetic to pick a single answer, and more
# combinations just means more of them balancing by coincidence.
MAX_DISPUTED = 12


@dataclass(slots=True)
class Repair:
    """What the second reading changed, and what it left alone."""

    # Keyed by line index, not by position in extraction.items: apply() inserts
    # recovered lines and re-sorts, which would invalidate a positional key.
    changes: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Line index -> price, for lines the first pass dropped and the sums need.
    recovered: dict[int, int] = field(default_factory=dict)
    disputed: int = 0

    @property
    def applied(self) -> bool:
        """Whether the repair contains any changes."""
        return bool(self.changes) or bool(self.recovered)


def _price_words(lines: list[Line], extraction: Extraction) -> dict[int, Any]:
    """Rightmost word per line of interest: the price, as the first pass read it.

    Both the lines that produced an item and the ones that were dropped for having
    no readable price. A dropped line is the worse of the two: a misread price is at
    least visible in the total, while a line that never became an item takes its
    money away in silence.
    """
    by_index = {line.index: line for line in lines}
    wanted = [item.line_index for item in extraction.items]
    wanted += extraction.skipped_lines
    out = {}
    for index in wanted:
        line = by_index.get(index)
        if line and line.words:
            out[index] = max(line.words, key=lambda w: w.right)
    return out


def second_opinion(
    prepared: Any, lines: list[Line], extraction: Extraction, profile: Profile
) -> dict[int, int]:
    """Read the price column on its own. Returns line index -> amount.

    One crop and one pass, not one per price. Reading each price in its own little
    box was tried and is worse as well as slower: cut that tight, Tesseract loses
    the run of text it uses to size glyphs, and it started misreading prices the
    full page had read correctly.
    """
    from .ocr import run  # noqa: PLC0415 - keeps Tesseract off the import path

    words = _price_words(lines, extraction)
    if not words:
        return {}

    left = max(0, min(w.left for w in words.values()) - PAD)
    right = min(prepared.width, max(w.right for w in words.values()) + PAD)
    if right - left < 1:
        return {}

    strip = prepared.crop((left, 0, right, prepared.height))
    strip = strip.resize((strip.width * SCALE, strip.height * SCALE))
    # psm 6 - a uniform block. The strip is one column of numbers and nothing else.
    strip_lines = group_lines(drop_speckle(load_tsv(run(strip, psm=6).tsv)))
    if not strip_lines:
        return {}

    centres = [
        (sum(w.cy for w in line.words) / len(line.words), line) for line in strip_lines
    ]

    out: dict[int, int] = {}
    for line_index, word in words.items():
        wanted = word.cy * SCALE
        centre, line = min(centres, key=lambda c: abs(c[0] - wanted))
        # Only trust a row that lines up with the one being asked about.
        if abs(centre - wanted) > word.height * SCALE:
            continue
        amount = parse_amount("".join(w.text for w in line.words), profile)
        if amount is not None:
            out[line_index] = amount
    return out


def repair(
    extraction: Extraction,
    validation: Validation,
    prepared: Any,
    profile: Profile,
    lines: list[Line],
) -> Repair | None:
    """Reconcile the two readings against the printed total.

    Returns None when there is nothing to do or nothing certain to say. The bar is
    exactly one combination that balances: two would mean the receipt cannot tell
    them apart, and picking either would be a guess wearing a proof's clothes.
    """
    target = validation.printed_total_minor
    if target is None or validation.delta_minor == 0 or not extraction.items:
        return None

    other = second_opinion(prepared, lines, extraction, profile)
    if not other:
        return None

    # A slot per item, plus one per dropped line. A dropped line is offered
    # either as nothing or as the price the column read.
    options: list[list[int]] = []
    for item in extraction.items:
        first = item.line_total_minor
        candidates = [first]
        second = other.get(item.line_index)
        if second is not None and second != first:
            candidates.append(second)
        options.append(candidates)

    recoverable: list[int] = []
    for line_index in extraction.skipped_lines:
        found = other.get(line_index)
        if found is not None and found > 0:
            recoverable.append(line_index)
            options.append([0, found])

    disputed = [i for i, c in enumerate(options) if len(c) > 1]
    if not disputed or len(disputed) > MAX_DISPUTED:
        return None

    adjustments = sum(a.amount_minor for a in extraction.adjustments)
    undisputed = set(range(len(options))) - set(disputed)
    settled = sum(
        options[i][0] for i in undisputed if options[i][0] is not None
    )

    balancing = [
        combination
        for combination in itertools.product(*[options[i] for i in disputed])
        if settled + adjustments + sum(combination) == target
    ]
    if len(balancing) != 1:
        return None

    chosen = dict(zip(disputed, balancing[0], strict=True))
    result = Repair(disputed=len(disputed))
    first_recovered = len(extraction.items)
    for index, value in chosen.items():
        if index >= first_recovered:
            # A dropped line the sums need. Zero means leave it dropped.
            if value:
                result.recovered[recoverable[index - first_recovered]] = value
            continue
        item = extraction.items[index]
        if value != item.line_total_minor:
            result.changes[item.line_index] = (item.line_total_minor, value)
    return result if result.applied else None


def _recovered_item(line: Line, price: int) -> LineItem:
    """Build the item for a line the first pass dropped.

    The description keeps the whole line, price and all. Trimming it would mean
    guessing where the price started, and the price is exactly the part the first
    pass could not find. It is flagged, so somebody reads it either way.
    """
    text = " ".join(w.text for w in line.words).strip()
    return LineItem(
        description_raw=text,
        description=" ".join(text.split()),
        quantity=Decimal(1),
        quantity_source="implicit",
        unit_price_minor=price,
        line_total_minor=price,
        vat_code=None,
        confidence=round(min(w.conf for w in line.words) / 100, 3),
        line_index=line.index,
        bbox=(line.left, line.top, line.right - line.left, line.bottom - line.top),
        flags=["line_recovered"],
    )


def apply(extraction: Extraction, result: Repair, lines: list[Line]) -> None:
    """Write the agreed prices back, and say so on everything that moved."""
    by_index = {line.index: line for line in lines}
    for line_index, price in sorted(result.recovered.items()):
        line = by_index.get(line_index)
        if line is None:
            continue
        extraction.items.append(_recovered_item(line, price))
    if result.recovered:
        # Items are read down the page, so a recovered one has to go back in
        # position or the review page shows it out of order.
        extraction.items.sort(key=lambda i: i.line_index)

    by_item = {item.line_index: item for item in extraction.items}
    for line_index, (_was, now) in result.changes.items():
        item = by_item.get(line_index)
        if item is None:
            continue
        item.line_total_minor = now
        # On a two-column receipt the unit price is not read separately, it is
        # the line total at quantity one. Move both or the per-line arithmetic
        # fails by exactly the correction just made.
        if item.quantity_source == "implicit":
            item.unit_price_minor = now
        # The number came from a different reading than the description did, so
        # flag the line even though it balances now.
        if "price_reread" not in item.flags:
            item.flags.append("price_reread")

"""Decide what a person has to check. Every rule for review lives here.

A receipt is "ok" only when its lines add up to a total nothing contradicts and
no field is in doubt. Anything else is "review", with one reason per doubtful
field, pointing at the item it concerns, so a person checks those fields only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from quadra_core.pipeline.extract import Extraction, name_words
from quadra_core.pipeline.lines import Line, text_block
from quadra_core.pipeline.money import format_minor
from quadra_core.pipeline.validate import Validation
from quadra_core.profiles.loader import Profile


@dataclass(frozen=True, slots=True)
class Reason:
    """One thing a person should check."""

    code: str
    field: str  # "total" | "items" | "price" | "quantity" | "description" | "discount"
    message: str
    item: int | None = None  # index into line_items, when the reason is about one


# A line worth pointing a person at has a name: two letters in a row.
_NAMED = re.compile(r"[^\W\d_]{2,}")


def item_for(line_index: int, extraction: Extraction) -> int | None:
    """Return the item printed nearest above a line, which a discount applies to."""
    above = [
        i for i, item in enumerate(extraction.items) if item.line_index < line_index
    ]
    return above[-1] if above else None


def _total_reasons(
    extraction: Extraction, validation: Validation, money
) -> list[Reason]:
    total = validation.printed_total_minor
    if total is None:
        return [Reason("total_missing", "total", "The total could not be read.")]
    if validation.total_disputed:
        paid = sum(extraction.payments_minor) - extraction.change_minor
        return [
            Reason(
                "total_disputed",
                "total",
                f"The total reads {money(total)} but the payment reads {money(paid)}.",
            )
        ]
    if not validation.balanced:
        return [
            Reason(
                "total_mismatch",
                "total",
                f"The lines add up to {money(validation.computed_total_minor)}, "
                f"but the total is {money(total)}.",
            )
        ]
    guessed = any("price_unconfirmed" in i.flags for i in extraction.items) or any(
        a.unconfirmed for a in extraction.adjustments
    )
    if guessed and not validation.total_confirmed:
        return [
            Reason(
                "total_unconfirmed",
                "total",
                f"The total {money(total)} was read only once, and prices were "
                "adjusted to match it.",
            )
        ]
    return []


def _item_reasons(extraction: Extraction, money) -> list[Reason]:
    out = []
    for index, item in enumerate(extraction.items):
        price = money(item.line_total_minor or 0)
        if "price_unconfirmed" in item.flags:
            before = (
                f"Read as {money(item.first_read_minor)}"
                if item.first_read_minor is not None
                else "The price was unreadable"
            )
            out.append(
                Reason(
                    "price_unconfirmed",
                    "price",
                    f"{before}; {price} would make the receipt add up.",
                    index,
                )
            )
        if "price_implausible" in item.flags:
            out.append(
                Reason(
                    "price_implausible",
                    "price",
                    f"The price {price} is more than the whole receipt.",
                    index,
                )
            )
        if {"quantity_unverified", "quantity_disagreement"} & set(item.flags):
            out.append(
                Reason(
                    "quantity_unverified",
                    "quantity",
                    f"The quantity {item.quantity} could not be confirmed.",
                    index,
                )
            )
        if "description_unreadable" in item.flags:
            out.append(
                Reason(
                    "description_unreadable",
                    "description",
                    f"The name may be misread: “{item.description}”.",
                    index,
                )
            )
    return out


def _missing_prices(extraction: Extraction, lines: Sequence[Line]) -> list[Reason]:
    """Name the lines with no readable price: where a missing amount most likely is."""
    by_index = {line.index: line for line in lines}
    block = text_block(list(lines))
    out = []
    for line_index in extraction.skipped_lines:
        line = by_index.get(line_index)
        if line is None:
            continue
        # The name alone: the unreadable price is noise to a person.
        words = name_words(line.words, block, extraction.vat_column)
        name = " ".join(w.text for w in words)
        if _NAMED.search(name):
            out.append(Reason("price_missing", "price", f"No price read for “{name}”."))
    return out


def reasons(
    extraction: Extraction,
    validation: Validation,
    profile: Profile,
    lines: Sequence[Line] = (),
) -> list[Reason]:
    """Everything a person has to check on this receipt, total first."""

    def money(minor: int) -> str:
        return format_minor(
            minor, places=profile.decimal_places, separator=profile.decimal_separator
        )

    if not extraction.items:
        return [Reason("no_items", "items", "No items were found on the receipt.")]

    out = _total_reasons(extraction, validation, money)
    if not validation.balanced:
        out += _missing_prices(extraction, lines)
    out += _item_reasons(extraction, money)
    for adjustment in extraction.adjustments:
        if adjustment.unconfirmed:
            out.append(
                Reason(
                    "discount_unconfirmed",
                    "discount",
                    f"The discount was unreadable; {money(adjustment.amount_minor)} "
                    "would make the receipt add up.",
                    item_for(adjustment.line_index, extraction),
                )
            )
    for _line, price in sorted(extraction.removed.items()):
        out.append(
            Reason(
                "line_removed",
                "items",
                f"A line with no name, priced {money(price)}, was taken for a "
                "repeated price and left out.",
            )
        )
    return out


def status(extraction: Extraction, found: list[Reason]) -> str:
    """'ok', 'review' or 'failed': the one field to look at."""
    if not extraction.items:
        return "failed"
    return "review" if found else "ok"

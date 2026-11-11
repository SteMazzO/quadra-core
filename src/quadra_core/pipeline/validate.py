"""Check a receipt against its own arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import NamedTuple

from quadra_core.pipeline.extract import MAX_QUANTITY, Charge, Extraction, LineItem
from quadra_core.pipeline.money import times

# A name the OCR is less sure of than this goes to review.
LOW_CONFIDENCE_DESCRIPTION = 0.6

# How far the items may miss a printed subtotal and still be the same sum, rounded.
ROUNDING = 1

# Flags owned by this module, cleared before each run since a repair validates twice.
CHECK_FLAGS = frozenset(
    {"quantity_unverified", "price_implausible", "description_unreadable"}
)


@dataclass(slots=True)
class Validation:
    """The result of running the arithmetic checks on a receipt extraction."""

    warnings: list[str] = field(default_factory=list)
    # The items alone, then the items with the discounts and the charges that
    # are added on top, which is what the reference total should equal.
    items_subtotal_minor: int = 0
    computed_total_minor: int = 0
    # The figure the items are checked against, and where it came from.
    printed_total_minor: int | None = None
    total_source: str = "none"  # "printed" | "payments" | "supplied" | "none"
    delta_minor: int | None = None
    # Two independent readings agree on the total: the printed total and the
    # payments, or a person typed it in.
    total_confirmed: bool = False
    # The printed total and the payments disagree, and the items side with neither.
    total_disputed: bool = False

    @property
    def balanced(self) -> bool:
        """Whether the receipt's arithmetic checks out."""
        return self.delta_minor == 0


def quantity_problem(item: LineItem) -> str | None:
    """Say why an item's quantity can't be trusted, or None if it can."""
    q = item.quantity
    if item.unit_price_minor is None:
        return "incomplete line"
    if q <= 0 or q > MAX_QUANTITY:
        return f"out of range: {q}"
    if abs(times(q, item.unit_price_minor) - item.line_total_minor) > 1:
        return f"{q} x {item.unit_price_minor} != {item.line_total_minor}"
    # Real and misread fractions look alike unless the fraction was printed.
    if q != q.to_integral_value() and item.quantity_source == "reconstructed":
        return f"uncorroborated fraction: {q}"
    return None


def _applied(
    base: int, charges: list[Charge], target: int | None
) -> tuple[Charge, ...]:
    """Pick the charges that, added to the items, come closest to the total.

    Tax is printed the same whether prices include it or not; only the sum
    tells. With no total to aim at, every charge is taken as added.
    """
    if target is None:
        return tuple(charges)
    options = [
        combo
        for size in range(len(charges), -1, -1)
        for combo in combinations(charges, size)
    ]
    return min(
        options, key=lambda c: abs(base + sum(x.amount_minor for x in c) - target)
    )


def _computed(extraction: Extraction, base: int, payments: int | None) -> int:
    """Mark the charges that are added on top, and return what the total should be."""
    printed = extraction.printed_total_minor

    def total(target: int | None) -> int:
        applied = _applied(base, extraction.charges, target)
        for charge in extraction.charges:
            charge.applied = charge in applied
        return base + sum(c.amount_minor for c in applied)

    computed = total(printed if printed is not None else payments)
    # The payments may fit where a misread printed total does not.
    misread = None not in {printed, payments} and computed != printed
    if (
        misread
        and not extraction.printed_total_supplied
        and total(payments) != payments
    ):
        computed = total(printed)
    return computed


class _Reference(NamedTuple):
    source: str
    total: int | None
    confirmed: bool = False
    disputed: bool = False
    warning: str | None = None


def _reference_total(  # noqa: PLR0911 - a decision table
    extraction: Extraction, computed: int, payments: int | None
) -> _Reference:
    """Pick the figure to check the items against, and say how sure it is."""
    printed = extraction.printed_total_minor
    if extraction.printed_total_supplied:
        return _Reference("supplied", printed, confirmed=True)

    if printed is not None and payments is not None:
        if printed == payments:
            return _Reference("printed", printed, confirmed=True)
        # Two readings disagree; the items, a third, can say which one was misread.
        if computed == payments:
            return _Reference(
                "payments", payments, True, warning=f"printed_total_misread:{printed}"
            )
        if computed == printed:
            return _Reference(
                "printed", printed, True, warning=f"payment_misread:{payments}"
            )
        return _Reference(
            "printed",
            printed,
            disputed=True,
            warning=f"payments_disagree:{payments}!={printed}",
        )

    if printed is not None:
        return _Reference("printed", printed)
    if payments is not None:
        # Payments survive when the printed total is too large or bold to read.
        return _Reference(
            "payments", payments, warning="printed_total_unreadable_used_payments"
        )
    return _Reference("none", None)


def validate(extraction: Extraction) -> Validation:
    """Run the checks, flagging each item that needs a second look."""
    result = Validation(warnings=list(extraction.warnings))
    result.items_subtotal_minor = sum(i.line_total_minor for i in extraction.items)
    base = result.items_subtotal_minor + sum(
        a.amount_minor for a in extraction.adjustments
    )
    payments = (
        sum(extraction.payments_minor) - extraction.change_minor
        if extraction.payments_minor
        else None
    )

    # Prices shown net of tax are rounded one by one, so they can miss the
    # printed subtotal by a unit. The subtotal is then the figure to carry on.
    subtotal = extraction.subtotal_minor
    if subtotal is not None and 0 < abs(subtotal - base) <= ROUNDING:
        result.warnings.append(f"items_rounded:{subtotal - base}")
        base = subtotal

    result.computed_total_minor = _computed(extraction, base, payments)

    reference = _reference_total(extraction, result.computed_total_minor, payments)
    total = reference.total
    result.total_source = reference.source
    result.printed_total_minor = total
    result.total_confirmed = reference.confirmed
    result.total_disputed = reference.disputed
    if reference.warning:
        result.warnings.append(reference.warning)
    if total is not None:
        result.delta_minor = result.computed_total_minor - total
    if subtotal not in {None, base, result.items_subtotal_minor}:
        result.warnings.append(f"subtotal_mismatch:{subtotal}")

    # No single line can cost more than the whole receipt before its discounts.
    ceiling = (total or 0) - sum(
        amount
        for amount in (
            *(a.amount_minor for a in extraction.adjustments),
            *(c.amount_minor for c in extraction.charges if c.applied),
        )
        if amount < 0
    )
    for item in extraction.items:
        item.flags[:] = [f for f in item.flags if f not in CHECK_FLAGS]
        problem = quantity_problem(item)
        if problem:
            item.flags.append("quantity_unverified")
            result.warnings.append(f"item[{item.line_index}]:quantity:{problem}")
        if total and item.line_total_minor > ceiling:
            item.flags.append("price_implausible")
        if item.confidence < LOW_CONFIDENCE_DESCRIPTION:
            item.flags.append("description_unreadable")

    result.warnings += [f"line_without_a_price:{i}" for i in extraction.skipped_lines]
    return result

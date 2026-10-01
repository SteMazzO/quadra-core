"""Check a receipt against its own arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple

from quadra_core.pipeline.extract import Extraction, LineItem
from quadra_core.profiles.loader import Profile

# A name the OCR is less sure of than this goes to review.
LOW_CONFIDENCE_DESCRIPTION = 0.6

# Flags owned by this module, cleared before each run since a repair validates twice.
CHECK_FLAGS = frozenset(
    {"quantity_unverified", "price_implausible", "description_unreadable"}
)


@dataclass(slots=True)
class Validation:
    """The result of running the arithmetic checks on a receipt extraction."""

    warnings: list[str] = field(default_factory=list)
    # The items alone, then the items plus the discounts, which is what the
    # reference total should equal.
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


def quantity_problem(item: LineItem, profile: Profile) -> str | None:
    """Say why an item's quantity can't be trusted, or None if it can."""
    q = item.quantity
    if q is None or item.unit_price_minor is None or item.line_total_minor is None:
        return "incomplete line"
    if q <= 0 or q > profile.max_quantity:
        return f"out of range: {q}"

    expected = (q * Decimal(item.unit_price_minor)).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    if abs(int(expected) - item.line_total_minor) > 1:
        return f"{q} x {item.unit_price_minor} != {item.line_total_minor}"

    # Real and misread fractions look alike unless the fraction was printed.
    if q != q.to_integral_value() and item.quantity_source not in {
        "parsed",
        "modifier",
    }:
        return f"uncorroborated fraction: {q}"
    return None


class _Reference(NamedTuple):
    source: str
    total: int | None
    confirmed: bool = False
    disputed: bool = False
    warning: str | None = None


def _reference_total(extraction: Extraction, computed: int) -> _Reference:  # noqa: PLR0911 - a decision table
    """Pick the figure to check the items against, and say how sure it is."""
    printed = extraction.printed_total_minor
    if extraction.printed_total_supplied:
        return _Reference("supplied", printed, confirmed=True)

    payments = (
        sum(extraction.payments_minor) - extraction.change_minor
        if extraction.payments_minor
        else None
    )

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


def validate(extraction: Extraction, profile: Profile) -> Validation:
    """Run the checks, flagging each item that needs a second look."""
    result = Validation(warnings=list(extraction.warnings))

    if not profile.calibrated:
        result.warnings.append("profile_not_calibrated")

    result.items_subtotal_minor = sum(
        i.line_total_minor for i in extraction.items if i.line_total_minor is not None
    )
    result.computed_total_minor = result.items_subtotal_minor + sum(
        a.amount_minor for a in extraction.adjustments
    )

    reference = _reference_total(extraction, result.computed_total_minor)
    total = reference.total
    result.total_source = reference.source
    result.printed_total_minor = total
    result.total_confirmed = reference.confirmed
    result.total_disputed = reference.disputed
    if reference.warning:
        result.warnings.append(reference.warning)
    if total is not None:
        result.delta_minor = result.computed_total_minor - total

    for item in extraction.items:
        item.flags[:] = [f for f in item.flags if f not in CHECK_FLAGS]
        problem = quantity_problem(item, profile)
        if problem:
            item.flags.append("quantity_unverified")
            result.warnings.append(f"item[{item.line_index}]:quantity:{problem}")
        # No single line can cost more than the whole receipt.
        if total and (item.line_total_minor or 0) > total:
            item.flags.append("price_implausible")
        if item.confidence < LOW_CONFIDENCE_DESCRIPTION:
            item.flags.append("description_unreadable")

    for line_index in extraction.skipped_lines:
        result.warnings.append(f"line_without_a_price:{line_index}")
    return result

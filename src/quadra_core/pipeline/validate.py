"""Check a receipt against its own arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from quadra_core.pipeline.extract import Extraction, LineItem
from quadra_core.profiles.loader import Profile

# Flags owned by this module, cleared before each run since a repair validates twice.
CHECK_FLAGS = frozenset({"line_arithmetic", "quantity_verified"})


@dataclass(frozen=True, slots=True)
class Check:
    """A single validation check, either passed or failed."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class Validation:
    """The result of running the arithmetic checks on a receipt extraction."""

    status: str  # "ok" | "partial" | "failed"
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # The items alone, then the items plus the discounts, which is what the
    # printed total should equal.
    items_subtotal_minor: int = 0
    computed_total_minor: int = 0
    printed_total_minor: int | None = None
    delta_minor: int | None = None
    total_source: str = "none"  # "printed" | "payments" | "none"

    @property
    def balanced(self) -> bool:
        """Whether the receipt's arithmetic checks out."""
        return self.delta_minor == 0


def check_line_arithmetic(item: LineItem) -> Check:
    """Quantity x unit_price == line_total, within one minor unit of rounding."""
    if (
        item.unit_price_minor is None
        or item.line_total_minor is None
        or item.quantity is None
    ):
        return Check("line_arithmetic", False, "incomplete line")

    expected = (item.quantity * Decimal(item.unit_price_minor)).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    delta = abs(int(expected) - item.line_total_minor)
    return Check("line_arithmetic", delta <= 1, f"delta={delta}")


def check_quantity_verified(item: LineItem, profile: Profile) -> Check:
    """Report whether the quantity was corroborated, not whether it looks right."""
    q = item.quantity
    if q is None:
        return Check("quantity_verified", False, "missing")
    if q <= 0 or q > profile.max_quantity:
        return Check("quantity_verified", False, f"out of range: {q}")
    if q == q.to_integral_value():
        return Check("quantity_verified", True)
    if item.quantity_source in {"parsed", "modifier"}:
        return Check("quantity_verified", True, "fractional but read directly")
    return Check("quantity_verified", False, f"uncorroborated fraction: {q}")


def _reference_total(
    extraction: Extraction, computed: int
) -> tuple[str, int | None, str | None]:
    """Pick the figure to check the items against: (source, total, warning)."""
    payments = (
        sum(extraction.payments_minor) - extraction.change_minor
        if extraction.payments_minor
        else None
    )
    printed = extraction.printed_total_minor

    # Items and payments agree but the printed total doesn't: the total was misread.
    if (
        not extraction.printed_total_supplied
        and printed is not None
        and payments is not None
        and payments != printed
        and computed == payments
    ):
        return "payments", payments, f"printed_total_misread:{printed}!={payments}"

    if printed is not None:
        disagree = payments is not None and payments != printed
        warning = (
            f"payments_disagree_with_printed_total:{payments}!={printed}"
            if disagree
            else None
        )
        return "printed", printed, warning

    if payments is not None:
        # Payments survive when the printed total is too large or bold to read.
        return "payments", payments, "printed_total_unreadable_used_payments"

    return "none", None, None


def validate(extraction: Extraction, profile: Profile) -> Validation:
    """Run the checks and work out an overall status."""
    result = Validation(status="ok", warnings=list(extraction.warnings))

    if not profile.calibrated:
        result.warnings.append("profile_not_calibrated")

    for item in extraction.items:
        item.flags[:] = [f for f in item.flags if f not in CHECK_FLAGS]
        for check in (
            check_line_arithmetic(item),
            check_quantity_verified(item, profile),
        ):
            if not check.passed:
                item.flags.append(check.name)
                result.warnings.append(
                    f"item[{item.line_index}]:{check.name}:{check.detail}"
                )

    result.items_subtotal_minor = sum(
        i.line_total_minor for i in extraction.items if i.line_total_minor is not None
    )
    result.computed_total_minor = result.items_subtotal_minor + sum(
        a.amount_minor for a in extraction.adjustments
    )

    source, total, warning = _reference_total(extraction, result.computed_total_minor)
    result.total_source = source
    result.printed_total_minor = total
    if warning:
        result.warnings.append(warning)

    if total is None:
        result.checks.append(Check("receipt_total", False, "no total or payments"))
    else:
        result.delta_minor = result.computed_total_minor - total
        balanced = result.delta_minor == 0
        result.checks.append(
            Check("receipt_total", balanced, f"delta={result.delta_minor}")
        )

    for line_index in extraction.skipped_lines:
        result.warnings.append(f"line_without_a_price:{line_index}")

    per_item_ok = sum(1 for i in extraction.items if not i.flags)
    result.checks.append(
        Check(
            "items_extracted", bool(extraction.items), f"{len(extraction.items)} items"
        )
    )
    result.checks.append(
        Check(
            "items_clean",
            per_item_ok == len(extraction.items),
            f"{per_item_ok}/{len(extraction.items)} without flags",
        )
    )

    if not extraction.items:
        result.status = "failed"
    elif any(not c.passed for c in result.checks) or result.warnings:
        result.status = "partial"
    return result

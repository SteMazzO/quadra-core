"""Check a receipt against its own arithmetic.

Quantity times unit price should equal the line total, and the lines should sum
to the printed total. That redundancy is what makes OCR output checkable.

Only numbers are checked here. A misread product name passes everything in this
module, which is why review sampling exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from quadra_core.pipeline.extract import Extraction, LineItem
from quadra_core.profiles.loader import Profile

# Flags this module owns. parse_tsv validates twice when a repair runs, so these
# are cleared before each pass; flags set elsewhere (price_reread, line_recovered)
# are left alone.
CHECK_FLAGS = frozenset({"line_arithmetic", "quantity_verified"})


@dataclass(frozen=True, slots=True)
class Check:
    """A single validation check, either passed or failed."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class Validation:
    """The result of running the arithmetic oracle on a receipt extraction."""

    status: str  # "ok" | "partial" | "failed"
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    items_subtotal_minor: int = 0
    printed_total_minor: int | None = None
    delta_minor: int | None = None
    # Which independent figure the item sum was checked against.
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
    # One minor unit of slack, since stores round weighed goods differently and
    # the convention is not something to assume.
    return Check("line_arithmetic", delta <= 1, f"delta={delta}")


def check_quantity_verified(item: LineItem, profile: Profile) -> Check:
    """Report whether the quantity was corroborated, not whether it looks right.

    Arithmetic cannot separate these two, both from the same photo and both
    non-integer:

        MELE GOLDEN KG   q=1.241  unit=2,49   correct, sold by weight
        YOGURT MAGRO     q=0.111  unit=31,15  wrong, '1,15' read as '31,15'

    Anything claiming to tell them apart would be guessing, so send both for review.
    """
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


def validate(extraction: Extraction, profile: Profile) -> Validation:
    """Run the checks and work out an overall status."""
    result = Validation(status="ok", warnings=list(extraction.warnings))

    if not profile.calibrated:
        result.warnings.append("profile_not_calibrated")

    subtotal = sum(a.amount_minor for a in extraction.adjustments)
    for item in extraction.items:
        item.flags[:] = [f for f in item.flags if f not in CHECK_FLAGS]
        if item.line_total_minor is not None:
            subtotal += item.line_total_minor

        for check in (
            check_line_arithmetic(item),
            check_quantity_verified(item, profile),
        ):
            if not check.passed:
                item.flags.append(check.name)
                result.warnings.append(
                    f"item[{item.line_index}]:{check.name}:{check.detail}"
                )

    result.items_subtotal_minor = subtotal

    # Amounts tendered less any change restates the total independently of the
    # item lines, so it is a real cross-check. It also rescues receipts whose
    # printed total is in a font Tesseract cannot read: on one photo
    # 'TOTALE EURO 26,44' came back as 'db, +' at every scale and threshold,
    # while 0,04 + 9,50 + 16,90 read cleanly.
    payments_total = (
        sum(extraction.payments_minor) - extraction.change_minor
        if extraction.payments_minor
        else None
    )
    printed = extraction.printed_total_minor

    # The items and the payment line are read independently of each other. When
    # they agree and the printed total does not, the printed total is the one that
    # was misread: 33 item prices do not conspire to match a payment line by
    # accident. Seen on a real photo where 'TOTALE EURO 101,29' came back as
    # '101,2' + '4' and joined to 101,24, against items and payments both at 101,29.
    corroborated = (
        not extraction.printed_total_supplied
        and printed is not None
        and payments_total is not None
        and payments_total != printed
        and subtotal == payments_total
    )

    if corroborated:
        result.total_source = "payments"
        result.printed_total_minor = payments_total
        result.delta_minor = 0
        result.warnings.append(f"printed_total_misread:{printed}!={payments_total}")
        result.checks.append(
            Check("receipt_total", True, f"delta=0 (printed {printed} misread)")
        )
    elif printed is not None:
        result.total_source = "printed"
        result.printed_total_minor = printed
        result.delta_minor = subtotal - printed
        result.checks.append(
            Check(
                "receipt_total", result.delta_minor == 0, f"delta={result.delta_minor}"
            )
        )
        # If both are readable they have to agree, otherwise one was misread.
        if payments_total is not None and payments_total != printed:
            result.warnings.append(
                f"payments_disagree_with_printed_total:{payments_total}!={printed}"
            )
    elif payments_total is not None:
        result.total_source = "payments"
        result.printed_total_minor = payments_total
        result.delta_minor = subtotal - payments_total
        result.warnings.append("printed_total_unreadable_used_payments")
        result.checks.append(
            Check(
                "receipt_total",
                result.delta_minor == 0,
                f"delta={result.delta_minor} (vs payments)",
            )
        )
    else:
        result.checks.append(
            Check("receipt_total", False, "no total or payments found")
        )

    # A line that looked like a product but had no readable price takes its
    # money out of the total. Warn about it, or the receipt is wrong with
    # nothing to point at.
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

    # Status. A receipt is always returned, graded rather than discarded.
    if not extraction.items:
        result.status = "failed"
    elif any(not c.passed for c in result.checks) or result.warnings:
        result.status = "partial"
    return result

"""Read printed amounts. Past this module, money is always whole minor units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True, slots=True)
class Format:
    """How one receipt prints money: '1.234,56', '1,234.56', '12,000' or '1.250'."""

    separator: str = ","
    places: int = 2


# What OCR returns for a decimal separator, e.g. '1;;19' or '2/76'.
_SEPARATORS = re.escape(",.;:/\u00b7\u2022\u201a")
_GROUPED = r"\d{1,3}(?:[.,']\d{3})+|\d+"
_SHAPES = {
    places: re.compile(
        rf"({_GROUPED})[{_SEPARATORS}]+(\d{{{places}}})"
        if places
        else rf"({_GROUPED})()"
    )
    for places in (0, 2, 3)
}

# A number with whatever is stuck to it: 'Rp58000.00', '7,20-S', '(5.00)', '$3.99'.
_TOKEN = re.compile(r"(?P<lead>\D*)(?P<body>\d(?:.*\d)?)(?P<tail>\D*)")
# The currency marks receipts print against an amount.
CURRENCY = r"rp|rm|idr|myr|eur|euro|usd|gbp|chf|sgd|rs|inr|kr|zl|pln|tl|try|[$€£¥₹₺₩₫₱]"
# Before: a currency mark. After: a tax letter. Around both, whatever specks OCR adds.
_LEAD = re.compile(rf"[^\w%]*(?:{CURRENCY})?[^\w%]*", re.IGNORECASE)
_TAIL = re.compile(r"[^\w%]*[^\W\d_]?[^\w%]*")
_TRAILING_MINUS = re.compile("(?<=\\d)[-\u2013\u2014]\\S?$")

# More digits than any receipt total: a barcode or a phone number.
_MAX_DIGITS = 9


def parse_amount(text: str, fmt: Format) -> int | None:
    """Read one word as money in this receipt's format, or None if it is not."""
    # A trailing minus, as in '7,20-S'. OCR sometimes reads that letter as a digit.
    text, minus = _TRAILING_MINUS.subn("", text.strip())
    token = _TOKEN.fullmatch(text)
    if not token:
        return None
    lead, body, tail = token.group("lead", "body", "tail")
    if not _LEAD.fullmatch(lead) or not _TAIL.fullmatch(tail):
        return None
    # Without decimals to go by, '500G' and '4X' are sizes, not prices.
    if not fmt.places and any(c.isalpha() for c in tail):
        return None
    shape = _SHAPES[fmt.places].fullmatch(body)
    if not shape:
        return None
    digits = re.sub(r"\D", "", shape.group(1)) + shape.group(2)
    if len(digits) > _MAX_DIGITS:
        return None
    negative = minus or "-" in lead or ("(" in lead and ")" in tail)
    return -int(digits) if negative else int(digits)


_DECIMAL = re.compile(rf"(?:{_GROUPED})([.,])\d{{2}}")
_THOUSANDS = re.compile(r"\d{1,3}(?:([.,])\d{3})+")


def detect_format(words: Iterable[str]) -> Format:
    """Work out how a receipt prints money from the numbers on it.

    Two digits after the last separator are decimals. Groups of three are
    thousands, unless a number starts '0.', which no grouped number does.
    """
    decimal: Counter[str] = Counter()
    thousands: Counter[str] = Counter()
    leading_zero = False
    for word in words:
        token = _TOKEN.fullmatch(word)
        if not token or "%" in token.group("tail"):
            continue
        body = token.group("body")
        if found := _DECIMAL.fullmatch(body):
            decimal[found.group(1)] += 1
        elif found := _THOUSANDS.fullmatch(body):
            thousands[found.group(1)] += 1
            leading_zero = leading_zero or body.startswith("0")
    if decimal and decimal.total() >= thousands.total():
        return Format(decimal.most_common(1)[0][0], 2)
    if thousands and leading_zero:
        return Format(thousands.most_common(1)[0][0], 3)
    return Format(".", 0)


_QUANTITY = re.compile(r"[x*@]?(\d{1,4}(?:[.,]\d{1,3})?)[x*@]?", re.IGNORECASE)


def parse_quantity(text: str) -> Decimal | None:
    """Read a quantity such as '2', 'x2', '3x' or the weight '1,240'."""
    found = _QUANTITY.fullmatch(text.strip())
    if not found:
        return None
    try:
        value = Decimal(found.group(1).replace(",", "."))
    except InvalidOperation:  # pragma: no cover - the pattern only admits numbers
        return None
    return value if value > 0 else None


def format_minor(minor: int, fmt: Format) -> str:
    """Write minor units the way the receipt prints them."""
    sign = "-" if minor < 0 else ""
    digits = str(abs(minor)).rjust(fmt.places + 1, "0")
    if not fmt.places:
        return f"{sign}{digits}"
    return f"{sign}{digits[: -fmt.places]}{fmt.separator}{digits[-fmt.places :]}"


def times(quantity: Decimal, unit_minor: int) -> int:
    """Quantity times unit price, rounded to a whole minor unit as tills do."""
    return int((quantity * unit_minor).quantize(Decimal(1), rounding="ROUND_HALF_UP"))

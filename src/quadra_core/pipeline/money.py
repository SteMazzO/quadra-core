"""Parse money and quantities. Past this module, money is always whole cents."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# Characters OCR mixes up inside numbers. Only applied to tokens that look numeric.
NUMERIC_CONFUSIONS: dict[str, str] = {
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "l": "1",
    "|": "1",
    "!": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
    "G": "6",
    "b": "6",
    "T": "7",
    "g": "9",
    "q": "9",
}

_STRIP = re.compile(r"[^0-9.,]")

# A trailing minus, as in Esselunga's '7,20-S'. Any character may follow the dash,
# since OCR sometimes reads that S as a digit.
_TRAILING_SIGN = re.compile(r"-\s*[^\s]?\s*$")

# Anything longer is garbage, and could overflow Decimal's default precision.
MAX_NUMERIC_LEN = 20

# What OCR returns for a decimal separator, e.g. '1;;19' or '2/76'.
_SEPARATOR_CONFUSIONS = ",.;:/\u00b7\u2022\u201a"
# Only between digits, so text like '8x.' is left alone.
_SEPARATOR_RUN = re.compile(f"(?<=\\d)[{re.escape(_SEPARATOR_CONFUSIONS)}]+(?=\\d)")


def normalize_separators(token: str, separator: str = ",") -> str:
    """Rewrite lookalike separators, including doubled ones, as `separator`."""
    if not token:
        return token
    sign = _TRAILING_SIGN.search(token)
    suffix = sign.group(0) if sign else ""
    core = token[: sign.start()] if sign else token
    return _SEPARATOR_RUN.sub(separator, core) + suffix


def strip_sign(token: str) -> str:
    """Strip a leading or trailing sign, and any VAT letter after it."""
    return _TRAILING_SIGN.sub("", token).lstrip("-").strip()


def digit_ratio(token: str) -> float:
    """Fraction of the characters that are digits."""
    if not token:
        return 0.0
    return sum(c.isdigit() for c in token) / len(token)


def repair_numeric(token: str) -> str:
    """Fix common misreads in a token we already believe is numeric."""
    return "".join(NUMERIC_CONFUSIONS.get(c, c) for c in token)


def parse_money(
    token: str,
    *,
    places: int = 2,
    repair: bool = True,
    min_digit_ratio: float = 0.4,
) -> int | None:
    """Parse a money token to whole cents, or None if it is not money.

    The decimal separator is worked out from the token, not passed in.
    """
    if not token:
        return None

    negative = bool(_TRAILING_SIGN.search(token)) or token.lstrip().startswith("-")
    token = _TRAILING_SIGN.sub("", token).lstrip("-")

    if repair and digit_ratio(token) >= min_digit_ratio:
        candidate = repair_numeric(token)
    else:
        candidate = token
    candidate = _STRIP.sub("", candidate)
    if not candidate or not any(c.isdigit() for c in candidate):
        return None
    if len(candidate) > MAX_NUMERIC_LEN:
        return None

    # A separator followed by exactly `places` digits is the decimal point.
    if places:
        tail = re.search(r"[.,](\d+)$", candidate)
        if tail and len(tail.group(1)) == places:
            _point = candidate[tail.start()]
            candidate = candidate[: tail.start()].replace(",", "").replace(".", "")
            candidate = f"{candidate or '0'}.{tail.group(1)}"
        elif tail:
            # '1,5' could be 1,50 or 15, so refuse.
            return None
        else:
            candidate = candidate.replace(",", "").replace(".", "")
    else:
        candidate = candidate.replace(",", "").replace(".", "")

    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return None

    scaled = (value * (10**places)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return -int(scaled) if negative else int(scaled)


def parse_quantity(
    token: str, *, separator: str = ",", min_digit_ratio: float = 0.6
) -> Decimal | None:
    """Parse a quantity, which can be fractional for goods sold by weight."""
    if not token or digit_ratio(token) < min_digit_ratio:
        return None
    candidate = _STRIP.sub("", repair_numeric(token))
    if not candidate or len(candidate) > MAX_NUMERIC_LEN:
        return None
    other = "." if separator == "," else ","
    candidate = candidate.replace(other, "").replace(separator, ".")
    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return None
    return value if value > 0 else None


def format_minor(minor: int, *, places: int = 2, separator: str = ",") -> str:
    """Format cents for display or export."""
    sign = "-" if minor < 0 else ""
    digits = str(abs(minor)).rjust(places + 1, "0")
    return (
        f"{sign}{digits[:-places]}{separator}{digits[-places:]}"
        if places
        else f"{sign}{digits}"
    )


def reconstruct_quantity(total_minor: int, unit_minor: int) -> Decimal | None:
    """Work out the quantity from the line total and the unit price."""
    if unit_minor <= 0 or total_minor <= 0:
        return None
    return (Decimal(total_minor) / Decimal(unit_minor)).quantize(Decimal("0.001"))

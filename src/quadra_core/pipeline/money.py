"""Parse money and quantities off a receipt.

Money is whole cents everywhere past this module: floats cannot hold 0.10
exactly, and the error accumulates line by line. Decimal is used only for
parsing.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# Characters Tesseract mixes up inside numbers. Only applied to tokens that
# already look numeric, since it would wreck a product name.
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

# Esselunga signs negatives on the right with a VAT letter after them, so
# 'SCONTO FIDATY 30%  7,20-S' means minus 7,20. Read as a leading minus, the
# receipt ends up wrong by twice the discount.
# The character after the dash is not required to be a letter. It should be the
# VAT letter, but OCR reads that S as an 8 often enough that '0,74-8' was being
# thrown out and the discount lost. Accepting a digit is safe: the rest of the
# token still has to look like money, and '2024-5' does not.
_TRAILING_SIGN = re.compile(r"-\s*[^\s]?\s*$")

# Decimal's default context holds 28 significant digits and raises past that. No
# real price is anywhere near this long, so reject early instead of letting
# garbled OCR crash the parse.
MAX_NUMERIC_LEN = 20


# At this resolution ',' ';' ':' and '.' look nearly identical and Tesseract
# swaps them freely in the decimal position. On a faded receipt '1,19' came back
# as '1;;19' and '2,99' as '2:99'. Normalising here keeps the profile money
# patterns simple.
# '/' is included because a comma printed over a fold reads as one: '2,76' came
# back as '2/76' and the item was dropped. Only applied between digits, so dates
# are left alone.
_SEPARATOR_CONFUSIONS = ",.;:/\u00b7\u2022\u201a"
# Only between digits, which is the only place a decimal point can be. Anywhere
# else it corrupts real text: '8x.' would become '8x,' and stop matching a
# quantity line.
_SEPARATOR_RUN = re.compile(f"(?<=\\d)[{re.escape(_SEPARATOR_CONFUSIONS)}]+(?=\\d)")


def normalize_separators(token: str, separator: str = ",") -> str:
    """Rewrite lookalike separators as the profile's separator.

    Runs are collapsed too, since a doubled misread like '1;;19' is common.
    """
    if not token:
        return token
    # Keep any trailing sign out of it, or it gets treated as a separator.
    sign = _TRAILING_SIGN.search(token)
    suffix = sign.group(0) if sign else ""
    core = token[: sign.start()] if sign else token
    return _SEPARATOR_RUN.sub(separator, core) + suffix


def strip_sign(token: str) -> str:
    """Strip a leading or trailing sign, and any VAT letter after it.

    Lets a profile's money pattern stay free of sign conventions. Parse the original
    token afterwards so the sign still applies.
    """
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

    Returns None for an invalid value instead of guessing.

    Repair is gated on the token already looking numeric. Ungated, the confusion map
    turns 'abc' into '6' and then into money.

    There is no separator argument: which character is the decimal point is
    worked out from the token itself, since receipts and OCR mix ',' and '.'.
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

    # Work out which character is the decimal point rather than trusting the
    # profile: a separator followed by exactly `places` digits is the decimal
    # point, whichever character it is. Receipts and OCR mix ',' and '.' freely.
    if places:
        tail = re.search(r"[.,](\d+)$", candidate)
        if tail and len(tail.group(1)) == places:
            _point = candidate[tail.start()]
            candidate = candidate[: tail.start()].replace(",", "").replace(".", "")
            candidate = f"{candidate or '0'}.{tail.group(1)}"
        elif tail:
            # Wrong digit count after the separator means it is not money.
            # '1,5' could be 1.50 or 15, and nothing says which, so refuse.
            return None
        else:
            # No separator at all: a whole-unit amount.
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
    """Parse a quantity, which can be fractional for things sold by weight.

    Gated on digit ratio so descriptive tokens are not read as quantities: '1L' is
    rejected at 0.5, '1,240' accepted at 0.83.
    """
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
    """Work out the quantity from the total and the unit price.

    On a degraded receipt the quantity column was the worst read field on the page,
    with '2' coming back as '2 deg' and '4' as "'4.", while every price on those lines
    survived intact. Digits with a decimal comma are a much stronger signal than a
    lone small integer, so derive the quantity when it will not parse.
    """
    if unit_minor <= 0 or total_minor <= 0:
        return None
    return (Decimal(total_minor) / Decimal(unit_minor)).quantize(Decimal("0.001"))

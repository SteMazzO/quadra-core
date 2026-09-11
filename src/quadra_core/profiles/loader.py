"""Loading shop profiles.

Everything shop-specific lives in a TOML profile rather than in Python. The code
knows how to find a price column; the profile says what a price looks like. So
recalibrating does not mean editing code.

Profiles are versioned and each receipt records the version it was parsed under,
so a layout change means adding v2 instead of quietly changing old results.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path

PROFILE_DIR = Path(__file__).parent


class ProfileError(ValueError):
    """A profile is missing, malformed, or incomplete."""


@dataclass(frozen=True, slots=True)
class Rule:
    """One line-classification rule. Order matters; first match wins."""

    name: str
    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class Profile:
    """A store-specific receipt layout and its parsing rules."""

    id: str
    version: int
    currency: str
    decimal_separator: str
    decimal_places: int
    max_quantity: Decimal
    has_quantity_column: bool
    has_unit_price_column: bool
    modifier_re: re.Pattern[str] | None
    modifier_position: str
    money_re: re.Pattern[str]
    quantity_re: re.Pattern[str]
    # None for a shop that prints no such column.
    vat_code_re: re.Pattern[str] | None
    fingerprint: tuple[str, ...]
    min_ratio: float
    items_start_after: tuple[str, ...]
    items_end_before: tuple[str, ...]
    rules: tuple[Rule, ...]
    calibrated: bool

    def rule_for(self, text: str) -> str:
        """Classify a line. Returns the first matching rule name, else 'unknown'."""
        for rule in self.rules:
            if rule.pattern.search(text):
                return rule.name
        return "unknown"


def _compile(pattern: str, field: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ProfileError(f"{field}: invalid regex {pattern!r}: {exc}") from exc


# "before" and "after" say which side of its item a 'N x UNIT' line is printed;
# "none" is for a shop that prints no such line. A typo here used to disable
# modifier handling silently, so unknown values are rejected.
MODIFIER_POSITIONS = frozenset({"before", "after", "none"})


def _modifier_position(layout: dict, path: Path) -> str:
    value = layout.get("modifier_position", "before")
    if value not in MODIFIER_POSITIONS:
        raise ProfileError(
            f"{path.name}: layout.modifier_position must be one of "
            f"{sorted(MODIFIER_POSITIONS)}, got {value!r}"
        )
    return value


def load(path: Path) -> Profile:
    """Read one profile from disk."""
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProfileError(f"cannot read profile {path}: {exc}") from exc

    meta = raw.get("profile", {})
    fmt = raw.get("format", {})
    fp = raw.get("fingerprint", {})
    regions = raw.get("regions", {})
    layout = raw.get("layout", {})

    for key, table in (("id", meta), ("version", meta), ("money", fmt)):
        if key not in table:
            raise ProfileError(f"{path.name}: missing required key '{key}'")

    rules = tuple(
        Rule(name=r["name"], pattern=_compile(r["pattern"], f"rule {r.get('name')}"))
        for r in raw.get("rules", [])
    )

    return Profile(
        id=meta["id"],
        version=int(meta["version"]),
        currency=meta.get("currency", "EUR"),
        decimal_separator=fmt.get("decimal_separator", ","),
        decimal_places=int(fmt.get("decimal_places", 2)),
        max_quantity=Decimal(str(fmt.get("max_quantity", 100))),
        # Esselunga prints DESCRIPTION + PRICE only, with quantity on its own
        # line when above one, so default to the two-column case.
        has_quantity_column=bool(layout.get("has_quantity_column", False)),
        has_unit_price_column=bool(layout.get("has_unit_price_column", False)),
        modifier_re=(
            _compile(layout["quantity_modifier"], "layout.quantity_modifier")
            if layout.get("quantity_modifier")
            else None
        ),
        modifier_position=_modifier_position(layout, path),
        money_re=_compile(fmt["money"], "format.money"),
        quantity_re=_compile(fmt.get("quantity", r"\d+"), "format.quantity"),
        vat_code_re=(
            _compile(fmt["vat_code"], "format.vat_code")
            if fmt.get("vat_code")
            else None
        ),
        fingerprint=tuple(fp.get("anchors", [])),
        min_ratio=float(fp.get("min_ratio", 0.8)),
        items_start_after=tuple(regions.get("items_start_after", [])),
        items_end_before=tuple(regions.get("items_end_before", [])),
        rules=rules,
        # An uncalibrated profile is still a scaffold, so callers should not
        # treat its output as trustworthy.
        calibrated=bool(meta.get("calibrated", False)),
    )


def available() -> list[Profile]:
    """Every profile shipped in the package, id-sorted for determinism."""
    return sorted(
        (load(p) for p in PROFILE_DIR.glob("*.toml")), key=lambda p: (p.id, p.version)
    )


def fuzzy_contains(haystack: str, needle: str, min_ratio: float) -> float:
    """Best fuzzy match score of `needle` against any window of `haystack`.

    OCR mangles the all-caps header text that anchors come from, so exact
    matching is too strict. 'TOTALE' against 'TOTALF' scores 0.833 and matches.
    """
    haystack, needle = haystack.upper(), needle.upper()
    if needle in haystack:
        return 1.0
    best = 0.0
    span = len(needle)
    if span == 0 or len(haystack) < span:
        return 0.0
    # Slide a needle-sized window; cheap enough for receipt-length strings.
    for start in range(len(haystack) - span + 1):
        score = SequenceMatcher(None, haystack[start : start + span], needle).ratio()
        if score > best:
            best = score
            if best == 1.0:
                break
    return best if best >= min_ratio else 0.0


def select(
    lines: list[str], profiles: list[Profile] | None = None
) -> tuple[Profile, float] | None:
    """Pick the profile whose fingerprint best matches these receipt lines.

    Returns (profile, confidence), or None if nothing matches well enough.
    None is worth surfacing: it usually means a new shop or a changed layout.
    """
    candidates = available() if profiles is None else profiles
    blob = "\n".join(lines)
    best: tuple[Profile, float] | None = None

    for profile in candidates:
        if not profile.fingerprint:
            continue
        scores = [
            fuzzy_contains(blob, a, profile.min_ratio) for a in profile.fingerprint
        ]
        matched = [s for s in scores if s > 0]
        if not matched:
            continue
        confidence = sum(matched) / len(scores)
        if best is None or confidence > best[1]:
            best = (profile, confidence)
    return best

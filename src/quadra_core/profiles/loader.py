"""Load shop profiles from TOML."""

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


# Which side of its item a 'N x UNIT' line is printed, or "none".
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
        calibrated=bool(meta.get("calibrated", False)),
    )


def available() -> list[Profile]:
    """Every profile shipped in the package, id-sorted for determinism."""
    return sorted(
        (load(p) for p in PROFILE_DIR.glob("*.toml")), key=lambda p: (p.id, p.version)
    )


def fuzzy_contains(haystack: str, needle: str, min_ratio: float) -> float:
    """Return how well `needle` fuzzily matches part of `haystack`, case-insensitive.

    Returns 0.0 when the best score is below `min_ratio`.
    """
    haystack, needle = haystack.upper(), needle.upper()
    if needle in haystack:
        return 1.0
    best = 0.0
    span = len(needle)
    if span == 0 or len(haystack) < span:
        return 0.0
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
    """Return (profile, confidence) for the best fingerprint match, or None."""
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

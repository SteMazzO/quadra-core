"""Compare a parsed receipt with the ground truth of a public dataset.

Amounts are compared as the integer the printed digits spell ('63,636' and
'636.36' are both 63636), so the score does not depend on how the annotators
wrote a currency's decimals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any

# A predicted name this close to the true one counts as the same product.
NAME_MATCH = 0.8


@dataclass(slots=True)
class Item:
    """One priced line of the receipt, as the annotators recorded it."""

    name: str
    total: int
    quantity: Decimal | None = None


@dataclass(slots=True)
class Truth:
    """What a dataset says is printed on one receipt. None means not annotated."""

    items: list[Item] | None = None
    total: int | None = None
    subtotal: int | None = None
    tax: int | None = None
    merchant: str | None = None
    date: str | None = None  # ISO, YYYY-MM-DD


@dataclass(slots=True)
class Sample:
    """One receipt of a dataset."""

    id: str
    image: bytes
    truth: Truth
    # Ground-truth OCR, where the dataset ships it: (text, left, top, width, height).
    words: list[tuple[str, int, int, int, int]] | None = None


def digits(text: str | None) -> int | None:
    """Return the integer a printed amount's digits spell, or None without digits."""
    found = re.sub(r"\D", "", text or "")
    return int(found) if found else None


def quantity(text: str | None) -> Decimal | None:
    """Read an annotated quantity such as '2', 'x2' or '1.5'."""
    match = re.search(r"\d+(?:[.,]\d+)?", text or "")
    try:
        return Decimal(match.group(0).replace(",", ".")) if match else None
    except InvalidOperation:
        return None


def similarity(a: str | None, b: str | None) -> float:
    """How alike two names are, ignoring case, spacing and punctuation: 0 to 1."""
    a, b = (
        "".join(c for c in (text or "").casefold() if c.isalnum()) for text in (a, b)
    )
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


@dataclass(slots=True)
class Outcome:
    """How one parsed receipt compares with its ground truth."""

    id: str
    status: str
    # Field name -> right or wrong, for the fields the dataset annotates.
    fields: dict[str, bool] = field(default_factory=dict)
    items_true: int = 0
    items_predicted: int = 0
    # Items found with the right line total, and those with the right name too.
    priced: int = 0
    named: int = 0
    quantities_true: int = 0
    quantities_right: int = 0
    # Every annotated amount right: each item's line total, and the total.
    exact: bool = False


def _match(truth: list[Item], predicted: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Pair true and predicted items with equal line totals, likeliest names first."""
    pairs = sorted(
        (-similarity(item.name, guess["description"]), t, p)
        for t, item in enumerate(truth)
        for p, guess in enumerate(predicted)
        if abs(guess["line_total_minor"]) == item.total
    )
    used_true: set[int] = set()
    used_predicted: set[int] = set()
    out = []
    for _score, t, p in pairs:
        if t not in used_true and p not in used_predicted:
            used_true.add(t)
            used_predicted.add(p)
            out.append((t, p))
    return out


def score(sample: Sample, document: dict[str, Any]) -> Outcome:
    """Score one parsed document against the sample's ground truth."""
    truth = sample.truth
    outcome = Outcome(sample.id, document["status"])
    totals = document["totals"]

    for name, expected, got in (
        ("total", truth.total, totals["total_minor"]),
        ("subtotal", truth.subtotal, totals["subtotal_minor"]),
        ("tax", truth.tax, totals["tax_minor"]),
    ):
        if expected is not None:
            outcome.fields[name] = got is not None and abs(got) == expected
    if truth.merchant is not None:
        merchant = (document.get("merchant") or {}).get("name")
        outcome.fields["merchant"] = similarity(truth.merchant, merchant) >= NAME_MATCH
    if truth.date is not None:
        outcome.fields["date"] = (document.get("date") or {}).get("value") == truth.date

    items_right = True
    if truth.items is not None:
        # A line priced at nothing changes no sum, so it is not scored either way.
        predicted = [i for i in document["line_items"] if i["line_total_minor"]]
        truth.items = [i for i in truth.items if i.total]
        pairs = _match(truth.items, predicted)
        outcome.items_true = len(truth.items)
        outcome.items_predicted = len(predicted)
        outcome.priced = len(pairs)
        for t, p in pairs:
            item, guess = truth.items[t], predicted[p]
            outcome.named += similarity(item.name, guess["description"]) >= NAME_MATCH
            if item.quantity is not None:
                outcome.quantities_true += 1
                outcome.quantities_right += Decimal(guess["quantity"]) == item.quantity
        items_right = len(pairs) == len(truth.items) == len(predicted)

    outcome.exact = items_right and outcome.fields.get("total", True)
    return outcome


def _ratio(hits: int, out_of: int) -> float | None:
    return round(hits / out_of, 4) if out_of else None


def summarise(outcomes: list[Outcome]) -> dict[str, Any]:
    """Aggregate per-receipt outcomes into the benchmark's headline numbers."""
    summary: dict[str, Any] = {"receipts": len(outcomes)}
    for name in ("total", "subtotal", "tax", "merchant", "date"):
        judged = [o.fields[name] for o in outcomes if name in o.fields]
        if judged:
            summary[f"{name}_accuracy"] = _ratio(sum(judged), len(judged))

    true = sum(o.items_true for o in outcomes)
    predicted = sum(o.items_predicted for o in outcomes)
    if true:
        for label, hits in (
            ("item_price", sum(o.priced for o in outcomes)),
            ("item_name_and_price", sum(o.named for o in outcomes)),
        ):
            precision, recall = _ratio(hits, predicted) or 0.0, hits / true
            f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
            summary[f"{label}_f1"] = round(f1, 4)
        summary["quantity_accuracy"] = _ratio(
            sum(o.quantities_right for o in outcomes),
            sum(o.quantities_true for o in outcomes),
        )

    summary["exact_receipts"] = _ratio(sum(o.exact for o in outcomes), len(outcomes))
    # The promise the library makes: a receipt it calls 'ok' needs no checking.
    ok = [o for o in outcomes if o.status == "ok"]
    summary["ok_share"] = _ratio(len(ok), len(outcomes))
    summary["ok_and_exact"] = _ratio(sum(o.exact for o in ok), len(ok))
    summary["ok_but_wrong"] = [o.id for o in ok if not o.exact]
    return summary

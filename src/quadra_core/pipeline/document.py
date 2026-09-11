"""Building the JSON document for one receipt.

Money leaves here as whole cents, with a _minor suffix so the unit is obvious at
the call site. Unreadable fields come out null with a warning, never guessed.
"""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from typing import Any

from quadra_core.pipeline.extract import Extraction
from quadra_core.pipeline.lines import Line
from quadra_core.pipeline.validate import Validation
from quadra_core.profiles.loader import Profile

SCHEMA_VERSION = "1.0.0"

# Below this, the least confident word in a description is too poor to file
# unseen. Arithmetic cannot catch a misread product name, so this is the only
# thing that sends one for review.
LOW_CONFIDENCE_DESCRIPTION = 0.5


def no_review(state: str = "not_required") -> dict[str, Any]:
    """Build the review block for a document nobody has decided anything about.

    Who gets checked is up to whatever uses this library, so the default says
    only that no decision has been made yet, rather than claiming the receipt
    was checked or does not need checking.
    """
    return {
        "required": False,
        "reason": None,
        "sampled": False,
        "mode": "none",
        "state": state,
        "blind_item_indexes": [],
    }


def default_review(
    validation: Validation, extraction: Extraction
) -> dict[str, Any]:
    """Whether a person has to look, when the caller has not said.

    The caller owns this decision and can pass its own block instead. This is
    the default, and it errs towards asking: a receipt that does not add up, or
    that carries a description nobody could read, is not one to file unseen.
    """
    reasons = []
    if not validation.balanced:
        reasons.append("totals_do_not_balance")
    if not extraction.items:
        reasons.append("no_items")
    worst = min((i.confidence for i in extraction.items), default=1.0)
    if worst < LOW_CONFIDENCE_DESCRIPTION:
        reasons.append(f"unreadable_description:{worst:.2f}")

    if not reasons:
        return no_review()
    return {
        "required": True,
        "reason": "; ".join(reasons),
        "sampled": False,
        "mode": "full",
        "state": "pending",
        "blind_item_indexes": [],
    }


def _quantity(value: Decimal | None) -> dict[str, Any] | None:
    if value is None:
        return None
    # A string rather than a float, since weighed goods need fractions that
    # binary floats cannot hold exactly.
    return {"value": format(value.normalize(), "f"), "unit": "each"}


def build(
    *,
    receipt_id: str,
    lines: list[Line],
    extraction: Extraction,
    validation: Validation,
    profile: Profile,
    profile_confidence: float,
    ocr_meta: dict[str, Any],
    source: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full receipt document."""
    mean_conf = sum(l.conf for l in lines) / len(lines) if lines else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_id": receipt_id,
        "source": source,
        "profile": {
            "id": profile.id,
            "version": profile.version,
            "calibrated": profile.calibrated,
            "match_confidence": round(profile_confidence, 3),
        },
        "ocr": {**ocr_meta, "mean_word_confidence": round(mean_conf, 1)},
        "merchant": {"name": None, "store_id": None, "address": None, "vat_id": None},
        "transaction": {
            "datetime": None,
            "currency": profile.currency,
            "receipt_number": None,
            "payment_method": None,
        },
        "line_items": [
            {
                "index": i,
                "description_raw": item.description_raw,
                "description": item.description,
                "quantity": _quantity(item.quantity),
                "quantity_source": item.quantity_source,
                "unit_price_minor": item.unit_price_minor,
                "line_total_minor": item.line_total_minor,
                "discounts_minor": 0,
                "vat_code": item.vat_code,
                "confidence": item.confidence,
                "provenance": {"line_index": item.line_index, "bbox": list(item.bbox)},
                "flags": item.flags,
            }
            for i, item in enumerate(extraction.items)
        ],
        "adjustments": [
            {
                "kind": adjustment.kind,
                "label": adjustment.label,
                "amount_minor": adjustment.amount_minor,
                "provenance": {"line_index": adjustment.line_index},
            }
            for adjustment in extraction.adjustments
        ],
        "totals": {
            "items_subtotal_minor": validation.items_subtotal_minor,
            "discounts_minor": sum(a.amount_minor for a in extraction.adjustments),
            "printed_total_minor": validation.printed_total_minor,
            "computed_total_minor": validation.items_subtotal_minor,
            "balanced": validation.balanced,
            "delta_minor": validation.delta_minor,
        },
        "validation": {
            "status": validation.status,
            "checks": [asdict(c) for c in validation.checks],
            "warnings": validation.warnings,
        },
        "review": {
            **review,
            "reviewed_at": None,
            "duration_ms": None,
            "rows_expanded": 0,
            "corrections": [],
        },
    }

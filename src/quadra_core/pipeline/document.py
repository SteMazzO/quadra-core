"""Build the JSON document for one receipt."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from typing import Any

from quadra_core.pipeline.extract import Extraction
from quadra_core.pipeline.lines import Line
from quadra_core.pipeline.review import item_for, reasons, status
from quadra_core.pipeline.validate import Validation

SCHEMA_VERSION = "3.0.0"


def _quantity(value: Decimal | None) -> str | None:
    # A string, since weighed quantities need exact fractions.
    return None if value is None else format(value.normalize(), "f")


def build(
    *,
    lines: list[Line],
    extraction: Extraction,
    validation: Validation,
    ocr_meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full receipt document."""
    found = reasons(extraction, validation, lines)
    doubtful = {r.item for r in found if r.item is not None}

    applies_to = [item_for(a.line_index, extraction) for a in extraction.adjustments]
    item_discounts = [0] * len(extraction.items)
    for adjustment, index in zip(extraction.adjustments, applies_to, strict=True):
        if index is not None:
            item_discounts[index] += adjustment.amount_minor

    mean_conf = sum(l.conf for l in lines) / len(lines) if lines else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status(extraction, found),
        "review": [asdict(r) for r in found],
        "decimal_places": extraction.fmt.places,
        "line_items": [
            {
                "index": i,
                "description": item.description,
                "quantity": _quantity(item.quantity),
                "unit_price_minor": item.unit_price_minor,
                "line_total_minor": item.line_total_minor,
                "discount_minor": item_discounts[i],
                "vat_code": item.vat_code,
                "confidence": item.confidence,
                "needs_review": i in doubtful,
                "flags": item.flags,
                "line_index": item.line_index,
                "bbox": list(item.bbox),
            }
            for i, item in enumerate(extraction.items)
        ],
        "adjustments": [
            {
                "label": adjustment.label,
                "amount_minor": adjustment.amount_minor,
                "applies_to": index,
                "needs_review": adjustment.unconfirmed,
                "line_index": adjustment.line_index,
            }
            for adjustment, index in zip(
                extraction.adjustments, applies_to, strict=True
            )
        ],
        "charges": [
            {
                "kind": charge.kind,
                "label": charge.label,
                "amount_minor": charge.amount_minor,
                "applied": charge.applied,
                "line_index": charge.line_index,
            }
            for charge in extraction.charges
        ],
        "totals": {
            "items_subtotal_minor": validation.items_subtotal_minor,
            "discounts_minor": sum(a.amount_minor for a in extraction.adjustments),
            "subtotal_minor": extraction.subtotal_minor,
            "tax_minor": sum(
                c.amount_minor for c in extraction.charges if c.kind == "tax"
            )
            or extraction.tax_breakdown_minor
            or None,
            "computed_total_minor": validation.computed_total_minor,
            "total_minor": validation.printed_total_minor,
            "total_source": validation.total_source,
            "balanced": validation.balanced,
            "delta_minor": validation.delta_minor,
        },
        "warnings": validation.warnings,
        "ocr": {**ocr_meta, "mean_word_confidence": round(mean_conf, 1)},
    }

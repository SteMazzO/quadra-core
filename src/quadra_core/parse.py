"""Photo or Tesseract TSV in, receipt document out."""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quadra_core.pipeline import document as document_module
from quadra_core.pipeline.document import build
from quadra_core.pipeline.extract import extract
from quadra_core.pipeline.lines import (
    drop_speckle,
    group_lines,
    load_tsv,
    median_glyph_height,
)
from quadra_core.pipeline.ocr import run_on_path
from quadra_core.pipeline.validate import validate
from quadra_core.profiles import loader

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

DEFAULT_OCR_META = {"engine": "tesseract", "lang": "ita", "psm": 4, "oem": 1}


def read_input(path: Path | None) -> tuple[str, dict]:
    """Return TSV plus OCR metadata, from an image, a TSV file, or stdin."""
    if path is None:
        return sys.stdin.read(), dict(DEFAULT_OCR_META)

    if path.suffix.lower() in IMAGE_SUFFIXES:
        result, report, prepared = run_on_path(path)
        return result.tsv, {
            "engine": "tesseract",
            "engine_version": result.engine_version,
            "lang": result.lang,
            "psm": result.psm,
            "oem": result.oem,
            "preprocessing": report.steps,
            "_prepared": prepared,
        }

    return path.read_text(), dict(DEFAULT_OCR_META)


def parse_tsv(
    tsv: str,
    *,
    profile_id: str | None,
    receipt_id: str | None = None,
    ocr_meta: dict | None = None,
    prepared: object | None = None,
    printed_total: int | None = None,
    review: dict[str, Any] | None = None,
) -> tuple[dict, str]:
    """Turn TSV into a receipt document. Never raises on bad receipt content.

    With `prepared`, the image the TSV came from, an unbalanced receipt gets its
    price column re-read. `printed_total` overrides the total read off the receipt.
    """
    words = drop_speckle(load_tsv(tsv))
    lines = group_lines(words)
    texts = [l.text for l in lines]

    confidence = 0.0
    if profile_id:
        matches = [p for p in loader.available() if p.id == profile_id]
        if not matches:
            raise loader.ProfileError(f"no such profile: {profile_id}")
        profile = matches[0]
        confidence = 1.0
    else:
        selected = loader.select(texts)
        if selected is None:
            raise loader.ProfileError(
                "no profile fingerprint matched. That is a real signal: either the "
                "receipt is from another store, or the layout changed."
            )
        profile, confidence = selected

    extraction = extract(lines, profile)
    if printed_total is not None:
        extraction.printed_total_minor = printed_total
        extraction.printed_total_supplied = True
    validation = validate(extraction, profile)

    if prepared is not None and not validation.balanced:
        from .pipeline import recheck  # noqa: PLC0415 - only needed with an image

        agreed = recheck.repair(extraction, validation, prepared, profile, lines)
        if agreed is not None:
            recheck.apply(extraction, agreed, lines, profile)
            # Validate again from scratch, so the per-line checks see the new prices.
            validation = validate(extraction, profile)
            validation.warnings.append(
                f"price_column_reread:{len(agreed.changes)}_of_{agreed.disputed}"
            )
            if agreed.recovered:
                validation.warnings.append(f"lines_recovered:{len(agreed.recovered)}")

    rid = receipt_id or uuid.uuid4().hex

    doc = build(
        receipt_id=rid,
        lines=lines,
        extraction=extraction,
        validation=validation,
        profile=profile,
        profile_confidence=confidence,
        ocr_meta=ocr_meta or dict(DEFAULT_OCR_META),
        source={"ingested_at": datetime.now(UTC).isoformat()},
        review=(
            review
            if review is not None
            else document_module.default_review(validation, extraction)
        ),
    )
    return doc, validation.status


# Below this glyph height Tesseract starts losing lines, so read again enlarged.
SMALL_GLYPH = 26
COMFORTABLE_GLYPH = 30


def _closer(candidate: dict, current: dict) -> bool:
    """Whether a second reading is better than the current one.

    Balancing wins, then the smaller delta, then more lines found.
    """
    a, b = candidate["totals"], current["totals"]
    if a["balanced"] != b["balanced"]:
        return bool(a["balanced"])
    if a["delta_minor"] is not None and b["delta_minor"] is not None:
        return abs(a["delta_minor"]) < abs(b["delta_minor"])
    if (a["printed_total_minor"] is None) != (b["printed_total_minor"] is None):
        return b["printed_total_minor"] is None
    return len(candidate["line_items"]) > len(current["line_items"])


def _read_larger(prepared, tsv: str, **kwargs):
    """Read the photo again, enlarged, if its text came out small; else None."""
    from PIL import Image  # noqa: PLC0415 - the parsing core runs without Pillow

    from .pipeline.ocr import run  # noqa: PLC0415

    glyph = median_glyph_height(drop_speckle(load_tsv(tsv)))
    if not glyph or glyph >= SMALL_GLYPH:
        return None

    scale = COMFORTABLE_GLYPH / glyph
    bigger = prepared.resize(
        (round(prepared.width * scale), round(prepared.height * scale)),
        Image.Resampling.LANCZOS,
    )
    result = run(bigger, psm=4)
    document, status = parse_tsv(result.tsv, prepared=bigger, **kwargs)
    return document, status, result.tsv, bigger


def parse_image(
    path: Path,
    *,
    profile_id: str | None,
    review: dict[str, Any] | None = None,
    printed_total: int | None = None,
):
    """Parse a photo into (document, status).

    The document carries two private keys to pop before storing it: `_tsv`, the OCR
    it was built from, and `_prepared`, the image whose coordinates item boxes use.
    """
    tsv, ocr_meta = read_input(path)
    # A PIL image in ocr_meta would make the document unserialisable.
    prepared = ocr_meta.pop("_prepared", None)
    shared = {
        "profile_id": profile_id,
        "review": review,
        "ocr_meta": ocr_meta,
        "printed_total": printed_total,
    }
    document, status = parse_tsv(tsv, prepared=prepared, **shared)

    if prepared is not None and not document["totals"]["balanced"]:
        second = _read_larger(prepared, tsv, **shared)
        if second is not None and _closer(second[0], document):
            document, status, tsv, prepared = second

    document["_tsv"] = tsv
    document["_prepared"] = prepared
    return document, status

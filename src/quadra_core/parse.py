"""Photo or Tesseract TSV in, receipt document out.

The entry point for the library. `parse_tsv` takes Tesseract output directly,
so the parsing side runs with neither Tesseract nor Pillow installed; the image
path imports both lazily to keep it that way.

Bad receipt content never raises. A receipt that could not be read comes back
as a document saying so, so a batch of receipts does not fail on one bad photo.
"""

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
    """Return TSV plus OCR metadata, from an image, a TSV file, or stdin.

    Taking TSV directly keeps the parsing core usable without Tesseract or
    Pillow installed; the image branch imports them lazily.
    """
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
    """TSV -> receipt document. Never raises on bad receipt content.

    `prepared` is the image the TSV was read from. Given one, a receipt that fails
    to add up gets its price column read a second time; see pipeline.recheck.

    `printed_total` overrides whatever total the receipt itself yielded, for the
    case where someone typed it in because it was unreadable.

    `review` says whether a person should check this receipt. That is the
    caller's decision, not this library's, so whatever is passed in is recorded
    as given.
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
        # Someone read the total off the paper because the parser could not, so
        # it beats anything OCR produced.
        extraction.printed_total_minor = printed_total
        extraction.printed_total_supplied = True
    validation = validate(extraction, profile)

    if prepared is not None and not validation.balanced:
        from .pipeline import recheck  # noqa: PLC0415 - only needed with an image

        agreed = recheck.repair(extraction, validation, prepared, profile, lines)
        if agreed is not None:
            recheck.apply(extraction, agreed, lines)
            # Re-run the whole check instead of patching the totals, so the
            # per-line arithmetic and warnings match the new prices.
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
        ocr_meta=ocr_meta or {"engine": "tesseract", "lang": "ita", "psm": 4, "oem": 1},
        source={"ingested_at": datetime.now(UTC).isoformat()},
        review=(
            review
            if review is not None
            else document_module.default_review(validation, extraction)
        ),
    )
    return doc, validation.status


# Below 26px Tesseract starts losing lines, so a second pass at a larger size is
# worth the time. A receipt at 25px was fine; one at 20px lost two lines.
SMALL_GLYPH = 26
COMFORTABLE_GLYPH = 30


def _closer(candidate: dict, current: dict) -> bool:
    """Whether a second reading is the better of the two.

    Balancing wins. Failing that, the smaller delta wins. With no total to
    compare against, more lines found wins, since a missed line takes its
    money with it.
    """
    a, b = candidate["totals"], current["totals"]
    if a["balanced"] != b["balanced"]:
        return bool(a["balanced"])
    if a["delta_minor"] is not None and b["delta_minor"] is not None:
        return abs(a["delta_minor"]) < abs(b["delta_minor"])
    if (a["printed_total_minor"] is None) != (b["printed_total_minor"] is None):
        return b["printed_total_minor"] is None
    return len(a and candidate["line_items"]) > len(current["line_items"])


def _read_larger(prepared, tsv: str, **kwargs):
    """Read the photo again, enlarged, when the first read came out small.

    Only runs on a receipt that did not add up, so a good one never pays for it.
    Enlarging adds no detail, but Tesseract's line model does better at the size
    it expects: on one real receipt it found 33 lines instead of 31.
    """
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
    """Parse a photo, carrying the raw TSV along for archiving.

    The document comes back with two private keys, `_tsv` and `_prepared`: the
    OCR it was built from and the image OCR actually saw. Whoever archives them
    pops them off first. Item bounding boxes use the prepared image's
    coordinates, so cropping by box needs that image, not the original photo.
    """
    tsv, ocr_meta = read_input(path)
    # Removed before parse_tsv: build() copies ocr_meta into the document, and a
    # PIL image in there makes it unserialisable.
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
            # The archived image must be the one the TSV describes, or crops on
            # the review page point at the wrong pixels.
            document, status, tsv, prepared = second

    # Handed to the archiver, which pops both before the document is written.
    document["_tsv"] = tsv
    document["_prepared"] = prepared
    return document, status



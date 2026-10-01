"""Photo or Tesseract TSV in, receipt document out."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quadra_core.pipeline import recheck
from quadra_core.pipeline.document import build
from quadra_core.pipeline.extract import extract
from quadra_core.pipeline.lines import (
    drop_speckle,
    group_lines,
    load_tsv,
    median_glyph_height,
)
from quadra_core.pipeline.validate import validate
from quadra_core.profiles import loader

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class Result:
    """A parsed receipt, and the OCR and image it was read from."""

    document: dict[str, Any]
    tsv: str
    image: Any = None

    @property
    def status(self) -> str:
        """'ok', 'review' or 'failed'. With 'review', see document['review']."""
        return self.document["status"]


def _profile(profile: str | None, texts: list[str]) -> loader.Profile:
    if profile:
        for candidate in loader.available():
            if candidate.id == profile:
                return candidate
        raise loader.ProfileError(f"no such profile: {profile}")
    selected = loader.select(texts)
    if selected is None:
        raise loader.ProfileError(
            "could not tell which shop this receipt is from; pass profile=... "
            "(see `quadra-core profiles`)"
        )
    return selected[0]


def parse_tsv(
    tsv: str,
    *,
    profile: str | None = None,
    total: int | None = None,
    image: Any = None,
    ocr: dict[str, Any] | None = None,
) -> Result:
    """Turn Tesseract TSV into a receipt. Never raises on a hard-to-read receipt.

    `profile` names the shop; without it the shop is recognised from the text.
    `total` is the receipt total in minor units, for when a person has read it.
    `image`, the image the TSV came from, lets an unbalanced receipt have its
    price column read again.
    """
    if total is not None and total <= 0:
        raise ValueError(f"total must be positive minor units, got {total}")

    lines = group_lines(drop_speckle(load_tsv(tsv)))
    shop = _profile(profile, [line.text for line in lines])

    extraction = extract(lines, shop)
    if total is not None:
        extraction.printed_total_minor = total
        extraction.printed_total_supplied = True
        if "printed_total_not_found" in extraction.warnings:
            extraction.warnings.remove("printed_total_not_found")
    validation = validate(extraction, shop)

    if image is not None and not validation.balanced:
        repair = recheck.repair(extraction, validation, image, shop, lines)
        if repair is not None:
            recheck.apply(extraction, repair, lines, shop)
            # Validate again from scratch, so the checks see the new prices.
            validation = validate(extraction, shop)

    document = build(
        lines=lines,
        extraction=extraction,
        validation=validation,
        profile=shop,
        ocr_meta=ocr or {},
    )
    return Result(document, tsv, image)


# Below this glyph height Tesseract starts losing lines, so read again enlarged.
SMALL_GLYPH = 26
COMFORTABLE_GLYPH = 30

# A finer flat field, as a fraction of the photo's width, for creased paper.
FINE_FLAT_FIELD = 0.025


def _rank(document: dict[str, Any]) -> tuple:
    """Order readings of one photo: balanced, then clean, then closest, then fullest."""
    totals = document["totals"]
    delta = totals["delta_minor"]
    return (
        totals["balanced"],
        document["status"] == "ok",
        -abs(delta) if delta is not None else float("-inf"),
        -len(document["review"]),
        len(document["line_items"]),
    )


def _settled(document: dict[str, Any]) -> bool:
    """Whether the money is all certain, so reading again could not improve it."""
    return document["totals"]["balanced"] and not any(
        reason["field"] in {"total", "price", "discount", "items"}
        for reason in document["review"]
    )


def _readings(path: Path) -> Iterator[tuple[str, Any, dict[str, Any]]]:
    """Read a photo several ways, cheapest first: (tsv, image, ocr metadata)."""
    from PIL import Image  # noqa: PLC0415 - the parsing core runs without Pillow

    from quadra_core.pipeline.ocr import run  # noqa: PLC0415
    from quadra_core.pipeline.preprocess import preprocess  # noqa: PLC0415

    def read(image, steps):
        result = run(image, psm=4)
        meta = {
            "engine_version": result.engine_version,
            "psm": result.psm,
            "preprocessing": steps,
        }
        return result.tsv, image, meta

    image, report = preprocess(path)
    tsv, image, meta = read(image, report.steps)
    yield tsv, image, meta

    # Small text: read the same image again, enlarged.
    glyph = median_glyph_height(drop_speckle(load_tsv(tsv)))
    if glyph and glyph < SMALL_GLYPH:
        scale = COMFORTABLE_GLYPH / glyph
        bigger = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        yield read(bigger, [*report.steps, f"enlarge x{scale:.2f}"])

    # Creased paper: even out the lighting over a smaller area.
    fine, fine_report = preprocess(path, flat_field_scale=FINE_FLAT_FIELD)
    yield read(fine, fine_report.steps)


def parse_image(
    path: str | Path, *, profile: str | None = None, total: int | None = None
) -> Result:
    """Read a receipt photo. Never raises on a hard-to-read receipt.

    If the first reading does not add up, the photo is read again in other ways
    and the best reading is kept.
    """
    best: Result | None = None
    for tsv, image, meta in _readings(Path(path)):
        result = parse_tsv(tsv, profile=profile, total=total, image=image, ocr=meta)
        if best is None or _rank(result.document) > _rank(best.document):
            best = result
        if _settled(best.document):
            break
    assert best is not None  # the first reading always yields
    return best

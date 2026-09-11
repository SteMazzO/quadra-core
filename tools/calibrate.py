"""Show what the parser sees on a receipt, so a profile can be written from it.

Prints the preprocessing steps, the reconstructed lines with their confidence, the
price column it found, and under a given profile which rule claimed each line and what
came out. Writing a profile without this is guesswork.

A new shop needs a stub profile before there is anything to report against;
with no --profile the receipt is fingerprinted against the shops already
shipped, which a new one will not match. See 'Adding a shop' in README.md.

Usage:
    python3 tools/calibrate.py receipt.jpg                     # detect the shop
    python3 tools/calibrate.py receipt.jpg --profile myshop    # force one
    python3 tools/calibrate.py ocr.tsv --profile myshop --save-fixture name
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quadra_core.pipeline.extract import extract
from quadra_core.pipeline.lines import (
    discover_price_column,
    drop_speckle,
    group_lines,
    load_tsv,
    median_glyph_height,
    text_block,
)
from quadra_core.pipeline.money import format_minor
from quadra_core.pipeline.ocr import run_on_path
from quadra_core.pipeline.validate import validate
from quadra_core.profiles import loader
from quadra_core.testdata import OCR as FIXTURE_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RULE = "-" * 78


def get_tsv(path: Path) -> str:
    """OCR an image, or read TSV straight from disk."""
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return path.read_text()

    result, report, _ = run_on_path(path)
    print(f"{RULE}\nPREPROCESSING")
    print(f"  {report.source_size} -> {report.output_size}")
    for step in report.steps:
        print(f"  - {step}")
    print(f"  engine: {result.engine_version}  psm={result.psm} oem={result.oem}")

    return result.tsv


def report_geometry(tsv: str, lines, profile_id: str | None):
    """Print geometry stats and resolve which profile to evaluate against."""
    words = load_tsv(tsv)
    kept = drop_speckle(words)
    block = text_block(lines)

    print(f"{RULE}\nGEOMETRY")
    print(f"  words {len(words)} -> {len(kept)} after speckle removal")
    print(f"  lines {len(lines)}  median glyph {median_glyph_height(kept):.0f}px")
    print(f"  text block x: {block.left}..{block.right} ({block.width}px wide)")
    if kept:
        print(f"  mean word confidence {sum(w.conf for w in kept) / len(kept):.1f}")

    profile = None
    if profile_id:
        matches = [p for p in loader.available() if p.id == profile_id]
        if not matches:
            print(f"error: no profile '{profile_id}'", file=sys.stderr)
            return None
        profile = matches[0]
    else:
        selected = loader.select([l.text for l in lines])
        if selected:
            profile, confidence = selected
            print(f"  fingerprint matched '{profile.id}' at {confidence:.2f}")
        else:
            print("  fingerprint matched NOTHING - add anchors to a profile")
            return None

    column = discover_price_column(lines, profile.money_re)
    if column:
        spread = (column[1] - column[0]) * 100
        print(
            f"  price column {column[0]:.3f}..{column[1]:.3f} of block "
            f"(spread {spread:.1f}%)"
        )
    else:
        print("  price column NOT FOUND - check format.money vs the lines below")
    return profile


def report_lines(lines, profile) -> None:
    """Print every reconstructed line with the rule that claimed it."""
    print(f"{RULE}\nLINES  (rule assigned by profile)")
    for line in lines:
        print(
            f"  [{line.index:3d}] {line.conf:5.1f} "
            f"{profile.rule_for(line.text):18} {line.text}"
        )


def report_extraction(extraction) -> None:
    """Print the items and adjustments that came out."""
    print(
        f"{RULE}\nEXTRACTED  ({len(extraction.items)} items, "
        f"{len(extraction.adjustments)} adjustments)"
    )
    for item in extraction.items:
        flags = f"  !{','.join(item.flags)}" if item.flags else ""
        print(
            f"  {item.description[:34]:36} x{item.quantity!s:>7} "
            f"{format_minor(item.line_total_minor):>9} "
            f"({item.quantity_source}){flags}"
        )
    for adjustment in extraction.adjustments:
        print(
            f"  {adjustment.label[:34]:36} {'':>8} "
            f"{format_minor(adjustment.amount_minor):>9} ({adjustment.kind})"
        )


def report_validation(result) -> None:
    """Print the arithmetic verdict."""
    print(f"{RULE}\nVALIDATION  status={result.status}")
    print(f"  items subtotal   {format_minor(result.items_subtotal_minor):>10}")
    printed = result.printed_total_minor
    shown = format_minor(printed) if printed is not None else "NOT FOUND"
    print(f"  printed total    {shown:>10}")
    if result.delta_minor is not None:
        verdict = "BALANCED" if result.balanced else "*** MISMATCH ***"
        print(f"  delta            {format_minor(result.delta_minor):>10}   {verdict}")
    for check in result.checks:
        print(f"  [{'ok  ' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    for warning in result.warnings:
        print(f"  warn: {warning}")


def save_fixture(tsv: str, name: str) -> None:
    """Write the OCR out where the test suite actually reads fixtures from."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIXTURE_DIR / f"{name}.tsv"
    out.write_text(tsv)
    print(f"{RULE}\nwrote {out}")
    print("  SCRUB THIS before committing: loyalty numbers, card digits, address.")


def main() -> int:
    """Run the calibration report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="receipt image or Tesseract TSV")
    ap.add_argument("--profile", help="profile id to evaluate against")
    ap.add_argument(
        "--save-fixture",
        metavar="NAME",
        help="save the OCR as a test fixture (scrub it before committing)",
    )
    args = ap.parse_args()

    tsv = get_tsv(args.path)
    lines = group_lines(drop_speckle(load_tsv(tsv)))
    if args.save_fixture:
        save_fixture(tsv, args.save_fixture)

    # No profile means no report, whether a named one was missing or the fingerprint
    # matched nothing. Either way this run produced nothing to act on.
    profile = report_geometry(tsv, lines, args.profile)
    if profile is None:
        return 2

    report_lines(lines, profile)
    extraction = extract(lines, profile)
    result = validate(extraction, profile)
    report_extraction(extraction)
    report_validation(result)

    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

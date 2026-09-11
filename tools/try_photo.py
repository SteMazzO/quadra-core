"""Process a receipt photograph and print pipeline diagnostics.

Requires Pillow and Tesseract. Usage:

    python3 tools/try_photo.py receipts/inbox/IMG_1234.jpg [--keep]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quadra_core.pipeline.extract import extract  # noqa: E402
from quadra_core.pipeline.lines import (  # noqa: E402
    discover_price_column,
    drop_speckle,
    group_lines,
    load_tsv,
)
from quadra_core.pipeline.money import format_minor  # noqa: E402
from quadra_core.pipeline.validate import validate  # noqa: E402
from quadra_core.profiles import loader  # noqa: E402

# ~300 DPI across 80mm thermal paper. Bigger costs time for no accuracy gain,
# smaller drops cap height below what the LSTM reads reliably.
TARGET_WIDTH = 1100

# Minimum whitespace around the text before Tesseract's layout analysis suffers.
# Only topped up when the image has less; see preprocess().
MIN_MARGIN = 12

# Limit upscaling for very small source images.
MAX_UPSCALE = 4.0


def preprocess(path: Path, out_dir: Path | None) -> tuple[Path, dict]:
    """Grayscale, orient, downscale, and add a margin. Returns the image and timings."""
    from PIL import Image, ImageOps  # noqa: PLC0415 - imported on use

    stats: dict[str, float] = {}
    t0 = time.perf_counter()

    with Image.open(path) as source:
        stats["source_px"] = source.width * source.height
        # DCT-domain downscale during JPEG decode. Never builds the full-size RGB
        # buffer, so it is the cheapest resize available.
        if source.format == "JPEG":
            source.draft("L", (TARGET_WIDTH, TARGET_WIDTH * 4))
        # EXIF orientation first. Phones store photos rotated with a tag, and
        # skipping this OCRs the receipt sideways with no warning.
        img = ImageOps.exif_transpose(source).convert("L")

    # Scale toward the target in both directions. Downscaling is the usual case
    # and the main latency lever. Upscaling matters when an image arrives too
    # small: Tesseract's LSTM needs a minimum glyph height, and a 444px-wide
    # receipt reads at 84.6 mean confidence natively but 89.0 when tripled.
    # It cannot add detail that was never there - a 338px receipt stays
    # unreadable at any factor - but it does recover merely small ones.
    if img.width != TARGET_WIDTH:
        scale = TARGET_WIDTH / img.width
        if scale > MAX_UPSCALE:
            scale = MAX_UPSCALE
            stats["upscale_capped"] = True
        height = round(img.height * scale)
        img = img.resize((round(img.width * scale), height), Image.LANCZOS)
        stats["scale"] = round(scale, 2)

    # Pad ONLY if the content is already close to an edge.
    #
    # Received wisdom says to always add a white margin because Tesseract's layout
    # analysis struggles when text touches the border. Measured on a degraded receipt
    # that already has whitespace, padding unconditionally hurts, because it
    # perturbs Tesseract's internal scaling and the receipt stopped balancing:
    #
    #     no padding   delta      0   (correct)
    #     +5px         delta  +1851
    #     +20px        delta   -149
    #     +40px        delta   -405   and one item lost entirely
    #
    # So measure what margin is there and top it up only when it is thin.
    content = ImageOps.invert(img).getbbox()
    if content:
        margin = min(
            content[0], content[1], img.width - content[2], img.height - content[3]
        )
        if margin < MIN_MARGIN:
            img = ImageOps.expand(img, border=MIN_MARGIN - margin, fill=255)
            stats["padded_by"] = MIN_MARGIN - margin
    stats["final_size"] = f"{img.width}x{img.height}"

    target = (out_dir or Path("/tmp")) / f"{path.stem}_pre.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target)

    stats["preprocess_ms"] = (time.perf_counter() - t0) * 1000
    return target, stats


def run_tesseract(image: Path, out_base: Path) -> tuple[str, float]:
    """OCR to TSV with the flags the design settled on. Returns (tsv, elapsed_ms)."""
    t0 = time.perf_counter()
    subprocess.run(
        [
            "tesseract",
            str(image),
            str(out_base),
            "--oem",
            "1",
            "--psm",
            "4",
            "-l",
            "ita",
            "-c",
            "preserve_interword_spaces=1",
            "-c",
            "load_system_dawg=0",
            "-c",
            "load_freq_dawg=0",
            "-c",
            "user_defined_dpi=300",
            "-c",
            "invert_threshold=0",
            "tsv",
        ],
        check=True,
        capture_output=True,
        env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"},
    )
    return out_base.with_suffix(".tsv").read_text(), (time.perf_counter() - t0) * 1000


def main() -> int:  # noqa: PLR0915 - a linear report, splitting it hides the flow
    """Preprocess, OCR, parse, and report on one photograph."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photo", type=Path)
    ap.add_argument("--profile", default=None, help="force a profile id")
    ap.add_argument(
        "--keep", action="store_true", help="write intermediates to receipts/out/"
    )
    ap.add_argument("--text", action="store_true", help="also dump reconstructed lines")
    args = ap.parse_args()

    if not args.photo.exists():
        print(f"no such file: {args.photo}")
        return 2

    out_dir = ROOT / "receipts" / "out" if args.keep else None
    image, stats = preprocess(args.photo, out_dir)
    tsv, ocr_ms = run_tesseract(image, (out_dir or Path("/tmp")) / args.photo.stem)

    words = load_tsv(tsv)
    kept = drop_speckle(words)
    lines = group_lines(kept)

    print(f"\n  {args.photo.name}")
    print(
        f"  {stats['source_px'] / 1e6:.1f}MP -> {stats['final_size']}   "
        f"preprocess {stats['preprocess_ms']:.0f}ms   ocr {ocr_ms:.0f}ms"
    )
    conf = sum(w.conf for w in kept) / len(kept) if kept else 0
    print(
        f"  {len(words)} words -> {len(kept)} after speckle -> {len(lines)} lines"
        f"   mean confidence {conf:.1f}"
    )

    selected = (
        (next((p for p in loader.available() if p.id == args.profile), None), 1.0)
        if args.profile
        else loader.select([l.text for l in lines])
    )
    if selected is None or selected[0] is None:
        print("\n  NO PROFILE MATCHED. The fingerprint anchors were not found.")
        print("  Either OCR mangled the header, or this is not an Esselunga receipt.")
        print("  Reconstructed lines:\n")
        for l in lines[:12]:
            print(f"    {l.text}")
        return 1

    profile, confidence = selected
    column = discover_price_column(lines, profile.money_re)
    col_text = f"{column[0]:.3f}-{column[1]:.3f}" if column else "NOT FOUND"
    print(
        f"  profile {profile.id} v{profile.version} (match {confidence:.2f})"
        f"   price column {col_text}"
    )

    if args.text:
        print("\n  --- reconstructed lines ---")
        for l in lines:
            print(f"    [{l.index:2d}] {l.text}")

    ex = extract(lines, profile)
    v = validate(ex, profile)

    print(f"\n  --- {len(ex.items)} items ---")
    for i in ex.items:
        note = ""
        if i.quantity_source == "modifier":
            note = f"  [{i.quantity} x {format_minor(i.unit_price_minor)}]"
        flags = f"  !{','.join(i.flags)}" if i.flags else ""
        total_text = format_minor(i.line_total_minor)
        print(f"    {i.description[:34]:36}{total_text:>9}{note}{flags}")
    for a in ex.adjustments:
        print(f"    {a.label[:34]:36}{format_minor(a.amount_minor):>9}  <{a.kind}>")

    printed = v.printed_total_minor
    total = format_minor(printed) if printed is not None else "?"
    print(f"    {'':36}{'-' * 9}")
    print(f"    {'computed':36}{format_minor(v.items_subtotal_minor):>9}")
    print(f"    {'printed':36}{total:>9}")

    mark = "OK" if v.balanced else f"MISMATCH {format_minor(v.delta_minor or 0)}"
    print(f"\n  status={v.status}  balanced={v.balanced}  {mark}")
    for w in v.warnings:
        print(f"    warn: {w}")
    if out_dir:
        print(f"\n  intermediates in {out_dir}")

    return 0 if v.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

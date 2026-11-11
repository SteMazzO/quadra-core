"""Read a receipt photo or Tesseract TSV and print the receipt as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from quadra_core.parse import IMAGE_SUFFIXES, parse_image, parse_tsv
from quadra_core.pipeline.ocr import OcrError

# The exit code carries the status, so scripts need not read the JSON.
EXIT_OK, EXIT_REVIEW, EXIT_FAILED, EXIT_ERROR = 0, 1, 2, 3

_STATUS_EXIT = {"ok": EXIT_OK, "review": EXIT_REVIEW, "failed": EXIT_FAILED}


def amount(text: str) -> int:
    """Read a total as a person writes it, '101,25' or '101.25', into cents."""
    try:
        value = Decimal(text.replace(",", "."))
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}") from None
    if value <= 0 or value != value.quantize(Decimal("0.01")):
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}")
    return int(value * 100)


def _run_parse(args: argparse.Namespace) -> int:
    source: Path | None = args.path
    if source is not None and source.suffix.lower() in IMAGE_SUFFIXES:
        result = parse_image(source, total=args.total)
    else:
        tsv = sys.stdin.read() if source is None else source.read_text("utf-8")
        result = parse_tsv(tsv, total=args.total)

    json.dump(result.document, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return _STATUS_EXIT[result.status]


def build_parser() -> argparse.ArgumentParser:
    """Assemble the command line."""
    ap = argparse.ArgumentParser(
        prog="quadra-core",
        description=__doc__,
        epilog="exit codes: 0 ok, 1 needs review, 2 failed, 3 error",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse", help="read a receipt photo or Tesseract TSV")
    parse.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="photo or .tsv file; omit to read TSV from stdin",
    )
    parse.add_argument(
        "--total",
        type=amount,
        help="the receipt total, e.g. 101,25, when it cannot be read",
    )
    parse.set_defaults(handler=_run_parse)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return its exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OcrError, OSError, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

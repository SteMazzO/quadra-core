"""`quadra-core` on the command line: read a receipt, print the document.

Small on purpose. It is here so you can try the library and check a profile
against a photo without writing a script. Storing, reviewing and exporting
receipts are jobs for whatever gets built on top.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quadra_core.parse import IMAGE_SUFFIXES, parse_image, parse_tsv, read_input
from quadra_core.profiles import loader

# Exit code carries the result, so a shell script or worker does not have to
# parse stdout.
EXIT_OK, EXIT_PARTIAL, EXIT_FAILED, EXIT_ERROR = 0, 1, 2, 3

_STATUS_EXIT = {"ok": EXIT_OK, "partial": EXIT_PARTIAL, "failed": EXIT_FAILED}


def _run_parse(args: argparse.Namespace) -> int:
    source: Path | None = args.path
    if source is not None and source.suffix.lower() in IMAGE_SUFFIXES:
        document, status = parse_image(
            source, profile_id=args.profile, printed_total=args.total
        )
        # Only of use to something archiving the originals.
        document.pop("_tsv", None)
        document.pop("_prepared", None)
    else:
        tsv, ocr_meta = read_input(source)
        document, status = parse_tsv(
            tsv, profile_id=args.profile, ocr_meta=ocr_meta, printed_total=args.total
        )

    json.dump(document, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return _STATUS_EXIT.get(status, EXIT_ERROR)


def _run_profiles(_args: argparse.Namespace) -> int:
    for profile in loader.available():
        state = "calibrated" if profile.calibrated else "UNCALIBRATED"
        print(f"{profile.id:<12} v{profile.version}  {state}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Assemble the command line."""
    ap = argparse.ArgumentParser(prog="quadra-core", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse", help="read a receipt image or Tesseract TSV")
    parse.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="image or .tsv file; omit to read TSV from stdin",
    )
    parse.add_argument("--profile", help="force a profile instead of fingerprinting")
    parse.add_argument(
        "--total",
        type=int,
        help="printed total in minor units, when the receipt's own is unreadable",
    )
    parse.set_defaults(handler=_run_parse)

    profiles = sub.add_parser("profiles", help="list the shop profiles available")
    profiles.set_defaults(handler=_run_profiles)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the exit code rather than raising on a bad receipt."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (loader.ProfileError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

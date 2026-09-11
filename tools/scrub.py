"""Replace personal data in a receipt's OCR with placeholders.

The rules are Esselunga's and they are Italian: Italian street words, the
`fidaty` card by name, Italian points wording. Another shop will match none of
them, and a receipt that matches nothing looks exactly like one that is already
clean, so check a new shop's text column by hand before trusting a quiet run.

It also does not touch cashier names, transaction numbers, dates, tax codes or
lottery codes, on any receipt.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

# Ordered, first match wins. The card line has to come before the looser digit
# patterns below it.
RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Street address. Every branch has one and it identifies the shop precisely.
    (
        re.compile(r"\bvia\b.*\d|\bviale\b|\bcorso\b|\bpiazza\b", re.I),
        "VIA ROMA 00 - CITTA",
    ),
    # Loyalty card. The receipt masks most of it, but the rest still names an
    # account.
    (
        re.compile(r"\bn\.?\s*carta\s*fidaty|\bfidaty\s*:", re.I),
        "N. CARTA FIDATY: 00XX*XX***00",
    ),
    # Points balance, which adds up to how much has been spent here over years.
    (re.compile(r"saldo.*punti|punti.*saldo", re.I), "SALDO PUNTI 0.000"),
    (re.compile(r"punti\s+sulla\s+spesa|tot\.?punti", re.I), "PUNTI SPESA 0"),
)


def _spread(replacement: str, count: int) -> list[str]:
    """Fit a replacement onto exactly `count` words."""
    words = replacement.split()
    if count <= 0:
        return []
    if len(words) >= count:
        # Leftovers join the last word so nothing is dropped without notice.
        return [*words[: count - 1], "".join(words[count - 1 :])]
    return words + ["x"] * (count - len(words))


def scrub(tsv: str) -> tuple[str, int]:
    """Rewrite the personal lines of a Tesseract TSV. Returns (tsv, lines changed)."""
    rows = list(csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE))
    header, body = rows[0], rows[1:]
    text_at = header.index("text")
    key_at = [header.index(c) for c in ("page_num", "block_num", "par_num", "line_num")]

    lines: dict[tuple, list[list[str]]] = {}
    for row in body:
        if len(row) <= text_at or not row[text_at].strip():
            continue
        lines.setdefault(tuple(row[i] for i in key_at), []).append(row)

    changed = 0
    for words in lines.values():
        joined = " ".join(w[text_at] for w in words)
        for pattern, replacement in RULES:
            if not pattern.search(joined):
                continue
            spread = _spread(replacement, len(words))
            # Do not report a line that already holds its replacement. The
            # replacements match the rules that produced them ("VIA ROMA 00" is
            # still an address), so without this the tool never reports itself
            # clean and is useless with --check.
            if [w[text_at] for w in words] != spread:
                for row, new in zip(words, spread, strict=True):
                    row[text_at] = new
                changed += 1
            break

    out = io.StringIO()
    writer = csv.writer(
        out, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n"
    )
    writer.writerows(rows)
    return out.getvalue(), changed


def main(argv: list[str] | None = None) -> int:
    """Scrub each TSV given, in place unless --check."""
    ap = argparse.ArgumentParser(prog="scrub", description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument(
        "--check",
        action="store_true",
        help="report what would change and exit non-zero if anything would",
    )
    args = ap.parse_args(argv)

    dirty = 0
    for path in args.paths:
        cleaned, changed = scrub(path.read_text())
        if changed and not args.check:
            path.write_text(cleaned)
        if changed:
            dirty += changed
            print(
                f"{path}: {changed} line(s) {'would be ' if args.check else ''}scrubbed"
            )
    if args.check and dirty:
        print(f"{dirty} line(s) still carry personal data", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

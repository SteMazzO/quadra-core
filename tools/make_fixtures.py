"""Regenerate the rendered test fixtures. Needs Pillow and Tesseract.

Usage:  python3 tools/make_fixtures.py [--out DIR]
"""

from __future__ import annotations

import argparse
import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
LINE_HEIGHT, WIDTH = 34, 1000
FIXTURES = (
    Path(__file__).resolve().parent.parent / "src" / "quadra_core" / "testdata" / "ocr"
)

ITEMS = [
    ("LATTE INTERO 1L", "2", "1,49", "2,98"),
    ("PANE INTEGRALE", "1", "2,30", "2,30"),
    ("PROSC.CRUDO 100G", "1", "4,95", "4,95"),
    ("MELE GOLDEN KG", "1,240", "2,49", "3,09"),
    ("YOGURT MAGRO 4X125", "3", "1,15", "3,45"),
    ("CAFFE MACINATO 250G", "1", "3,79", "3,79"),
    ("OLIO EXTRAV. 1L", "1", "7,90", "7,90"),
    ("PASTA PENNE 500G", "4", "0,89", "3,56"),
]
TOTAL = "32,02"


def synthetic(total: str) -> list[str]:
    """Build a four-column receipt: description, quantity, unit price, total."""
    return [
        "SUPERMERCATO PROVA",
        "VIA ROMA 00 - CITTA",
        "",
        "SCONTRINO N. 0042",
        "",
        *[f"{d:<28}{q:>6} {u:>7} {t:>8}" for d, q, u, t in ITEMS],
        "",
        f"{'TOTALE COMPLESSIVO':<28}{'':>6} {'':>7} {total:>8}",
        "",
        "ARRIVEDERCI E GRAZIE",
    ]


# Transcribed from two real Esselunga receipts, rendered rather than photographed.
ESSELUNGA_A = [
    "* Esselunga S.p.A. *",
    "VIA ROMA 00 - CITTA",
    "P.I.: 04916380159",
    "",
    f"{'':<24}{'EURO':>8}",
    f"{'IL RACC. POWER MIX':<24}{'0,99':>8}",
    f"{'CLEMENTINE FOGLIA':<24}{'0,61':>8}",
    f"{'SACCHETTO COMPOST.LE':<24}{'0,01':>8}",
    f"{'CLEMENTINE IGP 1,5 KG':<24}{'2,98':>8}",
    f"{'ARANCE':<24}{'0,65':>8}",
    f"{'SACCHETTO COMPOST.LE':<24}{'0,01':>8}",
    "",
    f"{'TOTALE EURO':<24}{'5,25':>8} *",
    f"{'Pagamento Carta di Cre':<24}{'5,25':>8}",
    "",
    f"{'RESTO':<24}{'0,00':>8} *",
    f"{'N. CARTA FIDATY:':<24}{'00XX*XX***00':>8}",
    "--- NUOVA RACCOLTA PUNTI ---",
    f"{'SALDO PUNTI':<24}{'0.000':>8}",
    f"{'SALDO PUNTI':<24}{'0.000':>8}",
]

ESSELUNGA_B = [
    "* Esselunga S.p.A. *",
    "VIA ROMA 00 - CITTA",
    "P.I.: 04916380159",
    "",
    f"{'':<24}{'EURO':>8}",
    *[
        f"{'GSK DENT. AQUAFRESH':<24}{p:>8}"
        for p in ("1,19", "0,99", "0,99", "0,99", "0,99")
    ],
    f"{'SCHIACCIATE OLIV&MARI':<24}{'1,29':>8}",
    f"{'CRACKERS OLIVIA&MARIN':<24}{'1,79':>8}",
    f"{'GRISS POMODORO OL&MAR':<24}{'1,49':>8}",
    f"{'    8 x':<24}{'2,99':>8}",
    f"{'BONDUE. COCCOLE SPINA':<24}{'23,92':>8}",
    f"{'SCONTO FIDATY 30%':<24}{'7,20-S':>8}",
    "",
    f"{'TOTALE EURO':<24}{'26,44':>8} *",
    f"{'Pagamento Sconto Euro':<24}{'0,04':>8}",
    f"{'Pagamento Buoni Sconto':<24}{'9,50':>8}",
    f"{'Pagamento Bancomat':<24}{'16,90':>8}",
    "",
    f"{'RESTO':<24}{'0,00':>8} *",
    f"{'N. CARTA FIDATY:':<24}{'00XX*XX***00':>8}",
    "--- NUOVA RACCOLTA PUNTI ---",
    f"{'SALDO PUNTI':<24}{'0.000':>8}",
    f"{'SALDO PUNTI':<24}{'0.000':>8}",
]

# name, lines, contrast (1.0 leaves it alone), blur radius, noise divisor, seed.
FIXTURE_SET = [
    ("synthetic_clean", synthetic(TOTAL), 1.0, 0.0, 0, 7),
    ("synthetic_faded", synthetic(TOTAL), 0.35, 0.9, 12, 11),
    # Printed total 1,00 below the items, so the arithmetic check must fail.
    ("synthetic_unbalanced", synthetic("31,02"), 1.0, 0.0, 0, 7),
    ("esselunga_a", ESSELUNGA_A, 1.0, 0.0, 0, 3),
    ("esselunga_b", ESSELUNGA_B, 1.0, 0.0, 0, 5),
    ("esselunga_b_faded", ESSELUNGA_B, 0.45, 0.8, 16, 13),
]


def render(
    lines: list[str],
    out: Path,
    *,
    contrast: float = 1.0,
    blur: float = 0.0,
    noise: int = 0,
    seed: int = 7,
) -> None:
    """Render receipt text to an image, optionally degraded."""
    random.seed(seed)
    image = Image.new("L", (WIDTH, LINE_HEIGHT * len(lines) + 80), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 26)
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * LINE_HEIGHT), line, font=font, fill=30)

    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        pixels = image.load()
        for _ in range(image.size[0] * image.size[1] // noise):
            x, y = random.randrange(image.size[0]), random.randrange(image.size[1])
            pixels[x, y] = max(0, min(255, pixels[x, y] + random.randint(-45, 45)))
    image.save(out)


def ocr(png: Path, out_base: Path) -> None:
    """Write out_base.tsv, with the flags the parser expects fixtures to carry."""
    subprocess.run(
        [
            "tesseract",
            str(png),
            str(out_base),
            "--oem",
            "1",
            "--psm",
            "4",
            "-l",
            "ita",
            "-c",
            "preserve_interword_spaces=1",
            "tsv",
        ],
        check=True,
        capture_output=True,
        env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"},
    )


def main(argv: list[str] | None = None) -> None:
    """Render every fixture and OCR it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=FIXTURES, help="where to write them")
    args = ap.parse_args(argv)

    assert sum(int(t.replace(",", "")) for *_, t in ITEMS) == int(
        TOTAL.replace(",", "")
    ), "ITEMS no longer sum to TOTAL, fix the constant rather than hardcoding a total"

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, lines, contrast, blur, noise, seed in FIXTURE_SET:
            png = Path(tmp) / f"{name}.png"
            render(lines, png, contrast=contrast, blur=blur, noise=noise, seed=seed)
            ocr(png, args.out / name)
            print(f"wrote {args.out / (name + '.tsv')}")


if __name__ == "__main__":
    main()

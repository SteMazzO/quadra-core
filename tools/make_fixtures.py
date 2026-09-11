"""Regenerate the synthetic test fixtures.

Renders a receipt three ways - clean, degraded, and with a printed total that
does not match its items - then runs Tesseract over each to produce the TSV the
tests parse. Keeping the generator here makes the fixtures reproducible and
records what damage each one represents.

Needs Pillow and Tesseract. The tests need neither.

Usage:  python3 tools/make_fixtures.py
"""

import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

LH, W = 34, 1000
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


def build(total, out, faded=False, seed=7):
    """Render a synthetic receipt and run Tesseract to produce the fixture TSV."""
    random.seed(seed)
    lines = ["SUPERMERCATO PROVA", "VIA ROMA 00 - CITTA", "", "SCONTRINO N. 0042", ""]
    lines += [f"{d:<28}{q:>6} {u:>7} {t:>8}" for d, q, u, t in ITEMS]
    lines += [
        "",
        f"{'TOTALE COMPLESSIVO':<28}{'':>6} {'':>7} {total:>8}",
        "",
        "ARRIVEDERCI E GRAZIE",
    ]
    img = Image.new("L", (W, LH * len(lines) + 80), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26)
    for i, l in enumerate(lines):
        d.text((40, 40 + i * LH), l, font=f, fill=30)
    if faded:
        img = (
            ImageEnhance.Contrast(img)
            .enhance(0.35)
            .filter(ImageFilter.GaussianBlur(0.9))
        )
        px = img.load()
        for _ in range(img.size[0] * img.size[1] // 12):
            x, y = random.randrange(img.size[0]), random.randrange(img.size[1])
            px[x, y] = max(0, min(255, px[x, y] + random.randint(-45, 45)))
    img.save(out)


# Transcribed from two real Esselunga receipts, then rendered rather than
# photographed. They exercise the layout the profile encodes: two columns, a
# quantity line above its item, and a discount signed on the right. They are not
# a substitute for OCR of the real photos, which also carry perspective, thermal
# fade and creases.
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

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ocr"
TOTAL = "32,02"  # the items really do sum to this; do not hardcode a guess

VARIANTS = [
    ("synthetic_clean", TOTAL, False, 7),
    ("synthetic_faded", TOTAL, True, 11),
    # A printed total 1,00 below the true sum, so the arithmetic check has
    # something it has to fail on.
    ("synthetic_unbalanced", "31,02", False, 7),
]


def render(lines, out, faded=False, seed=7):
    """Render arbitrary receipt text, optionally degraded."""
    random.seed(seed)
    img = Image.new("L", (W, LH * len(lines) + 80), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26)
    for i, l in enumerate(lines):
        d.text((40, 40 + i * LH), l, font=f, fill=30)
    if faded:
        img = ImageEnhance.Contrast(img).enhance(0.45)
        img = img.filter(ImageFilter.GaussianBlur(0.8))
        px = img.load()
        for _ in range(img.size[0] * img.size[1] // 16):
            x, y = random.randrange(img.size[0]), random.randrange(img.size[1])
            px[x, y] = max(0, min(255, px[x, y] + random.randint(-40, 40)))
    img.save(out)


ESSELUNGA_VARIANTS = [
    ("esselunga_a", ESSELUNGA_A, False, 3),
    ("esselunga_b", ESSELUNGA_B, False, 5),
    ("esselunga_b_faded", ESSELUNGA_B, True, 13),
]


def main() -> None:
    """Regenerate the synthetic test fixtures."""
    assert sum(int(t.replace(",", "")) for *_, t in ITEMS) == int(
        TOTAL.replace(",", "")
    ), "ITEMS no longer sum to TOTAL, fix the constant rather than hardcoding a total"
    with tempfile.TemporaryDirectory() as tmp:
        for name, total, faded, seed in VARIANTS:
            png = Path(tmp) / f"{name}.png"
            build(total, png, faded=faded, seed=seed)
            subprocess.run(
                [
                    "tesseract",
                    str(png),
                    str(OUT / name),
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
            print(f"wrote {OUT / (name + '.tsv')}")

        for name, lines, faded, seed in ESSELUNGA_VARIANTS:
            png = Path(tmp) / f"{name}.png"
            render(lines, png, faded=faded, seed=seed)
            subprocess.run(
                [
                    "tesseract",
                    str(png),
                    str(OUT / name),
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
            print(f"wrote {OUT / (name + '.tsv')}")


if __name__ == "__main__":
    main()

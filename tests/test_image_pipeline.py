"""The photo path end to end: preprocessing, Tesseract, parsing, repair.

Skipped where Tesseract is not installed, which includes CI.
"""

from __future__ import annotations

import shutil

import pytest

pytest.importorskip("PIL")
pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract is not installed"
)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from quadra_core import parse_image  # noqa: E402
from quadra_core.pipeline import recheck  # noqa: E402
from quadra_core.pipeline.extract import extract  # noqa: E402
from quadra_core.pipeline.lines import drop_speckle, group_lines, load_tsv  # noqa: E402
from quadra_core.pipeline.ocr import run_on_path  # noqa: E402
from quadra_core.profiles import loader  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# 2,30 + 1,49 - 0,45 = 3,34
RECEIPT = [
    "* Esselunga S.p.A. *",
    "VIA ROMA 00 - CITTA",
    "P.I.: 04916380159",
    "",
    f"{'':<24}{'EURO':>8}",
    f"{'PANE INTEGRALE':<24}{'2,30':>8}",
    f"{'LATTE INTERO 1L':<24}{'1,49':>8}",
    f"{'SCONTO FIDATY 30%':<24}{'0,45-S':>8}",
    "",
    f"{'TOTALE EURO':<24}{'3,34':>8}",
]


@pytest.fixture(scope="module")
def photo(tmp_path_factory):
    try:
        font = ImageFont.truetype(FONT, 26)
    except OSError:  # pragma: no cover - font missing on this machine
        pytest.skip("DejaVu font unavailable")
    image = Image.new("L", (1000, 34 * len(RECEIPT) + 80), 255)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(RECEIPT):
        draw.text((40, 40 + i * 34), line, font=font, fill=30)
    path = tmp_path_factory.mktemp("photo") / "receipt.png"
    image.save(path)
    return path


def test_a_rendered_receipt_parses_and_balances(photo):
    document, status = parse_image(photo, profile_id="esselunga")
    assert status == "ok", document["validation"]["warnings"]
    assert [i["line_total_minor"] for i in document["line_items"]] == [230, 149]
    assert [a["amount_minor"] for a in document["adjustments"]] == [-45]
    assert document["totals"] == {
        "items_subtotal_minor": 379,
        "discounts_minor": -45,
        "printed_total_minor": 334,
        "computed_total_minor": 334,
        "balanced": True,
        "delta_minor": 0,
    }


def test_the_shop_is_recognised_without_being_named(photo):
    document, _ = parse_image(photo, profile_id=None)
    assert document["profile"]["id"] == "esselunga"
    assert document["profile"]["match_confidence"] > 0.5


def test_the_ocr_and_the_image_it_read_come_back_for_archiving(photo):
    document, _ = parse_image(photo, profile_id="esselunga")
    assert document["_tsv"].splitlines()[0].startswith("level")
    # Item boxes are in the prepared image's coordinates, so they must fit it.
    prepared = document["_prepared"]
    for item in document["line_items"]:
        left, top, width, height = item["provenance"]["bbox"]
        assert left >= 0 and left + width <= prepared.width
        assert top >= 0 and top + height <= prepared.height
    assert document["ocr"]["engine_version"].startswith("tesseract")
    assert document["ocr"]["preprocessing"], "preprocessing steps were not recorded"


def test_the_price_column_can_be_read_on_its_own(photo):
    """What the repair leans on: the column read apart from the descriptions."""
    result, _report, prepared = run_on_path(photo)
    lines = group_lines(drop_speckle(load_tsv(result.tsv)))
    profile = next(p for p in loader.available() if p.id == "esselunga")
    extraction = extract(lines, profile)

    column = recheck.survey(lines, extraction)
    assert column.left is not None
    readings = recheck.second_opinion(prepared, column, profile)

    for item in extraction.items:
        assert item.line_total_minor in readings[item.line_index]

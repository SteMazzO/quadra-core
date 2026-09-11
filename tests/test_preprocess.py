"""Image preprocessing tests.

Each synthetic image isolates one defect: a known tilt, a known crop, a known
gradient. On a real photo they all appear at once and a failure says nothing
about which step caused it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PIL")

from PIL import Image, ImageDraw, ImageFont

from quadra_core.pipeline.preprocess import (
    MAX_WIDTH,
    MIN_WIDTH,
    Report,
    _otsu,
    add_margin,
    crop_to_paper,
    deskew,
    resize_to_target,
)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def text_page(width=800, height=600, rows=14):
    """A page of evenly spaced dark text rows on white."""
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT, 22)
    except OSError:  # pragma: no cover - font missing on this machine
        pytest.skip("DejaVu font unavailable")
    for i in range(rows):
        draw.text((40, 30 + i * 38), "ESSELUNGA TEST LINE 1,23", font=font, fill=20)
    return image


def on_desk(page, gradient=False, tilt=0.0, desk=95):
    """Place a page on a darker background, optionally lit unevenly and tilted."""
    canvas = Image.new("L", (page.width + 300, page.height + 300), desk)
    canvas.paste(page, (150, 150))
    if tilt:
        canvas = canvas.rotate(
            tilt, resample=Image.Resampling.BICUBIC, fillcolor=desk, expand=True
        )
    if gradient:
        ramp = Image.linear_gradient("L").resize(canvas.size)
        ramp = ramp.point(lambda v: 150 + v // 3)
        dim = canvas.point(lambda v: int(v * 0.55))
        canvas = Image.composite(canvas, dim, ramp)
    return canvas


@pytest.mark.parametrize("tilt", [-3.2, -1.5, 2.0, 4.0])
def test_deskew_recovers_a_known_tilt(tilt):
    """The correction should be the inverse of the applied rotation."""
    skewed = on_desk(text_page(), tilt=tilt)
    report = Report((0, 0), (0, 0))
    deskew(crop_to_paper(skewed), report=report)
    assert report.deskew_degrees == pytest.approx(-tilt, abs=0.6)


def test_deskew_is_not_fooled_by_uniform_borders():
    """Regression guard for a measured failure.

    Scoring row-mean *variance* let the desk corners left by a tilt peak at 0 degrees,
    so deskew never fired and OCR split every line. Adjacent-row contrast fixes it.
    """
    report = Report((0, 0), (0, 0))
    deskew(crop_to_paper(on_desk(text_page(), tilt=-3.2)), report=report)
    assert abs(report.deskew_degrees) > 1.0, "deskew failed to fire on a clear tilt"


def test_crop_survives_a_lighting_gradient():
    """A fixed threshold clipped the dimly lit edge and shaved off line items.

    Otsu picks the paper/desk split from the histogram instead, so the full page
    width survives.
    """
    page = text_page(width=800)
    lit = on_desk(page, gradient=True)
    cropped = crop_to_paper(lit)
    assert cropped.width < lit.width, "nothing was cropped"
    # The page is 800px inside a 1100px canvas; a correct crop keeps nearly all of it.
    assert cropped.width >= page.width * 0.9, f"crop shaved page to {cropped.width}px"


def test_crop_declines_when_there_is_no_background():
    """A scan with no visible desk must not be cropped speculatively."""
    page = text_page()
    assert crop_to_paper(page).size == page.size


def test_otsu_separates_two_populations():
    """Otsu returns the lower class boundary, so pixels ABOVE it are foreground."""
    image = Image.new("L", (100, 100), 30)
    image.paste(Image.new("L", (100, 50), 220), (0, 0))
    assert 30 <= _otsu(image) < 220


@pytest.mark.parametrize("width", [900, 1100, 1400, 3000, 6000])
def test_large_enough_images_land_in_the_target_band(width):
    image = Image.new("L", (width, width * 2), 255)
    result = resize_to_target(image)
    assert MIN_WIDTH <= result.width <= MAX_WIDTH
    assert result.height / result.width == pytest.approx(
        image.height / image.width, rel=0.02
    )


def test_small_images_are_upscaled_but_capped():
    """Below the band we upscale toward it, but never past 2x.

    Past that the extra pixels carry no extra information, so the result may sit
    below MIN_WIDTH by design rather than pretend to a resolution it does not have.
    """
    result = resize_to_target(Image.new("L", (400, 800), 255))
    assert result.width == 800


def test_resize_never_upscales_beyond_double():
    """Upscaling a blurry capture past a point adds pixels, not information."""
    tiny = Image.new("L", (200, 400), 255)
    assert resize_to_target(tiny).width <= 400


def test_margin_is_added_on_all_sides():
    image = Image.new("L", (100, 100), 255)
    padded = add_margin(image, margin=20)
    assert padded.size == (140, 140)
    assert padded.getpixel((0, 0)) == 255

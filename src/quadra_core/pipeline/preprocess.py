"""Cleaning up a photo before OCR.

Pillow only, no OpenCV or NumPy. Each step is separately callable so it can be timed or
turned off; preprocess() runs them in order.

Two rules throughout. Do detection at the smallest resolution that still answers
the question, then apply the result once to the full image. And do not make an
irreversible change on weak evidence: every step returns the image unchanged if
it cannot find what it is looking for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

# Tesseract's LSTM normalises text lines to a fixed internal height, so resolution far
# above its comfort band costs time and buys nothing. ~300 DPI across 80mm thermal
# paper is roughly 950px of receipt width.
TARGET_WIDTH = 1100
MIN_WIDTH = 900
MAX_WIDTH = 1400

# Layout analysis degrades when glyphs touch the image border.
MARGIN = 20

# Detection thumbnail. Small enough to be cheap, large enough to see text rows.
PROBE_WIDTH = 600

# Floor for the JPEG draft decode. draft() takes a minimum size and picks the
# largest power-of-two reduction that still clears it, so this is headroom, not
# a target. The receipt is only part of the frame and gets resampled again after
# cropping; drafting straight down to TARGET_WIDTH left too little to work with
# and cost real accuracy, losing the total line off a 4284px photo.
DRAFT_MIN_WIDTH = TARGET_WIDTH * 3


@dataclass(slots=True)
class Report:
    """Record preprocessing results for debugging and the review UI."""

    source_size: tuple[int, int]
    output_size: tuple[int, int]
    draft_scale: int = 1
    cropped: bool = False
    deskew_degrees: float = 0.0
    flat_fielded: bool = False
    steps: list[str] = field(default_factory=list)


def load_grayscale(path: Path, report: Report | None = None) -> Image.Image:
    """Open an image as grayscale, decoding at reduced scale where possible.

    `draft` downscales during JPEG decode, so the full-size RGB buffer is never
    built. On a phone photo this is the biggest single saving in both time and
    memory.
    """
    image = Image.open(path)
    if report is not None:
        report.source_size = image.size

    # draft() has to come first: it only works while the JPEG is still
    # undecoded, and exif_transpose() decodes it to make its copy.
    original_width = image.width
    if image.format == "JPEG":
        image.draft("L", (DRAFT_MIN_WIDTH, DRAFT_MIN_WIDTH))
        if report is not None and image.width:
            report.draft_scale = max(1, round(original_width / image.width))

    # Phone photos are often stored rotated with an orientation tag. Apply it
    # or sideways text gets OCRed with no warning.
    image = ImageOps.exif_transpose(image)
    return image.convert("L")


def _probe(image: Image.Image) -> Image.Image:
    """Build a thumbnail for detection work."""
    if image.width <= PROBE_WIDTH:
        return image
    height = max(1, round(image.height * PROBE_WIDTH / image.width))
    return image.resize((PROBE_WIDTH, height), Image.Resampling.BILINEAR)


# Row and column statistics run on every deskew candidate, so they are done with
# Pillow rather than by walking pixels in Python. Squeezing an axis to one pixel
# with a BOX filter is exactly a mean over that axis, in C; "F" mode keeps it in
# floats so the scores match what the arithmetic would have given.
def _values(band: Image.Image) -> list[float]:
    """Pixels as a flat list.

    getdata() is deprecated and older Pillow lacks the replacement, so this has
    to work either way.
    """
    flattened = getattr(band, "get_flattened_data", None)
    return list(flattened() if flattened is not None else band.getdata())


def _squeeze(image: Image.Image, size: tuple[int, int]) -> list[float]:
    if not image.width or not image.height:
        return []
    return _values(image.convert("F").resize(size, Image.Resampling.BOX))


def _column_means(image: Image.Image) -> list[float]:
    return _squeeze(image, (image.width, 1))


def _row_means(image: Image.Image) -> list[float]:
    return _squeeze(image, (1, image.height))


def _otsu(image: Image.Image) -> float:
    """Otsu's threshold, the split with the most contrast between the two sides.

    Used instead of a fixed fraction of the range because a hand-held photo nearly
    always has a lighting gradient, and a fixed threshold then clips the dim edge of
    the paper. Measured: it took the left half off every description.
    """
    histogram = image.histogram()[:256]
    total = sum(histogram)
    if not total:
        return 128.0

    sum_all = sum(i * h for i, h in enumerate(histogram))
    sum_bg = weight_bg = 0.0
    best_threshold, best_variance = 128.0, -1.0

    for level, count in enumerate(histogram):
        weight_bg += count
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += level * count
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_threshold, best_variance = float(level), variance
    return best_threshold


# Maxima the same way: fold the image in half against itself, keeping the
# brighter pixel, until the axis is one deep. Halves overlap by a row when the
# count is odd, which a maximum does not mind.
def _column_maxima(image: Image.Image) -> list[float]:
    band = image
    while band.height > 1:
        half = (band.height + 1) // 2
        band = ImageChops.lighter(
            band.crop((0, 0, band.width, half)),
            band.crop((0, band.height - half, band.width, band.height)),
        )
    return [float(v) for v in _values(band)]


def _row_maxima(image: Image.Image) -> list[float]:
    band = image
    while band.width > 1:
        half = (band.width + 1) // 2
        band = ImageChops.lighter(
            band.crop((0, 0, half, band.height)),
            band.crop((band.width - half, 0, band.width, band.height)),
        )
    return [float(v) for v in _values(band)]


def _bright_span(means: list[float], floor: float) -> tuple[int, int] | None:
    """Longest run of values above `floor`, as (start, end_exclusive)."""
    best = current = None
    for i, value in enumerate(means):
        if value >= floor:
            current = i if current is None else current
        elif current is not None:
            if best is None or i - current > best[1] - best[0]:
                best = (current, i)
            current = None
    if current is not None and (
        best is None or len(means) - current > best[1] - best[0]
    ):
        best = (current, len(means))
    return best


def crop_to_paper(image: Image.Image, report: Report | None = None) -> Image.Image:
    """Crop to the receipt when it stands out clearly against its background.

    Detection runs on a thumbnail and only the box is applied to the full image.
    Returns the input unchanged unless the crop looks both substantial and sensible,
    since a wrong crop silently deletes items.
    """
    probe = _probe(image)
    darkest, brightest = probe.getextrema()
    if brightest - darkest < 40:
        return image  # no contrast to work with; a surface photographed alone

    floor = _otsu(probe)
    # Use maxima rather than means. On a tilted page the edge columns are mostly
    # background, so their mean falls below any threshold and the crop eats into
    # the paper: at 3 degrees, 800px of page became a 530px crop. Any column with
    # paper in it has a bright maximum whatever the angle. Blur first so a single
    # noisy pixel cannot stretch the box.
    smoothed = probe.filter(ImageFilter.BoxBlur(2))
    columns = _bright_span(_column_maxima(smoothed), floor)
    rows = _bright_span(_row_maxima(smoothed), floor)
    if not columns or not rows:
        return image

    scale = image.width / probe.width
    left, right = (round(v * scale) for v in columns)
    top, bottom = (round(v * scale) for v in rows)

    # Pad outwards. Cropping into the paper loses a line item for good, while a
    # bit of desk in the image is harmless, so err wide.
    pad_x = round((right - left) * 0.02)
    pad_y = round((bottom - top) * 0.02)
    left, top = max(0, left - pad_x), max(0, top - pad_y)
    right, bottom = min(image.width, right + pad_x), min(image.height, bottom + pad_y)

    # Reject implausible crops: too small to be the receipt, or so large it
    # barely changed anything.
    area = (right - left) * (bottom - top)
    full = image.width * image.height
    if area < 0.15 * full or area > 0.98 * full:
        return image

    if report is not None:
        report.cropped = True
        report.steps.append(f"crop {image.size}->{(right - left, bottom - top)}")
    return image.crop((left, top, right, bottom))


def _central(image: Image.Image, inset: float = 0.12) -> Image.Image:
    """Trim the border, where crop artefacts and desk corners live."""
    width, height = image.size
    return image.crop(
        (
            int(width * inset),
            int(height * inset),
            int(width * (1 - inset)),
            int(height * (1 - inset)),
        )
    )


def _row_profile_score(image: Image.Image) -> float:
    """Score how level the text rows are.

    Measures contrast between neighbouring rows, not variance across all of them.
    Variance is dominated by large flat areas: on a cropped photo the background
    corners peaked at 0 degrees and deskew never fired.
    """
    means = _row_means(_central(image))
    if len(means) < 2:
        return 0.0
    return sum((means[i + 1] - means[i]) ** 2 for i in range(len(means) - 1)) / (
        len(means) - 1
    )


def deskew(
    image: Image.Image,
    *,
    limit: float = 5.0,
    step: float = 0.25,
    report: Report | None = None,
) -> Image.Image:
    """Rotate so text rows run horizontally.

    Tries angles on a thumbnail and scores each one, then applies the winner once to
    the full image.
    """
    probe = _probe(image)
    best_angle, best_score = 0.0, _row_profile_score(probe)

    angle = -limit
    while angle <= limit:
        if angle != 0.0:
            score = _row_profile_score(
                probe.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
            )
            if score > best_score:
                best_angle, best_score = angle, score
        angle += step

    if abs(best_angle) < step / 2:
        return image
    if report is not None:
        report.deskew_degrees = best_angle
        report.steps.append(f"deskew {best_angle:+.1f}deg")
    return image.rotate(
        best_angle, resample=Image.Resampling.BICUBIC, fillcolor=255, expand=True
    )


def flat_field(
    image: Image.Image, *, radius: int = 40, report: Report | None = None
) -> Image.Image:
    """Even out uneven lighting.

    A big box blur estimates the paper background; subtracting it removes the shadow
    gradients that break thresholding. BoxBlur is O(1) per pixel, so the radius is
    almost free.

    Skipped when the lighting is already even, since it costs contrast.
    """
    probe = _probe(image)
    means = _row_means(probe) + _column_means(probe)
    if not means or max(means) - min(means) < 35:
        return image

    # Subtract rather than divide. background - image is ink density, zero on
    # bare paper however brightly lit, so inverting gives an evenly lit page.
    # Dividing needs ImageMath.eval, which Pillow 12 removed and which Ubuntu
    # 24.04's Pillow predates the replacement for. ImageChops works on both.
    background = image.filter(ImageFilter.BoxBlur(radius))
    corrected = ImageChops.invert(ImageChops.subtract(background, image))
    if report is not None:
        report.flat_fielded = True
        report.steps.append(f"flat-field r={radius}")
    return corrected


def resize_to_target(
    image: Image.Image, *, target: int = TARGET_WIDTH, report: Report | None = None
) -> Image.Image:
    """Scale so the receipt is about `target` pixels wide.

    The primary latency lever: Tesseract's cost scales with pixel count, and detail
    beyond its comfort band is wasted. Upscaling beyond the source is never worth it,
    so only shrink, unless the image is too small to read at all.
    """
    if MIN_WIDTH <= image.width <= MAX_WIDTH:
        return image
    # Anything between MIN_WIDTH and MAX_WIDTH already returned above, so this
    # is either too wide or too narrow.
    if image.width > target:
        scale = target / image.width
    else:
        scale = min(target / image.width, 2.0)  # a blurry upscale helps only so much

    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    if report is not None:
        report.steps.append(f"resize {image.size}->{size}")
    return image.resize(size, Image.Resampling.LANCZOS)


def add_margin(image: Image.Image, *, margin: int = MARGIN) -> Image.Image:
    """Pad with white so no glyph touches the border."""
    return ImageOps.expand(image, border=margin, fill=255)


def autocontrast(image: Image.Image) -> Image.Image:
    """Stretch contrast, clipping a little at each end.

    Till-roll printing is faint, so a small clip recovers a lot. We do not threshold
    to black and white here: Tesseract does that itself, tuned for its own recogniser,
    and doing it by hand usually comes out worse.
    """
    return ImageOps.autocontrast(image, cutoff=1)


def preprocess(path: Path) -> tuple[Image.Image, Report]:
    """Full pipeline: photograph in, OCR-ready grayscale out."""
    report = Report(source_size=(0, 0), output_size=(0, 0))
    image = load_grayscale(path, report)
    image = crop_to_paper(image, report)
    image = deskew(image, report=report)
    image = flat_field(image, report=report)
    image = resize_to_target(image, report=report)
    image = autocontrast(image)
    image = add_margin(image)
    report.output_size = image.size
    return image, report

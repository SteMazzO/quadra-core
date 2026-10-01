"""Clean up a photo for OCR, using Pillow only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

# About 300 DPI across 80mm paper. More resolution only costs time.
TARGET_WIDTH = 1100
MIN_WIDTH = 900
MAX_WIDTH = 1400

# Layout analysis degrades when glyphs touch the image border.
MARGIN = 20

# Thumbnail width for detection work.
PROBE_WIDTH = 600

# Smallest size to draft-decode a JPEG at. Going lower loses text after cropping.
DRAFT_MIN_WIDTH = TARGET_WIDTH * 3


@dataclass(slots=True)
class Report:
    """Record preprocessing results for debugging and the review UI."""

    source_size: tuple[int, int]
    output_size: tuple[int, int]
    deskew_degrees: float = 0.0
    steps: list[str] = field(default_factory=list)


def load_grayscale(path: Path, report: Report | None = None) -> Image.Image:
    """Open an image as grayscale, decoding JPEGs at reduced scale."""
    image = Image.open(path)
    if report is not None:
        report.source_size = image.size

    # draft() only works before exif_transpose() decodes the image.
    if image.format == "JPEG":
        image.draft("L", (DRAFT_MIN_WIDTH, DRAFT_MIN_WIDTH))

    image = ImageOps.exif_transpose(image)
    return image.convert("L")


def _probe(image: Image.Image) -> Image.Image:
    """Build a thumbnail for detection work."""
    if image.width <= PROBE_WIDTH:
        return image
    height = max(1, round(image.height * PROBE_WIDTH / image.width))
    return image.resize((PROBE_WIDTH, height), Image.Resampling.BILINEAR)


# Row and column statistics run once per deskew angle, so they stay inside Pillow.
def _values(band: Image.Image) -> list[float]:
    """Pixels as a flat list, on old and new Pillow alike."""
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
    """Otsu's threshold, which copes with a lighting gradient across the photo."""
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


# Maxima by folding the image onto itself, keeping the brighter pixel.
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
    """Crop to the receipt, or return the image unchanged if the paper isn't clear."""
    probe = _probe(image)
    darkest, brightest = probe.getextrema()
    if brightest - darkest < 40:
        return image

    floor = _otsu(probe)
    # Maxima, not means: on a tilted page the edge columns are mostly background.
    smoothed = probe.filter(ImageFilter.BoxBlur(2))
    columns = _bright_span(_column_maxima(smoothed), floor)
    rows = _bright_span(_row_maxima(smoothed), floor)
    if not columns or not rows:
        return image

    scale = image.width / probe.width
    left, right = (round(v * scale) for v in columns)
    top, bottom = (round(v * scale) for v in rows)

    # Err wide: cropping into the paper loses items, a bit of desk is harmless.
    pad_x = round((right - left) * 0.02)
    pad_y = round((bottom - top) * 0.02)
    left, top = max(0, left - pad_x), max(0, top - pad_y)
    right, bottom = min(image.width, right + pad_x), min(image.height, bottom + pad_y)

    area = (right - left) * (bottom - top)
    full = image.width * image.height
    if area < 0.15 * full or area > 0.98 * full:
        return image

    if report is not None:
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
    """Score how level the text rows are, by contrast between neighbouring rows."""
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
    """Rotate so text rows run horizontally, trying angles on a thumbnail."""
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
    """Even out uneven lighting, unless it is already even."""
    probe = _probe(image)
    means = _row_means(probe) + _column_means(probe)
    if not means or max(means) - min(means) < 35:
        return image

    # Subtract rather than divide: ImageMath.eval is gone from Pillow 12.
    background = image.filter(ImageFilter.BoxBlur(radius))
    corrected = ImageChops.invert(ImageChops.subtract(background, image))
    if report is not None:
        report.steps.append(f"flat-field r={radius}")
    return corrected


def resize_to_target(
    image: Image.Image, *, target: int = TARGET_WIDTH, report: Report | None = None
) -> Image.Image:
    """Scale so the receipt is about `target` pixels wide, enlarging at most 2x."""
    if MIN_WIDTH <= image.width <= MAX_WIDTH:
        return image
    if image.width > target:
        scale = target / image.width
    else:
        scale = min(target / image.width, 2.0)

    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    if report is not None:
        report.steps.append(f"resize {image.size}->{size}")
    return image.resize(size, Image.Resampling.LANCZOS)


def add_margin(image: Image.Image, *, margin: int = MARGIN) -> Image.Image:
    """Pad with white so no glyph touches the border."""
    return ImageOps.expand(image, border=margin, fill=255)


def autocontrast(image: Image.Image) -> Image.Image:
    """Stretch contrast, clipping a little at each end; Tesseract does thresholding."""
    return ImageOps.autocontrast(image, cutoff=1)


def preprocess(
    path: Path, *, flat_field_scale: float | None = None
) -> tuple[Image.Image, Report]:
    """Full pipeline: photograph in, OCR-ready grayscale out.

    `flat_field_scale` sets the lighting correction's radius as a fraction of the
    width; a small one irons out creases that the default leaves in.
    """
    report = Report(source_size=(0, 0), output_size=(0, 0))
    image = load_grayscale(path, report)
    image = crop_to_paper(image, report)
    image = deskew(image, report=report)
    if flat_field_scale is None:
        image = flat_field(image, report=report)
    else:
        radius = max(5, round(image.width * flat_field_scale))
        image = flat_field(image, radius=radius, report=report)
    image = resize_to_target(image, report=report)
    image = autocontrast(image)
    image = add_margin(image)
    report.output_size = image.size
    return image, report

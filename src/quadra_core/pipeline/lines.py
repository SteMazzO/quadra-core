"""Turn Tesseract TSV into lines of words, and find the price column."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

# Tesseract uses -1 for page/block/para/line rows, which carry no text.
_NO_TEXT_CONF = -1.0

# Below this slope the page counts as straight.
_SLOPE_IGNORED = 0.005


@dataclass(frozen=True, slots=True)
class Word:
    """One recognised word and where it sits."""

    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float

    @property
    def right(self) -> int:
        """Right edge, in pixels."""
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Bottom edge, in pixels."""
        return self.top + self.height

    @property
    def cx(self) -> float:
        """Horizontal centre."""
        return self.left + self.width / 2

    @property
    def cy(self) -> float:
        """Vertical centre."""
        return self.top + self.height / 2


@dataclass(slots=True)
class Line:
    """Words on one line of the receipt, left to right."""

    words: list[Word]
    index: int = 0

    @property
    def text(self) -> str:
        """Concatenate the words into a single string, with spaces."""
        return " ".join(w.text for w in self.words)

    @property
    def left(self) -> int:
        """Left edge of the line's bounding box, in pixels."""
        return min(w.left for w in self.words)

    @property
    def right(self) -> int:
        """Right edge of the line's bounding box, in pixels."""
        return max(w.right for w in self.words)

    @property
    def top(self) -> int:
        """Top edge of the line's bounding box, in pixels."""
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> int:
        """Bottom edge of the line's bounding box, in pixels."""
        return max(w.bottom for w in self.words)

    @property
    def conf(self) -> float:
        """Mean word confidence, 0 to 100."""
        return sum(w.conf for w in self.words) / len(self.words)


def load_tsv(tsv: str) -> list[Word]:
    """Parse Tesseract TSV into words, skipping blank and structural rows."""
    words: list[Word] = []
    # QUOTE_NONE, or a stray quote in receipt text swallows the rest of the row.
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row["conf"])
            if conf <= _NO_TEXT_CONF:
                continue
            words.append(
                Word(
                    text=text,
                    left=int(row["left"]),
                    top=int(row["top"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    conf=conf,
                )
            )
        except (KeyError, ValueError):
            continue
    return words


def median_glyph_height(words: list[Word]) -> float:
    """Median word height. Other tolerances are a fraction of this."""
    if not words:
        return 0.0
    heights = sorted(w.height for w in words)
    return float(heights[len(heights) // 2])


def drop_speckle(words: list[Word], *, min_height_ratio: float = 0.4) -> list[Word]:
    """Drop specks of noise, judged by height since their confidence is unreliable."""
    if not words:
        return []
    threshold = median_glyph_height(words) * min_height_ratio
    kept = [w for w in words if w.height >= threshold]
    # Recompute once: the specks drag the median down.
    if kept and len(kept) < len(words):
        threshold = median_glyph_height(kept) * min_height_ratio
        kept = [w for w in kept if w.height >= threshold]
    return kept


# Low, so short footer rows like 'TOTALE EURO 26,44' still count towards the slope.
_SLOPE_MIN_WORDS = 2
_SLOPE_MIN_SPREAD = 0.10

# How many nearby rows vote on the slope at a given height.
_SLOPE_NEIGHBOURS = 5


def _fit_slope(line: Line) -> float | None:
    """Least-squares dy/dx through one line's word centres, or None if too short."""
    if len(line.words) < _SLOPE_MIN_WORDS:
        return None
    xs = [w.cx for w in line.words]
    ys = [w.cy for w in line.words]
    spread = max(xs) - min(xs)
    if spread <= 0:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return covariance / variance


def _slope_samples(lines: list[Line], page_width: float) -> list[tuple[float, float]]:
    """(y, slope) for every row long enough to say which way it runs, sorted by y."""
    samples = []
    for line in lines:
        xs = [w.cx for w in line.words]
        if not xs or (max(xs) - min(xs)) < page_width * _SLOPE_MIN_SPREAD:
            continue
        slope = _fit_slope(line)
        if slope is not None:
            samples.append((sum(w.cy for w in line.words) / len(line.words), slope))
    samples.sort()
    return samples


def _median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def local_slope(samples: list[tuple[float, float]], y: float) -> float:
    """Median slope of the rows nearest this height, so a curled page is followed."""
    if not samples:
        return 0.0
    nearest = sorted(samples, key=lambda s: abs(s[0] - y))[:_SLOPE_NEIGHBOURS]
    return _median([slope for _y, slope in nearest])


def _cluster(
    words: list[Word], limit: float, samples: list[tuple[float, float]]
) -> list[Line]:
    """Group words whose centres agree once the page's slope is taken out."""

    def level(word: Word) -> float:
        # Where this word's row would cross the left edge, following the local slope.
        return word.cy - local_slope(samples, word.cy) * word.cx

    lines: list[Line] = []
    centres: list[float] = []
    for word in sorted(words, key=lambda w: (level(w), w.left)):
        here = level(word)
        for index in range(len(lines) - 1, -1, -1):
            if abs(here - centres[index]) <= limit:
                line = lines[index]
                # Running mean, so one stray box does not drag the centre.
                centres[index] = (centres[index] * len(line.words) + here) / (
                    len(line.words) + 1
                )
                line.words.append(word)
                break
        else:
            lines.append(Line(words=[word]))
            centres.append(here)

    for i, line in enumerate(lines):
        line.words.sort(key=lambda w: w.left)
        line.index = i
    return lines


def group_lines(words: list[Word], *, tolerance: float = 0.6) -> list[Line]:
    """Group words into lines by vertical centre, following the page's slope."""
    if not words:
        return []

    limit = median_glyph_height(words) * tolerance
    # The first pass only finds which way the rows slope.
    first = _cluster(words, limit, [])

    page_width = max(w.right for w in words) - min(w.left for w in words)
    samples = _slope_samples(first, page_width)
    if not samples or max(abs(s) for _y, s in samples) < _SLOPE_IGNORED:
        return first
    return _cluster(words, limit, samples)


@dataclass(slots=True)
class Block:
    """Horizontal extent of the text, so x positions can be scale independent."""

    left: int
    right: int

    @property
    def width(self) -> int:
        """Width in pixels, at least 1 to avoid divide-by-zero."""
        return max(1, self.right - self.left)

    def fraction(self, x: float) -> float:
        """Pixel x as a fraction of the block width, clamped to 0..1."""
        return min(1.0, max(0.0, (x - self.left) / self.width))


def text_block(lines: list[Line]) -> Block:
    """Horizontal extent of the text on the page."""
    if not lines:
        return Block(0, 1)
    return Block(min(l.left for l in lines), max(l.right for l in lines))


def cluster_1d(values: list[float], *, gap: float) -> list[list[float]]:
    """Split sorted values wherever the step between them exceeds `gap`."""
    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] > gap:
            clusters.append([])
        clusters[-1].append(value)
    return clusters


def discover_price_column(
    lines: list[Line],
    price_re: re.Pattern[str],
    *,
    gap: float = 0.08,
) -> tuple[float, float] | None:
    """Find the price column from the right edges of price-like tokens.

    Returns (min, max) as a fraction of block width, or None if not found.
    """
    block = text_block(lines)
    edges: list[float] = []
    for line in lines:
        matches = [w for w in line.words if price_re.fullmatch(w.text)]
        if matches:
            edges.append(block.fraction(matches[-1].right))

    if not edges:
        return None

    clusters = cluster_1d(edges, gap=gap)
    largest = max(clusters, key=len)
    # One stray price-shaped token does not make a column.
    if len(largest) < 2:
        return None
    return (min(largest), max(largest))

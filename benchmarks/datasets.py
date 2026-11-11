"""Load public receipt datasets into one shape the benchmark can score.

Nothing is redistributed: each dataset is fetched from its own home into a
cache directory, under that dataset's own licence.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from scoring import Item, Sample, Truth, digits, quantity

HUB = "https://huggingface.co/datasets"

# CORD v2 (CC BY 4.0): Indonesian shop and restaurant receipts, 100 per file here.
CORD_FILES = {
    "train": "train-00000-of-00004-b4aaeceff1d90ecb.parquet",
    "validation": "validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    "test": "test-00000-of-00001-9c204eb3f4e11791.parquet",
}


def _fetch(url: str, target: Path) -> Path:
    """Download a file once."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.rename(target)
    return target


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value) -> str:
    """CORD repeats a key as a list when a field spans several lines."""
    return " ".join(str(part) for part in _as_list(value))


def _cord_items(menu) -> list[Item]:
    """Every priced line: the dishes, and the priced extras printed under them."""
    items = []
    for entry in _as_list(menu):
        if not isinstance(entry, dict):
            continue
        for line in (entry, *_as_list(entry.get("sub"))):
            total = digits(_text(line.get("price"))) if isinstance(line, dict) else None
            if total is not None:
                items.append(
                    Item(_text(line.get("nm")), total, quantity(_text(line.get("cnt"))))
                )
    return items


def _cord_words(valid_line: list[dict]) -> list[tuple[str, int, int, int, int]]:
    words = []
    for line in valid_line:
        for word in line["words"]:
            xs = [word["quad"][f"x{i}"] for i in range(1, 5)]
            ys = [word["quad"][f"y{i}"] for i in range(1, 5)]
            if word["text"].strip():
                words.append(
                    (
                        word["text"],
                        min(xs),
                        min(ys),
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                    )
                )
    return words


def cord(cache: Path, split: str) -> Iterator[Sample]:
    """Yield the receipts of one CORD split."""
    import pyarrow.parquet as pq  # noqa: PLC0415 - only the benchmark needs it

    name = CORD_FILES[split]
    url = f"{HUB}/naver-clova-ix/cord-v2/resolve/main/data/{name}"
    for index, row in enumerate(
        pq.read_table(_fetch(url, cache / "cord" / name)).to_pylist()
    ):
        truth = json.loads(row["ground_truth"])
        parsed = truth["gt_parse"]
        sub_total, total = parsed.get("sub_total") or {}, parsed.get("total") or {}
        if not isinstance(sub_total, dict) or not isinstance(total, dict):
            continue
        yield Sample(
            id=f"cord-{split}-{index}",
            image=row["image"]["bytes"],
            words=_cord_words(truth["valid_line"]),
            truth=Truth(
                items=_cord_items(parsed.get("menu")),
                total=digits(_text(total.get("total_price"))),
                subtotal=digits(_text(sub_total.get("subtotal_price"))),
                tax=digits(_text(sub_total.get("tax_price"))),
            ),
        )


DATASETS: dict[str, Callable[[Path, str], Iterator[Sample]]] = {"cord": cord}

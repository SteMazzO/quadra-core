"""Load public receipt datasets into one shape the benchmark can score.

Nothing is redistributed: each dataset is fetched from its own home into a
cache directory, under that dataset's own licence.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterator
from decimal import Decimal
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


def _cents(value) -> int | None:
    return None if value is None else round(float(value) * 100)


def _day_first(text: str | None) -> str | None:
    """'01/06/2022' as Italy, Germany, France and Britain write it -> '2022-06-01'."""
    parts = (text or "").replace(".", "/").replace("-", "/").split("/")
    if len(parts) != 3:
        return None
    day, month, year = parts
    return f"{year}-{month:0>2}-{day:0>2}"


SYNTHETIC_FILE = "eval-00000-of-00002.parquet"


def synthetic(cache: Path, split: str) -> Iterator[Sample]:
    """Yield rendered receipts of one locale: IT, DE, FR, UK or US.

    albertobarnabo/synthetic-receipts-ocr (Apache-2.0). Every label is exact, but
    the receipts are generated from a few templates: use it to catch regressions,
    never to claim accuracy on real paper.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    repo = f"{HUB}/albertobarnabo/synthetic-receipts-ocr"
    url = f"{repo}/resolve/main/data/{SYNTHETIC_FILE}"
    table = pq.read_table(_fetch(url, cache / "synthetic" / SYNTHETIC_FILE))
    for row in table.to_pylist():
        if row["locale"] != split.upper():
            continue
        fields = json.loads(row["fields"])
        taxes = fields.get("taxes") or []
        yield Sample(
            id=f"synthetic-{row['id']}",
            image=row["image_photo"]["bytes"],
            words=[
                (w["text"], w["x0"], w["y0"], w["x1"] - w["x0"], w["y1"] - w["y0"])
                for w in json.loads(row["words"])
            ],
            truth=Truth(
                items=[
                    Item(line["name"], _cents(line["total"]), Decimal(str(line["qty"])))
                    for line in fields["lines"]
                ],
                total=_cents(fields["total"]),
                subtotal=_cents(fields.get("subtotal")),
                tax=sum(_cents(t["amount"]) for t in taxes) if taxes else None,
                merchant=fields.get("merchant"),
                date=None if split.upper() == "US" else _day_first(fields.get("date")),
            ),
        )


OPEN_PRICES = "https://prices.openfoodfacts.org/img"


def openprices(cache: Path, split: str) -> Iterator[Sample]:
    """Yield real receipt photos from one country of Open Prices, e.g. IT.

    Open Food Facts volunteers photograph a receipt and type in some of its
    prices (data ODbL, images CC BY-SA 4.0). Usually only one or two prices are
    entered and the total almost never, so this measures whether known prices
    are found on real paper from many chains, not whether a receipt is complete.
    """
    import pyarrow.compute as pc  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    url = f"{HUB}/openfoodfacts/open-prices/resolve/main/prices.parquet"
    table = pq.read_table(_fetch(url, cache / "openprices" / "prices.parquet"))
    country = pc.equal(table["location_osm_address_country_code"], split.upper())
    rows = table.filter(pc.and_(country, pc.equal(table["proof_type"], "RECEIPT")))

    receipts: dict[int, list[dict]] = {}
    for row in rows.to_pylist():
        receipts.setdefault(row["proof_id"], []).append(row)
    for proof, prices in sorted(receipts.items()):
        first = prices[0]
        # A price per kilo, or for several units, is not the line's printed total.
        items = [
            Item(row["product_name"] or "", _cents(row["price"]))
            for row in prices
            if row["price"]
            and not row["price_per"]
            and row["receipt_quantity"] in {None, 1}
        ]
        if not items:
            continue
        name = first["proof_file_path"]
        photo = _fetch(f"{OPEN_PRICES}/{name}", cache / "openprices" / "img" / name)
        yield Sample(
            id=f"openprices-{split.lower()}-{proof}",
            image=photo.read_bytes(),
            truth=Truth(
                items=items,
                total=_cents(first["proof_receipt_price_total"]),
                date=str(first["proof_date"]) if first["proof_date"] else None,
                partial=True,
            ),
        )


DATASETS: dict[str, Callable[[Path, str], Iterator[Sample]]] = {
    "cord": cord,
    "openprices": openprices,
    "synthetic": synthetic,
}

# Quadra core

[![License: MIT](https://img.shields.io/github/license/SteMazzO/quadra-core?logo=Open+Source+Initiative)](https://opensource.org/license/mit)
[![PyPI version](https://img.shields.io/pypi/v/quadra-core?logo=pypi)](https://pypi.org/project/quadra-core/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/quadra-core?logo=pypi)](https://pypi.org/project/quadra-core/)
[![codecov](https://codecov.io/github/SteMazzO/quadra-core/coverage.svg?branch=main)](https://codecov.io/github/SteMazzO/quadra-core?branch=main)

Reads a shop or restaurant receipt from a photo, from any shop and any country. No LLM, no network calls, no per-shop configuration: Tesseract does the OCR, then geometry and arithmetic do the rest.

```python
from quadra_core import parse_image

receipt = parse_image("receipt.jpg")

receipt.status  # "ok", "review" or "failed"
receipt.document  # the items, discounts and totals, ready for JSON
```

    pip install quadra-core
    sudo apt install tesseract-ocr tesseract-ocr-ita

## One status to look at

| status | meaning |
|---|---|
| `ok` | Every line adds up to the receipt's total, and nothing on the receipt contradicts it. Nothing to check. |
| `review` | Usable, but a person should check the fields listed in `document["review"]`. Only those. |
| `failed` | No items were found. |

Each review entry names the field, the item it concerns, and what went wrong, in simple terms:

```json
"review": [
  {"code": "total_mismatch", "field": "total", "item": null,
   "message": "The lines add up to 90,09, but the total is 101,25."},
  {"code": "price_missing", "field": "price", "item": null,
   "message": "No price read for “BANANE”."},
  {"code": "price_unconfirmed", "field": "price", "item": 3,
   "message": "Read as 1,98; 1,95 would make the receipt add up."}
]
```

Items with an entry also carry `"needs_review": true`, so a review screen can highlight just those rows. Everything else in `warnings` is for debugging and never changes the status.

## How it decides

Nothing is configured per shop. The parser works out from the receipt itself how money is printed (`1.234,56`, `1,234.56`, or whole numbers like `12,000`), where the price column is, and whether quantities are printed before the name, before the price, or on a line of their own. The footer is found by the words receipts use for it (total, subtotal, tax, service, discount, cash, change) in the main languages written in Latin script.

The line totals plus the discounts have to equal the total, so the parser can check its own work. Tax is printed the same way whether prices include it or not, so the arithmetic decides: a tax or service line counts as added on top only if the total needs it.

When they don't, it reads the price column again on its own, three different ways, and looks for the prices that make the receipt add up. The arithmetic only *proposes*. A correction is accepted without review only when both of these hold:

- the total was confirmed by a second reading (the printed total and the payment line agree, or you typed it in), and
- every re-read of that price agrees on the new value.

Anything less is kept as a suggestion, and that line goes to review. Prices are never adjusted towards a total that the payment line contradicts.

## When the total is unreadable

Type it in, and every line is checked against it:

```python
receipt = parse_image("receipt.jpg", total=10125)  # minor units: 101,25
```

## Command line

    quadra-core parse receipt.jpg                 # prints the JSON
    quadra-core parse receipt.jpg --total 101,25

The exit code is the status: 0 ok, 1 review, 2 failed, 3 error.

## The document

| key | what it holds |
|---|---|
| `status`, `review` | as above |
| `line_items` | description, quantity (a decimal string), unit price, line total, the discount printed under it, VAT code, and where it sits on the photo |
| `adjustments` | the discounts printed among the items, each pointing at the item it applies to |
| `charges` | tax, service, rounding and footer discounts, each with `applied`: whether the total includes it on top of the items |
| `totals` | the items' sum, discounts, the printed subtotal and tax, the computed total, the total it was checked against and where that came from (`printed`, `payments` or `supplied`), and the difference |
| `decimal_places` | how many digits of every amount are decimals on this receipt: 2 for euros, 0 for rupiah |
| `warnings`, `ocr` | debugging detail |

Money is always in integer minor units (cents, or whole units where `decimal_places` is 0). A field the parser could not read is `null`; it is never guessed. [The JSON Schema](src/quadra_core/schema/receipt-3.0.0.schema.json) is exported as `quadra_core.SCHEMA_PATH`.

`receipt.tsv` and `receipt.image` hold the OCR and the prepared photo that item boxes refer to, if you want to archive them. You can re-parse the TSV later with `parse_tsv`, without the photo.

## Accuracy

`benchmarks/` scores the parser on public datasets. On [CORD](https://github.com/clovaai/cord) (Indonesian shop and restaurant receipts), fed the dataset's own transcription so that only the parsing is measured:

    uv run --group bench python benchmarks/run.py cord --split validation

| CORD | validation, 100 receipts (tuned on) | train sample, 200 receipts (not tuned on) |
|---|---|---|
| total read exactly | 95% | 85% |
| line totals, F1 | 0.97 | 0.94 |
| receipts with every amount right | 84% | 80% |
| receipts called `ok` | 80% | 75% |
| ...of which every amount is right | 97.5% | 98.7% |

The two validation receipts called `ok` but scored wrong are annotation slips in the dataset. Reading photos end to end adds the OCR engine's own errors on top; those numbers are not measured yet.

Two more sets cover Italy:

    uv run --group bench python benchmarks/run.py synthetic --split IT
    uv run --group bench python benchmarks/run.py openprices --split IT --engine photo

- `synthetic` is rendered receipts with exact labels (also `DE`, `FR`, `UK`, `US`). On its 155 Italian receipts 98% come out with every amount right. They are generated from a few templates, so this catches regressions; it says little about real paper.
- `openprices` is 383 real photos from Italian chains (Poli, MD, Esselunga, Lidl, Conad, Coop and more), where volunteers typed in a few of each receipt's prices. No public set of real Italian receipts has every line annotated, so this scores whether those known prices are found.

`benchmarks/show.py` prints what the parser saw on any receipt it got wrong. Datasets are downloaded from their own homes into `~/.cache/quadra-bench`.

## Development

    uv run pytest

## Licence

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
